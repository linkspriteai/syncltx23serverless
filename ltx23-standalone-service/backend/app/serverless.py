from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

import requests
import runpod

from app.main import (  # noqa: E402
    SyncReferenceAvRequest,
    _artifacts_dir,
    _best_effort_cuda_cleanup,
    normalize_frame_count,
    now_utc,
    on_startup,
    runtime,
    settings,
)

logger = logging.getLogger("ltx23-serverless-worker")
logging.basicConfig(level=logging.INFO)

_BOOT_LOCK = threading.Lock()
_BOOTED = False


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_frame_count_for_request(raw_count: int, *, hard_max: int) -> int:
    return normalize_frame_count(raw_count, hard_max)


def _mktemp_path(prefix: str, suffix: str) -> Path:
    fd, raw = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(raw)


def _guess_suffix(path_or_url: str, default_suffix: str) -> str:
    suffix = Path(path_or_url).suffix.strip()
    if not suffix:
        return default_suffix
    if len(suffix) > 12:
        return default_suffix
    return suffix


def _download_to_temp(url: str, *, prefix: str, default_suffix: str, max_mb: int, timeout_s: int = 180) -> Path:
    suffix = _guess_suffix(url, default_suffix)
    destination = _mktemp_path(prefix=prefix, suffix=suffix)
    max_bytes = int(max_mb) * 1024 * 1024
    total = 0
    try:
        with requests.get(url, stream=True, timeout=timeout_s) as resp:
            resp.raise_for_status()
            with destination.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"download exceeds limit {max_mb}MB: {url}")
                    f.write(chunk)
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _decode_b64_to_temp(data: str, *, prefix: str, suffix: str, max_mb: int) -> Path:
    payload = str(data or "").strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1].strip()

    raw = base64.b64decode(payload, validate=False)
    max_bytes = int(max_mb) * 1024 * 1024
    if len(raw) > max_bytes:
        raise ValueError(f"base64 payload exceeds limit {max_mb}MB")

    destination = _mktemp_path(prefix=prefix, suffix=suffix)
    destination.write_bytes(raw)
    return destination


def _resolve_input_file(
    *,
    job_input: dict[str, Any],
    path_key: str,
    url_key: str,
    b64_key: str,
    required: bool,
    prefix: str,
    default_suffix: str,
    max_mb: int,
    temp_paths: list[Path],
) -> Path | None:
    local_path_value = str(job_input.get(path_key) or "").strip()
    if local_path_value:
        path = Path(local_path_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path_key} file not found: {path}")
        return path

    url_value = str(job_input.get(url_key) or "").strip()
    if url_value:
        downloaded = _download_to_temp(
            url_value,
            prefix=prefix,
            default_suffix=default_suffix,
            max_mb=max_mb,
        )
        temp_paths.append(downloaded)
        return downloaded

    b64_value = job_input.get(b64_key)
    if b64_value:
        decoded = _decode_b64_to_temp(
            str(b64_value),
            prefix=prefix,
            suffix=default_suffix,
            max_mb=max_mb,
        )
        temp_paths.append(decoded)
        return decoded

    if required:
        raise ValueError(
            f"Missing required input. Provide one of: {path_key}, {url_key}, or {b64_key}"
        )
    return None


def _upload_output(video_path: Path, upload_url: str, upload_headers: dict[str, Any] | None = None) -> dict[str, Any]:
    headers: dict[str, str] = {"Content-Type": "video/mp4"}
    for key, value in (upload_headers or {}).items():
        if value is None:
            continue
        headers[str(key)] = str(value)

    with video_path.open("rb") as f:
        resp = requests.put(upload_url, data=f, headers=headers, timeout=600)
    if resp.status_code >= 400:
        raise RuntimeError(f"output upload failed status={resp.status_code} body={resp.text[:500]}")
    return {"status_code": resp.status_code, "ok": True}


def _ensure_booted() -> None:
    global _BOOTED
    if _BOOTED:
        return
    with _BOOT_LOCK:
        if _BOOTED:
            return
        on_startup()
        _BOOTED = True


def _extract_job_input(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    raise TypeError("Runpod job input must be a JSON object")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    temp_paths: list[Path] = []
    output_path: Path | None = None
    try:
        _ensure_booted()
        job_input = _extract_job_input(job)

        if _to_bool(job_input.get("healthcheck"), False):
            return {
                "status": "ok",
                "service": settings.service_name,
                "mode": "serverless",
                "time": now_utc().isoformat(),
                "capabilities": {
                    "standard_av": bool(settings.enable_standard_av),
                    "sync_av": bool(settings.enable_sync_av),
                },
            }

        if not bool(settings.enable_sync_av):
            raise RuntimeError("sync_av_disabled (set LTX_ENABLE_SYNC_AV=true)")

        prompt = str(job_input.get("prompt") or "").strip()
        if len(prompt) < 3:
            raise ValueError("prompt must have at least 3 characters")

        fps = max(1, min(60, _to_int(job_input.get("fps"), 24)))
        duration_s = _to_float(job_input.get("duration"), 0.0)
        frame_count_in = _to_int(job_input.get("frame_count"), 0)
        if frame_count_in <= 0:
            frame_count_in = int(round(max(duration_s, 0.0) * float(fps)))
            if frame_count_in <= 0:
                frame_count_in = 97
        frame_count = _normalize_frame_count_for_request(frame_count_in, hard_max=int(settings.frame_count_max))

        width = max(128, _to_int(job_input.get("width"), 768))
        height = max(128, _to_int(job_input.get("height"), 512))
        seed_raw = job_input.get("seed")
        seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None

        reference_video_path = _resolve_input_file(
            job_input=job_input,
            path_key="reference_video_path",
            url_key="reference_video_url",
            b64_key="reference_video_base64",
            required=True,
            prefix="ltx_srv_ref_video_",
            default_suffix=".mp4",
            max_mb=_to_int(job_input.get("max_reference_video_mb"), 600),
            temp_paths=temp_paths,
        )
        reference_image_path = _resolve_input_file(
            job_input=job_input,
            path_key="reference_image_path",
            url_key="reference_image_url",
            b64_key="reference_image_base64",
            required=False,
            prefix="ltx_srv_ref_image_",
            default_suffix=".png",
            max_mb=_to_int(job_input.get("max_reference_image_mb"), 40),
            temp_paths=temp_paths,
        )
        override_audio_path = _resolve_input_file(
            job_input=job_input,
            path_key="override_audio_path",
            url_key="override_audio_url",
            b64_key="override_audio_base64",
            required=False,
            prefix="ltx_srv_override_audio_",
            default_suffix=".wav",
            max_mb=_to_int(job_input.get("max_override_audio_mb"), 80),
            temp_paths=temp_paths,
        )

        preset = str(job_input.get("preset") or "quality").strip().lower()
        if preset not in {"fast", "quality"}:
            preset = "quality"

        negative_prompt_raw = job_input.get("negative_prompt")
        negative_prompt: str | None = None
        if negative_prompt_raw is not None:
            candidate = str(negative_prompt_raw).strip()
            if candidate:
                negative_prompt = candidate

        metadata = job_input.get("metadata") if isinstance(job_input.get("metadata"), dict) else {}

        request = SyncReferenceAvRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            preset=preset,  # type: ignore[arg-type]
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            seed=seed,
            model_variant=str(job_input.get("model_variant") or settings.default_model_variant),
            enhance_prompt=_to_bool(job_input.get("enhance_prompt"), False),
            metadata=metadata,
            reference_video_upload_id="runpod-reference-video",
            reference_image_upload_id=("runpod-reference-image" if reference_image_path is not None else None),
            override_audio_upload_id=("runpod-override-audio" if override_audio_path is not None else None),
            use_video_audio=_to_bool(job_input.get("use_video_audio"), True),
            conditioning_strength=max(0.0, min(1.0, _to_float(job_input.get("conditioning_strength"), 0.85))),
        )

        artifacts_dir = _artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"runpod_sync_{job.get('id', 'job')}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = artifacts_dir / output_filename

        result_meta = runtime.generate_sync_reference_av(
            request=request,
            reference_video_path=reference_video_path,
            reference_image_path=reference_image_path,
            override_audio_path=override_audio_path,
            output_path=output_path,
        )

        response: dict[str, Any] = {
            "status": "succeeded",
            "job_id": str(job.get("id") or ""),
            "time": now_utc().isoformat(),
            "metadata": result_meta,
            "output": {
                "filename": output_filename,
                "size_bytes": output_path.stat().st_size,
                "local_path": str(output_path),
            },
        }

        upload_url = str(job_input.get("output_upload_url") or "").strip()
        if upload_url:
            upload_headers = job_input.get("output_upload_headers")
            if upload_headers is not None and not isinstance(upload_headers, dict):
                raise TypeError("output_upload_headers must be a JSON object if provided")
            upload_result = _upload_output(
                output_path,
                upload_url=upload_url,
                upload_headers=upload_headers if isinstance(upload_headers, dict) else None,
            )
            response["output"]["uploaded"] = upload_result
            response["output"]["output_upload_url"] = upload_url

        return_base64 = _to_bool(job_input.get("return_base64"), False)
        if return_base64:
            max_return_mb = _to_int(job_input.get("max_return_base64_mb"), 40)
            size_bytes = output_path.stat().st_size
            if size_bytes > max_return_mb * 1024 * 1024:
                raise ValueError(
                    f"Output video is {size_bytes} bytes and exceeds return_base64 limit of {max_return_mb}MB"
                )
            response["output"]["video_base64"] = base64.b64encode(output_path.read_bytes()).decode("ascii")

        cleanup_output = _to_bool(job_input.get("cleanup_output"), False)
        if cleanup_output and output_path.exists():
            output_path.unlink(missing_ok=True)
            response["output"]["local_path"] = None
            response["output"]["cleaned_up"] = True

        return response
    except Exception as exc:
        logger.error("Serverless job failed: %s", exc)
        logger.debug(traceback.format_exc())
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
            "time": now_utc().isoformat(),
        }
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        _best_effort_cuda_cleanup()


if __name__ == "__main__":
    logger.info("Starting Runpod serverless worker")
    runpod.serverless.start({"handler": handler})

