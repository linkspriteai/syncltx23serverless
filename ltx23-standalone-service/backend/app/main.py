from __future__ import annotations

import logging
import mimetypes
import os
import subprocess
import threading
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Keep torch.compile disabled to match HF Space runtime behavior.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("ltx23-standalone-service")
logging.basicConfig(level=logging.INFO)

SYNC_SM80_FALLBACK_REPO = "Lightricks/LTX-2.3-fp8"
SYNC_SM80_FALLBACK_CHECKPOINT = "ltx-2.3-22b-distilled-fp8.safetensors"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field(default="ltx23-standalone-service", alias="SERVICE_NAME")
    service_host: str = Field(default="0.0.0.0", alias="SERVICE_HOST")
    service_port: int = Field(default=8080, alias="SERVICE_PORT")
    service_api_key: str = Field(default="replace-me", alias="SERVICE_API_KEY")

    data_dir: str = Field(default="./data", alias="DATA_DIR")
    upload_max_mb: int = Field(default=20, alias="UPLOAD_MAX_MB")
    frame_count_max: int = Field(default=257, alias="FRAME_COUNT_MAX")
    enable_standard_av: bool = Field(default=True, alias="LTX_ENABLE_STANDARD_AV")
    enable_sync_av: bool = Field(default=True, alias="LTX_ENABLE_SYNC_AV")

    hf_token: str = Field(default="", alias="HF_TOKEN")
    ltx_repo: str = Field(default="Lightricks/LTX-2.3", alias="LTX_MODEL_REPO")
    ltx_checkpoint_repo: str = Field(default="", alias="LTX_MODEL_CHECKPOINT_REPO")
    gemma_repo: str = Field(
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        alias="LTX_GEMMA_REPO",
    )

    distilled_checkpoint_name: str = Field(
        default="ltx-2.3-22b-distilled.safetensors",
        alias="LTX_MODEL_CHECKPOINT_DISTILLED",
    )
    full_checkpoint_name: str = Field(
        default="ltx-2.3-22b-dev.safetensors",
        alias="LTX_MODEL_CHECKPOINT_FULL",
    )
    distilled_lora_name: str = Field(
        default="ltx-2.3-22b-distilled-lora-384.safetensors",
        alias="LTX_MODEL_DISTILLED_LORA",
    )
    spatial_upsampler_name: str = Field(
        default="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        alias="LTX_MODEL_SPATIAL_UPSAMPLER",
    )

    default_model_variant: str = Field(default="distilled", alias="LTX_DEFAULT_MODEL_VARIANT")
    distilled_use_fp8_cast: bool = Field(default=True, alias="LTX_DISTILLED_USE_FP8_CAST")
    full_use_fp8_cast: bool = Field(default=False, alias="LTX_FULL_USE_FP8_CAST")
    full_num_inference_steps: int = Field(default=30, alias="LTX_FULL_NUM_INFERENCE_STEPS")

    video_cfg_scale: float = Field(default=3.0, alias="LTX_FULL_VIDEO_CFG_SCALE")
    video_stg_scale: float = Field(default=1.0, alias="LTX_FULL_VIDEO_STG_SCALE")
    video_rescale_scale: float = Field(default=0.7, alias="LTX_FULL_VIDEO_RESCALE_SCALE")
    video_modality_scale: float = Field(default=3.0, alias="LTX_FULL_VIDEO_MODALITY_SCALE")
    audio_cfg_scale: float = Field(default=7.0, alias="LTX_FULL_AUDIO_CFG_SCALE")
    audio_stg_scale: float = Field(default=1.0, alias="LTX_FULL_AUDIO_STG_SCALE")
    audio_rescale_scale: float = Field(default=0.7, alias="LTX_FULL_AUDIO_RESCALE_SCALE")
    audio_modality_scale: float = Field(default=3.0, alias="LTX_FULL_AUDIO_MODALITY_SCALE")

    preload_on_start: bool = Field(default=False, alias="LTX_PRELOAD_ON_START")
    local_checkpoints_dir: str = Field(
        default="/data/models/checkpoints",
        alias="LTX_LOCAL_CHECKPOINTS_DIR",
    )
    sync_ic_lora_repo: str = Field(
        default="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        alias="LTX_SYNC_IC_LORA_REPO",
    )
    sync_ic_lora_filename: str = Field(
        default="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        alias="LTX_SYNC_IC_LORA_FILENAME",
    )
    sync_reference_downscale_factor: int = Field(
        default=2,
        alias="LTX_SYNC_REFERENCE_DOWNSCALE_FACTOR",
    )
    sync_video_preprocess_mode: str = Field(
        default="pose_dwpose",
        alias="LTX_SYNC_VIDEO_PREPROCESS_MODE",
    )
    sync_force_distilled: bool = Field(
        default=True,
        alias="LTX_SYNC_FORCE_DISTILLED",
    )
    sync_assume_fused_checkpoint: bool = Field(
        default=False,
        alias="LTX_SYNC_ASSUME_FUSED_CHECKPOINT",
    )
    sync_pre_hopper_fallback_enabled: bool = Field(
        default=True,
        alias="LTX_SYNC_PRE_HOPPER_FALLBACK_ENABLED",
    )

    @field_validator("default_model_variant")
    @classmethod
    def _validate_default_model_variant(cls, value: str) -> str:
        normalized = str(value or "distilled").strip().lower()
        if normalized not in {"distilled", "full"}:
            return "distilled"
        return normalized


settings = Settings()


class JobType(str, Enum):
    text_to_video = "text_to_video"
    image_to_video = "image_to_video"
    sync_reference_av = "sync_reference_av"


class JobState(str, Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


class PresetMode(str, Enum):
    fast = "fast"
    quality = "quality"


class BaseGenerationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=2000)
    preset: PresetMode = Field(default=PresetMode.fast)
    width: int = Field(default=768)
    height: int = Field(default=512)
    frame_count: int = Field(default=121)
    fps: int = Field(default=24, ge=1, le=60)
    seed: int | None = Field(default=None)
    model_variant: str | None = Field(default=None)
    enhance_prompt: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("width")
    @classmethod
    def validate_width(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("width must be divisible by 32")
        return value

    @field_validator("height")
    @classmethod
    def validate_height(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("height must be divisible by 32")
        return value

    @field_validator("frame_count")
    @classmethod
    def validate_frame_count(cls, value: int) -> int:
        if value < 9:
            raise ValueError("frame_count must be at least 9")
        if (value - 1) % 8 != 0:
            raise ValueError("frame_count must satisfy (frame_count - 1) divisible by 8")
        return value


class TextToVideoRequest(BaseGenerationRequest):
    pass


class ImageToVideoRequest(BaseGenerationRequest):
    image_upload_id: str
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0)


class SyncReferenceAvRequest(BaseGenerationRequest):
    reference_video_upload_id: str
    reference_image_upload_id: str | None = None
    override_audio_upload_id: str | None = None
    use_video_audio: bool = Field(default=True)
    conditioning_strength: float = Field(default=0.85, ge=0.0, le=1.0)


class JobCreatedResponse(BaseModel):
    job_id: str
    state: JobState
    workflow_id: str
    prompt_id: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: JobType
    state: JobState
    workflow_id: str
    prompt_id: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    patched_workflow_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobArtifactsResponse(BaseModel):
    job_id: str
    state: JobState
    artifacts: list[dict[str, Any]]


@dataclass
class JobRecord:
    job_id: str
    job_type: JobType
    state: JobState
    workflow_id: str
    prompt_id: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, JobRecord] = {}
        self.uploads: dict[str, Path] = {}

    def create_job(self, record: JobRecord) -> None:
        with self.lock:
            self.jobs[record.job_id] = record

    def get_job(self, job_id: str) -> JobRecord:
        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record

    def update_job(self, job_id: str, **changes: Any) -> JobRecord:
        with self.lock:
            record = self.jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = now_utc()
            return record

    def put_upload(self, upload_id: str, path: Path) -> None:
        with self.lock:
            self.uploads[upload_id] = path

    def get_upload(self, upload_id: str) -> Path:
        with self.lock:
            path = self.uploads.get(upload_id)
            if path is None:
                raise KeyError(upload_id)
            return path


_XFORMERS_PATCH_LOCK = threading.Lock()
_XFORMERS_PATCH_DONE = False


def _enable_xformers_attention_patch() -> None:
    """Match HF Space behavior by forcing xformers attention into LTX transformer attention."""
    global _XFORMERS_PATCH_DONE
    with _XFORMERS_PATCH_LOCK:
        if _XFORMERS_PATCH_DONE:
            return
        try:
            from ltx_core.model.transformer import attention as attn_mod
            from xformers.ops import memory_efficient_attention as xformers_mea
        except Exception as exc:
            logger.warning("xformers attention patch unavailable: %s", exc)
            _XFORMERS_PATCH_DONE = True
            return

        try:
            before = getattr(attn_mod, "memory_efficient_attention", None)
            attn_mod.memory_efficient_attention = xformers_mea
            logger.info(
                "Enabled xformers memory_efficient_attention patch for LTX transformer (before=%s after=%s)",
                before,
                attn_mod.memory_efficient_attention,
            )
        except Exception as exc:
            logger.warning("xformers attention patch failed: %s", exc)
        finally:
            _XFORMERS_PATCH_DONE = True


class PipelineRuntime:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        # Re-entrant lock: _get_* methods call _shared(), which also acquires
        # this lock. A plain Lock deadlocks on first pipeline init.
        self._lock = threading.RLock()
        # Single active generation protects A100 VRAM from concurrent OOM.
        self._generation_lock = threading.Lock()
        self._pipelines: dict[str, Any] = {}
        self._shared_assets: dict[str, str] = {}
        self._sync_pipeline: Any | None = None

    def _resolve_fp8_cast(self, requested: bool, *, context: str) -> bool:
        """Enable FP8 cast only on supported CUDA architectures."""
        if not requested:
            return False
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("FP8 cast requested for %s but CUDA is unavailable; disabling FP8 cast", context)
                return False
            major, minor = torch.cuda.get_device_capability()
            # Hopper (SM90+) supports the FP8 cast path used here.
            if int(major) < 9:
                logger.warning(
                    "FP8 cast requested for %s but GPU capability is %s.%s (requires >= 9.0); disabling FP8 cast",
                    context,
                    major,
                    minor,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Failed to verify FP8 support for %s (%s); disabling FP8 cast", context, exc)
            return False

    def _is_pre_hopper_cuda(self) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            major, _minor = torch.cuda.get_device_capability()
            return int(major) < 9
        except Exception:
            return False

    def _download_model_file(self, filename: str, repo_id: str | None = None) -> str:
        selected_repo = (repo_id or self.cfg.ltx_repo).strip()
        return hf_hub_download(
            repo_id=selected_repo,
            filename=filename,
            token=self.cfg.hf_token or None,
        )

    def _checkpoint_repo(self) -> str:
        value = str(self.cfg.ltx_checkpoint_repo or "").strip()
        return value or self.cfg.ltx_repo

    def _download_checkpoint_file(self, filename: str) -> str:
        local_path = self._local_checkpoint_path(filename)
        if local_path is not None:
            logger.info("Using local checkpoint %s", local_path)
            return local_path
        return self._download_model_file(filename, repo_id=self._checkpoint_repo())

    def _local_checkpoint_path(self, filename: str) -> str | None:
        base = str(self.cfg.local_checkpoints_dir or "").strip()
        if not base:
            return None
        candidate = Path(base).expanduser().resolve() / filename
        if candidate.is_file():
            return str(candidate)
        return None

    def prewarm_checkpoints(self) -> dict[str, str]:
        """Download shared assets plus both model checkpoints before serving traffic."""
        with self._lock:
            shared = self._shared()
            checkpoint_repo = self._checkpoint_repo()
            logger.info("Prewarming checkpoints from repo=%s", checkpoint_repo)
            distilled_checkpoint_path = self._download_checkpoint_file(self.cfg.distilled_checkpoint_name)
            full_checkpoint_path = self._download_checkpoint_file(self.cfg.full_checkpoint_name)
            return {
                "gemma_root": shared["gemma_root"],
                "spatial_upsampler_path": shared["spatial_upsampler_path"],
                "distilled_lora_path": shared["distilled_lora_path"],
                "sync_ic_lora_path": shared.get("sync_ic_lora_path", ""),
                "distilled_checkpoint_path": distilled_checkpoint_path,
                "full_checkpoint_path": full_checkpoint_path,
            }

    def _shared(self) -> dict[str, str]:
        with self._lock:
            if self._shared_assets:
                return dict(self._shared_assets)

            logger.info("Downloading Gemma and shared LTX assets")
            gemma_root = snapshot_download(
                repo_id=self.cfg.gemma_repo,
                token=self.cfg.hf_token or None,
            )
            spatial_upsampler_path = self._download_model_file(self.cfg.spatial_upsampler_name)
            distilled_lora_path = self._download_model_file(self.cfg.distilled_lora_name)
            sync_ic_lora_path = ""
            if self.cfg.enable_sync_av and self.cfg.sync_ic_lora_repo and self.cfg.sync_ic_lora_filename:
                try:
                    sync_ic_lora_path = self._download_model_file(
                        self.cfg.sync_ic_lora_filename,
                        repo_id=self.cfg.sync_ic_lora_repo,
                    )
                except Exception as exc:
                    logger.warning("Sync IC-LoRA download skipped: %s", exc)
                    sync_ic_lora_path = ""

            self._shared_assets = {
                "gemma_root": gemma_root,
                "spatial_upsampler_path": spatial_upsampler_path,
                "distilled_lora_path": distilled_lora_path,
                "sync_ic_lora_path": sync_ic_lora_path,
            }
            return dict(self._shared_assets)

    def _get_sync_pipeline(self):
        with self._lock:
            if self._sync_pipeline is not None:
                return self._sync_pipeline

            _enable_xformers_attention_patch()
            shared = self._shared()
            fused_marker = "fused-union-control"
            checkpoint_repo = self._checkpoint_repo()
            checkpoint_name = str(self.cfg.distilled_checkpoint_name or "").strip()
            checkpoint_repo_l = checkpoint_repo.lower()
            checkpoint_name_l = checkpoint_name.lower()
            uses_fused_checkpoint = bool(self.cfg.sync_assume_fused_checkpoint) or (
                fused_marker in checkpoint_repo_l or fused_marker in checkpoint_name_l
            )

            if uses_fused_checkpoint and self._is_pre_hopper_cuda():
                if self.cfg.sync_pre_hopper_fallback_enabled:
                    logger.warning(
                        "Sync fused checkpoint is too memory-heavy on pre-Hopper GPUs; "
                        "falling back to repo=%s checkpoint=%s",
                        SYNC_SM80_FALLBACK_REPO,
                        SYNC_SM80_FALLBACK_CHECKPOINT,
                    )
                    checkpoint_repo = SYNC_SM80_FALLBACK_REPO
                    checkpoint_name = SYNC_SM80_FALLBACK_CHECKPOINT
                    uses_fused_checkpoint = False
                else:
                    logger.warning(
                        "Running fused sync checkpoint on pre-Hopper GPU because "
                        "LTX_SYNC_PRE_HOPPER_FALLBACK_ENABLED=false; this may OOM."
                    )

            checkpoint_path = self._download_model_file(checkpoint_name, repo_id=checkpoint_repo)
            ic_lora_path = None if uses_fused_checkpoint else (shared.get("sync_ic_lora_path") or None)
            if uses_fused_checkpoint and shared.get("sync_ic_lora_path"):
                logger.info(
                    "Sync pipeline using fused union-control checkpoint; skipping external IC-LoRA load"
                )
            sync_pipeline = HFStyleSyncPipeline(
                checkpoint_path=checkpoint_path,
                gemma_root=shared["gemma_root"],
                spatial_upsampler_path=shared["spatial_upsampler_path"],
                ic_lora_path=ic_lora_path,
                use_fp8_cast=self._resolve_fp8_cast(
                    bool(self.cfg.distilled_use_fp8_cast),
                    context="sync pipeline",
                ),
                reference_downscale_factor=max(1, int(self.cfg.sync_reference_downscale_factor or 1)),
                fused_ic_lora=uses_fused_checkpoint,
            )
            self._sync_pipeline = sync_pipeline
            return sync_pipeline

    def _get_distilled_pipeline(self):
        with self._lock:
            cached = self._pipelines.get("distilled")
            if cached is not None:
                return cached

            from ltx_core.quantization import QuantizationPolicy
            from ltx_pipelines.distilled import DistilledPipeline

            _enable_xformers_attention_patch()
            shared = self._shared()
            checkpoint_path = self._download_checkpoint_file(self.cfg.distilled_checkpoint_name)
            quantization = (
                QuantizationPolicy.fp8_cast()
                if self._resolve_fp8_cast(
                    bool(self.cfg.distilled_use_fp8_cast),
                    context="distilled pipeline",
                )
                else None
            )

            logger.info("Initializing DistilledPipeline checkpoint=%s", checkpoint_path)
            pipeline = DistilledPipeline(
                distilled_checkpoint_path=checkpoint_path,
                spatial_upsampler_path=shared["spatial_upsampler_path"],
                gemma_root=shared["gemma_root"],
                loras=[],
                quantization=quantization,
            )
            self._pipelines["distilled"] = pipeline
            return pipeline

    def _get_full_pipeline(self):
        with self._lock:
            cached = self._pipelines.get("full")
            if cached is not None:
                return cached

            from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
            from ltx_core.quantization import QuantizationPolicy
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

            _enable_xformers_attention_patch()
            shared = self._shared()
            checkpoint_path = self._download_checkpoint_file(self.cfg.full_checkpoint_name)
            quantization = (
                QuantizationPolicy.fp8_cast()
                if self._resolve_fp8_cast(
                    bool(self.cfg.full_use_fp8_cast),
                    context="full pipeline",
                )
                else None
            )
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    shared["distilled_lora_path"],
                    1.0,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]

            logger.info("Initializing TI2VidTwoStagesPipeline checkpoint=%s", checkpoint_path)
            pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=checkpoint_path,
                distilled_lora=distilled_lora,
                spatial_upsampler_path=shared["spatial_upsampler_path"],
                gemma_root=shared["gemma_root"],
                loras=[],
                quantization=quantization,
            )
            self._pipelines["full"] = pipeline
            return pipeline

    def generate(
        self,
        *,
        request: BaseGenerationRequest,
        output_path: Path,
        image_path: Path | None = None,
        image_strength: float = 1.0,
    ) -> dict[str, Any]:
        import torch
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
        from ltx_pipelines.utils.media_io import encode_video

        with self._generation_lock:
            model_variant = normalize_model_variant(request.model_variant or self.cfg.default_model_variant)
            if model_variant == "distilled":
                pipeline = self._get_distilled_pipeline()
            else:
                pipeline = self._get_full_pipeline()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            tiling_config = TilingConfig.default()
            frame_count = normalize_frame_count(request.frame_count, self.cfg.frame_count_max)
            frame_rate = float(request.fps)
            images: list[Any] = []
            if image_path is not None:
                images = [
                    ImageConditioningInput(
                        path=str(image_path),
                        frame_idx=0,
                        strength=max(0.0, min(1.0, float(image_strength))),
                    )
                ]

            logger.info(
                "Generation start variant=%s %sx%s frames=%s fps=%s image=%s",
                model_variant,
                request.width,
                request.height,
                frame_count,
                frame_rate,
                bool(image_path),
            )

            with torch.inference_mode():
                if model_variant == "distilled":
                    video, audio = pipeline(
                        prompt=request.prompt,
                        seed=normalize_seed(request.seed),
                        height=request.height,
                        width=request.width,
                        num_frames=frame_count,
                        frame_rate=frame_rate,
                        images=images,
                        tiling_config=tiling_config,
                        enhance_prompt=bool(request.enhance_prompt),
                    )
                else:
                    video_guider = MultiModalGuiderParams(
                        cfg_scale=self.cfg.video_cfg_scale,
                        stg_scale=self.cfg.video_stg_scale,
                        rescale_scale=self.cfg.video_rescale_scale,
                        modality_scale=self.cfg.video_modality_scale,
                        skip_step=0,
                        stg_blocks=[28],
                    )
                    audio_guider = MultiModalGuiderParams(
                        cfg_scale=self.cfg.audio_cfg_scale,
                        stg_scale=self.cfg.audio_stg_scale,
                        rescale_scale=self.cfg.audio_rescale_scale,
                        modality_scale=self.cfg.audio_modality_scale,
                        skip_step=0,
                        stg_blocks=[28],
                    )
                    video, audio = pipeline(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                        seed=normalize_seed(request.seed),
                        height=request.height,
                        width=request.width,
                        num_frames=frame_count,
                        frame_rate=frame_rate,
                        num_inference_steps=self.cfg.full_num_inference_steps,
                        video_guider_params=video_guider,
                        audio_guider_params=audio_guider,
                        images=images,
                        tiling_config=tiling_config,
                        enhance_prompt=bool(request.enhance_prompt),
                    )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                video_chunks_number = get_video_chunks_number(frame_count, tiling_config)
                encode_video(
                    video=video,
                    fps=frame_rate,
                    audio=audio,
                    output_path=str(output_path),
                    video_chunks_number=video_chunks_number,
                )

            peak_gb = None
            if torch.cuda.is_available():
                peak_gb = float(torch.cuda.max_memory_allocated()) / (1024**3)
                torch.cuda.empty_cache()

            return {
                "model_variant": model_variant,
                "frame_count": frame_count,
                "fps": frame_rate,
                "width": request.width,
                "height": request.height,
                "peak_vram_gb": peak_gb,
            }

    def generate_sync_reference_av(
        self,
        *,
        request: SyncReferenceAvRequest,
        reference_video_path: Path,
        reference_image_path: Path | None,
        override_audio_path: Path | None,
        output_path: Path,
    ) -> dict[str, Any]:
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.media_io import encode_video

        with self._generation_lock:
            requested_variant = normalize_model_variant(request.model_variant or self.cfg.default_model_variant)
            if requested_variant != "distilled":
                if not self.cfg.sync_force_distilled:
                    raise ValueError("HF-style sync pipeline currently supports distilled variant only")
                logger.warning(
                    "HF-style sync pipeline requested model=%s, forcing distilled variant",
                    requested_variant,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            frame_count = normalize_frame_count(request.frame_count, self.cfg.frame_count_max)
            frame_rate = float(request.fps)
            pre_mode = str(self.cfg.sync_video_preprocess_mode or "pose_dwpose").strip().lower()
            attempts = _build_sync_fallback_attempts(
                width=int(request.width),
                height=int(request.height),
                frame_count=frame_count,
                hard_max=self.cfg.frame_count_max,
            )

            sync_pipeline = self._get_sync_pipeline()
            if not bool(getattr(sync_pipeline, "has_ic_lora", False)):
                raise ValueError(
                    "HF-style sync parity requires IC-LoRA weights. "
                    "Set HF_TOKEN and ensure LTX_SYNC_IC_LORA_* model access is approved."
                )

            for attempt_index, (attempt_width, attempt_height, attempt_frames, attempt_reason) in enumerate(
                attempts,
                start=1,
            ):
                # HF parity: preprocess reference video to pose/control video at Stage-1 resolution.
                stage1_width = max(32, (int(attempt_width) // 2) // 32 * 32)
                stage1_height = max(32, (int(attempt_height) // 2) // 32 * 32)
                conditioning_video_path, first_frame_path, temp_paths = _preprocess_conditioning_video(
                    video_path=reference_video_path,
                    mode=pre_mode,
                    width=stage1_width,
                    height=stage1_height,
                    num_frames=attempt_frames,
                    fps=frame_rate,
                )
                extracted_audio_temp: Path | None = None
                try:
                    image_input_path = reference_image_path or first_frame_path
                    images = [
                        ImageConditioningInput(
                            path=str(image_input_path),
                            frame_idx=0,
                            strength=1.0,
                        )
                    ]

                    audio_input_path: Path | None = override_audio_path
                    if audio_input_path is None and request.use_video_audio:
                        extracted_audio_temp = _extract_audio_from_video(reference_video_path)
                        if extracted_audio_temp is not None:
                            audio_input_path = extracted_audio_temp

                    logger.info(
                        "Sync parity generation start %sx%s frames=%s fps=%.2f preprocess=%s audio=%s attempt=%s/%s reason=%s",
                        attempt_width,
                        attempt_height,
                        attempt_frames,
                        frame_rate,
                        pre_mode,
                        bool(audio_input_path),
                        attempt_index,
                        len(attempts),
                        attempt_reason,
                    )
                    try:
                        video, audio = sync_pipeline(
                            prompt=request.prompt,
                            seed=normalize_seed(request.seed),
                            height=int(attempt_height),
                            width=int(attempt_width),
                            num_frames=attempt_frames,
                            frame_rate=frame_rate,
                            images=images,
                            audio_path=str(audio_input_path) if audio_input_path is not None else None,
                            video_conditioning=[(str(conditioning_video_path), 1.0)],
                            conditioning_strength=float(request.conditioning_strength),
                            enhance_prompt=bool(request.enhance_prompt),
                        )
                    finally:
                        # Keep only one stage ledger on GPU at a time; force release between retries.
                        if hasattr(sync_pipeline, "_drop_stage_1_model_ledger"):
                            try:
                                sync_pipeline._drop_stage_1_model_ledger()
                            except Exception:
                                pass
                        if hasattr(sync_pipeline, "_drop_stage_2_model_ledger"):
                            try:
                                sync_pipeline._drop_stage_2_model_ledger()
                            except Exception:
                                pass

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    tiling_config = TilingConfig.default()
                    video_chunks_number = get_video_chunks_number(attempt_frames, tiling_config)
                    encode_video(
                        video=video,
                        fps=frame_rate,
                        audio=audio,
                        output_path=str(output_path),
                        video_chunks_number=video_chunks_number,
                    )

                    peak_gb = None
                    if torch.cuda.is_available():
                        peak_gb = float(torch.cuda.max_memory_allocated()) / (1024**3)
                        torch.cuda.empty_cache()

                    return {
                        "model_variant": "distilled",
                        "requested_model_variant": requested_variant,
                        "frame_count": attempt_frames,
                        "fps": frame_rate,
                        "width": attempt_width,
                        "height": attempt_height,
                        "conditioning_strength": float(request.conditioning_strength),
                        "preprocess_mode": pre_mode,
                        "audio_source": (
                            "override_audio"
                            if request.override_audio_upload_id
                            else ("reference_video" if audio_input_path is not None else "generated")
                        ),
                        "peak_vram_gb": peak_gb,
                        "pipeline": "hf_style_ic_lora_sync",
                        "attempt": {
                            "index": attempt_index,
                            "total": len(attempts),
                            "reason": attempt_reason,
                        },
                    }
                except Exception as exc:
                    if _is_cuda_oom_error(exc) and attempt_index < len(attempts):
                        logger.warning(
                            "Sync parity OOM at attempt=%s/%s (%sx%s frames=%s); retrying fallback profile",
                            attempt_index,
                            len(attempts),
                            attempt_width,
                            attempt_height,
                            attempt_frames,
                        )
                        _best_effort_cuda_cleanup()
                        continue
                    raise
                finally:
                    cleanup_paths = [conditioning_video_path, first_frame_path]
                    if extracted_audio_temp is not None:
                        cleanup_paths.append(extracted_audio_temp)
                    cleanup_paths.extend(temp_paths)
                    for path in cleanup_paths:
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            continue
                    _best_effort_cuda_cleanup()


def _read_lora_reference_downscale_factor(lora_path: str) -> int:
    try:
        from safetensors import safe_open
    except Exception:
        return 1

    try:
        with safe_open(lora_path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            return max(1, int(metadata.get("reference_downscale_factor", 1)))
    except Exception as exc:
        logger.warning("Could not read IC-LoRA metadata from %s: %s", lora_path, exc)
        return 1


_POSE_DETECTOR_LOCK = threading.Lock()
_POSE_DETECTOR = None


def _get_pose_detector():
    global _POSE_DETECTOR
    with _POSE_DETECTOR_LOCK:
        if _POSE_DETECTOR is not None:
            return _POSE_DETECTOR
        from dwpose import DwposeDetector

        _POSE_DETECTOR = DwposeDetector.from_pretrained_default()
        logger.info("Loaded DWPose detector")
        return _POSE_DETECTOR


def _load_video_frames(video_path: Path) -> list[Any]:
    import imageio

    frames: list[Any] = []
    with imageio.get_reader(str(video_path)) as reader:
        for frame in reader:
            frames.append(frame)
    return frames


def _write_video_mp4(frames_float_01: list[Any], fps: float, output_path: Path) -> None:
    import imageio
    import numpy as np

    frames_uint8 = [
        (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        for frame in frames_float_01
    ]
    with imageio.get_writer(str(output_path), fps=float(fps), macro_block_size=1) as writer:
        for frame in frames_uint8:
            writer.append_data(frame)


def _preprocess_video_pose(frames: list[Any], width: int, height: int) -> list[Any]:
    import numpy as np
    from PIL import Image

    detector = _get_pose_detector()
    result: list[Any] = []
    for frame in frames:
        pil = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        pose_img = detector(pil, include_body=True, include_hand=True, include_face=True)
        if not isinstance(pose_img, Image.Image):
            pose_img = Image.fromarray(np.array(pose_img).astype(np.uint8))
        pose_img = pose_img.convert("RGB").resize((int(width), int(height)), Image.BILINEAR)
        result.append(np.array(pose_img).astype(np.float32) / 255.0)
    return result


def _preprocess_video_canny(
    frames: list[Any],
    width: int,
    height: int,
    *,
    low_threshold: int = 50,
    high_threshold: int = 100,
) -> list[Any]:
    import cv2
    import numpy as np

    result: list[Any] = []
    for frame in frames:
        resized = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_3ch = np.stack([edges, edges, edges], axis=-1)
        result.append(edges_3ch.astype(np.float32) / 255.0)
    return result


def _preprocess_video_depth(frames: list[Any], width: int, height: int) -> list[Any]:
    import cv2
    import numpy as np

    result: list[Any] = []
    for frame in frames:
        resized = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=5))
        lap = lap / (float(lap.max()) + 1e-8)
        depth_3ch = np.stack([lap, lap, lap], axis=-1)
        result.append(depth_3ch.astype(np.float32))
    return result


def _preprocess_conditioning_video(
    *,
    video_path: Path,
    mode: str,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
) -> tuple[Path, Path, list[Path]]:
    import numpy as np
    from PIL import Image

    frames = _load_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames decoded from reference video: {video_path}")

    if len(frames) < int(num_frames):
        last = frames[-1]
        frames.extend([last] * (int(num_frames) - len(frames)))
    else:
        frames = frames[: int(num_frames)]

    fd_first, first_frame_raw = tempfile.mkstemp(prefix="ltx_sync_first_", suffix=".png")
    os.close(fd_first)
    first_frame_path = Path(first_frame_raw)
    Image.fromarray(frames[0].astype(np.uint8)).save(first_frame_path)

    normalized_mode = str(mode or "pose_dwpose").strip().lower()
    if normalized_mode in {"pose", "pose_dwpose", "dwpose"}:
        processed_frames = _preprocess_video_pose(frames, width=width, height=height)
        preprocess_used = "pose_dwpose"
    elif normalized_mode in {"canny", "canny_edge", "edge"}:
        processed_frames = _preprocess_video_canny(frames, width=width, height=height)
        preprocess_used = "canny"
    elif normalized_mode in {"depth", "depth_laplacian", "laplacian"}:
        processed_frames = _preprocess_video_depth(frames, width=width, height=height)
        preprocess_used = "depth_laplacian"
    else:
        # Fallback/raw mode for operators who explicitly disable DWPose.
        processed_frames = []
        for frame in frames:
            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
            image = image.resize((int(width), int(height)), Image.BILINEAR)
            processed_frames.append(np.array(image).astype(np.float32) / 255.0)
        preprocess_used = "raw"

    fd_cond, cond_raw = tempfile.mkstemp(prefix="ltx_sync_cond_", suffix=".mp4")
    os.close(fd_cond)
    cond_video_path = Path(cond_raw)
    _write_video_mp4(processed_frames, fps=fps, output_path=cond_video_path)
    logger.info(
        "Prepared sync conditioning video mode=%s frames=%s size=%sx%s path=%s",
        preprocess_used,
        len(processed_frames),
        width,
        height,
        cond_video_path,
    )
    return cond_video_path, first_frame_path, []


class HFStyleSyncPipeline:
    """
    Distilled sync pipeline with IC-LoRA video conditioning, matching HF space behavior.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        ic_lora_path: str | None,
        use_fp8_cast: bool,
        reference_downscale_factor: int = 2,
        fused_ic_lora: bool = False,
    ) -> None:
        import torch
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.utils.helpers import get_device
        from ltx_pipelines.utils.types import PipelineComponents

        self.device = get_device()
        self.dtype = torch.bfloat16

        self._checkpoint_path = checkpoint_path
        self._spatial_upsampler_path = spatial_upsampler_path
        self._gemma_root = gemma_root
        self._quantization = QuantizationPolicy.fp8_cast() if use_fp8_cast else None

        ic_loras: list[Any] = []
        if ic_lora_path:
            ic_loras.append(
                LoraPathStrengthAndSDOps(
                    ic_lora_path,
                    1.0,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            )
        self._ic_loras = ic_loras
        self.has_ic_lora = bool(self._ic_loras) or bool(fused_ic_lora)
        self._requires_dual_ledger = bool(self._ic_loras)
        self._stage_1_model_ledger: Any | None = None
        self._stage_2_model_ledger: Any | None = None

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )
        self.reference_downscale_factor = max(1, int(reference_downscale_factor or 1))
        if self.has_ic_lora and ic_lora_path:
            self.reference_downscale_factor = _read_lora_reference_downscale_factor(ic_lora_path)
        logger.info(
            "HF sync pipeline ready has_ic_lora=%s source=%s",
            self.has_ic_lora,
            "external_ic_lora" if ic_lora_path else ("fused_checkpoint" if fused_ic_lora else "none"),
        )

    def _build_model_ledger(self, *, with_ic_lora: bool):
        from ltx_pipelines.utils import ModelLedger

        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self._checkpoint_path,
            spatial_upsampler_path=self._spatial_upsampler_path,
            gemma_root_path=self._gemma_root,
            loras=self._ic_loras if with_ic_lora else [],
            quantization=self._quantization,
        )

    def _ensure_stage_1_model_ledger(self):
        if self._stage_1_model_ledger is None:
            self._stage_1_model_ledger = self._build_model_ledger(with_ic_lora=True)
        return self._stage_1_model_ledger

    def _ensure_stage_2_model_ledger(self):
        if not self._requires_dual_ledger:
            return self._ensure_stage_1_model_ledger()
        if self._stage_2_model_ledger is None:
            self._stage_2_model_ledger = self._build_model_ledger(with_ic_lora=False)
        return self._stage_2_model_ledger

    def _drop_stage_1_model_ledger(self) -> None:
        self._stage_1_model_ledger = None
        _best_effort_cuda_cleanup()

    def _drop_stage_2_model_ledger(self) -> None:
        self._stage_2_model_ledger = None
        _best_effort_cuda_cleanup()

    def _create_ic_conditionings(
        self,
        *,
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: Any,
        conditioning_strength: float,
    ) -> list[Any]:
        from ltx_core.conditioning import (
            ConditioningItemAttentionStrengthWrapper,
            VideoConditionByReferenceLatent,
        )
        from ltx_pipelines.utils.media_io import load_video_conditioning

        conditionings: list[Any] = []
        scale = max(1, int(self.reference_downscale_factor))
        ref_height = max(32, int(height) // scale)
        ref_width = max(32, int(width) // scale)

        for video_path, strength in video_conditioning:
            video = load_video_conditioning(
                video_path=video_path,
                height=ref_height,
                width=ref_width,
                frame_cap=num_frames,
                dtype=self.dtype,
                device=self.device,
            )
            encoded_video = video_encoder(video)
            cond = VideoConditionByReferenceLatent(
                latent=encoded_video,
                downscale_factor=scale,
                strength=float(strength),
            )
            if conditioning_strength < 1.0:
                cond = ConditioningItemAttentionStrengthWrapper(
                    cond,
                    attention_mask=max(0.0, min(1.0, float(conditioning_strength))),
                )
            conditionings.append(cond)
        return conditionings

    def __call__(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[Any],
        audio_path: str | None,
        video_conditioning: list[tuple[str, float]],
        conditioning_strength: float,
        enhance_prompt: bool,
    ):
        import torch
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.video_vae import TilingConfig
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils import euler_denoising_loop
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.helpers import (
            assert_resolution,
            cleanup_memory,
            combined_image_conditionings,
            denoise_audio_video,
            denoise_video_only,
            encode_prompts,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import decode_audio_from_file

        assert_resolution(height=height, width=width, is_two_stage=True)
        sync_prompt = f"{str(prompt or '').strip()} synchronized lipsync".strip()

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        stage_1_model_ledger = self._ensure_stage_1_model_ledger()
        stage_2_model_ledger = None

        (ctx_p,) = encode_prompts(
            [sync_prompt],
            stage_1_model_ledger,
            enhance_first_prompt=bool(enhance_prompt),
            enhance_prompt_image=images[0].path if images else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        has_audio = bool(audio_path)
        encoded_audio_latent = None
        decoded_audio_for_output = None
        if has_audio:
            video_duration = float(num_frames) / float(frame_rate)
            decoded_audio = decode_audio_from_file(audio_path, self.device, 0.0, video_duration)
            if decoded_audio is None:
                raise ValueError(f"Could not decode audio stream from {audio_path}")

            encoded_audio_latent = vae_encode_audio(
                decoded_audio,
                stage_1_model_ledger.audio_encoder(),
            )
            audio_shape = AudioLatentShape.from_duration(
                batch=1,
                duration=video_duration,
                channels=8,
                mel_bins=16,
            )
            expected_frames = int(audio_shape.frames)
            actual_frames = int(encoded_audio_latent.shape[2])
            if actual_frames > expected_frames:
                encoded_audio_latent = encoded_audio_latent[:, :, :expected_frames, :]
            elif actual_frames < expected_frames:
                pad = torch.zeros(
                    encoded_audio_latent.shape[0],
                    encoded_audio_latent.shape[1],
                    expected_frames - actual_frames,
                    encoded_audio_latent.shape[3],
                    device=encoded_audio_latent.device,
                    dtype=encoded_audio_latent.dtype,
                )
                encoded_audio_latent = torch.cat([encoded_audio_latent, pad], dim=2)
            decoded_audio_for_output = Audio(
                waveform=decoded_audio.waveform.squeeze(0),
                sampling_rate=decoded_audio.sampling_rate,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=int(num_frames),
            width=int(width) // 2,
            height=int(height) // 2,
            fps=float(frame_rate),
        )
        video_encoder = stage_1_model_ledger.video_encoder()
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        stage_1_conditionings.extend(
            self._create_ic_conditionings(
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=int(num_frames),
                video_encoder=video_encoder,
                conditioning_strength=float(conditioning_strength),
            )
        )

        transformer_stage1 = stage_1_model_ledger.transformer()
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(sigmas, video_state, audio_state, current_stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=current_stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer_stage1,
                ),
            )

        if has_audio:
            video_state = denoise_video_only(
                output_shape=stage_1_output_shape,
                conditionings=stage_1_conditionings,
                noiser=noiser,
                sigmas=stage_1_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop,
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
                initial_audio_latent=encoded_audio_latent,
            )
            audio_state = None
        else:
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_1_output_shape,
                conditionings=stage_1_conditionings,
                noiser=noiser,
                sigmas=stage_1_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop,
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cleanup_memory()

        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=stage_1_model_ledger.spatial_upsampler(),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cleanup_memory()

        if self._requires_dual_ledger:
            del transformer_stage1, video_encoder
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cleanup_memory()
            self._drop_stage_1_model_ledger()
            stage_2_model_ledger = self._ensure_stage_2_model_ledger()
            video_encoder = stage_2_model_ledger.video_encoder()
            transformer_stage2 = stage_2_model_ledger.transformer()
        else:
            stage_2_model_ledger = stage_1_model_ledger
            transformer_stage2 = stage_2_model_ledger.transformer()

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop_stage2(sigmas, video_state, audio_state, current_stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=current_stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer_stage2,
                ),
            )

        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=int(num_frames),
            width=int(width),
            height=int(height),
            fps=float(frame_rate),
        )
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )

        if has_audio:
            video_state = denoise_video_only(
                output_shape=stage_2_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop_stage2,
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
                noise_scale=stage_2_sigmas[0],
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=encoded_audio_latent,
            )
            audio_state = None
        else:
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_2_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop_stage2,
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
                noise_scale=stage_2_sigmas[0],
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=audio_state.latent,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._requires_dual_ledger:
            del transformer_stage2, video_encoder
        else:
            del transformer_stage1, transformer_stage2, video_encoder
        cleanup_memory()

        decoded_video = vae_decode_video(
            video_state.latent,
            stage_2_model_ledger.video_decoder(),
            TilingConfig.default(),
            generator,
        )
        if has_audio:
            output_audio = decoded_audio_for_output
        else:
            output_audio = vae_decode_audio(
                audio_state.latent,
                stage_2_model_ledger.audio_decoder(),
                stage_2_model_ledger.vocoder(),
            )
        if self._requires_dual_ledger:
            self._drop_stage_2_model_ledger()
        return decoded_video, output_audio


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def normalize_seed(seed: int | None) -> int:
    if seed is None:
        return int(uuid.uuid4().int % 2_147_483_647)
    if seed < 0:
        return int(uuid.uuid4().int % 2_147_483_647)
    return min(int(seed), 2_147_483_647)


def normalize_model_variant(value: str | None) -> str:
    normalized = str(value or "distilled").strip().lower()
    return normalized if normalized in {"distilled", "full"} else "distilled"


def normalize_frame_count(frame_count: int, hard_max: int) -> int:
    value = max(9, int(frame_count))
    if (value - 1) % 8 != 0:
        value = ((value - 1 + 7) // 8) * 8 + 1
    if value > hard_max:
        value = ((hard_max - 1) // 8) * 8 + 1
    return max(9, value)


def _to_divisible_32(value: int) -> int:
    return max(32, (int(value) // 32) * 32)


def _to_divisible(value: int, divisor: int, *, min_value: int | None = None) -> int:
    divisor_i = max(1, int(divisor))
    min_v = divisor_i if min_value is None else max(divisor_i, int(min_value))
    return max(min_v, (int(value) // divisor_i) * divisor_i)


def _to_sync_divisible_128(value: int) -> int:
    # Sync stage-2 reshapes latent dims in factor-2 chunks; 128 alignment avoids odd latent axes.
    return _to_divisible(value, 128, min_value=128)


def _scale_resolution_to_max_pixels(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    width_i = max(32, int(width))
    height_i = max(32, int(height))
    pixels = width_i * height_i
    if pixels <= max_pixels:
        return _to_divisible_32(width_i), _to_divisible_32(height_i)

    scale = (float(max_pixels) / float(pixels)) ** 0.5
    scaled_w = _to_divisible_32(int(width_i * scale))
    scaled_h = _to_divisible_32(int(height_i * scale))
    return max(32, scaled_w), max(32, scaled_h)


def _is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc or "").strip().lower()
    if not message:
        return False
    return ("out of memory" in message and "cuda" in message) or ("cuda out of memory" in message)


def _best_effort_cuda_cleanup() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _build_sync_fallback_attempts(
    *,
    width: int,
    height: int,
    frame_count: int,
    hard_max: int,
) -> list[tuple[int, int, int, str]]:
    attempts: list[tuple[int, int, int, str]] = []
    seen_profiles: set[tuple[int, int, int]] = set()

    def _append(w: int, h: int, f: int, reason: str) -> None:
        w = _to_sync_divisible_128(w)
        h = _to_sync_divisible_128(h)
        f = normalize_frame_count(f, hard_max)
        profile = (w, h, f)
        if profile in seen_profiles:
            return
        seen_profiles.add(profile)
        attempts.append((w, h, f, reason))

    primary_w = _to_sync_divisible_128(width)
    primary_h = _to_sync_divisible_128(height)
    primary_f = normalize_frame_count(frame_count, hard_max)
    _append(primary_w, primary_h, primary_f, "primary")

    # Fallback 1: cap to 768x512-equivalent pixels and trim very long sequences.
    fb1_w, fb1_h = _scale_resolution_to_max_pixels(primary_w, primary_h, 768 * 512)
    _append(fb1_w, fb1_h, min(primary_f, 73), "oom_fallback_standard")

    # Fallback 2: more conservative emergency profile.
    fb2_w, fb2_h = _scale_resolution_to_max_pixels(primary_w, primary_h, 640 * 384)
    _append(fb2_w, fb2_h, min(primary_f, 49), "oom_fallback_conservative")

    return attempts


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.service_api_key:
        return
    if x_api_key != settings.service_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title=settings.service_name)
store = Store()
runtime = PipelineRuntime(settings)


def _artifacts_dir() -> Path:
    return Path(settings.data_dir).resolve() / "artifacts"


def _uploads_dir() -> Path:
    return Path(settings.data_dir).resolve() / "uploads"


def _extract_first_frame(video_path: Path) -> Path:
    first_frame_path = Path(tempfile.mkstemp(prefix="ltx_sync_frame_", suffix=".png")[1])
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(first_frame_path),
    ]
    subprocess.run(cmd, check=True)
    return first_frame_path


def _extract_audio_from_video(video_path: Path) -> Path | None:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        return None

    extracted_audio_path = Path(tempfile.mkstemp(prefix="ltx_sync_audio_", suffix=".wav")[1])
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        str(extracted_audio_path),
    ]
    subprocess.run(cmd, check=True)
    return extracted_audio_path


def _replace_audio_track(video_path: Path, audio_path: Path) -> None:
    remux_path = Path(tempfile.mkstemp(prefix="ltx_sync_remux_", suffix=".mp4")[1])
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(remux_path),
    ]
    subprocess.run(cmd, check=True)
    remux_path.replace(video_path)


def _job_to_status(record: JobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=record.job_id,
        job_type=record.job_type,
        state=record.state,
        workflow_id=record.workflow_id,
        prompt_id=record.prompt_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        patched_workflow_path=None,
        error=record.error,
        metadata=record.metadata,
    )


def _build_workflow_id(job_type: JobType, model_variant: str, preset: PresetMode) -> str:
    return f"{job_type.value}_{model_variant}_{preset.value}"


def _run_job(job_id: str, payload: dict[str, Any], job_type: JobType) -> None:
    try:
        store.update_job(job_id, state=JobState.running, started_at=now_utc())
        if job_type is JobType.text_to_video:
            request = TextToVideoRequest.model_validate(payload)
            image_path = None
            image_strength = 1.0
            reference_video_path = None
            override_audio_path = None
        elif job_type is JobType.image_to_video:
            request = ImageToVideoRequest.model_validate(payload)
            image_path = store.get_upload(request.image_upload_id)
            image_strength = request.image_strength
            reference_video_path = None
            override_audio_path = None
        else:
            request = SyncReferenceAvRequest.model_validate(payload)
            reference_video_path = store.get_upload(request.reference_video_upload_id)
            if request.reference_image_upload_id:
                image_path = store.get_upload(request.reference_image_upload_id)
            else:
                image_path = None
            image_strength = request.conditioning_strength

            override_audio_path = None
            if request.override_audio_upload_id:
                override_audio_path = store.get_upload(request.override_audio_upload_id)

        artifact_id = str(uuid.uuid4())
        output_name = f"{job_id}.mp4"
        output_path = _artifacts_dir() / output_name

        if job_type is JobType.sync_reference_av:
            metadata = runtime.generate_sync_reference_av(
                request=request,
                reference_video_path=reference_video_path,
                reference_image_path=image_path,
                override_audio_path=override_audio_path,
                output_path=output_path,
            )
        else:
            metadata = runtime.generate(
                request=request,
                output_path=output_path,
                image_path=image_path,
                image_strength=image_strength,
            )

        artifact = {
            "artifact_id": artifact_id,
            "filename": output_name,
            "subfolder": "",
            "type": "output",
            "size_bytes": output_path.stat().st_size,
            "abs_path": str(output_path),
        }

        store.update_job(
            job_id,
            state=JobState.succeeded,
            completed_at=now_utc(),
            metadata={**metadata, "request_metadata": request.metadata},
            artifacts=[artifact],
            error=None,
        )
    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        logger.debug(traceback.format_exc())
        store.update_job(
            job_id,
            state=JobState.failed,
            completed_at=now_utc(),
            error=str(exc),
        )


@app.on_event("startup")
def on_startup() -> None:
    _artifacts_dir().mkdir(parents=True, exist_ok=True)
    _uploads_dir().mkdir(parents=True, exist_ok=True)
    logger.info(
        "Service started data_dir=%s standard_av=%s sync_av=%s",
        settings.data_dir,
        settings.enable_standard_av,
        settings.enable_sync_av,
    )
    if settings.preload_on_start:
        try:
            logger.info(
                "Prewarming shared assets and both checkpoints at startup "
                "(distilled=%s, full=%s)",
                settings.distilled_checkpoint_name,
                settings.full_checkpoint_name,
            )
            warmed = runtime.prewarm_checkpoints()
            logger.info(
                "Prewarm complete distilled=%s full=%s",
                warmed["distilled_checkpoint_path"],
                warmed["full_checkpoint_path"],
            )
        except Exception as exc:
            logger.warning("Pipeline preload failed: %s", exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.service_name,
        "time": now_utc().isoformat(),
        "capabilities": {
            "standard_av": bool(settings.enable_standard_av),
            "sync_av": bool(settings.enable_sync_av),
        },
    }


def _require_standard_av_enabled() -> None:
    if not settings.enable_standard_av:
        raise HTTPException(status_code=503, detail="standard_av_disabled")


def _require_sync_av_enabled() -> None:
    if not settings.enable_sync_av:
        raise HTTPException(status_code=503, detail="sync_av_disabled")


async def _save_upload(upload: UploadFile, fallback_name: str, max_mb: int | None = None) -> dict[str, Any]:
    content = await upload.read()
    effective_max_mb = int(max_mb or settings.upload_max_mb)
    max_bytes = int(effective_max_mb * 1024 * 1024)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload too large (>{effective_max_mb}MB)")

    upload_id = str(uuid.uuid4())
    ext = Path(upload.filename or fallback_name).suffix or Path(fallback_name).suffix
    destination = _uploads_dir() / f"{upload_id}{ext}"
    destination.write_bytes(content)
    store.put_upload(upload_id, destination)

    return {
        "upload_id": upload_id,
        "filename": destination.name,
        "content_type": upload.content_type,
        "size_bytes": len(content),
    }


@app.post("/uploads/image", dependencies=[Depends(require_api_key)])
async def upload_image(image: UploadFile = File(...)) -> dict[str, Any]:
    return await _save_upload(image, "image.png", max_mb=25)


@app.post("/uploads/video", dependencies=[Depends(require_api_key)])
async def upload_video(video: UploadFile = File(...)) -> dict[str, Any]:
    return await _save_upload(video, "reference.mp4", max_mb=300)


@app.post("/uploads/audio", dependencies=[Depends(require_api_key)])
async def upload_audio(audio: UploadFile = File(...)) -> dict[str, Any]:
    return await _save_upload(audio, "override.wav", max_mb=100)


@app.post("/jobs/text-to-video", response_model=JobCreatedResponse, dependencies=[Depends(require_api_key)])
def create_text_to_video_job(request: TextToVideoRequest, background_tasks: BackgroundTasks) -> JobCreatedResponse:
    _require_standard_av_enabled()
    model_variant = normalize_model_variant(request.model_variant or settings.default_model_variant)
    request = request.model_copy(update={"model_variant": model_variant})

    job_id = str(uuid.uuid4())
    record = JobRecord(
        job_id=job_id,
        job_type=JobType.text_to_video,
        state=JobState.queued,
        workflow_id=_build_workflow_id(JobType.text_to_video, model_variant, request.preset),
        prompt_id=job_id,
        created_at=now_utc(),
        updated_at=now_utc(),
    )
    store.create_job(record)
    background_tasks.add_task(_run_job, job_id, request.model_dump(), JobType.text_to_video)

    return JobCreatedResponse(
        job_id=job_id,
        state=record.state,
        workflow_id=record.workflow_id,
        prompt_id=record.prompt_id,
    )


@app.post("/jobs/image-to-video", response_model=JobCreatedResponse, dependencies=[Depends(require_api_key)])
def create_image_to_video_job(request: ImageToVideoRequest, background_tasks: BackgroundTasks) -> JobCreatedResponse:
    _require_standard_av_enabled()
    model_variant = normalize_model_variant(request.model_variant or settings.default_model_variant)
    request = request.model_copy(update={"model_variant": model_variant})

    try:
        store.get_upload(request.image_upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="image_upload_id not found") from exc

    job_id = str(uuid.uuid4())
    record = JobRecord(
        job_id=job_id,
        job_type=JobType.image_to_video,
        state=JobState.queued,
        workflow_id=_build_workflow_id(JobType.image_to_video, model_variant, request.preset),
        prompt_id=job_id,
        created_at=now_utc(),
        updated_at=now_utc(),
    )
    store.create_job(record)
    background_tasks.add_task(_run_job, job_id, request.model_dump(), JobType.image_to_video)

    return JobCreatedResponse(
        job_id=job_id,
        state=record.state,
        workflow_id=record.workflow_id,
        prompt_id=record.prompt_id,
    )


@app.post("/jobs/sync-reference-av", response_model=JobCreatedResponse, dependencies=[Depends(require_api_key)])
def create_sync_reference_av_job(
    request: SyncReferenceAvRequest,
    background_tasks: BackgroundTasks,
) -> JobCreatedResponse:
    _require_sync_av_enabled()
    model_variant = normalize_model_variant(request.model_variant or settings.default_model_variant)
    request = request.model_copy(update={"model_variant": model_variant})

    try:
        store.get_upload(request.reference_video_upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="reference_video_upload_id not found") from exc

    if request.reference_image_upload_id:
        try:
            store.get_upload(request.reference_image_upload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="reference_image_upload_id not found") from exc

    if request.override_audio_upload_id:
        try:
            store.get_upload(request.override_audio_upload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="override_audio_upload_id not found") from exc

    job_id = str(uuid.uuid4())
    record = JobRecord(
        job_id=job_id,
        job_type=JobType.sync_reference_av,
        state=JobState.queued,
        workflow_id=_build_workflow_id(JobType.sync_reference_av, model_variant, request.preset),
        prompt_id=job_id,
        created_at=now_utc(),
        updated_at=now_utc(),
    )
    store.create_job(record)
    background_tasks.add_task(_run_job, job_id, request.model_dump(), JobType.sync_reference_av)

    return JobCreatedResponse(
        job_id=job_id,
        state=record.state,
        workflow_id=record.workflow_id,
        prompt_id=record.prompt_id,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_api_key)])
def get_job_status(job_id: str, refresh: bool = True) -> JobStatusResponse:  # noqa: ARG001
    try:
        record = store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return _job_to_status(record)


@app.get("/jobs/{job_id}/artifacts", response_model=JobArtifactsResponse, dependencies=[Depends(require_api_key)])
def get_job_artifacts(job_id: str) -> JobArtifactsResponse:
    try:
        record = store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    public_artifacts: list[dict[str, Any]] = []
    for artifact in record.artifacts:
        public_artifacts.append({
            "artifact_id": artifact.get("artifact_id"),
            "filename": artifact.get("filename"),
            "subfolder": artifact.get("subfolder", ""),
            "type": artifact.get("type", "output"),
            "size_bytes": artifact.get("size_bytes"),
        })

    return JobArtifactsResponse(
        job_id=job_id,
        state=record.state,
        artifacts=public_artifacts,
    )


@app.get("/jobs/{job_id}/artifacts/{artifact_id}/content", dependencies=[Depends(require_api_key)])
def download_artifact(job_id: str, artifact_id: str) -> FileResponse:
    try:
        record = store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    for artifact in record.artifacts:
        if str(artifact.get("artifact_id")) != artifact_id:
            continue
        path = Path(str(artifact.get("abs_path") or "")).resolve()
        if not path.exists():
            raise HTTPException(status_code=404, detail="artifact file missing")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(str(path), media_type=media_type, filename=path.name)

    raise HTTPException(status_code=404, detail="artifact not found")


@app.get("/presets", dependencies=[Depends(require_api_key)])
def list_presets() -> dict[str, Any]:
    presets: list[dict[str, Any]] = []
    if settings.enable_standard_av:
        presets.extend(
            [
                {
                    "workflow_id": "text_to_video_distilled_fast",
                    "task": "text_to_video",
                    "preset": "fast",
                    "description": "Standalone distilled model",
                },
                {
                    "workflow_id": "text_to_video_full_quality",
                    "task": "text_to_video",
                    "preset": "quality",
                    "description": "Standalone full model",
                },
                {
                    "workflow_id": "image_to_video_distilled_fast",
                    "task": "image_to_video",
                    "preset": "fast",
                    "description": "Standalone distilled model",
                },
                {
                    "workflow_id": "image_to_video_full_quality",
                    "task": "image_to_video",
                    "preset": "quality",
                    "description": "Standalone full model",
                },
            ]
        )
    if settings.enable_sync_av:
        presets.append(
            {
                "workflow_id": "sync_reference_av_distilled_quality",
                "task": "sync_reference_av",
                "preset": "quality",
                "description": "Sync mode with reference video/audio",
            }
        )
    return {"presets": presets}
