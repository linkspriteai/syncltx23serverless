# LTX23 Standalone Service

Standalone HTTP service for LTX-2.3 generation without ComfyUI.
This repo also includes a Runpod serverless worker entrypoint (`backend/app/serverless.py`) for sync AV workloads.

## What It Does

- Exposes a job-based API compatible with the existing AWS `Ltx23RemoteServiceClient` flow.
- Uses official Lightricks LTX-2 pipelines directly (`DistilledPipeline` / `TI2VidTwoStagesPipeline`).
- Supports text-to-video, image-to-video, and sync AV from reference video/audio.
- Sync mode now uses HF-style IC-LoRA conditioning parity:
  - DWPose preprocessing on reference video
  - IC-LoRA video conditioning latent path
  - Optional override audio or extracted reference-video audio
  - Sync generation currently runs on the distilled pipeline for parity/stability
- Returns MP4 artifacts with embedded audio.

## API Endpoints

- `POST /uploads/image`
- `POST /uploads/video`
- `POST /uploads/audio`
- `POST /jobs/text-to-video`
- `POST /jobs/image-to-video`
- `POST /jobs/sync-reference-av`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/artifacts`
- `GET /jobs/{job_id}/artifacts/{artifact_id}/content`
- `GET /health`

All non-health endpoints require header `x-api-key`.

## Model Variants

Request field `model_variant` supports:

- `distilled` (default)
- `full`

## Run With Docker Compose

From `ltx23-standalone-service/infra/docker`:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Default exposed port: `13482` (container `8080`).

## Key Environment Variables

- `SERVICE_API_KEY`
- `HF_TOKEN`
- `LTX_DEFAULT_MODEL_VARIANT=distilled|full`
- `LTX_MODEL_CHECKPOINT_DISTILLED`
- `LTX_MODEL_CHECKPOINT_FULL`
- `LTX_MODEL_DISTILLED_LORA`
- `LTX_MODEL_SPATIAL_UPSAMPLER`
- `LTX_SYNC_FORCE_DISTILLED`
- `LTX_SYNC_ASSUME_FUSED_CHECKPOINT` (`true` when using fused union-control checkpoint)
- `LTX_SYNC_IC_LORA_REPO`
- `LTX_SYNC_IC_LORA_FILENAME`
- `LTX_SYNC_REFERENCE_DOWNSCALE_FACTOR`
- `LTX_SYNC_VIDEO_PREPROCESS_MODE` (`pose_dwpose` recommended)

Sync parity requires access to the configured IC-LoRA repository (accept license on Hugging Face and provide `HF_TOKEN`).
If you use `linoyts/ltx-2.3-22b-fused-union-control`, set `LTX_MODEL_CHECKPOINT_REPO` and `LTX_MODEL_CHECKPOINT_DISTILLED` to that repo/file and enable `LTX_SYNC_ASSUME_FUSED_CHECKPOINT=true`.

## Runpod Serverless Mode

- Entry: `backend/app/serverless.py`
- Dockerfile: `infra/docker/Dockerfile.serverless`
- Deploy docs/scripts: `deploy/runpod/`

The serverless worker supports sync AV input via URL/base64 payload fields and optional `output_upload_url` for direct artifact upload.
