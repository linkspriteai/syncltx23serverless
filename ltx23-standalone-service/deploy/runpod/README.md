# Runpod Serverless Deployment (H200 Sync AV)

This folder deploys `ltx23-standalone-service` as a **Runpod serverless endpoint** for sync AV generation.

## Files

- `build_and_push_image.sh`: builds/pushes the serverless image (`Dockerfile.serverless` by default).
- `.env.runpod.serverless.sync-h200.example`: serverless H200 profile.
- `create_or_update_serverless_endpoint.sh`: creates or updates Runpod template + endpoint.
- `SERVERLESS-H200-PREP.md`: quick deployment and routing notes.

## Local Deploy Flow

1. Prepare env.

```bash
cd ltx23-standalone-service/deploy/runpod
cp .env.runpod.serverless.sync-h200.example .env.runpod.serverless.sync-h200
# edit values (RUNPOD_API_KEY, HF_TOKEN, image tag, etc.)
```

2. Build and push image.

```bash
./build_and_push_image.sh ghcr.io/<owner>/ltx23-sync sync-h200
```

3. Set image tag in env file.

```bash
RUNPOD_IMAGE=ghcr.io/<owner>/ltx23-sync:sync-h200
```

4. Create/update template + endpoint.

```bash
./create_or_update_serverless_endpoint.sh ./.env.runpod.serverless.sync-h200
```

5. Smoke test endpoint.

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input":{"healthcheck":true}}'
```

## GitHub Actions Secrets

- `RUNPOD_API_KEY`
- `HF_TOKEN`
- `LTX_SERVICE_API_KEY`
- `RUNPOD_CONTAINER_REGISTRY_AUTH_ID` (optional, needed for private GHCR pulls)

## Runtime Split (recommended)

- Vietnam A100 standard AV service:
  - `LTX_ENABLE_STANDARD_AV=true`
  - `LTX_ENABLE_SYNC_AV=false`
- Runpod H200 serverless sync service:
  - `LTX_ENABLE_STANDARD_AV=false`
  - `LTX_ENABLE_SYNC_AV=true`

## Serverless Input Shape (sync job)

Core fields:

- `prompt`
- `reference_video_url` or `reference_video_base64` (or `reference_video_path` inside container)
- optional: `reference_image_url|base64|path`
- optional: `override_audio_url|base64|path`

Optional controls:

- `duration` or `frame_count`
- `fps`, `width`, `height`
- `conditioning_strength`, `enhance_prompt`, `model_variant`
- `use_video_audio`
- `output_upload_url` (+ optional `output_upload_headers`)
