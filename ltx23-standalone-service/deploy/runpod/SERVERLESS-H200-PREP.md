# Runpod H200 Serverless Prep (Sync AV)

This repository is configured for **Runpod serverless** sync AV workers.

## 1) Build and push image

Use GitHub Actions:

- `.github/workflows/build-runpod-sync-h200-image.yml`

Or local build:

```bash
./build_and_push_image.sh ghcr.io/<owner>/ltx23-sync sync-h200
```

## 2) Prepare env

Use:

- `.env.runpod.serverless.sync-h200.example`

Required split flags:

- `LTX_ENABLE_STANDARD_AV=false`
- `LTX_ENABLE_SYNC_AV=true`

## 3) Create/update template + endpoint

```bash
./create_or_update_serverless_endpoint.sh ./.env.runpod.serverless.sync-h200
```

This script:

- creates or updates a **serverless template**
- creates or updates a **serverless endpoint**
- prints:
  - `template_id`
  - `endpoint_id`
  - runsync URL (`https://api.runpod.ai/v2/<endpoint_id>/runsync`)

## 4) Smoke test

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input":{"healthcheck":true}}'
```

## 5) AWS routing note

This is serverless, so it is not compatible with the old job-based headless HTTP API.
If you route from AWS, call Runpod `/run` or `/runsync` and pass sync payload directly.
