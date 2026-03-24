# syncltx23serverless

Runpod serverless deployment repo for LTX23 sync AV generation.

## Main workflow

- Build image: `.github/workflows/build-runpod-sync-h200-image.yml`
- Build + deploy endpoint: `.github/workflows/deploy-runpod-serverless-sync.yml`

## Required GitHub secrets

- `RUNPOD_API_KEY`
- `HF_TOKEN`
- `LTX_SERVICE_API_KEY`
- `RUNPOD_CONTAINER_REGISTRY_AUTH_ID` (optional, required for private GHCR image pull)

## Service code

- `ltx23-standalone-service/backend/app/serverless.py` (Runpod handler)
- `ltx23-standalone-service/infra/docker/Dockerfile.serverless`
- `ltx23-standalone-service/deploy/runpod/create_or_update_serverless_endpoint.sh`

