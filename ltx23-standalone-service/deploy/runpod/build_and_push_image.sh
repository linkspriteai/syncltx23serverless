#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

IMAGE_REPO=${1:-ghcr.io/linkspriteai/ltx23-sync}
IMAGE_TAG=${2:-$(date -u +%Y%m%d-%H%M)-$(git -C "$ROOT_DIR" rev-parse --short HEAD)}
DOCKERFILE=${3:-"$ROOT_DIR/ltx23-standalone-service/infra/docker/Dockerfile.serverless"}

FULL_IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"

echo "building $FULL_IMAGE using $DOCKERFILE"
docker build \
  --platform linux/amd64 \
  -f "$DOCKERFILE" \
  -t "$FULL_IMAGE" \
  "$ROOT_DIR/ltx23-standalone-service"

echo "pushing $FULL_IMAGE"
docker push "$FULL_IMAGE"

echo "done: $FULL_IMAGE"
