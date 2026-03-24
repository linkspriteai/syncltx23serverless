#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${1:-"$SCRIPT_DIR/.env.runpod"}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

require_var() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "missing required variable: $name" >&2
    exit 1
  fi
}

require_var RUNPOD_API_KEY
require_var RUNPOD_NAME
require_var RUNPOD_GPU_TYPE
require_var RUNPOD_IMAGE
require_var SERVICE_API_KEY
require_var HF_TOKEN

RUNPOD_CLOUD_TYPE=${RUNPOD_CLOUD_TYPE:-SECURE}
RUNPOD_GPU_COUNT=${RUNPOD_GPU_COUNT:-1}
RUNPOD_PORTS=${RUNPOD_PORTS:-8080/http}
RUNPOD_CONTAINER_DISK_GB=${RUNPOD_CONTAINER_DISK_GB:-120}
RUNPOD_VOLUME_GB=${RUNPOD_VOLUME_GB:-400}
RUNPOD_VOLUME_MOUNT_PATH=${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}
RUNPOD_INTERRUPTIBLE=${RUNPOD_INTERRUPTIBLE:-false}
RUNPOD_SUPPORT_PUBLIC_IP=${RUNPOD_SUPPORT_PUBLIC_IP:-true}
RUNPOD_MIN_VCPU_PER_GPU=${RUNPOD_MIN_VCPU_PER_GPU:-8}
RUNPOD_MIN_RAM_PER_GPU=${RUNPOD_MIN_RAM_PER_GPU:-30}
RUNPOD_ALLOWED_CUDA=${RUNPOD_ALLOWED_CUDA:-13.0}
RUNPOD_POLL_SECONDS=${RUNPOD_POLL_SECONDS:-300}
RUNPOD_DRY_RUN=${RUNPOD_DRY_RUN:-false}

PAYLOAD=$(python3 - <<'PY'
import json
import os

def to_bool(v: str, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def split_csv(name: str):
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]

app_env = {}
for k, v in os.environ.items():
    if k.startswith(("LTX_", "SERVICE_", "HF_", "DATA_", "FRAME_", "PYTORCH_", "TORCH_")):
        app_env[k] = v

payload = {
    "name": os.environ["RUNPOD_NAME"],
    "cloudType": os.getenv("RUNPOD_CLOUD_TYPE", "SECURE"),
    "computeType": "GPU",
    "gpuCount": int(os.getenv("RUNPOD_GPU_COUNT", "1")),
    "gpuTypeIds": [os.environ["RUNPOD_GPU_TYPE"]],
    "gpuTypePriority": "availability",
    "imageName": os.environ["RUNPOD_IMAGE"],
    "ports": split_csv("RUNPOD_PORTS") or ["8080/http"],
    "containerDiskInGb": int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "120")),
    "volumeInGb": int(os.getenv("RUNPOD_VOLUME_GB", "400")),
    "volumeMountPath": os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace"),
    "interruptible": to_bool(os.getenv("RUNPOD_INTERRUPTIBLE"), False),
    "supportPublicIp": to_bool(os.getenv("RUNPOD_SUPPORT_PUBLIC_IP"), True),
    "minVCPUPerGPU": int(os.getenv("RUNPOD_MIN_VCPU_PER_GPU", "8")),
    "minRAMPerGPU": int(os.getenv("RUNPOD_MIN_RAM_PER_GPU", "30")),
    "allowedCudaVersions": [os.getenv("RUNPOD_ALLOWED_CUDA", "13.0")],
    "env": app_env,
}

country_codes = split_csv("RUNPOD_COUNTRY_CODES")
if country_codes:
    payload["countryCodes"] = country_codes

data_center_ids = split_csv("RUNPOD_DATA_CENTER_IDS")
if data_center_ids:
    payload["dataCenterIds"] = data_center_ids

network_volume_id = os.getenv("RUNPOD_NETWORK_VOLUME_ID", "").strip()
if network_volume_id:
    payload["networkVolumeId"] = network_volume_id

registry_auth_id = os.getenv("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "").strip()
if registry_auth_id:
    payload["containerRegistryAuthId"] = registry_auth_id

print(json.dumps(payload, separators=(",", ":")))
PY
)

if [[ "$RUNPOD_DRY_RUN" == "true" ]]; then
  echo "$PAYLOAD" | python3 -m json.tool
  exit 0
fi

CREATE_RESP=$(curl -sS -X POST \
  https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data "$PAYLOAD")

POD_ID=$(python3 - <<'PY' "$CREATE_RESP"
import json, sys
obj=json.loads(sys.argv[1])
print(obj.get("id", ""))
PY
)

if [[ -z "$POD_ID" ]]; then
  echo "failed to create pod:" >&2
  echo "$CREATE_RESP" >&2
  exit 1
fi

echo "pod created: $POD_ID"

declare -i elapsed=0
declare -i sleep_s=10
while (( elapsed < RUNPOD_POLL_SECONDS )); do
  RESP=$(curl -sS -X GET \
    "https://rest.runpod.io/v1/pods/$POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY")

  STATUS=$(python3 - <<'PY' "$RESP"
import json, sys
obj=json.loads(sys.argv[1])
print(obj.get("desiredStatus", ""))
PY
)

  PUBLIC_IP=$(python3 - <<'PY' "$RESP"
import json, sys
obj=json.loads(sys.argv[1])
print(obj.get("publicIp") or "")
PY
)

  PORT_8080=$(python3 - <<'PY' "$RESP"
import json, sys
obj=json.loads(sys.argv[1])
pm=obj.get("portMappings") or {}
print(pm.get("8080") or "")
PY
)

  echo "status=$STATUS elapsed=${elapsed}s"

  if [[ -n "$PUBLIC_IP" && -n "$PORT_8080" ]]; then
    echo "ready endpoint: http://$PUBLIC_IP:$PORT_8080"
    echo "health check: curl -H 'x-api-key: $SERVICE_API_KEY' http://$PUBLIC_IP:$PORT_8080/health"
    exit 0
  fi

  sleep "$sleep_s"
  elapsed=$((elapsed + sleep_s))
done

echo "pod created but endpoint not ready within ${RUNPOD_POLL_SECONDS}s" >&2
echo "check pod in console: https://console.runpod.io/pods" >&2
exit 2
