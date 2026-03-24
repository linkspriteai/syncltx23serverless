#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${1:-"$SCRIPT_DIR/.env.runpod.serverless.sync-h200"}

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
require_var RUNPOD_IMAGE
require_var RUNPOD_TEMPLATE_NAME
require_var RUNPOD_ENDPOINT_NAME
require_var RUNPOD_GPU_TYPE
require_var SERVICE_API_KEY
require_var HF_TOKEN

API_BASE=${RUNPOD_API_BASE:-https://rest.runpod.io/v1}
RUNPOD_GPU_COUNT=${RUNPOD_GPU_COUNT:-1}
RUNPOD_WORKERS_MIN=${RUNPOD_WORKERS_MIN:-0}
RUNPOD_WORKERS_MAX=${RUNPOD_WORKERS_MAX:-1}
RUNPOD_IDLE_TIMEOUT=${RUNPOD_IDLE_TIMEOUT:-1}
RUNPOD_EXECUTION_TIMEOUT_MS=${RUNPOD_EXECUTION_TIMEOUT_MS:-1800000}
RUNPOD_SCALER_TYPE=${RUNPOD_SCALER_TYPE:-QUEUE_DELAY}
RUNPOD_SCALER_VALUE=${RUNPOD_SCALER_VALUE:-1}
RUNPOD_FLASHBOOT=${RUNPOD_FLASHBOOT:-true}
RUNPOD_ALLOWED_CUDA=${RUNPOD_ALLOWED_CUDA:-13.0}
RUNPOD_MIN_CUDA_VERSION=${RUNPOD_MIN_CUDA_VERSION:-13.0}
RUNPOD_CONTAINER_DISK_GB=${RUNPOD_CONTAINER_DISK_GB:-160}
RUNPOD_VOLUME_GB=${RUNPOD_VOLUME_GB:-500}
RUNPOD_VOLUME_MOUNT_PATH=${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}
RUNPOD_COMPUTE_TYPE=${RUNPOD_COMPUTE_TYPE:-GPU}
RUNPOD_TEMPLATE_CATEGORY=${RUNPOD_TEMPLATE_CATEGORY:-NVIDIA}
RUNPOD_TEMPLATE_README=${RUNPOD_TEMPLATE_README:-LTX23 sync AV serverless worker}
RUNPOD_DRY_RUN=${RUNPOD_DRY_RUN:-false}

api_call() {
  local method=$1
  local path=$2
  local body=${3:-}
  local url="${API_BASE}${path}"
  local attempts=${RUNPOD_API_RETRIES:-3}
  local sleep_s=${RUNPOD_API_RETRY_SLEEP_SECONDS:-2}
  local attempt=1

  while (( attempt <= attempts )); do
    local tmp
    tmp=$(mktemp)
    local status
    if [[ -n "$body" ]]; then
      status=$(curl -sS -o "$tmp" -w "%{http_code}" -X "$method" "$url" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        --data "$body")
    else
      status=$(curl -sS -o "$tmp" -w "%{http_code}" -X "$method" "$url" \
        -H "Authorization: Bearer $RUNPOD_API_KEY")
    fi

    if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
      cat "$tmp"
      rm -f "$tmp"
      return 0
    fi

    if [[ "$status" =~ ^5[0-9][0-9]$ && $attempt -lt $attempts ]]; then
      echo "Runpod API transient error status=$status method=$method path=$path (attempt $attempt/$attempts), retrying..." >&2
      rm -f "$tmp"
      sleep "$sleep_s"
      attempt=$((attempt + 1))
      continue
    fi

    echo "Runpod API request failed status=$status method=$method path=$path" >&2
    echo "Runpod response body:" >&2
    cat "$tmp" >&2
    rm -f "$tmp"
    return 1
  done
}

CSV_SPLIT_SCRIPT='
import json
import os
import sys

raw = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
if not raw:
    print("[]")
    raise SystemExit(0)
parts = [p.strip() for p in raw.split(",") if p.strip()]
print(json.dumps(parts))
'

APP_ENV_JSON=$(python3 - <<'PY'
import json
import os

payload = {}
for key, value in os.environ.items():
    if key.startswith(("LTX_", "HF_", "SERVICE_", "DATA_", "FRAME_", "PYTORCH_", "TORCH_")):
        payload[key] = value
print(json.dumps(payload, separators=(",", ":")))
PY
)

GPU_TYPE_IDS_JSON=$(python3 -c "$CSV_SPLIT_SCRIPT" "${RUNPOD_GPU_TYPE_IDS:-$RUNPOD_GPU_TYPE}")
DATA_CENTER_IDS_JSON=$(python3 -c "$CSV_SPLIT_SCRIPT" "${RUNPOD_DATA_CENTER_IDS:-}")
ALLOWED_CUDA_JSON=$(python3 -c "$CSV_SPLIT_SCRIPT" "${RUNPOD_ALLOWED_CUDA:-}")

TEMPLATE_CREATE_PAYLOAD=$(APP_ENV_JSON="$APP_ENV_JSON" \
GPU_TYPE_IDS_JSON="$GPU_TYPE_IDS_JSON" \
DATA_CENTER_IDS_JSON="$DATA_CENTER_IDS_JSON" \
ALLOWED_CUDA_JSON="$ALLOWED_CUDA_JSON" \
python3 - <<'PY'
import json
import os

def to_bool(v: str, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

payload = {
    "name": os.environ["RUNPOD_TEMPLATE_NAME"],
    "imageName": os.environ["RUNPOD_IMAGE"],
    "isServerless": True,
    "category": os.getenv("RUNPOD_TEMPLATE_CATEGORY", "NVIDIA"),
    "containerDiskInGb": int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "160")),
    # Runpod serverless template API may default this when omitted; enforce zero.
    "volumeInGb": 0,
    "dockerStartCmd": ["python", "-u", "backend/app/serverless.py"],
    "readme": os.getenv("RUNPOD_TEMPLATE_README", "LTX23 sync AV serverless worker"),
    "env": json.loads(os.environ["APP_ENV_JSON"]),
}

registry_auth_id = str(os.getenv("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "")).strip()
if registry_auth_id:
    payload["containerRegistryAuthId"] = registry_auth_id

print(json.dumps(payload, separators=(",", ":")))
PY
)

TEMPLATE_UPDATE_PAYLOAD=$(APP_ENV_JSON="$APP_ENV_JSON" \
python3 - <<'PY'
import json
import os

payload = {
    "name": os.environ["RUNPOD_TEMPLATE_NAME"],
    "imageName": os.environ["RUNPOD_IMAGE"],
    "containerDiskInGb": int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "160")),
    # Keep zero during update to avoid server-side defaulting to unsupported value.
    "volumeInGb": 0,
    "dockerStartCmd": ["python", "-u", "backend/app/serverless.py"],
    "readme": os.getenv("RUNPOD_TEMPLATE_README", "LTX23 sync AV serverless worker"),
    "env": json.loads(os.environ["APP_ENV_JSON"]),
}

registry_auth_id = str(os.getenv("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "")).strip()
if registry_auth_id:
    payload["containerRegistryAuthId"] = registry_auth_id

print(json.dumps(payload, separators=(",", ":")))
PY
)

ENDPOINT_CREATE_PAYLOAD_BASE=$(GPU_TYPE_IDS_JSON="$GPU_TYPE_IDS_JSON" \
DATA_CENTER_IDS_JSON="$DATA_CENTER_IDS_JSON" \
ALLOWED_CUDA_JSON="$ALLOWED_CUDA_JSON" \
python3 - <<'PY'
import json
import os

def to_bool(v: str, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

payload = {
    "name": os.environ["RUNPOD_ENDPOINT_NAME"],
    "computeType": os.getenv("RUNPOD_COMPUTE_TYPE", "GPU"),
    "gpuCount": int(os.getenv("RUNPOD_GPU_COUNT", "1")),
    "gpuTypeIds": json.loads(os.environ["GPU_TYPE_IDS_JSON"]),
    "workersMin": int(os.getenv("RUNPOD_WORKERS_MIN", "0")),
    "workersMax": int(os.getenv("RUNPOD_WORKERS_MAX", "1")),
    "idleTimeout": int(os.getenv("RUNPOD_IDLE_TIMEOUT", "5")),
    "executionTimeoutMs": int(os.getenv("RUNPOD_EXECUTION_TIMEOUT_MS", "1800000")),
    "scalerType": os.getenv("RUNPOD_SCALER_TYPE", "QUEUE_DELAY"),
    "scalerValue": int(os.getenv("RUNPOD_SCALER_VALUE", "1")),
    "flashboot": to_bool(os.getenv("RUNPOD_FLASHBOOT", "true"), True),
}

allowed_cuda = json.loads(os.environ["ALLOWED_CUDA_JSON"])
if allowed_cuda:
    payload["allowedCudaVersions"] = allowed_cuda

min_cuda = str(os.getenv("RUNPOD_MIN_CUDA_VERSION", "")).strip()
if min_cuda:
    payload["minCudaVersion"] = min_cuda

network_volume_id = str(os.getenv("RUNPOD_NETWORK_VOLUME_ID", "")).strip()
if network_volume_id:
    payload["networkVolumeId"] = network_volume_id

data_centers = json.loads(os.environ["DATA_CENTER_IDS_JSON"])
if data_centers:
    payload["dataCenterIds"] = data_centers

vcpu_count = str(os.getenv("RUNPOD_VCPU_COUNT", "")).strip()
if vcpu_count:
    payload["vcpuCount"] = int(vcpu_count)

print(json.dumps(payload, separators=(",", ":")))
PY
)

ENDPOINT_UPDATE_PAYLOAD_BASE=$(GPU_TYPE_IDS_JSON="$GPU_TYPE_IDS_JSON" \
DATA_CENTER_IDS_JSON="$DATA_CENTER_IDS_JSON" \
ALLOWED_CUDA_JSON="$ALLOWED_CUDA_JSON" \
python3 - <<'PY'
import json
import os

def to_bool(v: str, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

payload = {
    "name": os.environ["RUNPOD_ENDPOINT_NAME"],
    "gpuCount": int(os.getenv("RUNPOD_GPU_COUNT", "1")),
    "gpuTypeIds": json.loads(os.environ["GPU_TYPE_IDS_JSON"]),
    "workersMin": int(os.getenv("RUNPOD_WORKERS_MIN", "0")),
    "workersMax": int(os.getenv("RUNPOD_WORKERS_MAX", "1")),
    "idleTimeout": int(os.getenv("RUNPOD_IDLE_TIMEOUT", "5")),
    "executionTimeoutMs": int(os.getenv("RUNPOD_EXECUTION_TIMEOUT_MS", "1800000")),
    "scalerType": os.getenv("RUNPOD_SCALER_TYPE", "QUEUE_DELAY"),
    "scalerValue": int(os.getenv("RUNPOD_SCALER_VALUE", "1")),
    "flashboot": to_bool(os.getenv("RUNPOD_FLASHBOOT", "true"), True),
}

allowed_cuda = json.loads(os.environ["ALLOWED_CUDA_JSON"])
if allowed_cuda:
    payload["allowedCudaVersions"] = allowed_cuda

min_cuda = str(os.getenv("RUNPOD_MIN_CUDA_VERSION", "")).strip()
if min_cuda:
    payload["minCudaVersion"] = min_cuda

network_volume_id = str(os.getenv("RUNPOD_NETWORK_VOLUME_ID", "")).strip()
if network_volume_id:
    payload["networkVolumeId"] = network_volume_id

data_centers = json.loads(os.environ["DATA_CENTER_IDS_JSON"])
if data_centers:
    payload["dataCenterIds"] = data_centers

vcpu_count = str(os.getenv("RUNPOD_VCPU_COUNT", "")).strip()
if vcpu_count:
    payload["vcpuCount"] = int(vcpu_count)

print(json.dumps(payload, separators=(",", ":")))
PY
)

find_id_by_name() {
  local json_arr=$1
  local expected_name=$2
  python3 - <<'PY' "$json_arr" "$expected_name"
import json
import sys

items = json.loads(sys.argv[1] or "[]")
name = str(sys.argv[2] or "").strip()
match = ""
for item in items:
    if str(item.get("name", "")).strip() == name:
        match = str(item.get("id", "")).strip()
        break
print(match)
PY
}

merge_with_template_id() {
  local base_json=$1
  local template_id=$2
  python3 - <<'PY' "$base_json" "$template_id"
import json
import sys

payload = json.loads(sys.argv[1])
payload["templateId"] = sys.argv[2]
print(json.dumps(payload, separators=(",", ":")))
PY
}

graphql_save_template() {
  local existing_template_id=${1:-}
  EXISTING_TEMPLATE_ID="$existing_template_id" \
  APP_ENV_JSON="$APP_ENV_JSON" \
  python3 - <<'PY'
import json
import os
import sys
import urllib.request

api_key = os.environ["RUNPOD_API_KEY"]
url = f"https://api.runpod.io/graphql?api_key={api_key}"

env_obj = json.loads(os.environ["APP_ENV_JSON"])
env_items = ", ".join(
    [
        f'{{ key: {json.dumps(str(k))}, value: {json.dumps(str(v))} }}'
        for k, v in env_obj.items()
    ]
)

fields = []
existing_id = str(os.getenv("EXISTING_TEMPLATE_ID", "")).strip()
if existing_id:
    fields.append(f"id: {json.dumps(existing_id)}")
fields.append(f'containerDiskInGb: {int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "160"))}')
fields.append(f'dockerArgs: {json.dumps("python -u backend/app/serverless.py")}')
fields.append(f"env: [ {env_items} ]")
fields.append(f'imageName: {json.dumps(os.environ["RUNPOD_IMAGE"])}')
fields.append("isServerless: true")
fields.append(f'name: {json.dumps(os.environ["RUNPOD_TEMPLATE_NAME"])}')
fields.append(
    f'readme: {json.dumps(os.getenv("RUNPOD_TEMPLATE_README", "LTX23 sync AV serverless worker"))}'
)
fields.append("volumeInGb: 0")
registry_auth_id = str(os.getenv("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "")).strip()
if registry_auth_id:
    fields.append(f"containerRegistryAuthId: {json.dumps(registry_auth_id)}")

query = (
    "mutation { saveTemplate(input: { "
    + ", ".join(fields)
    + " }) { id name isServerless imageName } }"
)

req = urllib.request.Request(
    url,
    data=json.dumps({"query": query}).encode("utf-8"),
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    payload = json.loads(resp.read().decode("utf-8"))

if payload.get("errors"):
    print(f"Runpod GraphQL saveTemplate failed: {payload['errors']}", file=sys.stderr)
    raise SystemExit(1)

tmpl = payload.get("data", {}).get("saveTemplate") or {}
template_id = str(tmpl.get("id") or "").strip()
if not template_id:
    print(f"Runpod GraphQL saveTemplate missing id: {payload}", file=sys.stderr)
    raise SystemExit(1)

print(template_id)
PY
}

if [[ "$RUNPOD_DRY_RUN" == "true" ]]; then
  echo "Template CREATE payload:"
  echo "$TEMPLATE_CREATE_PAYLOAD" | python3 -m json.tool
  echo
  echo "Template UPDATE payload:"
  echo "$TEMPLATE_UPDATE_PAYLOAD" | python3 -m json.tool
  echo
  echo "Endpoint CREATE payload base:"
  echo "$ENDPOINT_CREATE_PAYLOAD_BASE" | python3 -m json.tool
  echo
  echo "Endpoint UPDATE payload base:"
  echo "$ENDPOINT_UPDATE_PAYLOAD_BASE" | python3 -m json.tool
  exit 0
fi

TEMPLATE_ID=${RUNPOD_TEMPLATE_ID:-}
if [[ -z "${TEMPLATE_ID}" ]]; then
  TEMPLATES_JSON=$(api_call GET /templates)
  TEMPLATE_ID=$(find_id_by_name "$TEMPLATES_JSON" "$RUNPOD_TEMPLATE_NAME")
fi

if [[ -z "${TEMPLATE_ID}" ]]; then
  TEMPLATE_ID=$(graphql_save_template "")
  echo "template created: $TEMPLATE_ID"
else
  TEMPLATE_ID=$(graphql_save_template "$TEMPLATE_ID")
  echo "template updated: $TEMPLATE_ID"
fi

ENDPOINT_ID=${RUNPOD_ENDPOINT_ID:-}
if [[ -z "${ENDPOINT_ID}" ]]; then
  ENDPOINTS_JSON=$(api_call GET /endpoints)
  ENDPOINT_ID=$(find_id_by_name "$ENDPOINTS_JSON" "$RUNPOD_ENDPOINT_NAME")
fi

ENDPOINT_CREATE_PAYLOAD=$(merge_with_template_id "$ENDPOINT_CREATE_PAYLOAD_BASE" "$TEMPLATE_ID")
ENDPOINT_UPDATE_PAYLOAD=$(merge_with_template_id "$ENDPOINT_UPDATE_PAYLOAD_BASE" "$TEMPLATE_ID")

if [[ -z "${ENDPOINT_ID}" ]]; then
  CREATED_ENDPOINT_JSON=$(api_call POST /endpoints "$ENDPOINT_CREATE_PAYLOAD")
  ENDPOINT_ID=$(python3 - <<'PY' "$CREATED_ENDPOINT_JSON"
import json, sys
obj = json.loads(sys.argv[1])
print(str(obj.get("id", "")).strip())
PY
)
  if [[ -z "$ENDPOINT_ID" ]]; then
    echo "failed to create endpoint" >&2
    echo "$CREATED_ENDPOINT_JSON" >&2
    exit 1
  fi
  echo "endpoint created: $ENDPOINT_ID"
else
  UPDATED_ENDPOINT_JSON=$(api_call PATCH "/endpoints/${ENDPOINT_ID}" "$ENDPOINT_UPDATE_PAYLOAD")
  echo "endpoint updated: $ENDPOINT_ID"
  if [[ -n "${UPDATED_ENDPOINT_JSON:-}" ]]; then
    true
  fi
fi

RUNSYNC_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync"
RUN_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/run"

echo "template_id=$TEMPLATE_ID"
echo "endpoint_id=$ENDPOINT_ID"
echo "runsync_url=$RUNSYNC_URL"
echo "run_url=$RUN_URL"
echo
echo "Smoke test:"
echo "curl -X POST '$RUNSYNC_URL' \\"
echo "  -H 'Authorization: Bearer \$RUNPOD_API_KEY' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"input\":{\"healthcheck\":true}}'"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "template_id=$TEMPLATE_ID"
    echo "endpoint_id=$ENDPOINT_ID"
    echo "runsync_url=$RUNSYNC_URL"
    echo "run_url=$RUN_URL"
  } >> "$GITHUB_OUTPUT"
fi
