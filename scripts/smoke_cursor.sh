#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
TOKEN="${TOKEN:-${CODEX_GATEWAY_TOKEN:-}}"
MODEL="${MODEL:-gpt-5.4}"
FIXTURE_DIR="${FIXTURE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/fixtures/cursor}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"

declare -a AUTH_ARGS=()
if [[ -n "${TOKEN}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${TOKEN}")
fi

post_fixture() {
  local fixture_name="$1"
  local mode="$2"
  local fixture_path="${FIXTURE_DIR}/${fixture_name}"

  if [[ ! -f "${fixture_path}" ]]; then
    echo "[cursor-smoke] missing fixture: ${fixture_path}" >&2
    exit 1
  fi

  echo "[cursor-smoke] POST ${BASE_URL}/chat/completions (${fixture_name}; ${mode})"
  set +e
  uv run python -c '
import json
import sys

fixture_path, model, mode = sys.argv[1:4]
payload = json.load(open(fixture_path, encoding="utf-8"))["request"]
payload["model"] = model
if mode == "stream":
    payload["stream"] = True
print(json.dumps(payload, separators=(",", ":")))
' "${fixture_path}" "${MODEL}" "${mode}" | \
    curl -sS --max-time "${TIMEOUT_SECONDS}" ${mode:+-N} "${BASE_URL}/chat/completions" \
      -H "Content-Type: application/json" \
      "${AUTH_ARGS[@]}" \
      -d @- | head -c 2000
  local statuses=("${PIPESTATUS[@]}")
  local payload_status=${statuses[0]}
  local curl_status=${statuses[1]}
  local head_status=${statuses[2]}
  set -e
  if [[ "${payload_status}" -ne 0 ]]; then
    exit "${payload_status}"
  fi
  if [[ "${curl_status}" -ne 0 && "${curl_status}" -ne 23 ]]; then
    exit "${curl_status}"
  fi
  if [[ "${head_status}" -ne 0 ]]; then
    exit "${head_status}"
  fi
  echo
  echo
}

echo "[cursor-smoke] gateway=${BASE_URL}"
echo "[cursor-smoke] health"
curl -sS --max-time "${TIMEOUT_SECONDS}" "${BASE_URL%/v1}/healthz" "${AUTH_ARGS[@]}" | cat
echo
echo

post_fixture "text_chat_request.json" "stream"
post_fixture "tool_prompt_request.json" "stream"
post_fixture "tool_result_followup_request.json" "stream"
post_fixture "image_upload_request.json" ""

echo "[cursor-smoke] done"
