#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/uv_runtime_common.sh"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROFILE="${1:-${TORCH_PROFILE:-auto}}"
WITH_ONNX="${WITH_ONNX:-0}"
UV_EXTRA_ARGS=()
PROFILE_FILE="$SCRIPT_DIR/.uv-profile"

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv was not found. Install it first with: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
  exit 1
fi

choose_profile() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "cpu"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local cuda_version
    cuda_version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
    if [[ -n "$cuda_version" ]]; then
      local major="${cuda_version%%.*}"
      local minor="${cuda_version#*.}"
      if (( major >= 13 )); then
        echo "cu130"
        return
      elif (( major == 12 && minor >= 8 )); then
        echo "cu128"
        return
      elif (( major == 12 && minor >= 6 )); then
        echo "cu126"
        return
      fi
    fi
  fi

  echo "cpu"
}

if [[ "$PROFILE" == "auto" ]]; then
  PROFILE="$(choose_profile)"
fi

if [[ "$PROFILE" != "cpu" && "$PROFILE" != "cu126" && "$PROFILE" != "cu128" && "$PROFILE" != "cu130" ]]; then
  echo "Invalid profile: $PROFILE" >&2
  exit 1
fi

if [[ "$WITH_ONNX" == "1" || "$WITH_ONNX" == "true" || "$WITH_ONNX" == "TRUE" ]]; then
  UV_EXTRA_ARGS+=(--extra onnx)
  WITH_ONNX="1"
else
  WITH_ONNX="0"
fi

configure_uv_runtime "$SCRIPT_DIR" "face_mosaic_streamlit_bundle_uv"

printf '\n=== YOLO26 / uv setup ===\n'
echo "Profile: $PROFILE"
echo "TMP: $TMP"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  echo "WSL mounted-drive workaround: enabled"
  echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
  echo "UV_CACHE_DIR: $UV_CACHE_DIR"
  echo "UV_LINK_MODE: $UV_LINK_MODE"
fi
if (( ${#UV_EXTRA_ARGS[@]} > 0 )); then
  echo "Extra: onnx"
fi
echo

echo "[1/4] Installing Python ${PYTHON_VERSION} via uv"
uv python install "$PYTHON_VERSION"

echo "[2/4] Syncing dependencies from uv.lock"
uv sync --locked --extra "$PROFILE" "${UV_EXTRA_ARGS[@]}"

persist_uv_runtime_profile "$PROFILE_FILE" "$PROFILE" "$WITH_ONNX"

echo "[3/4] Verifying torch runtime"
uv run --no-sync --extra "$PROFILE" "${UV_EXTRA_ARGS[@]}" -- python verify_torch_env.py || true

echo "[4/4] Done"
cat <<EOF

Saved launch profile: $PROFILE_FILE

Recommended next step:
  ./run_app.sh

Alternative launch command:
  uv run --no-sync --extra $PROFILE ${UV_EXTRA_ARGS[*]} -- streamlit run face_mosaic_streamlit_app.py
EOF
