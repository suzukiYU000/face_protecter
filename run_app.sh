#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/uv_runtime_common.sh"
PROFILE_FILE="$SCRIPT_DIR/.uv-profile"
UV_PROFILE="cpu"
UV_WITH_ONNX="0"
UV_EXTRA_ARGS=()
STREAMLIT_ARGS=()

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv was not found. Install it first with: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
  exit 1
fi

if [[ ! -f "$PROFILE_FILE" ]]; then
  echo "Launch profile was not found. Run ./setup_uv.sh first." >&2
  exit 1
fi

load_uv_runtime_profile "$PROFILE_FILE"

if [[ "$UV_WITH_ONNX" == "1" ]]; then
  UV_EXTRA_ARGS+=(--extra onnx)
fi

if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  STREAMLIT_ARGS+=(--server.headless true)
fi

uv run --no-sync --extra "$UV_PROFILE" "${UV_EXTRA_ARGS[@]}" -- \
  streamlit run face_mosaic_streamlit_app.py "${STREAMLIT_ARGS[@]}"
