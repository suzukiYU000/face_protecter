#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/uv_runtime_common.sh"

MODE="${1:-smoke}"
PROFILE="${2:-${TORCH_PROFILE:-auto}}"
shift $(( $# > 0 ? 1 : 0 )) || true
if [[ $# -gt 0 ]]; then
  shift 1 || true
fi
EXTRA_TRAIN_ARGS=("$@")

WITH_ONNX="${WITH_ONNX:-0}"
MODEL="${MODEL:-models/yolo26l.pt}"
EPOCHS="${EPOCHS:-0}"
IMGSZ="${IMGSZ:-0}"
BATCH="${BATCH:-0}"
WORKERS="${WORKERS:-0}"
RUN_NAME="${YOLO_RUN_NAME:-}"
COPY_BEST_TO="${COPY_BEST_TO:-}"
NO_SYNC="${NO_SYNC:-0}"
UV_EXTRA_ARGS=()

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Run ./setup_uv.sh first." >&2
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
fi

configure_uv_runtime "$SCRIPT_DIR" "face_mosaic_streamlit_bundle_uv"

if [[ "$EPOCHS" == "0" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    EPOCHS=1
  else
    EPOCHS=50
  fi
fi

if [[ "$IMGSZ" == "0" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    IMGSZ=320
  else
    IMGSZ=512
  fi
fi

if [[ "$BATCH" == "0" ]]; then
  BATCH=4
fi

if [[ -z "$RUN_NAME" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    RUN_NAME="smoke"
  else
    RUN_NAME="full_wider_face"
  fi
fi

if [[ -z "$COPY_BEST_TO" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    COPY_BEST_TO="models/yolo26l_face_smoke.pt"
  else
    COPY_BEST_TO="models/yolo26l_face_full.pt"
  fi
fi

printf '\n=== YOLO26 / uv training ===\n'
echo "Mode: $MODE"
echo "Profile: $PROFILE"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Image size: $IMGSZ"
echo "Batch: $BATCH"
echo "Workers: $WORKERS"
echo "TMP: $TMP"
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  echo "WSL mounted-drive workaround: enabled"
  echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
  echo "UV_CACHE_DIR: $UV_CACHE_DIR"
  echo "UV_LINK_MODE: $UV_LINK_MODE"
fi
echo

if [[ "$NO_SYNC" != "1" ]]; then
  uv sync --locked --extra "$PROFILE" "${UV_EXTRA_ARGS[@]}"
fi

RUN_ARGS=(run)
if [[ "$NO_SYNC" == "1" ]]; then
  RUN_ARGS+=(--no-sync)
fi
RUN_ARGS+=(--extra "$PROFILE")
RUN_ARGS+=("${UV_EXTRA_ARGS[@]}")
RUN_ARGS+=(-- python train_face_detector.py)

uv "${RUN_ARGS[@]}" \
  --mode "$MODE" \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --name "$RUN_NAME" \
  --copy-best-to "$COPY_BEST_TO" \
  "${EXTRA_TRAIN_ARGS[@]}"
