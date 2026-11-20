#!/usr/bin/env bash
set -euo pipefail
RECAP_ROLLOUT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export RECAP_PACKAGE="${RECAP_PACKAGE:-/data/checkpoints/kele/usb/usb_recap_epoch10_step15490}"
export RECAP_SERVER_PYTHON="${RECAP_SERVER_PYTHON:-python3}"
export RECAP_CONTROL_PYTHON="${RECAP_CONTROL_PYTHON:-python3}"
export RECAP_RECORD_OUT="${RECAP_RECORD_OUT:-$RECAP_ROLLOUT_ROOT/recordings/usb_recap_epoch10_step15490}"
export XR1_RESET_CONFIG="${XR1_RESET_CONFIG:-$RECAP_ROLLOUT_ROOT/config/vr_reset.yaml}"
export XR1_VLM_PATH="$RECAP_PACKAGE/assets/Qwen3-VL-4B-Instruct"
export PYTHONPATH="$RECAP_ROLLOUT_ROOT/server:$RECAP_ROLLOUT_ROOT/runtime"
export PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$RECAP_ROLLOUT_ROOT/control/runs"
