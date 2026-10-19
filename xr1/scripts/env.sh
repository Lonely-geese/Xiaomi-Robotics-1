#!/usr/bin/env bash
# Source this file from any working directory.
XR1_REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$XR1_REPO_DIR/.venv/bin/activate"
export XR1_VLM_PATH=${XR1_VLM_PATH:-"$XR1_REPO_DIR/models/Qwen3-VL-4B-Instruct"}
export PYTHONPATH="$XR1_REPO_DIR/xr1${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export XR1_VIDEO_BACKEND=${XR1_VIDEO_BACKEND:-pyav}
export WANDB_MODE=${WANDB_MODE:-offline}
# Training uses PyTorch fused AdamW, so no DeepSpeed CUDA-13 JIT extension is needed.
export DS_BUILD_OPS=0
