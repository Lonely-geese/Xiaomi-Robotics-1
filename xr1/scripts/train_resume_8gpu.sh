#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XR1_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${XR1_DIR}/.." && pwd)"
VENV_ACTIVATE="${REPO_DIR}/.venv/bin/activate"
CUDA_TOOLKIT="/root/yxq/software/miniconda3/envs/lingbotvla"
DATA_DIR="${XR1_DIR}/data/Data_0904_clean_v2_xr1"
CHECKPOINT="${XR1_DIR}/outputs/project_xiaomi-robotics-1/tianji_0903_g05_bs128_ep5/last.ckpt/checkpoint/mp_rank_00_model_states.pt"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    echo "Virtual environment not found: ${VENV_ACTIVATE}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -x "${CUDA_TOOLKIT}/bin/nvcc" ]]; then
    echo "CUDA Toolkit not found: ${CUDA_TOOLKIT}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
cd "${XR1_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME="${CUDA_TOOLKIT}"
# Keep the project virtualenv ahead of the CUDA conda environment so torchrun
# uses this project's Python and dependencies.
export PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export RESOURCE_GPU=8
export HYDRA_FULL_ERROR=1
export NETWORK_XR1_EPISODES="${DATA_DIR}/episodes"
export NETWORK_XR1_NORMALIZATION="${DATA_DIR}/normalize.json"

# The original DeepSpeed checkpoint used four ranks, so an eight-GPU run loads
# its model weights and starts fresh optimizer/scheduler state for two epochs.
# Per-GPU batch size 16 preserves the original global batch size of 128.
bash scripts/train.sh \
    data=network \
    model=posttrain \
    trainer.project=xiaomi-robotics-1 \
    trainer.exp_name=tianji_0903_g05_bs128_ep5_8gpu_continue \
    trainer.default_root_dir="${XR1_DIR}/outputs" \
    +trainer.devices=8 \
    +trainer.max_epochs=2 \
    trainer.max_steps=6222 \
    trainer.save_interval=3111 \
    data.params.train_datasets.batch_size=16 \
    model.params.pretrained="${CHECKPOINT}" \
    trainer.ckpt_path=null \
    trainer.optimizer.type=torch.optim.AdamW \
    +trainer.optimizer.params.fused=true \
    trainer.scheduler.params.num_warmup_steps=0 \
    trainer.scheduler.params.max_lr=0.0000105
