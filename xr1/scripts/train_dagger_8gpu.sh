#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
export XR1_REPO_DIR
cd "$XR1_REPO_DIR/xr1"
MODE=${1:-config}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in train|config|audit) ;; *) echo 'Usage: train_dagger_8gpu.sh [config|audit|train] [Hydra overrides]' >&2; exit 2 ;; esac
export DAGGER_DATA_DIR=${DAGGER_DATA_DIR:-"$PWD/data/hg_dagger_1563_20261021"}
export PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$PWD/outputs/project_xiaomi-robotics-1/middle_mixed_old2000_5ep_lr2e6_20261013/last.ckpt/checkpoint/mp_rank_00_model_states.pt"}
export XR1_STRICT_BATCH=1 MAX_LENGTH=${MAX_LENGTH:-22000} OMP_NUM_THREADS=4
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
EXP_NAME=${EXP_NAME:-hg_dagger_1563_10ep_bs256_20261021}
if [[ "$MODE" == audit || "$MODE" == train ]]; then
    python tools/verify_dagger_dataset.py --data "$DAGGER_DATA_DIR"
    [[ "$MODE" == train ]] || exit 0
fi
ARGS=(trainer.exp_name="$EXP_NAME" hydra.run.dir=. hydra.output_subdir=null)
if [[ "$MODE" == config ]]; then
    # Resolve derived epoch/scheduler settings without constructing the model.
    python - "trainer.exp_name=$EXP_NAME" "$@" <<'PY'
import json, os, sys
from hydra import compose, initialize_config_dir
sys.path.insert(0, 'tools')
from train_dagger import configure
with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
    cfg, root, manifest = configure(compose(config_name='dagger', overrides=sys.argv[1:]))
print(cfg.pretty_text)
print(json.dumps(dict(steps_per_epoch=cfg.trainer.max_steps//cfg.trainer.max_epochs,
    total_steps=cfg.trainer.max_steps, counts=manifest['counts'], training_started=False), indent=2))
PY
    exit 0
fi
if [[ -n "${DAGGER_RESUME:-}" ]]; then
    ARGS+=("trainer.ckpt_path=$DAGGER_RESUME")
fi
GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
GPU_COUNT=0
while read -r used util; do
    GPU_COUNT=$((GPU_COUNT+1))
    (( used <= 1024 && util <= 5 )) || { echo 'Eight idle GPUs are required; another job is running.' >&2; exit 3; }
done <<< "${GPU_STATS//,/}"
(( GPU_COUNT == 8 )) || { echo 'Exactly eight GPUs are required.' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESOURCE_GPU=8 WORLD_SIZE=1
TASK_TMP_ROOT=${XR1_TMP_ROOT:-/tmp}
mkdir -p "$TASK_TMP_ROOT"
export TMPDIR
TMPDIR=$(mktemp -d "$TASK_TMP_ROOT/xr1-dagger.XXXXXX")
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=localhost \
    --master_port="${MASTER_PORT:-29671}" tools/train_dagger.py "${ARGS[@]}" "$@"
