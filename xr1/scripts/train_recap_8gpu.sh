#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
export XR1_REPO_DIR
cd "$XR1_REPO_DIR/xr1"
MODE=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in train|smoke|config) ;; *) echo 'Usage: train_recap_8gpu.sh [train|smoke|config] [Hydra overrides]' >&2; exit 2 ;; esac
export RECAP_DATA_DIR=${RECAP_DATA_DIR:-"$PWD/data/recap_20261020"}
export PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$PWD/outputs/project_xiaomi-robotics-1/middle_mixed_old2000_5ep_lr2e6_20261013/last.ckpt/checkpoint/mp_rank_00_model_states.pt"}
[[ -f "$PRETRAINED_CKPT" && -f "$RECAP_DATA_DIR/prepared.json" && -f "$RECAP_DATA_DIR/advantages_ready.json" ]] || {
    echo 'Initial weights, prepared pools and completed advantage sidecars are required.' >&2; exit 2;
}
export RECAP_STEPS RECAP_WARMUP
read -r RECAP_STEPS RECAP_WARMUP < <(python - <<'PY'
import json,math,os
m=json.load(open(os.path.join(os.environ['RECAP_DATA_DIR'],'prepared.json')))
steps=max(m['sft_frames'],m['rollout_frames'])//128*10
assert steps > 0
print(steps,math.ceil(steps*.03))
PY
)
EXP_NAME=${EXP_NAME:-recap_10ep_bs256_20261020}
EXTRA=()
if [[ "$MODE" == smoke ]]; then
    EXP_NAME="${EXP_NAME}_smoke"
    EXTRA=(trainer.max_steps=2 trainer.max_epochs=1 +trainer.limit_train_batches=2 +trainer.enable_checkpointing=false)
fi
if [[ "$MODE" != config ]]; then
    RUN_DIR="$PWD/outputs/project_xiaomi-robotics-1/$EXP_NAME"
    if [[ -n "${RECAP_RESUME:-}" ]]; then
        [[ "$MODE" == train && -f "$RECAP_RESUME/restart_ready.json" && ! -f "$RUN_DIR/completed.json" ]] || {
            echo 'Recovery requires an unfinished run and a complete distributed checkpoint.' >&2; exit 3;
        }
        EXTRA+=("trainer.ckpt_path=$RECAP_RESUME")
    else
        [[ ! -e "$RUN_DIR/config.yaml" && ! -e "$RUN_DIR/completed.json" ]] || {
            echo "Refusing to overwrite $RUN_DIR" >&2; exit 3;
        }
    fi
    GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    GPU_COUNT=0
    while read -r used util; do
        GPU_COUNT=$((GPU_COUNT+1))
        (( used <= 1024 && util <= 5 )) || { echo 'Eight idle GPUs are required.' >&2; exit 3; }
    done <<< "${GPU_STATS//,/}"
    (( GPU_COUNT == 8 )) || { echo 'Exactly eight GPUs are required.' >&2; exit 3; }
    TASK_TMP_ROOT=${XR1_TMP_ROOT:-/tmp}
    mkdir -p "$TASK_TMP_ROOT"
    export TMPDIR
    TMPDIR=$(mktemp -d "$TASK_TMP_ROOT/xr1-recap.XXXXXX")
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESOURCE_GPU=8 WORLD_SIZE=1
export XR1_STRICT_BATCH=1 MAX_LENGTH=${MAX_LENGTH:-22000} OMP_NUM_THREADS=4
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
ARGS=(trainer.exp_name="$EXP_NAME" hydra.run.dir=. hydra.output_subdir=null)
if [[ "$MODE" == config ]]; then
    python tools/train_recap.py "${ARGS[@]}" "$@" --cfg job --resolve
else
    torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=localhost \
        --master_port="${MASTER_PORT:-29661}" tools/train_recap.py "${ARGS[@]}" "${EXTRA[@]}" "$@"
fi
