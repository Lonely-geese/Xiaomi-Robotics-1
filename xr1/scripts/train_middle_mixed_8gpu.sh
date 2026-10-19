#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
cd "$XR1_REPO_DIR/xr1"

MODE=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in train|smoke|config) ;; *) echo 'Usage: train_middle_mixed_8gpu.sh [train|smoke|config] [Hydra overrides...]' >&2; exit 2 ;; esac
export ETHERNET_XR1_DIR=${MIXED_DATA_DIR:-"$PWD/data/ethernet_middle_mixed_old2000_20261013"}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
MICRO_BATCH=${MICRO_BATCH:-32}
EPOCHS=${EPOCHS:-5}
LOG_BACKEND=${LOG_BACKEND:-swanlab}
LOG_MODE=${LOG_MODE:-online}
export NO_PROXY="${NO_PROXY:-},swanlab.cn,.swanlab.cn"
export no_proxy="$NO_PROXY"
export XR1_MIXED_EPOCHS="$EPOCHS"
(( GLOBAL_BATCH_SIZE > 0 && MICRO_BATCH > 0 && EPOCHS > 0 && GLOBAL_BATCH_SIZE % (8 * MICRO_BATCH) == 0 )) || {
    echo 'Global batch must be divisible by eight times the positive microbatch.' >&2; exit 2;
}
ACCUMULATE=$((GLOBAL_BATCH_SIZE / (8 * MICRO_BATCH)))
PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$PWD/outputs/project_xiaomi-robotics-1/left_relative_10ep_bs256_20261002/last.ckpt/checkpoint/mp_rank_00_model_states.pt"}
[[ -f "$PRETRAINED_CKPT" && -f "$ETHERNET_XR1_DIR/prepared.json" && -f "$ETHERNET_XR1_DIR/normalize.json" && -d "$ETHERNET_XR1_DIR/episodes" ]] || {
    echo 'The initial model and completed mixed dataset are required.' >&2; exit 2;
}
read -r FRAMES STEPS WARMUP TAIL < <(python - <<'PY'
import json, math, os
with open(os.path.join(os.environ['ETHERNET_XR1_DIR'], 'prepared.json')) as handle:
    metadata = json.load(handle)
assert metadata['active_action_dims'] == list(range(7))
frames = int(metadata.get('train_frames', metadata.get('splits', {}).get('train', {}).get('frames', 0)))
batch = int(os.environ['GLOBAL_BATCH_SIZE'])
steps = (frames // batch) * int(os.environ['XR1_MIXED_EPOCHS'])
assert steps > 0
print(frames, steps, math.ceil(steps * 0.03), frames % batch)
PY
)
EXP_NAME=${EXP_NAME:-middle_mixed_old2000_5ep_lr2e6_20261013}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/outputs"}
EXTRA=()
if [[ "$MODE" == smoke ]]; then
    LOG_BACKEND=csv
    LOG_MODE=offline
    EXP_NAME="${EXP_NAME}_smoke_mb${MICRO_BATCH}"
    EXTRA=(trainer.max_steps=2 ++trainer.max_epochs=1 +trainer.limit_train_batches="$((2 * ACCUMULATE))" +trainer.enable_checkpointing=false)
fi
RUN_DIR="$OUTPUT_ROOT/project_xiaomi-robotics-1/$EXP_NAME"
if [[ "$MODE" != config ]]; then
    if [[ -f "$RUN_DIR/config.yaml" || -f "$RUN_DIR/completed.json" || -d "$RUN_DIR/csv" || -e "$RUN_DIR/last.ckpt" ]]; then
        echo "Refusing to overwrite an existing training run: $RUN_DIR" >&2; exit 3
    fi
    GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    GPU_COUNT=0
    while read -r used util; do
        GPU_COUNT=$((GPU_COUNT + 1))
        (( used <= 1024 && util <= 5 )) || { echo 'All eight GPUs must be idle before launch.' >&2; exit 3; }
    done <<< "${GPU_STATS//,/}"
    (( GPU_COUNT == 8 )) || { echo 'Exactly eight GPUs are required.' >&2; exit 3; }
    # multiprocessing creates AF_UNIX sockets here; their paths must fit 108 bytes.
    # Keep caches on CFS without using the much longer experiment directory.
    TASK_TMP_ROOT=${XR1_TMP_ROOT:-/mnt/cfs/7rnh3z/.xr1tmp}
    mkdir -p "$TASK_TMP_ROOT"
    export TMPDIR
    TMPDIR=$(mktemp -d "$TASK_TMP_ROOT/middle.XXXXXX")
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESOURCE_GPU=8 WORLD_SIZE=1
export XR1_STRICT_BATCH=1 OMP_NUM_THREADS=4
export MASTER_PORT=${MASTER_PORT:-29643}
export MAX_LENGTH=${MAX_LENGTH:-20000}
ARGS=(data=ethernet_left model=posttrain
    trainer.project=xiaomi-robotics-1 trainer.exp_name="$EXP_NAME" trainer.default_root_dir="$OUTPUT_ROOT"
    +trainer.devices=8 trainer.num_nodes=1 ++trainer.max_epochs="$EPOCHS"
    trainer.max_steps="$STEPS" trainer.accumulate_grad_batches="$ACCUMULATE" trainer.ckpt_path=null
    trainer.scheduler.params.num_training_steps="$STEPS" trainer.scheduler.params.num_warmup_steps="$WARMUP"
    trainer.scheduler.params.warmup_lr_start=5e-8 trainer.scheduler.params.max_lr=2e-6 trainer.scheduler.params.min_lr=5e-7
    data.params.train_datasets.batch_size="$MICRO_BATCH" data.params.global_batch_size="$GLOBAL_BATCH_SIZE"
    "data.params.train_datasets.paths=[$ETHERNET_XR1_DIR/episodes]"
    data.params.train_datasets.repeat_to_max_steps=false data.params.train_datasets.sample_stride=1
    '~data.params.val_datasets' '~data.params.test_datasets' evaluation.test_after_fit=false
    model.params.pretrained="$PRETRAINED_CKPT"
    +model.params.model.full_finetune=true '+model.params.model.active_action_dims=[0,1,2,3,4,5,6]'
    trainer.optimizer.type=torch.optim.AdamW +trainer.optimizer.params.fused=true
    +trainer.strategy.params.stage=2 +trainer.use_distributed_sampler=false
    +trainer.log_every_n_steps=1 +trainer.num_sanity_val_steps=0 +trainer.limit_val_batches=0
    trainer.enable_progress_bar=false logging.type="$LOG_BACKEND" logging.mode="$LOG_MODE"
    hydra.run.dir=. hydra.output_subdir=null)
echo "mixed full fine-tune: frames=$FRAMES steps=$STEPS warmup=$WARMUP epochs=$EPOCHS global_batch=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH accumulation=$ACCUMULATE discarded_epoch_tail=$TAIL initial_model=$PRETRAINED_CKPT" >&2
if [[ "$MODE" == config ]]; then
    python tools/train_middle_mixed.py "${ARGS[@]}" "$@" --cfg job --resolve
else
    torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=localhost --master_port="$MASTER_PORT" \
        tools/train_middle_mixed.py "${ARGS[@]}" "${EXTRA[@]}" "$@"
fi
