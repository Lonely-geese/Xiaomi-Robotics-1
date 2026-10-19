#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
cd "$XR1_REPO_DIR/xr1"
# Direct access avoids the unreliable inherited localhost proxy for SwanLab.
export NO_PROXY="${NO_PROXY:-},swanlab.cn,.swanlab.cn"
export no_proxy="$NO_PROXY"
export ETHERNET_XR1_DIR="$PWD/data/ethernet_temporal_left_v1"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESOURCE_GPU=8 WORLD_SIZE=1
export XR1_STRICT_BATCH=1 OMP_NUM_THREADS=4
export MASTER_PORT=${MASTER_PORT:-29641}
MICRO_BATCH=${MICRO_BATCH:-32}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
(( GLOBAL_BATCH_SIZE > 0 && MICRO_BATCH > 0 && GLOBAL_BATCH_SIZE == 8 * MICRO_BATCH )) || { echo 'Global batch must equal 8 times per-GPU batch; gradient accumulation is disabled.'; exit 1; }
ACCUMULATE=1
read -r FRAMES STEPS WARMUP < <(python - <<'PY'
import json,os,math
p=json.load(open(os.environ['ETHERNET_XR1_DIR']+'/prepared.json'))
assert p['active_action_dims']==list(range(7)) and not p['quality_filtering']
f=p['splits']['train']['frames'];s=(f//int(os.environ['GLOBAL_BATCH_SIZE']))*10
print(f,s,math.ceil(s*0.03))
PY
)
MODE=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
EXP_NAME=${EXP_NAME:-left_relative_10ep_20261002}
EXTRA=()
if [[ "$MODE" == smoke ]]; then
    EXP_NAME="${EXP_NAME}_smoke_mb${MICRO_BATCH}"
    EXTRA=(trainer.max_steps=1 ++trainer.max_epochs=1 +trainer.limit_train_batches="$ACCUMULATE" +trainer.limit_val_batches=2 +trainer.enable_checkpointing=false evaluation.test_after_fit=false)
elif [[ "$MODE" != train && "$MODE" != config ]]; then
    echo 'Usage: train_left_8gpu.sh [train|smoke|config] [Hydra overrides...]' >&2; exit 2
fi
if [[ "$MODE" != config ]]; then
    GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    GPU_COUNT=0
    while read -r used util; do
        GPU_COUNT=$((GPU_COUNT + 1))
        (( used <= 1024 && util <= 5 )) || { echo 'All eight GPUs must be idle before launch.' >&2; exit 3; }
    done <<< "${GPU_STATS//,/}"
    (( GPU_COUNT == 8 )) || { echo 'Exactly eight GPUs are required.' >&2; exit 3; }
fi
ARGS=(data=ethernet_left model=posttrain trainer.project=xiaomi-robotics-1 trainer.exp_name="$EXP_NAME"
    trainer.default_root_dir="$PWD/outputs" +trainer.devices=8 ++trainer.max_epochs=10
    trainer.max_steps="$STEPS" trainer.accumulate_grad_batches="$ACCUMULATE"
    trainer.scheduler.params.num_warmup_steps="$WARMUP"
    data.params.train_datasets.batch_size="$MICRO_BATCH"
    data.params.global_batch_size="$GLOBAL_BATCH_SIZE"
    model.params.pretrained="$PWD/pretrained_ckpt/model_states.pt"
    +model.params.model.full_finetune=true '+model.params.model.active_action_dims=[0,1,2,3,4,5,6]'
    trainer.optimizer.type=torch.optim.AdamW +trainer.optimizer.params.fused=true
    +trainer.strategy.params.stage=2 +trainer.use_distributed_sampler=false
    +trainer.log_every_n_steps=10 +trainer.num_sanity_val_steps=2 trainer.enable_progress_bar=false)
echo "left-only full fine-tune: train_frames=$FRAMES steps=$STEPS warmup=$WARMUP epochs=10 global_batch=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH accumulation=$ACCUMULATE"
if [[ "$MODE" == config ]]; then
    python tools/train.py "${ARGS[@]}" "$@" --cfg job --resolve
else
    bash scripts/train.sh "${ARGS[@]}" "${EXTRA[@]}" "$@"
fi
