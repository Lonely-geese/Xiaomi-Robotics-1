#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh"
cd "$XR1_REPO_DIR/xr1"
export ETHERNET_XR1_DIR=${ETHERNET_XR1_DIR:-"$PWD/data/ethernet_9007_verified"}
export CUDA_VISIBLE_DEVICES=${GPU_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
export RESOURCE_GPU=${#gpu_ids[@]}
MIN_FREE_VRAM_MB=${MIN_FREE_VRAM_MB:-46080}
GPU_MEMORY=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)
while read -r free_mb; do
    if (( free_mb < MIN_FREE_VRAM_MB )); then
        echo "Selected GPU has only ${free_mb} MiB free; requires at least ${MIN_FREE_VRAM_MB} MiB for this launch. Select idle GPUs using GPU_IDS." >&2
        exit 1
    fi
done <<< "$GPU_MEMORY"
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATE=${ACCUMULATE:-4}
EPOCHS=${EPOCHS:-5}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$PWD/pretrained_ckpt/model_states.pt"}
[[ -f "$PRETRAINED_CKPT" && -f "$ETHERNET_XR1_DIR/prepared.json" ]] || {
    echo 'Missing checkpoint or prepared dataset.' >&2; exit 1;
}
FRAMES=$(python -c 'import json,os; print(json.load(open(os.environ["ETHERNET_XR1_DIR"]+"/prepared.json"))["summary"]["train_frames"])')
# Match DistributedSampler padding and per-rank DataLoader/accumulation boundaries.
NODES=${WORLD_SIZE:-1}
SAMPLES_PER_RANK=$(((FRAMES + RESOURCE_GPU*NODES - 1)/(RESOURCE_GPU*NODES)))
BATCHES=$(((SAMPLES_PER_RANK + BATCH_SIZE - 1)/BATCH_SIZE))
STEPS_PER_EPOCH=$(((BATCHES + ACCUMULATE - 1)/ACCUMULATE))
MAX_STEPS=${MAX_STEPS:-$((STEPS_PER_EPOCH*EPOCHS))}
SAVE_INTERVAL=${SAVE_INTERVAL:-$STEPS_PER_EPOCH}
echo "Full fine-tuning: $FRAMES frames, $RESOURCE_GPU GPUs/node, batch/GPU=$BATCH_SIZE, accumulation=$ACCUMULATE, steps=$MAX_STEPS"
bash scripts/train.sh \
    data=ethernet model=posttrain \
    trainer.project=xiaomi-robotics-1 trainer.exp_name="${EXP_NAME:-ethernet_full}" \
    trainer.default_root_dir="$PWD/outputs" \
    +trainer.devices="$RESOURCE_GPU" +trainer.max_epochs="$EPOCHS" \
    trainer.max_steps="$MAX_STEPS" trainer.save_interval="$SAVE_INTERVAL" \
    trainer.accumulate_grad_batches="$ACCUMULATE" \
    data.params.train_datasets.batch_size="$BATCH_SIZE" \
    model.params.pretrained="$PRETRAINED_CKPT" \
    +model.params.model.full_finetune=true \
    trainer.optimizer.type=torch.optim.AdamW +trainer.optimizer.params.fused=true \
    +trainer.strategy.params.stage=2 "$@"
