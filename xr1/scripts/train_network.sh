#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
XR1_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd -- "$XR1_DIR/.." && pwd)
VENV_DIR=${VENV_DIR:-"$REPO_DIR/.venv"}

if [[ ! -x "$VENV_DIR/bin/python" || ! -x "$VENV_DIR/bin/torchrun" ]]; then
    echo "Python training environment not found: $VENV_DIR" >&2
    exit 1
fi
export PATH="$VENV_DIR/bin:$PATH"

DATASET_DIR=${DATASET_DIR:-"/mnt/cfs/l1x67e/ye/Data_0904_clean_v2"}
PREPARED_DIR=${PREPARED_DIR:-"$XR1_DIR/data/Data_0904_clean_v2_xr1"}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$XR1_DIR/pretrained_ckpt/model_states.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"$XR1_DIR/outputs"}
PROJECT=${PROJECT:-"xiaomi-robotics-1"}
EXP_NAME=${EXP_NAME:-"network-0904-clean-v2"}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-5}
GPU_IDS=${GPU_IDS:-4,5,6,7}
RESOURCE_GPU=${RESOURCE_GPU:-4}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ! "$RESOURCE_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "RESOURCE_GPU must be a positive integer, got: $RESOURCE_GPU" >&2
    exit 1
fi
if (( ${#VISIBLE_GPU_IDS[@]} != RESOURCE_GPU )); then
    echo "GPU count mismatch: GPU_IDS=$GPU_IDS exposes ${#VISIBLE_GPU_IDS[@]} GPUs, but RESOURCE_GPU=$RESOURCE_GPU" >&2
    exit 1
fi

if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
    echo "Dataset not found: $DATASET_DIR" >&2
    exit 1
fi
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "Pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

TOTAL_FRAMES=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_frames"])' \
    "$DATASET_DIR/meta/info.json")
NUM_NODES=${WORLD_SIZE:-1}
TOTAL_GPUS=$((NUM_NODES * RESOURCE_GPU))
GLOBAL_BATCH_SIZE=$((TOTAL_GPUS * BATCH_SIZE))
STEPS_PER_EPOCH=$(((TOTAL_FRAMES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))
MAX_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
SAVE_INTERVAL=$STEPS_PER_EPOCH

echo "Training setup: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, ${TOTAL_GPUS} GPUs, batch/GPU=${BATCH_SIZE}, global_batch=${GLOBAL_BATCH_SIZE}"
echo "Training schedule: ${EPOCHS} epochs, ${STEPS_PER_EPOCH} steps/epoch, ${MAX_STEPS} total steps"

cd "$XR1_DIR"
prepare_args=("$DATASET_DIR" --output "$PREPARED_DIR")
if [[ ${FORCE_PREPARE:-0} == 1 ]]; then
    prepare_args+=(--force)
fi
python tools/prepare_network_dataset.py "${prepare_args[@]}"

export NETWORK_XR1_EPISODES="$PREPARED_DIR/episodes"
export NETWORK_XR1_NORMALIZATION="$PREPARED_DIR/normalize.json"
export XR1_VIDEO_BACKEND=${XR1_VIDEO_BACKEND:-pyav}
export DS_IGNORE_CUDA_DETECTION=${DS_IGNORE_CUDA_DETECTION:-1}
export RESOURCE_GPU

bash scripts/train.sh \
    data=network \
    model=posttrain \
    trainer.project="$PROJECT" \
    trainer.exp_name="$EXP_NAME" \
    trainer.default_root_dir="$OUTPUT_DIR" \
    +trainer.devices="$RESOURCE_GPU" \
    +trainer.max_epochs="$EPOCHS" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.save_interval="$SAVE_INTERVAL" \
    data.params.train_datasets.batch_size="$BATCH_SIZE" \
    model.params.pretrained="$PRETRAINED_CKPT" \
    trainer.optimizer.type=torch.optim.AdamW \
    +trainer.optimizer.params.fused=true \
    "$@"
