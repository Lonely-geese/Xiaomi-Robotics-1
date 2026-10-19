#!/usr/bin/env bash
# Copyright (C) 2026 Xiaomi Corporation.
set -euo pipefail

TRACK_NAMES=(
    track_1_in_distribution
    track_2_cross_category
    track_3_common_sense
    track_4_semantic_instruction
    track_6_unseen_texture
)

BASE_PORT="${BASE_PORT:-10086}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
ROBOT_TYPE="${ROBOT_TYPE:-vlabench_choice}"
VISUALIZATION="${VISUALIZATION:-1}"
COT="${COT:-0}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"

NUM_PORTS="${1:-}"
LOG_PATH="${2:-}"
MODEL_PATH="${3:-XiaomiRobotics/Xiaomi-Robotics-1-VLABench}"

if [[ -z "${NUM_PORTS}" || -z "${LOG_PATH}" ]]; then
    echo "Usage: bash scripts/launch_vlabench.sh <num_ports> <log_path> [model_path]"
    echo "Example: bash scripts/launch_vlabench.sh 8 ./eval_vlabench/eval_logs XiaomiRobotics/Xiaomi-Robotics-1-VLABench"
    exit 1
fi
if ! [[ "${NUM_PORTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_ports must be a positive integer"
    exit 1
fi
if [[ -z "${VLABENCH_ROOT:-}" ]]; then
    echo "Error: VLABENCH_ROOT is not set. Export VLABENCH_ROOT=/path/to/VLABench"
    exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "${LOG_PATH}"

visualization_flag="--visualization"
if [[ "${VISUALIZATION}" == "0" ]]; then
    visualization_flag="--no-visualization"
fi
cot_flag="--no-cot"
if [[ "${COT}" == "1" ]]; then
    cot_flag="--cot"
fi

for track in "${TRACK_NAMES[@]}"; do
    track_dir="${LOG_PATH}/${track}"
    mkdir -p "${track_dir}"
    echo "Starting VLABench track: ${track}"
    python -u eval_vlabench/dispatch.py \
        --model-path "${MODEL_PATH}" \
        --eval-track "${track}" \
        --save-dir "${track_dir}" \
        --host 127.0.0.1 \
        --base-port "${BASE_PORT}" \
        --num-workers "${NUM_PORTS}" \
        --robot-type "${ROBOT_TYPE}" \
        --n-episode "${NUM_EVAL_EPISODES}" \
        --action-chunk-size "${ACTION_CHUNK_SIZE}" \
        --replan-steps "${REPLAN_STEPS}" \
        "${visualization_flag}" \
        "${cot_flag}" 2>&1 | tee "${track_dir}/dispatch.log"
done

python -u eval_vlabench/merge_results.py --eval-log-dir "${LOG_PATH}"
echo "VLABench evaluation complete. Results are stored in: ${LOG_PATH}"
