# Copyright (C) 2026 Xiaomi Corporation.
#!/usr/bin/env bash

# ======================== Configuration Section (Adjust as needed) ========================
TASK_NAMES=(
    PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter
    PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter
    OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor
    OpenDrawer CloseDrawer TurnOnStove TurnOffStove
    TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout
    CoffeeSetupMug CoffeeServeMug CoffeePressButton
    TurnOnMicrowave TurnOffMicrowave
)
BASE_PORT=10086                          # Base port number for task assignment
NUM_EVAL_EPISODES=100                    # Number of episodes per task

# ======================== Main Execution Logic ========================
# Parse input parameters
NUM_PORTS="$1"
LOG_PATH="$2"
MODEL_PATH="${3:-XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa}"

if [ -z "$NUM_PORTS" ] || [ -z "$LOG_PATH" ]; then
    echo "Usage: bash scripts/launch_robocasa.sh <num_ports> <log_path> [model_path]"
    echo "Example: bash scripts/launch_robocasa.sh 8 ./eval_robocasa/eval_logs XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa"
    exit 1
fi

# Create log directory
mkdir -p "${LOG_PATH}"

# Launch tasks in batch with concurrency control
echo "Starting RoboCasa batch evaluation, concurrency limit: ${NUM_PORTS}"
echo "Total tasks: ${#TASK_NAMES[@]}, episodes per task: ${NUM_EVAL_EPISODES}"
echo "Model: ${MODEL_PATH}"
declare -a jobs=()

for i in "${!TASK_NAMES[@]}"; do
    TASK_NAME="${TASK_NAMES[$i]}"
    PORT_OFFSET=$((i % NUM_PORTS))
    PORT=$((BASE_PORT + PORT_OFFSET))
    LOG_FILE="${LOG_PATH}/${TASK_NAME}_eval.log"

    # Activate conda environment
    conda activate robocasa

    # Execute the task
    echo "Starting task: ${TASK_NAME}, port=${PORT}, log=${LOG_FILE}"
    python -u eval_robocasa/main.py \
        --task-name "${TASK_NAME}" \
        --model-path "${MODEL_PATH}" \
        --num-eval-episodes "${NUM_EVAL_EPISODES}" \
        --start-seed 0 \
        --server-port "${PORT}" \
        --save-root-dir "${LOG_PATH}" 2>&1 | tee "${LOG_FILE}" &

    # Track background job PID
    JOB_PID=$!
    jobs+=("${JOB_PID}")
    echo "Started task ${TASK_NAME} on port ${PORT} with job ID ${JOB_PID}"

    # Enforce concurrency limit
    if (( ${#jobs[@]} >= NUM_PORTS )); then
        echo "Concurrent task limit (${NUM_PORTS}) reached, waiting for any task to complete..."
        wait -n 2>/dev/null || true
        jobs=()
        for job in $(jobs -p); do
            jobs+=("${job}")
        done
    fi
done

# Wait for all remaining background tasks to finish
echo "Waiting for all remaining tasks to complete..."
wait

# Merge task results
echo "All tasks completed successfully! Merging results..."
python -u eval_robocasa/merge_results.py --eval-log-dir "${LOG_PATH}"

# Final status message
echo "Whole process completed! All results and logs are stored in: ${LOG_PATH}"
