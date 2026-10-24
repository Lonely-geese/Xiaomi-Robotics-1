#!/usr/bin/env bash
# Sequential STEAM stages; checkpoint generations permit restarting this script.
set -euo pipefail
STEAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$STEAM_ROOT"
STEAM_PYTHON="${STEAM_PYTHON:-/data/wz/Xiaomi-Robotics-1/.venv/bin/python}"
STEAM_LOG_DIR="${STEAM_LOG_DIR:-/data/wz/STEAM/runs/pipeline_20261024}"
export PYTHONPATH="$STEAM_ROOT/xr1${PYTHONPATH:+:$PYTHONPATH}"
export XR1_VIDEO_BACKEND=pyav
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$STEAM_LOG_DIR"
trap 'printf "failed\n" > "$STEAM_LOG_DIR/status"' ERR
printf 'critic\n' > "$STEAM_LOG_DIR/status"
resume_args=()
if [[ -f /data/wz/STEAM/runs/critic/latest.json ]]; then
  resume_args=(--resume /data/wz/STEAM/runs/critic)
fi
"$STEAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  xr1/tools/train_steam_critic.py --config xr1/configs/steam_critic.json "${resume_args[@]}" \
  >> "$STEAM_LOG_DIR/critic.log" 2>&1
printf 'labeling\n' > "$STEAM_LOG_DIR/status"
CUDA_VISIBLE_DEVICES=0 "$STEAM_PYTHON" xr1/tools/label_steam_advantages.py \
  --checkpoint /data/wz/STEAM/runs/critic \
  --manifest /data/wz/STEAM/lwd_20261024/manifest.json \
  --output /data/wz/STEAM/lwd_20261024/advantages.parquet \
  >> "$STEAM_LOG_DIR/labeling.log" 2>&1
printf 'policy\n' > "$STEAM_LOG_DIR/status"
resume_args=()
if [[ -f /data/wz/STEAM/runs/policy/latest.json ]]; then
  resume_args=(--resume /data/wz/STEAM/runs/policy)
fi
"$STEAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  xr1/tools/train_steam_policy.py --config xr1/configs/steam_policy.json "${resume_args[@]}" \
  >> "$STEAM_LOG_DIR/policy.log" 2>&1
printf 'complete\n' > "$STEAM_LOG_DIR/status"
