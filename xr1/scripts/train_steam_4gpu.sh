#!/usr/bin/env bash
set -euo pipefail
STEAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=1,2,3,4
export STEAM_NPROC=4
export STEAM_CRITIC_CONFIG="$STEAM_ROOT/xr1/configs/steam_critic_4gpu.json"
export STEAM_POLICY_CONFIG="$STEAM_ROOT/xr1/configs/steam_policy_4gpu.json"
export STEAM_LOG_DIR=/data/wz/STEAM/runs/pipeline_4gpu_20261025
exec bash "$STEAM_ROOT/xr1/scripts/train_steam_pipeline.sh"
