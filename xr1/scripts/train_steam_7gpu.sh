#!/usr/bin/env bash
set -euo pipefail
STEAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export STEAM_NPROC=7
export STEAM_CRITIC_CONFIG="$STEAM_ROOT/xr1/configs/steam_critic_7gpu.json"
export STEAM_POLICY_CONFIG="$STEAM_ROOT/xr1/configs/steam_policy_7gpu.json"
export STEAM_LOG_DIR=/data/wz/STEAM/runs/pipeline_7gpu_20261025
exec bash "$STEAM_ROOT/xr1/scripts/train_steam_pipeline.sh"
