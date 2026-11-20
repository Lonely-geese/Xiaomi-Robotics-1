#!/usr/bin/env bash
set -euo pipefail
STEAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$STEAM_ROOT"
STEAM_PYTHON="${STEAM_PYTHON:-$STEAM_ROOT/.venv/bin/python}"
STEAM_POLICY_CONFIG="${STEAM_POLICY_CONFIG:-$STEAM_ROOT/xr1/configs/steam_policy_8gpu_bs256.json}"
export PYTHONPATH="$STEAM_ROOT/xr1${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XR1_VIDEO_BACKEND=pyav
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
STEAM_POLICY_OUTPUT="$("$STEAM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$STEAM_POLICY_CONFIG")"
resume_args=()
if [[ -f "$STEAM_POLICY_OUTPUT/latest.json" ]]; then
  resume_args=(--resume "$STEAM_POLICY_OUTPUT")
fi
exec "$STEAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  xr1/tools/train_steam_policy.py --config "$STEAM_POLICY_CONFIG" "${resume_args[@]}" \
  --loader-processes 4 --decode-workers 4 --prefetch-batches 3 \
  --ddp --fused-optimizer --coalesced-broadcast "$@"
