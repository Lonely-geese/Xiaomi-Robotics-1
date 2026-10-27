#!/usr/bin/env bash
# Sequential STEAM stages; checkpoint generations permit restarting this script.
set -euo pipefail
STEAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$STEAM_ROOT"
STEAM_PYTHON="${STEAM_PYTHON:-/data/wz/Xiaomi-Robotics-1/.venv/bin/python}"
STEAM_LOG_DIR="${STEAM_LOG_DIR:-/data/wz/STEAM/runs/pipeline_20261024}"
STEAM_NPROC="${STEAM_NPROC:-8}"
STEAM_CRITIC_CONFIG="${STEAM_CRITIC_CONFIG:-xr1/configs/steam_critic.json}"
STEAM_POLICY_CONFIG="${STEAM_POLICY_CONFIG:-xr1/configs/steam_policy.json}"
config_value() {
  "$STEAM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}
STEAM_CRITIC_OUTPUT="$(config_value "$STEAM_CRITIC_CONFIG" output)"
STEAM_POLICY_OUTPUT="$(config_value "$STEAM_POLICY_CONFIG" output)"
STEAM_MANIFEST="$(config_value "$STEAM_CRITIC_CONFIG" manifest)"
STEAM_ADVANTAGES="$(config_value "$STEAM_POLICY_CONFIG" advantages)"
STEAM_LABEL_GPU="${CUDA_VISIBLE_DEVICES:-0}"
STEAM_LABEL_GPU="${STEAM_LABEL_GPU%%,*}"
export PYTHONPATH="$STEAM_ROOT/xr1${PYTHONPATH:+:$PYTHONPATH}"
export XR1_VIDEO_BACKEND=pyav
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$STEAM_LOG_DIR"
trap 'printf "failed\n" > "$STEAM_LOG_DIR/status"' ERR
printf 'critic\n' > "$STEAM_LOG_DIR/status"
resume_args=()
if [[ -f "$STEAM_CRITIC_OUTPUT/latest.json" ]]; then
  resume_args=(--resume "$STEAM_CRITIC_OUTPUT")
fi
"$STEAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$STEAM_NPROC" \
  xr1/tools/train_steam_critic.py --config "$STEAM_CRITIC_CONFIG" "${resume_args[@]}" \
  --decode-workers 4 --loader-processes 4 --prefetch-batches 3 --ddp --fused-optimizer \
  >> "$STEAM_LOG_DIR/critic.log" 2>&1
# A graceful early stop saves successfully, but must not start advantage labeling.
if ! "$STEAM_PYTHON" - "$STEAM_CRITIC_CONFIG" <<'PY'
import json
import re
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
latest = json.loads((Path(cfg['output']) / 'latest.json').read_text())
step = latest.get('step')
if step is None:
    match = re.fullmatch(r'step_(\d+)_\d+', Path(latest['checkpoint']).parent.name)
    if match is None:
        raise ValueError('Cannot verify critic completion from checkpoint')
    step = int(match.group(1))
raise SystemExit(0 if step >= cfg['max_steps'] else 1)
PY
then
  printf 'paused_critic\n' > "$STEAM_LOG_DIR/status"
  exit 0
fi
printf 'labeling\n' > "$STEAM_LOG_DIR/status"
CUDA_VISIBLE_DEVICES="$STEAM_LABEL_GPU" "$STEAM_PYTHON" xr1/tools/label_steam_advantages.py \
  --checkpoint "$STEAM_CRITIC_OUTPUT" \
  --manifest "$STEAM_MANIFEST" \
  --output "$STEAM_ADVANTAGES" \
  >> "$STEAM_LOG_DIR/labeling.log" 2>&1
printf 'policy\n' > "$STEAM_LOG_DIR/status"
resume_args=()
if [[ -f "$STEAM_POLICY_OUTPUT/latest.json" ]]; then
  resume_args=(--resume "$STEAM_POLICY_OUTPUT")
fi
"$STEAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$STEAM_NPROC" \
  xr1/tools/train_steam_policy.py --config "$STEAM_POLICY_CONFIG" "${resume_args[@]}" \
  >> "$STEAM_LOG_DIR/policy.log" 2>&1
printf 'complete\n' > "$STEAM_LOG_DIR/status"
