#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
cd "$RECAP_ROLLOUT_ROOT/control"
exec "$RECAP_CONTROL_PYTHON" run_robot.py --record-out "$RECAP_RECORD_OUT" \
    --gripper-snap-close-m 0.012 "$@"
