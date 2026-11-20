#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
exec "$RECAP_CONTROL_PYTHON" "$RECAP_ROLLOUT_ROOT/scripts/wait_port.py" "$@"
