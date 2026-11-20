#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
cd "$RECAP_ROLLOUT_ROOT/control"
exec "$RECAP_SERVER_PYTHON" serve_robot_adapter.py "$@"
