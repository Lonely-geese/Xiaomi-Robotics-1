#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
cd "$RECAP_ROLLOUT_ROOT/server"
"$RECAP_SERVER_PYTHON" "$RECAP_ROLLOUT_ROOT/scripts/check_package.py" --full
exec "$RECAP_SERVER_PYTHON" -m mibot.server.deploy --model "$RECAP_PACKAGE/model" \
    --recap-guidance-scale 1.0 --host 127.0.0.1 --port 10090 "$@"
