#!/usr/bin/env bash
# Recreate the local runtime using the already downloaded, verified wheel.
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_DIR=${XR1_ENV_DIR:-/opt/xr1-env}
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    uv venv --python python3.12 --seed "$ENV_DIR"
fi
if [[ ! -e "$REPO_DIR/.venv" && ! -L "$REPO_DIR/.venv" ]]; then
    ln -s "$ENV_DIR" "$REPO_DIR/.venv"
elif [[ $(readlink -f "$REPO_DIR/.venv") != $(readlink -f "$ENV_DIR") ]]; then
    echo "Project .venv points elsewhere; use XR1_ENV_DIR to select that environment." >&2
    exit 1
fi
export DS_BUILD_OPS=0 UV_NATIVE_TLS=true
INDEX_URL=${XR1_PYPI_INDEX:-https://pypi.org/simple}
uv pip install --python "$ENV_DIR/bin/python" torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$ENV_DIR/bin/python" wheel ninja packaging --index-url "$INDEX_URL"
uv pip install --python "$ENV_DIR/bin/python" --no-build-isolation -e "$REPO_DIR/xr1" pytest --index-url "$INDEX_URL"
WHEEL="$REPO_DIR/.setup/wheels/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
[[ -f "$WHEEL" ]] || { echo "Missing verified Flash Attention wheel: $WHEEL" >&2; exit 1; }
"$ENV_DIR/bin/python" - "$WHEEL" <<'PY'
import hashlib,sys
with open(sys.argv[1],'rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb'
PY
uv pip install --python "$ENV_DIR/bin/python" "$WHEEL" --index-url "$INDEX_URL"
"$ENV_DIR/bin/python" -m pip check
