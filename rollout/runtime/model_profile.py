"""Paths and reviewed digests for the bundled robot execution runtime."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STABLE = ROOT / 'stable'
STOP_HELPER = ROOT / 'hal_compat.py'
STOP_SHA = 'daf07158a79377bd073530a20ca363cb13c7c76e61d54d386fcc2b5eb3f51b7f'
DEPENDENCIES = {
  "conversion.py": "05caab52a02475ae8246c74a2efcdcd149bcfa1485fea275c645ffa040bee60e",
  "wire.py": "5e61df6c1d95dc457fc5034b59e508107645a60c5f1996ec11578dcbe603b741",
  "run_robot.py": "796a40c6178e49845c4930e5cd88a33b7fb68bb9233ababf07ab77ca1710d0df",
  "serve_adapter.py": "2cc2e54aaae2e65659ae2079c191a258378ef35805b9abdd609ab2ee6fc2b2a5",
  "tianji_image_decoder.py": "1e4020b20da1918adcedfddc00e86bd358ac9690de0e7a5fa4eb7fc26ed29e23"
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module