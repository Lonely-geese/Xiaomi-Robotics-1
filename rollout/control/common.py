"""Identity and dependencies for the RECAP robot integration."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent / 'runtime'
NATIVE_CODE = ROOT.parent / 'server'
PACKAGE = Path(os.environ.get('RECAP_PACKAGE', '/data/checkpoints/kele/usb/usb_recap_epoch10_step15490')).expanduser().resolve()
MODEL = PACKAGE / 'model'
SOURCE = BUNDLE / 'vr_runtime.py'
RESET_CONFIG = os.environ.get('XR1_RESET_CONFIG', str(ROOT.parent / 'config/vr_reset.yaml'))
TASK = 'Pick up the USB cable and insert it into the USB port.'
EXTENDED_IDENTITY = ('normalization_sha256', 'camera_layout', 'zero_state',
                     'left_wrist_crop', 'recap_enabled', 'recap_conditioning',
                     'recap_guidance_scale')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def manifest():
    return json.loads((ROOT / 'model_manifest.json').read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_dependencies():
    for path, expected in json.loads((ROOT / 'runtime_dependencies.json').read_text()).items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f'Robot dependency changed: {path}')
    sys.path.insert(0, str(BUNDLE))
    import model_profile as profile
    for name, expected in profile.DEPENDENCIES.items():
        if digest(profile.STABLE / name) != expected:
            raise RuntimeError(f'Stable robot dependency changed: {name}')
    if digest(profile.STOP_HELPER) != profile.STOP_SHA:
        raise RuntimeError('Stop helper changed')
    return profile


def verify_package(full=False, *, require_weights=True):
    identity = manifest()
    config_root = MODEL if require_weights or MODEL.is_dir() else ROOT.parent / 'config/model'
    for name, key in [('normalize.json', 'normalization_sha256'), ('config.py', 'config_sha256')]:
        if digest(config_root / name) != identity[key]:
            raise RuntimeError(f'RECAP {name} does not match manifest')
    checkpoint = MODEL / 'last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    if require_weights and checkpoint.stat().st_size != identity['checkpoint_bytes']:
        raise RuntimeError('RECAP checkpoint size mismatch')
    if full and digest(checkpoint) != identity['checkpoint_sha256']:
        raise RuntimeError('RECAP checkpoint SHA-256 mismatch')
    cfg = runpy.run_path(str(config_root / 'config.py'))
    model = cfg['model']['params']['model']
    expected = {'recap_enabled': True, 'recap_conditioning': 'language',
                'recap_guidance_scale': 1.0, 'async_train': True,
                'active_action_dims': list(range(7))}
    if any(model.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Not the expected RECAP language / async / left-arm configuration')
    if cfg['data']['params']['train_datasets']['action_length'] != 30:
        raise RuntimeError('Unexpected action horizon')
    return identity


def check_handshake(metadata, identity):
    for key in EXTENDED_IDENTITY:
        if key not in metadata or metadata[key] != identity[key]:
            raise RuntimeError(f'RECAP handshake mismatch: {key}={metadata.get(key)!r}')


def verify_native(port):
    listeners = subprocess.check_output(['ss', '-ltnp', f'sport = :{port}'], text=True)
    pids = {int(pid) for pid in re.findall(r'pid=(\d+),', listeners)}
    owners = []
    for pid in pids:
        proc = Path('/proc') / str(pid)
        argv = (proc / 'cmdline').read_bytes().decode().split('\0')
        def value(flag):
            return argv[argv.index(flag) + 1] if flag in argv else None
        env = dict(item.split('=', 1) for item in (proc / 'environ').read_bytes().decode().split('\0') if '=' in item)
        if (value('-m') == 'mibot.server.deploy' and value('--model') == str(MODEL)
                and value('--port') == str(port) and value('--recap-guidance-scale') == '1.0'
                and proc.joinpath('cwd').resolve() == NATIVE_CODE
                and env.get('PYTHONPATH', '').split(':')[0] == str(NATIVE_CODE)
                and env.get('XR1_VLM_PATH') == str(PACKAGE / 'assets/Qwen3-VL-4B-Instruct')):
            owners.append(pid)
    if len(owners) != 1:
        raise RuntimeError(f'Port {port} is not the verified RECAP native server')
    return owners[0]


def controller_conflicts():
    conflicts = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if (any(Path(arg).name in ('run_robot.py', 'run_robot_async.py', 'record_xr1_async_rollout.py') for arg in argv)
                or any(Path(arg).name == 'entry.py' for arg in argv) and 'client' in argv):
            conflicts.append(int(proc.name))
    return conflicts
