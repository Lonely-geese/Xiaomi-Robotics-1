"""USB epoch10 model profile and integrity checks for the local inference code."""
import ast
import hashlib
import importlib.util
import json
import os
import pprint
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT
STABLE = BUNDLE / 'tianji_adapter/experiments/pre_7dof_interpolation_1257_smooth_async'
STOP_HELPER = STABLE.with_name(STABLE.name + '_accel') / 'hal_compat.py'
MODEL = ROOT / 'model'
PORT = 10090
CHECKPOINT_BYTES = 10227522621
EXPECTED = {'checkpoint_step': 12050, 'checkpoint_sha256': 'ba975a3d5bb4a68273096b1bc49d5273db5d05328e558658bcb374338a926abd', 'checkpoint_epoch': 9, 'completed_epochs': 10, 'model_profile': 'xiaomi_xr1_usb_epoch10_step12050'}
DEPENDENCIES = {
    'run_robot.py': '796a40c6178e49845c4930e5cd88a33b7fb68bb9233ababf07ab77ca1710d0df',
    'serve_adapter.py': '4d0fc64f2fe82e0d6be407c40087ca36a84422edf9d247e40e80656c62c15ec5',
    'conversion.py': '05caab52a02475ae8246c74a2efcdcd149bcfa1485fea275c645ffa040bee60e',
    'wire.py': '5e61df6c1d95dc457fc5034b59e508107645a60c5f1996ec11578dcbe603b741',
    'tianji_image_decoder.py': '1e4020b20da1918adcedfddc00e86bd358ac9690de0e7a5fa4eb7fc26ed29e23',
}
CONFIG_SHA = '84d474e9b3f61ed678e75b375e8db48c4039a50d36da4efe3526b6c094349bf1'
NORMALIZATION_SHA = '9ee242381ea2a1a0c03951e1b7d6619ff5dcc3767fece219f96eaa26ee299b4d'
INFERENCE_CONFIG_SHA = '47877687467ae45b1ff18447f32291de5d83d1722735561c37923eb1cc96897f'
RUNTIME_DEPENDENCIES = {'mibot/models/VLA/XR1.py': 'a11e8d10de7047537832a2e59c1b129461cfe5a7c277a70d05f3a4b5204d35d0', 'mibot/models/VLM/qwen3vl.py': 'e440a69c3c79b84e7c3133787786de4d30045e379ace72487fb438928ef41fb5', 'mibot/server/deploy.py': '04c10076a0c5b351df4d2a2cd3daf6a65c39b1ff23daa4441f611ccf96ffcc8f', 'mibot/server/runtime/client.py': 'a6362e1aa263af1390d842913012ac662b54fc35262844b0efb8f3396d99199e', 'mibot/server/runtime/server.py': 'd017f218621841b421782b93146e9fe10b777ab2615f088d4e12e8ed1959fac4', 'mibot/utils/io.py': 'a0e1a603343bc8bf01b60e29635d2050d767bb6e6837237e27ddc408b140852c'}
STOP_SHA = 'daf07158a79377bd073530a20ca363cb13c7c76e61d54d386fcc2b5eb3f51b7f'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify(verify_weights=False):
    manifest = json.loads((MODEL / 'model_manifest.json').read_text())
    actual = {key: manifest.get('profile' if key == 'model_profile' else key) for key in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(f'Wrong model identity: {actual}')
    checkpoint = MODEL / 'last.ckpt/checkpoint/mp_rank_00_model_states.pt'
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or manifest['checkpoint_bytes'] != CHECKPOINT_BYTES:
        raise RuntimeError('Checkpoint size mismatch')
    for name, expected in [('config.py', CONFIG_SHA), ('normalize.json', NORMALIZATION_SHA)]:
        if digest(MODEL / name) != expected:
            raise RuntimeError(f'Model {name} checksum mismatch')
    if manifest['config_sha256'] != CONFIG_SHA or manifest['normalization_sha256'] != NORMALIZATION_SHA:
        raise RuntimeError('Manifest configuration checksum mismatch')
    if manifest['active_action_dims'] != list(range(7)) or manifest['action_horizon'] != 30:
        raise RuntimeError('Expected left-arm 30-frame profile')
    tree = ast.parse((MODEL / 'config.py').read_text())
    config = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name)}
    model = config['model']['params']['model']
    data = config['data']['params']['train_datasets']
    normalization = json.loads((MODEL / 'normalize.json').read_text())
    if model.get('async_train') is not True or model.get('active_action_dims') != list(range(7)):
        raise RuntimeError('Expected official left-arm async-prefix configuration')
    if data['action_length'] != 30 or any(data[k] != normalization[k] for k in ('mean', 'std', 'q01', 'q99')):
        raise RuntimeError('Embedded inference statistics do not match normalization')
    exported = json.loads((MODEL / 'inference_config.json').read_text())
    if digest(MODEL / 'inference_config.json') != INFERENCE_CONFIG_SHA:
        raise RuntimeError('Export inference configuration checksum mismatch')
    if exported['model'] != model:
        raise RuntimeError('Inference architecture differs from the exported model')
    if (exported['checkpoint_epoch_zero_based'], exported['training_epoch_ordinal'],
            exported['global_step']) != (EXPECTED['checkpoint_epoch'], EXPECTED['completed_epochs'], EXPECTED['checkpoint_step']):
        raise RuntimeError('Export checkpoint epoch/step mismatch')
    if (exported['action_length'], exported['action_dim'], exported['action_rate_hz']) != (30, 60, 30):
        raise RuntimeError('Export action layout mismatch')
    for name, expected in RUNTIME_DEPENDENCIES.items():
        if digest(BUNDLE / 'xr1' / name) != expected:
            raise RuntimeError(f'Official inference runtime changed: {name}')
    for name, expected in DEPENDENCIES.items():
        if digest(STABLE / name) != expected:
            raise RuntimeError(f'Stable dependency changed; review before use: {name}')
    if digest(STOP_HELPER) != STOP_SHA:
        raise RuntimeError('Stop verification dependency changed; review before use')
    if verify_weights and digest(checkpoint) != EXPECTED['checkpoint_sha256']:
        raise RuntimeError('Checkpoint SHA-256 mismatch; refusing to start')
    return dict(EXPECTED)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selection():
    return dict(model_dir=str(MODEL.resolve()),
                checkpoint=str((MODEL / 'last.ckpt').resolve()), expected=dict(EXPECTED),
                checkpoint_bytes=CHECKPOINT_BYTES, config_sha256=CONFIG_SHA,
                normalization_sha256=NORMALIZATION_SHA,
                inference_config_sha256=INFERENCE_CONFIG_SHA)


def configure(selected):
    global MODEL, EXPECTED, CHECKPOINT_BYTES, CONFIG_SHA, NORMALIZATION_SHA, INFERENCE_CONFIG_SHA
    MODEL = Path(selected['model_dir']).resolve(strict=True)
    EXPECTED = dict(selected['expected'])
    CHECKPOINT_BYTES = selected['checkpoint_bytes']
    CONFIG_SHA = selected['config_sha256']
    NORMALIZATION_SHA = selected['normalization_sha256']
    INFERENCE_CONFIG_SHA = selected['inference_config_sha256']


DEFAULT_SELECTION = selection()


def use_active(port=PORT):
    path = ROOT / f'active_model_{port}.json'
    configure(json.loads(path.read_text()) if path.exists() else DEFAULT_SELECTION)


def select_checkpoint(checkpoint):
    requested = Path(checkpoint).expanduser().resolve(strict=True)
    if requested.name == 'mp_rank_00_model_states.pt':
        requested = requested.parent.parent
    registry = json.loads((ROOT / 'model_registry.json').read_text())
    for selected in registry:
        if requested == Path(selected['checkpoint']).resolve():
            configure(selected)
            return
    configure(prepare_exported_checkpoint(requested))


def prepare_exported_checkpoint(checkpoint):
    """Accept compatible exports with explicit, adjacent inference statistics."""
    source = checkpoint.parent
    if not (source / 'inference_config.json').is_file() or not (source / 'normalize.json').is_file():
        raise ValueError('Checkpoint has no verified inference profile or adjacent '
                         'inference_config.json and normalize.json: ' + str(checkpoint))
    exported = json.loads((source / 'inference_config.json').read_text())
    weight = checkpoint / 'checkpoint/mp_rank_00_model_states.pt'
    if (source / exported['checkpoint']).resolve() != weight.resolve():
        raise ValueError('Adjacent inference_config.json describes a different checkpoint')
    tree = ast.parse((BUNDLE / 'model/config.py').read_text())
    base = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}
    if exported['model'] != base['model']['params']['model']:
        raise ValueError('Exported architecture is incompatible with this XR1 runtime')
    if (exported['action_length'], exported['action_dim'], exported['action_rate_hz']) != (30, 60, 30):
        raise ValueError('Exported action layout must be 30 x 60 at 30 Hz')
    if exported['active_action_dims'] != list(range(7)):
        raise ValueError('Only left-arm XR1 exports are supported')
    stats = json.loads((source / 'normalize.json').read_text())
    for key in ('mean', 'std', 'q01', 'q99'):
        base['data']['params']['train_datasets'][key] = stats[key]
    config_text = '\n\n'.join(k + ' = ' + pprint.pformat(v) for k, v in base.items()) + '\n'
    weight_sha = digest(weight)
    directory = ROOT / 'model_profiles'
    directory.mkdir(exist_ok=True)
    # A unique directory keeps already running services bound to their own files.
    target = Path(tempfile.mkdtemp(prefix='selected_' + weight_sha[:12] + '_', dir=directory))
    (target / 'config.py').write_text(config_text)
    (target / 'normalize.json').write_text((source / 'normalize.json').read_text())
    (target / 'inference_config.json').write_text((source / 'inference_config.json').read_text())
    (target / 'last.ckpt').symlink_to(checkpoint)
    expected = dict(checkpoint_step=exported['global_step'], checkpoint_sha256=weight_sha,
                    checkpoint_epoch=exported['checkpoint_epoch_zero_based'],
                    completed_epochs=exported['training_epoch_ordinal'],
                    model_profile='xiaomi_xr1_' + checkpoint.stem + '_' + weight_sha[:12])
    selected = dict(model_dir=str(target), checkpoint=str(checkpoint), expected=expected,
                    checkpoint_bytes=weight.stat().st_size, config_sha256=digest(target / 'config.py'),
                    normalization_sha256=digest(target / 'normalize.json'),
                    inference_config_sha256=digest(target / 'inference_config.json'))
    manifest = dict(expected, profile=expected['model_profile'], active_action_dims=list(range(7)),
                    action_horizon=30, checkpoint_bytes=selected['checkpoint_bytes'],
                    config_sha256=selected['config_sha256'], normalization_sha256=selected['normalization_sha256'],
                    checkpoint_source=str(weight), inference_shift_frames=0)
    (target / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return selected


def publish_selection(port):
    # Called only after loading the model and binding the serving socket.
    # Readers see either the previous complete selection or the new one.
    path = ROOT / f'active_model_{port}.json'
    descriptor = selection()
    with tempfile.NamedTemporaryFile(mode='w', dir=ROOT, prefix='.active_model_',
                                     suffix='.tmp', delete=False) as stream:
        json.dump(descriptor, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)


use_active()
