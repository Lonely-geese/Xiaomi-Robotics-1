import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from preserve_recap_checkpoint import checksum, preserve


def test_preserved_exact_step_survives_recovery_checkpoint_cleanup(tmp_path):
    source = tmp_path/'recovery/step5000.ckpt'
    weights = source/'checkpoint/mp_rank_00_model_states.pt'
    weights.parent.mkdir(parents=True)
    torch.save(dict(global_step=5000,module={'model.recap_condition_embedding':torch.ones(1,6,4)}),weights)
    (source/'restart_ready.json').write_text(json.dumps(dict(global_step=5000,epoch=0,world_size=8)))
    (tmp_path/'config.py').write_text('model = {}\n')
    saved = preserve(tmp_path,5000)
    assert saved['global_step'] == 5000
    assert checksum(saved['model']) == saved['sha256']
    shutil.rmtree(source)
    loaded = torch.load(saved['model'],weights_only=True)
    assert loaded['global_step'] == 5000
    assert preserve(tmp_path,5000) == saved
    assert (tmp_path/'milestones/step5000/config.py').exists()


def test_checkpoint_is_not_copied_until_save_is_complete(tmp_path):
    assert preserve(tmp_path,5000) is None
