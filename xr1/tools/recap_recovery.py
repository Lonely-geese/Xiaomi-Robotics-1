"""Small, explicit recovery operations shared by the unattended supervisor."""
import json
import os
import shutil
from pathlib import Path
import time


def process_alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    torch.save(value, temporary)
    if path.name == 'resume.pt' and path.exists():
        previous = path.with_name('resume.previous.pt')
        backup = previous.with_name(previous.name+'.partial')
        shutil.copyfile(path, backup)
        backup.replace(previous)
    temporary.replace(path)


def latest_policy_checkpoint(root):
    """Only select checkpoints whose distributed save finished on every rank."""
    candidates = []
    for marker in Path(root).glob('**/restart_ready.json'):
        try:
            state = json.loads(marker.read_text())
            checkpoint = marker.parent
            if not (checkpoint/'checkpoint/mp_rank_00_model_states.pt').is_file():
                continue
            if len(list((checkpoint/'checkpoint').glob('*optim_states.pt'))) < int(state['world_size']):
                continue
            candidates.append((int(state['global_step']), checkpoint))
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, key=lambda item:item[0])[1] if candidates else None


def recovery_adjustments(error, attempt):
    """Change execution mechanics only; batch/ratio/epochs/LR stay intact."""
    error = error.lower()
    adjustments = dict(workers=4, dit_checkpointing=False, disable_ib=False)
    if any(word in error for word in ('out of memory', 'cuda error: memory allocation')):
        adjustments['dit_checkpointing'] = True
    if any(word in error for word in ('dataloader', 'worker', 'bus error', 'too many open files', 'resource temporarily unavailable')):
        adjustments['workers'] = 2 if attempt < 3 else 0
    if any(word in error for word in ('nccl', 'socket', 'connection reset', 'connection closed', 'timed out')):
        adjustments['disable_ib'] = True
    return adjustments
