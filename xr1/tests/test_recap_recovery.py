import json
import copy
import sys
from types import MethodType
from pathlib import Path

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from recap_recovery import atomic_torch_save, latest_policy_checkpoint, recovery_adjustments
from run_recap_pipeline import Pipeline
from mibot.data.datasets.recap_dataset import ResumablePoolLoader
from mibot.data.recap import BalancedPoolSampler
from mibot.models.VLA.XR1 import DiT


class SmallData(LightningDataModule):
    def train_dataloader(self):
        self.loader = ResumablePoolLoader(torch.arange(32, dtype=torch.float32), batch_size=4,
            sampler=BalancedPoolSampler(16,16,4), num_workers=0)
        return self.loader


class SmallModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def training_step(self, batch, batch_idx):
        self.seen.extend(batch.int().tolist())
        return (self.weight*batch.mean())**2

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['seen'] = self.seen

    def on_load_checkpoint(self, checkpoint):
        self.seen = checkpoint['seen']


class ConsumedBatches(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        trainer.train_dataloader.consumed_batches = batch_idx+1


def test_lightning_resume_preserves_sampling_position_and_total_epochs(tmp_path):
    checkpoint = ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=1,
        every_n_epochs=0, save_top_k=1, monitor='step', mode='max', save_on_train_epoch_end=False)
    first = Trainer(accelerator='cpu', devices=1, max_epochs=2, max_steps=3,
        logger=False, callbacks=[ConsumedBatches(),checkpoint], enable_progress_bar=False,
        enable_model_summary=False, use_distributed_sampler=False)
    first.fit(SmallModel(), datamodule=SmallData())
    model = SmallModel()
    resumed = Trainer(accelerator='cpu', devices=1, max_epochs=2, max_steps=16,
        logger=False, callbacks=[ConsumedBatches()], enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, use_distributed_sampler=False)
    resumed.fit(model, datamodule=SmallData(), ckpt_path=checkpoint.best_model_path)
    sampler = BalancedPoolSampler(16,16,4)
    expected = list(sampler)
    sampler.set_epoch(1)
    expected += list(sampler)
    assert model.seen == expected
    assert resumed.global_step == 16 and resumed.current_epoch == 2


def test_only_complete_distributed_checkpoints_are_selected(tmp_path):
    for step in (1000,2000):
        checkpoint = tmp_path/f'step{step}.ckpt'
        (checkpoint/'checkpoint').mkdir(parents=True)
        (checkpoint/'checkpoint/mp_rank_00_model_states.pt').touch()
        (checkpoint/'restart_ready.json').write_text(json.dumps(dict(global_step=step,world_size=8)))
        for rank in range(8 if step == 1000 else 7):
            (checkpoint/f'checkpoint/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').touch()
    assert latest_policy_checkpoint(tmp_path).name == 'step1000.ckpt'


def test_recovery_adjustments_preserve_training_hyperparameters():
    assert recovery_adjustments('CUDA out of memory',1)['dit_checkpointing']
    assert recovery_adjustments('DataLoader worker bus error',3)['workers'] == 0
    assert recovery_adjustments('NCCL connection reset',1)['disable_ib']
    assert 'batch_size' not in recovery_adjustments('CUDA out of memory',1)


def test_controller_recovers_from_injected_failure(tmp_path, monkeypatch):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.root = tmp_path
    pipeline.status = dict(stage='policy_train')
    pipeline.settings_path = tmp_path/'settings.json'
    pipeline.settings = dict(workers=4,dit_checkpointing=False,disable_ib=False,attempts={})
    pipeline.last_error_log = ''
    calls = []
    def run(self):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('CUDA out of memory')
        self.status['state'] = 'completed'
    def record(self, **changes):
        self.status.update(changes)
    pipeline.run = MethodType(run,pipeline)
    pipeline.record = MethodType(record,pipeline)
    monkeypatch.setattr('run_recap_pipeline.time.sleep', lambda seconds:None)
    pipeline.run_forever()
    assert len(calls) == 2 and pipeline.status['state'] == 'completed'
    assert pipeline.settings['dit_checkpointing']
    assert pipeline.settings['attempts']['policy_train'] == 1


def test_dit_memory_recovery_keeps_outputs_and_gradients():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4,4))
        def forward(self, hidden, past, positions, timestep, mask):
            return torch.nn.functional.silu(hidden@self.weight+past[0].mean()+timestep.mean())
    original = DiT(layer_num=0)
    original.layers = torch.nn.ModuleList([Layer(),Layer()])
    original.layer_num = 2
    recovered = copy.deepcopy(original)
    recovered.gradient_checkpointing = True
    first = torch.randn(2,3,4,requires_grad=True)
    second = first.detach().clone().requires_grad_()
    past = [(torch.randn(2,2,3,4),torch.randn(2,2,3,4)) for _ in range(2)]
    timestep = torch.randn(2,6,4)
    a = original(first,past,None,None,timestep)
    b = recovered(second,past,None,None,timestep)
    a.sum().backward();b.sum().backward()
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(first.grad,second.grad)
    for p,q in zip(original.parameters(),recovered.parameters()):
        torch.testing.assert_close(p.grad,q.grad)


def test_atomic_critic_checkpoint_keeps_previous_copy(tmp_path):
    path = tmp_path/'resume.pt'
    atomic_torch_save(dict(step=500),path)
    atomic_torch_save(dict(step=1000),path)
    assert torch.load(path,weights_only=True)['step'] == 1000
    assert torch.load(tmp_path/'resume.previous.pt',weights_only=True)['step'] == 500
