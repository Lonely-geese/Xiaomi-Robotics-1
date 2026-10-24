"""STEAM numerical contracts, source fidelity and XR1 integration boundaries."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mibot.steam.policy import SteamPolicy
from mibot.steam.upstream.advantage import apply_boolean_label, quantile_threshold
from mibot.steam.upstream.data.binning import (
    _scaled_signed_stride_to_bin,
    _signed_stride_to_bin,
)
from mibot.steam.upstream.data.pair_dataset import (
    BinaryPairInferenceDataset,
    PairDataset,
    TrajectorySource,
)
from mibot.steam.upstream.models.steam.ensemble_modeling_critic import (
    EnsembleSteamCriticModel,
)
from mibot.steam.upstream.models.steam.modeling_critic import SteamCriticModel
from mibot.steam.upstream.routing import compute_cfg_routing_masks
from torch import nn


def test_vendored_source_hashes():
    root = Path(__file__).parents[1] / "mibot/steam/upstream"
    info = json.loads((root / "PROVENANCE.json").read_text())
    for record in info["files"]:
        content = (root / record["local"]).read_bytes()
        if record["local"] == "data/pair_dataset.py":
            content = (
                content.decode()
                .replace("        source: Optional[TrajectorySource] = None,\n", "")
                .replace(
                    "self._source = source if source is not None else _LeRobotSource(",
                    "self._source = _LeRobotSource(",
                )
                .encode()
            )
        assert hashlib.sha256(content).hexdigest() == record["sha256"], record["local"]

    for name, record in info["extractions"].items():
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest() == record["sha256"]
        )


@pytest.mark.parametrize("bins", [2, 4, 8, 16, 32, 64])
def test_all_signed_bins_and_symmetry(bins):
    for stride in range(1, 33):
        pos = _signed_stride_to_bin(stride, 32, bins)
        neg = _signed_stride_to_bin(-stride, 32, bins)
        assert pos + neg == bins - 1
        assert pos == ((stride + 31) * bins) // 64
        assert _scaled_signed_stride_to_bin(float(stride), 32, bins) == pos
    assert _scaled_signed_stride_to_bin(100.0, 32, bins) == bins - 1
    assert _scaled_signed_stride_to_bin(-100.0, 32, bins) == 0
    with pytest.raises(ValueError):
        _signed_stride_to_bin(0, 32, bins)


class TinySource(TrajectorySource):
    base = None

    def num_episodes(self):
        return 2

    def episode_length(self, e):
        return [4, 7][e]

    def episode_is_success(self, e):
        return True

    def get_raw_sample(self, e, f):
        return {"e": e, "f": f}

    def get_view_from_sample(self, s, key):
        return self.get_view(s["e"], s["f"], key)

    def get_view(self, e, f, key):
        return np.full((4, 4, 3), 10 * e + f, dtype=np.uint8)

    def get_prompt(self, e, f):
        return f"task {e}"


def test_pairs_keep_episode_bounds_and_true_stride():
    ds = PairDataset(
        "test",
        source=TinySource(),
        camera_keys=["camera"],
        k=32,
        num_bins=32,
        dataset_type="rollout",
        only_success=False,
    )
    ds._rng = np.random.default_rng(42)
    assert len(ds) == 18
    for i in range(len(ds)):
        x = ds[i]
        t, tk = x["frame_idx_t"], x["frame_idx_tk"]
        assert 0 <= t < ds._source.episode_length(x["episode"])
        assert 0 <= tk < ds._source.episode_length(x["episode"])
        assert (tk > t) == (i % 2 == 0)
        assert x["label"] == _signed_stride_to_bin(tk - t, 32, 32)
        assert int(x["image_t"]["camera"][0, 0, 0]) == 10 * x["episode"] + t
    inf = BinaryPairInferenceDataset(
        dataset_path="test",
        source=TinySource(),
        camera_keys=["camera"],
        k=32,
        prompt=None,
        dataset_type="rollout",
    )
    assert len(inf) == 9
    samples = inf.__getitems__(list(range(len(inf))))
    assert len({(s["episode"], s["frame_idx_t"]) for s in samples}) == 9


def test_signed_expectation_and_worst_member_not_mean():
    single = SteamCriticModel.__new__(SteamCriticModel)
    nn.Module.__init__(single)
    single.config = SimpleNamespace(num_bins=32)
    probabilities = torch.eye(32)
    expected = torch.cat([torch.arange(-16, 0), torch.arange(1, 17)]).float() / 16
    assert torch.equal(single._predicted_signed_value(probabilities), expected)
    ensemble = EnsembleSteamCriticModel.__new__(EnsembleSteamCriticModel)
    nn.Module.__init__(ensemble)
    probs = torch.stack(
        [probabilities[[30, 0]], probabilities[[0, 31]], probabilities[[20, 20]]]
    )
    values = torch.stack([single._predicted_signed_value(p) for p in probs])
    result = ensemble._aggregate_member_predictions(probs, probs, values)
    assert torch.equal(result[0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(single._predicted_signed_value(result[2]), result[0])
    assert torch.all(result[3] > result[0])


def test_strict_quantile_ties_and_cfg_routes():
    scores = np.array([0.0, 0.0, 1.0, 2.0])
    threshold = quantile_threshold(scores, 0.5)
    assert threshold == 0.5
    assert not apply_boolean_label(np.array([threshold]), threshold, inclusive=False)[0]
    routes = compute_cfg_routing_masks(
        torch.tensor([True, True, False, False]),
        positive_only_conditional=True,
        unconditional_prob=0.1,
        random_values=torch.tensor([0.1, 0.8, 0.1, 0.8]),
    )
    assert routes["conditional_mask"].tolist() == [False, True, False, False]
    assert not routes["negative_conditional_mask"].any()


class FakeXR1(nn.Module):
    def __init__(self):
        super().__init__()
        self.sink = nn.Embedding(1, 1)
        for name in [
            "state_projector_choice",
            "action_projector_choice",
            "score_projector_choice",
        ]:
            setattr(self, name, nn.Linear(1, 1))
        self.vlm = nn.Identity()
        self.beta = torch.distributions.Beta(1.5, 1.0)
        self.coefficient = nn.Parameter(torch.tensor(2.0))

    def prepare_flow_condition(self, observation, action_mask):
        return {"factor": observation["factor"]}

    def dit_forward(self, x, time, factor):
        assert time.shape == (len(x), 1, 1)
        return torch.ones_like(x) * factor * self.coefficient


def test_flow_loss_valid_dims_and_gradient():
    policy = SteamPolicy(FakeXR1())
    mask = torch.zeros(2, 30, 60)
    mask[..., :7] = 1
    mask[1, 10:] = 0
    action = torch.ones_like(mask) * 3
    noise = torch.ones_like(mask)
    loss = policy(
        {"factor": 1.0}, action, mask, noise=noise, time=torch.tensor([0.2, 0.8])
    )
    assert loss.item() == 0.0
    loss = policy(
        {"factor": 2.0}, action, mask, noise=noise, time=torch.tensor([0.2, 0.8])
    )
    loss.backward()
    assert loss.item() == 4.0 and torch.isfinite(policy.model.coefficient.grad)


@pytest.mark.parametrize("scale,expected", [(0, 2), (1, 6), (2, 10)])
def test_cfg_same_noise_and_time_and_mask(scale, expected):
    p = SteamPolicy(FakeXR1()).eval()
    mask = torch.zeros(1, 30, 60)
    mask[..., :7] = 1
    out = p.generate(
        {"factor": 1.0},
        {"factor": 3.0},
        mask,
        guidance_scale=scale,
        num_steps=5,
        noise=torch.zeros_like(mask),
    )
    assert torch.allclose(out[..., :7], torch.full_like(out[..., :7], expected))
    assert torch.count_nonzero(out[..., 7:]) == 0


def test_thresholds_fit_train_only_and_keep_terminal_convention():
    import pandas as pd
    from tools.label_steam_advantages import label_scores

    frame = pd.DataFrame(
        {
            "type": ["sft"] * 5 + ["rollout"] * 5,
            "split": (["train"] * 4 + ["validation"]) * 2,
            "advantage_continuous": [0.0, 1.0, 2.0, 3.0, 1e6] * 2,
        }
    )
    result, thresholds = label_scores(frame, 0.5, 0.5)
    assert thresholds == {"sft": 1.5, "rollout": 1.5}
    assert result.advantage_boolean.tolist() == [False, False, True, True, True] * 2
    with pytest.raises(ValueError):
        label_scores(frame, 1.0, 0.5)


def test_cfg_prompt_retains_task_images_state_and_does_not_mutate():
    from mibot.steam.data import PolicyCollator

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return {"input_ids": torch.ones(len(messages), 2, dtype=torch.long)}

    collator = PolicyCollator.__new__(PolicyCollator)
    collator.processor = Processor()
    original = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "test-image"},
                {"type": "text", "text": "insert cable"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]
    item = {"messages": original, "state": torch.zeros(1, 60)}
    batch = collator.observations([item, item], [False, True])
    prompts = collator.processor.messages
    assert prompts[0] == original
    assert prompts[1][0]["content"][:-1] == original[0]["content"]
    assert prompts[1][0]["content"][-1]["text"] == "\nAdvantage: positive"
    assert len(original[0]["content"]) == 2
    assert batch["state"].shape == (2, 1, 60)


def test_smoke_labels_are_rejected_by_production_dataset(tmp_path):
    import pandas as pd
    from mibot.steam.data import PolicyDataset, digest

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "xr1-steam-v1", "episodes": []}))
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        {"episode_id": ["fake"], "frame_index": [0], "advantage_boolean": [True]}
    ).to_parquet(labels)
    Path(str(labels) + ".json").write_text(
        json.dumps(
            {"manifest_sha256": digest(manifest), "complete": True, "smoke_only": True}
        )
    )
    with pytest.raises(ValueError, match="smoke labels"):
        PolicyDataset(manifest, labels)
