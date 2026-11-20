"""Prefetch must preserve pair choices, labels and the checkpoint RNG boundary."""

import copy
import json

import numpy as np
import pytest
from mibot.steam.data import XR1TrajectorySource
from mibot.steam.prefetch import DeferredPairDataset, PairPrefetcher
from mibot.steam.upstream.data.pair_dataset import PairDataset
from test_steam import TinySource


def datasets(cls, rng):
    groups = [
        cls(
            dataset_path="tiny",
            source=TinySource(),
            camera_keys=["cam"],
            k=4,
            num_bins=4,
            dataset_type=kind,
            only_success=False,
        )
        for kind in ["sft", "rollout"]
    ]
    for ds in groups:
        ds._rng = rng
    return groups


def summary(items):
    return [
        (
            x["source_name"],
            x["episode"],
            x["frame_idx_t"],
            x["frame_idx_tk"],
            x["label"],
            x["prompt"],
            x["image_t"]["cam"].tolist(),
            x["image_tk"]["cam"].tolist(),
        )
        for x in items
    ]


def test_prefetch_preserves_sampling_and_resume_with_pending_batches():
    rng = np.random.default_rng(120)
    reference_rng = np.random.default_rng(120)
    groups = datasets(PairDataset, reference_rng)
    probabilities = np.array([0.3, 0.7])
    stream = PairPrefetcher(
        datasets(DeferredPairDataset, rng),
        probabilities,
        rng,
        5,
        summary,
        workers=3,
        depth=3,
        pin_memory=False,
    )
    try:
        for _ in range(7):
            expected = []
            for _ in range(5):
                ds = groups[int(reference_rng.choice(2, p=probabilities))]
                expected.append(ds[int(reference_rng.integers(len(ds)))])
            assert next(stream) == summary(expected)
            assert rng.bit_generator.state == reference_rng.bit_generator.state
        saved = copy.deepcopy(rng.bit_generator.state)
        next_expected = next(stream)
    finally:
        stream.close()
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    resumed = PairPrefetcher(
        datasets(DeferredPairDataset, restored),
        probabilities,
        restored,
        5,
        summary,
        workers=2,
        pin_memory=False,
    )
    try:
        assert next(resumed) == next_expected
    finally:
        resumed.close()


def test_prefetch_propagates_decode_errors():
    rng = np.random.default_rng(3)

    def fail(items):
        raise ValueError("broken image")

    stream = PairPrefetcher(
        datasets(DeferredPairDataset, rng), [0.5, 0.5], rng, 2, fail, pin_memory=False
    )
    try:
        with pytest.raises(ValueError, match="broken image"):
            next(stream)
    finally:
        stream.close()


@pytest.mark.parametrize("crop", [None, [3, 4, 24, 36]])
def test_pair_decode_matches_independent_seeks_with_offsets_and_reversal(
    tmp_path, crop
):
    import av

    video = tmp_path / "video.mp4"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 48, 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 10
        stream.codec_context.max_b_frames = 2
        for index in range(42):
            image = np.full((32, 48, 3), index * 5, dtype=np.uint8)
            image[5:15, 12:24, 1] = 255 - index * 5
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    episode = tmp_path / "episode.json"
    episode.write_text(
        json.dumps(
            {
                "observations": {
                    "cam": [{"path": str(video), "start": 3, "crop_bbox": crop}]
                }
            }
        )
    )
    source = XR1TrajectorySource([{"path": str(episode), "usable_frames": 39}])
    for t, u in [(0, 1), (6, 25), (25, 6), (37, 38), (8, 8)]:
        pair = source.load_pair_views(0, t, u, ["cam"])
        for frame, sample in zip((t, u), pair, strict=True):
            np.testing.assert_array_equal(
                source.get_view(0, frame, "cam"),
                source.get_view_from_sample(sample, "cam"),
            )
    with pytest.raises(IndexError, match="outside"):
        source.load_pair_views(0, 0, 39, ["cam"])
