import json
from pathlib import Path
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from prepare_ethernet_dataset import canonical_arrays, trajectory_arrays
from prepare_network_dataset import iter_packed_action_steps
from mibot.data.datasets.json_dataset import JsonDataset


def example():
    state = np.zeros((32, 30), dtype=np.float32)
    state[:, 13] = state[:, 28] = 1  # identity XYZW quaternions
    state[:, :7] = 90
    state[:, 15:22] = -90
    state[:, 14] = 0.02
    state[:, 29] = 0.04
    action = state.copy()
    action[:, 7] = np.arange(32) * 0.01
    action[:, 22] = np.arange(32) * 0.02
    action[:, 14] = 0.03
    action[:, 29] = 0.01
    return state, action


def test_canonical_layout_and_units_do_not_shift_labels_again():
    state, action = example()
    arrays = trajectory_arrays(state, action)
    np.testing.assert_allclose(arrays['proprios.left_arm_joint'], np.pi/2)
    np.testing.assert_allclose(arrays['proprios.right_arm_joint'], -np.pi/2)
    np.testing.assert_array_equal(arrays['actions.right_ee_pos'], action[:, 22:25])
    np.testing.assert_array_equal(arrays['actions.left_gripper_pos'], action[:, 14:15])
    # Asymmetric values ensure an old-layout indexing mistake cannot pass.
    np.testing.assert_array_equal(arrays['actions.right_gripper_pos'], action[:, 29:30])
    _, packed = next(iter_packed_action_steps(canonical_arrays(state, action), 30))
    np.testing.assert_allclose(packed[:, 0], [0, 0.01, 0.02])
    np.testing.assert_allclose(packed[:, 8], [0, 0.02, 0.04])
    np.testing.assert_allclose(packed[:, 6], 0.01, atol=1e-8)
    np.testing.assert_allclose(packed[:, 14], -0.03, atol=1e-8)


def test_numeric_sidecar_matches_training_relative_actions(tmp_path):
    state, action = example()
    # Rotate current left tool 90 degrees around z. World x becomes local -y.
    state[:, 12:14] = np.sqrt(0.5)
    action[:, 10:14] = state[:, 10:14]
    arrays = trajectory_arrays(state, action)
    np.savez(tmp_path/'values.npz', **arrays)
    path=tmp_path/'episode.json'
    path.write_text(json.dumps({'num_frames':32,'arrays_path':'values.npz'}))
    trajectory=JsonDataset._read_json(str(path))
    dataset=JsonDataset.__new__(JsonDataset);dataset.action_length=30
    actual=dataset._arm_action(trajectory,'left',0,30)
    np.testing.assert_allclose(actual[0][:,0],0,atol=1e-7)
    np.testing.assert_allclose(actual[0][:,1],-action[:30,7],atol=1e-7)
    np.testing.assert_allclose(actual[1],0,atol=1e-7)
