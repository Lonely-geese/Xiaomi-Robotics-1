from pathlib import Path
import sys

import numpy as np
import pyarrow as pa

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_mixed_middle_dataset import metadata_rejection, stratified_selection, edge_crop, new_temporal_segments


def test_success_is_required_even_when_data_valid():
    assert metadata_rejection({'operator_verdict':'failure','data_valid':True}) == 'explicit_failure'
    assert metadata_rejection({'data_valid':True}) == 'unknown_verdict'
    assert metadata_rejection({'operator_verdict':'success','structurally_valid':False}) == 'explicit_invalid'
    assert metadata_rejection({'operator_verdict':'success','quality_status':'degraded'}) is None


def test_selection_is_by_original_episode_reproducible_and_covers_sessions():
    records = [dict(dataset=f'd{s//3}',session=f's{s}',episode_index=i) for s in range(6) for i in range(20)]
    a,q = stratified_selection(records,31,42)
    b,q2 = stratified_selection(list(reversed(records)),31,42)
    assert a == b and q == q2
    assert len(a) == 31 and len({r['session'] for r in a}) == 6
    assert len({(r['dataset'],r['session'],r['episode_index']) for r in a}) == 31


def test_edge_crop_preserves_internal_wait_and_context():
    state = np.zeros((250,30),np.float32)
    state[:,13] = state[:,28] = 1
    state[50:100,7] = np.arange(50)*0.001
    state[100:150,7] = .049
    state[150:200,7] = .049+np.arange(50)*.001
    state[200:,7] = .098
    a,b,report = edge_crop(state)
    assert 0 < a <= 40 and 215 <= b < 250
    assert a < 100 < 150 < b
    still = state[:40].copy()
    x,y,details = edge_crop(still)
    assert (x,y) == (0,40) and details['status'] == 'no_confident_motion_preserved'


def test_new_shift_targets_cannot_cross_acquisition_or_camera_gap():
    n = 140
    seq = np.arange(n);seq[70:] += 5
    audit = pa.table({'observation_sequence':seq,'camera_sample_sequence.leftCamera':seq})
    segments,_ = new_temporal_segments(audit,np.arange(n)/30,n)
    assert segments == [(0,67),(70,137)]
    for a,b in segments:
        assert np.all(seq[a+3:b+3]-seq[a:b] == 3)
        assert b+3 <= n
