"""Actual Replay sampling annotations and Prefetch/Consec lifecycle."""
from functools import partial
import copy

import embodied
import numpy as np
import pytest

from embodied.tests.test_fg_s3_source_transfer import source_bundle
from embodied.run.fg_s3_teacher_adapter import attach_teacher,admit_replay_directory
from embodied.run.fg_s3c_stream import TeacherJoinStream


def make_replay(b):
    r=embodied.replay.Replay(length=129,capacity=3000,chunksize=1024,directory=b['root']/'replay_scratch',seed=202)
    r.load(directory=b['trace']/'replay')
    return r


def test_native_sampling_flags_are_authenticated_not_raw_flags(source_bundle):
    b=source_bundle;r=make_replay(b)
    assert admit_replay_directory(b['trace']/'replay',b['catalog'],diagnostic=True)>0
    raw=r.sample(2)
    assert raw['is_first'][:,0].all()
    raw['consec']=np.zeros(raw['is_first'].shape,np.int32)
    expected_native=[b['catalog'].lookup(x).record['is_first'] for x in raw['stepid'][:,0]]
    assert not all(expected_native)
    with pytest.raises(ValueError,match='MISMATCH'):
        attach_teacher(raw,b['catalog'],mode='train')
    joined=attach_teacher(raw,b['catalog'],mode='train',sampling_flags=True)
    assert joined['teacher_valid'].any()
    bad=copy.deepcopy(raw);bad['is_first'][:,0]=False
    with pytest.raises(ValueError,match='MISMATCH'):
        attach_teacher(bad,b['catalog'],mode='train',sampling_flags=True)


def test_real_prefetch_consec_restore_keeps_current_and_teacher(source_bundle,tmp_path):
    from embodied.run.fg_checkpoint import save_replay,load_replay
    b=source_bundle;r=make_replay(b)
    def stream(replay):
        return embodied.streams.Prefetch(TeacherJoinStream(embodied.streams.Consec(
            embodied.streams.Stateless(partial(replay.sample,2)),length=64,consec=2,prefix=1),b['catalog'],mode='train'))
    s=stream(r);s.enable_deterministic();iter(s);next(s)
    with s.pause():
        state=s.save_full();rs=save_replay(r,tmp_path/'replay')
    expected=[next(s) for _ in range(4)]
    restored=make_replay(b);load_replay(restored,rs,tmp_path/'replay')
    target=stream(restored);target.load_full(state);target.enable_deterministic();iter(target)
    for a in expected:
        z=next(target)
        for k in a:np.testing.assert_array_equal(a[k],z[k])
