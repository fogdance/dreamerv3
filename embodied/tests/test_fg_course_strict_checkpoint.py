"""External Reader strict transaction faults; no model or market I/O."""
import json
import pickle
from pathlib import Path

import elements
import pytest

from embodied.run.fg_checkpoint import FullCheckpoint
from embodied.run.fg_s3_e_quality_online_state import JournalState


class State:
    def __init__(self):self.value=0;self.loads=0
    def save(self):return self.value
    def validate_snapshot(self,state=None):
        if state is not None and type(state) is not int:raise ValueError('TEST_STATE_INVALID')
    def load(self,value):self.loads+=1;self.value=value
    load_exact=load


def checkpoint(root):
    step=elements.Counter()
    cp=FullCheckpoint(root/'ckpt',step,dict(test='EXTERNAL_STRICT_TRANSACTION_V2'),(),
        reserve_gib=0,keep_all=True,backend='external_immutable_course_reader',strict_latest=True)
    cp.step=step;cp.agent=State();cp.course_reader=State();cp.runtime=State()
    cp.logs=JournalState(root,('updates.jsonl',))
    return cp,step


@pytest.fixture
def committed(tmp_path):
    cp,step=checkpoint(tmp_path)
    for n in (1,2):
        step.load(n);cp.agent.value=n
        with (tmp_path/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(position=n))+'\n')
        cp.save()
    return cp,step,tmp_path


def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('attack',['missing_done','missing_manifest','corrupt_member','bad_identity',
    'backend','strict_flag','step_name','rollback_pointer','missing_pointer','symlink_pointer',
    'external_pointer','symlink_node','orphan'])
def test_latest_fault_never_falls_back_or_mutates_state(committed,attack,tmp_path):
    cp,step,root=committed;directory=root/'ckpt';node=directory/'step_000000000002'
    if attack=='missing_done':(node/'done').unlink()
    elif attack=='missing_manifest':(node/'manifest.json').unlink()
    elif attack=='corrupt_member':(node/'agent.pkl').write_bytes(b'corrupt')
    elif attack in ('bad_identity','backend','strict_flag'):
        path=node/'manifest.json';m=json.loads(path.read_text())
        if attack=='bad_identity':m['identity']={'changed':True}
        elif attack=='backend':m['backend']='native_replay'
        else:m['strict_latest']=False
        path.write_text(json.dumps(m))
    elif attack=='step_name':
        node.rename(directory/'step_000000000003');(directory/'latest').write_text('step_000000000003')
    elif attack in ('rollback_pointer','orphan'):(directory/'latest').write_text('step_000000000001')
    elif attack=='missing_pointer':(directory/'latest').unlink()
    elif attack=='symlink_pointer':
        p=directory/'latest';value=p.read_bytes();p.unlink();target=root/'external_pointer';target.write_bytes(value);p.symlink_to(target)
    elif attack=='external_pointer':(directory/'latest').write_text(str(root/'outside'))
    else:
        external=root/'outside';node.rename(external);node.symlink_to(external,target_is_directory=True)
    before=snapshot(root)
    with pytest.raises((ValueError,OSError)):cp.latest()
    with pytest.raises((ValueError,OSError)):cp.load()
    assert cp.agent.loads==cp.course_reader.loads==cp.runtime.loads==0
    assert snapshot(root)==before and int(step)==2


def test_explicit_old_load_and_old_save_cannot_bypass_published_latest(committed):
    cp,step,root=committed;before=snapshot(root)
    with pytest.raises(ValueError,match='published latest'):
        cp.load(root/'ckpt/step_000000000001')
    step.load(1)
    with pytest.raises(ValueError,match='backward|published latest'):cp.save()
    assert snapshot(root)==before and cp.agent.loads==0


def test_existing_incomplete_node_is_never_overwritten_or_repaired(committed):
    cp,_,root=committed;(root/'ckpt/step_000000000002/done').unlink()
    before=snapshot(root)
    with pytest.raises(ValueError):cp.save(final=True)
    assert snapshot(root)==before


def test_rename_before_pointer_publish_requires_review(committed,monkeypatch):
    cp,step,root=committed;step.load(3)
    def crash(dest):raise OSError('injected rename-before-publish crash')
    monkeypatch.setattr(cp,'_publish',crash)
    with pytest.raises(OSError,match='rename-before-publish'):cp.save()
    assert (root/'ckpt/step_000000000003/done').exists()
    assert (root/'ckpt/latest').read_text()=='step_000000000002'
    before=snapshot(root)
    with pytest.raises(ValueError,match='UNPUBLISHED_NODE_REQUIRES_REVIEW'):cp.latest()
    assert before==snapshot(root)


def test_pending_write_is_retained_but_never_a_recovery_candidate(committed,monkeypatch):
    cp,step,root=committed;step.load(3)
    def crash():raise OSError('injected pending write crash')
    monkeypatch.setattr(cp.agent,'save',crash)
    with pytest.raises(OSError,match='pending write'):cp.save()
    pending=list((root/'ckpt').glob('.pending_*'));assert pending
    assert cp.latest().name=='step_000000000002'
    cp.load()
    assert int(step)==2 and all(p.exists() for p in pending)


def test_committed_journal_prefix_validates_before_any_mutable_restore(committed):
    cp,_,root=committed
    with (root/'updates.jsonl').open('r+b') as f:f.write(b'CHANGED')
    before=snapshot(root)
    with pytest.raises(ValueError,match='COMMITTED_JOURNAL_MISSING_OR_CHANGED'):cp.load()
    assert cp.agent.loads==cp.course_reader.loads==cp.runtime.loads==0
    assert before==snapshot(root)


def test_valid_cold_load_restores_latest_and_archives_only_uncommitted_tail(committed):
    _,_,root=committed;committed_bytes=(root/'updates.jsonl').read_bytes()
    with (root/'updates.jsonl').open('ab') as f:f.write(b'crash tail\n')
    cold,step=checkpoint(root);cold.load()
    assert int(step)==cold.agent.value==2
    assert (root/'updates.jsonl').read_bytes()==committed_bytes
    assert [p.read_bytes() for p in (root/'recovery_logs').iterdir()]==[b'crash tail\n']
    assert cold.latest().name=='step_000000000002'
