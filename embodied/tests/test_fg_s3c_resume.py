"""Actual neural FullCheckpoint + Replay + two Prefetches + Consec + teacher join."""
import copy
import json

import jax
import numpy as np
import pytest

from embodied.tests.test_fg_s3_source_transfer import source_bundle
from embodied.tests.test_fg_s3c_agent import config_for
from embodied.run.fg_s3c_train import S3CRun
from dreamerv3.fg_s3a_actor import parameter_digests


def tree_equal(a,b):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        if isinstance(x,np.ndarray) or hasattr(x,'dtype'):np.testing.assert_array_equal(x,y)
        else:assert x==y


def test_real_neural_joint_runtime_exact_resume_no_source_reload(source_bundle,tmp_path,monkeypatch):
    b=source_bundle;cfg=config_for(b)
    cfg['agent']['avail_threshold']=.3
    # Controlled diagnostic thresholds open the real controller after report 2.
    # Production thresholds remain unchanged; no manual READY call is used.
    cfg['agent']['avail_actor_enabled']=False
    cfg['action_mask_warmup'].update(enabled=True,min_train_samples=1,consecutive_reports=2,
        add_recall_enabled=False,post_accuracy=0.,post_exact=0.,prior_accuracy=0.,
        prior_exact=0.,prior_fpr=1.,img_fallback=1.)
    run=S3CRun(tmp_path/'run',catalog=b['root']/'catalog',config=cfg,source_kind='direct_init',
               seed=0,bc_lambda=1.,replay_directory=b['trace']/'replay',capacity=2048,
               consec=2,report_every=1,save_every=2,target_updates=3)
    first=run.update();run.cp.save()
    expected=[run.update() for _ in range(2)]
    assert [bool(first['metrics']['avail/ready']),*[bool(x['metrics']['avail/ready']) for x in expected]]==[False,False,True]
    assert run.counters.state['bc_effective_updates']==1 and run.warmup.switched_step==2
    final=run.agent.save_exact();carry=copy.deepcopy(jax.device_get(run.carry))
    counters=run.counters.save();gate=run.warmup.save();sampler=copy.deepcopy(run.replay.sampler.rng.bit_generator.state)
    # Only the K=1 full bundle is the chosen recovery point. Later automatic
    # nodes are kept intact; restore selects a validated node explicitly below.
    import embodied.run.fg_s3c_train as module
    original_load=module.FullCheckpoint.load
    def load_at_one(cp,*args,**kwargs):return original_load(cp,cp._directory/'step_000000000001')
    monkeypatch.setattr(module.FullCheckpoint,'load',load_at_one)
    monkeypatch.setattr(module,'inspect_source',lambda *a,**k:(_ for _ in ()).throw(AssertionError('resume re-admitted source')))
    monkeypatch.setattr(module,'apply_and_audit_transfer',lambda *a,**k:(_ for _ in ()).throw(AssertionError('resume re-transferred source')))
    restored=S3CRun(tmp_path/'run',catalog=b['root']/'catalog',resume=True)
    actual=[restored.update() for _ in range(2)]
    for x,y in zip(expected,actual):
        for k in ('stepid','teacher_action','teacher_valid'):np.testing.assert_array_equal(x[k],y[k])
        tree_equal(x['metrics'],y['metrics'])
    assert parameter_digests(restored.agent.params)==parameter_digests(final['params'])
    tree_equal(restored.agent.save_exact(),final)
    tree_equal(jax.device_get(restored.carry),carry)
    assert restored.counters.save()==counters and restored.warmup.save()==gate
    assert restored.replay.sampler.rng.bit_generator.state==sampler
    rows=[json.loads(line) for line in (tmp_path/'run/updates.jsonl').read_text().splitlines()]
    assert [row['local_update'] for row in rows]==[1,2,3]
    archives=list((tmp_path/'run/recovery_logs').glob('*.jsonl'))
    assert len(archives)==1
    assert [json.loads(line)['local_update'] for line in archives[0].read_text().splitlines()]==[1,2,3]
    # Teacher fields were not persisted into native chunks.
    assert all(not any(k.startswith('teacher_') for k in chunk.data) for chunk in restored.replay.chunks.values())
    with pytest.raises(ValueError,match='source transfer'):
        S3CRun(tmp_path/'run',catalog=b['root']/'catalog',resume=True,source_kind='direct_init')
    identity_path=tmp_path/'run'/'run_settings.json'
    identity_path.write_text(identity_path.read_text()+' ')
    with pytest.raises(ValueError,match='settings changed'):
        S3CRun(tmp_path/'run',catalog=b['root']/'catalog',resume=True)


@pytest.mark.parametrize('kind',['direct_full','after_bc_s3a'])
@pytest.mark.parametrize('source_bundle',[True],indirect=True)
def test_real_joint_runner_transferred_sources(source_bundle,tmp_path,kind):
    b=source_bundle
    source=b['base'] if kind=='direct_full' else b['bc']
    run=S3CRun(tmp_path/kind,catalog=b['root']/'catalog',config=config_for(b),
        source_kind=kind,source_checkpoint=source,base_checkpoint=b['base'] if kind=='after_bc_s3a' else None,
        bc_lambda=1.,replay_directory=b['trace']/'replay',capacity=2048,report_every=1,save_every=1,target_updates=2)
    run.train_to(2)
    assert run.counters.state['bc_effective_updates']==2
    restored=S3CRun(tmp_path/kind,catalog=b['root']/'catalog',resume=True)
    assert parameter_digests(restored.agent.params)==parameter_digests(run.agent.params)
    assert int(restored.agent.n_updates)==run.offset+2
    assert restored.counters.state['all_updates']==2
    bad=tmp_path/(kind+'_wrong_source')
    with pytest.raises(ValueError,match='immutable preparation'):
        S3CRun(bad,catalog=b['root']/'catalog',config=config_for(b),source_kind=kind,
            source_checkpoint=source,base_checkpoint=b['base'] if kind=='after_bc_s3a' else None,
            bc_lambda=1.,expected_source_hash='foreign')
    assert not bad.exists()
