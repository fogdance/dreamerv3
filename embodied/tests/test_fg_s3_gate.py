"""Shared Gate contracts; actual CPU models use explicitly diagnostic data."""
import copy
import json
from pathlib import Path
import pickle
import shutil
import subprocess

import elements
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import yaml
import cloudpickle

from dreamerv3.agent import availability_prediction, hard_nonempty_mask
from dreamerv3.fg_s3c_agent import S3CAgent
from dreamerv3.fg_s3a_actor import parameter_digests
from embodied.tests.test_fg_s3_source_transfer import source_bundle
from embodied.run.fg_s3_gate import SINGLE3, gate_policy, make_gate_controller
from embodied.run.fg_s3_source_admission import (
    audit_code_transition, inspect_source, prepare_transfer, transfer_control_state)
from embodied.run.fg_s3_source_transfer import make_agent, apply_and_audit_transfer
from embodied.run.fg_s3c_train import S3CRun
from embodied.run.fg_checkpoint import file_hash, learning_spec, plain
from gym_trading_env.envs.entry_contract import fingerprint
from gym_trading_env.research.fg_replay import read_manifest, signed, write_json


def test_threshold_boundaries_float32_bfloat16_and_safe_wait():
    for dtype in (jnp.float32, jnp.bfloat16):
        logits=jnp.asarray([-20.,-1.,-1e-9,0.,1e-9,1.],dtype)
        prob,half=availability_prediction(logits,.5)
        np.testing.assert_array_equal(half,np.asarray(logits,dtype=np.float32)>=0)
        assert prob[2]==.5 and not half[2]
        prob,raw=availability_prediction(logits,.3)
        np.testing.assert_array_equal(raw,np.asarray(prob)>=np.float32(.3))
        mask,fallback=hard_nonempty_mask(prob[None,:2],.3,0,raw_mask=raw[None,:2])
        np.testing.assert_array_equal(mask,[[True,False]])
        assert fallback[0] and not raw[:2].any()
    for threshold in (0.,1.,-1.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):availability_prediction(jnp.zeros(2),threshold)


def test_audited_ast_accepts_only_the_reviewed_transition():
    root=Path(__file__).resolve().parents[2]
    old=subprocess.check_output(['git','show','09e2e5c:dreamerv3/agent.py'],cwd=root)
    current=(root/'dreamerv3/agent.py').read_bytes()
    assert len(audit_code_transition(old,current,'dreamerv3/agent.py'))==64
    bad_helper=current.replace(b'pred = logits >= 0',b'pred = logits > 0')
    with pytest.raises(ValueError,match='Unreviewed availability helper'):
        audit_code_transition(old,bad_helper,'dreamerv3/agent.py')
    bad_loss=current.replace(b'postpred = availability_prediction',b'postpred = 1 + availability_prediction')
    with pytest.raises(ValueError,match='semantics changed'):
        audit_code_transition(old,bad_loss,'dreamerv3/agent.py')
    bad_factory=current.replace(b'return self.loss',b'return self.report',1)
    with pytest.raises(ValueError,match='loss factory'):
        audit_code_transition(old,bad_factory,'dreamerv3/agent.py')


def test_direct_init_real_runner_uses_twenty_reports_and_regression_stop(source_bundle,tmp_path):
    b=source_bundle;cfg=copy.deepcopy(b['config'])
    cfg.update(batch_length=64,report_length=16)
    cfg['agent']['avail_threshold']=.3
    cfg['action_mask_warmup'].update(enabled=True,add_recall_enabled=True)
    run=S3CRun(tmp_path/'formal_policy_diagnostic_model',catalog=b['root']/'catalog',
        config=cfg,source_kind='direct_init',seed=0,bc_lambda=1.,diagnostic=True,
        gate_protocol=SINGLE3,replay_directory=b['trace']/'replay',capacity=2048)
    w=run.warmup
    assert w.required_passes==20 and w.opening_thresholds['prior_accuracy']==.9915
    assert w.opening_thresholds['prior_fpr']==.004
    passing={k:1. for k in ('avail/post_accuracy','avail/post_exact_accuracy',
        'avail/prior_accuracy','avail/prior_exact_accuracy','avail/post_add_recall','avail/prior_add_recall')}
    passing.update({'avail/prior_fpr':0.,'avail/img_fallback_rate':0.,
        'avail/post_add_positives':16,'avail/prior_add_positives':16})
    for step in range(1,20):assert not w.update(passing,step)
    assert w.consecutive_passes==19 and not w.actor_enabled
    failed=dict(passing,**{'avail/prior_accuracy':.990,'avail/prior_fpr':.005})
    assert not w.update(failed,20) and w.consecutive_passes==0
    for step in range(21,40):assert not w.update(passing,step)
    assert w.update(passing,40) and w.actor_enabled
    run.agent.set_avail_actor_enabled(w.actor_enabled)
    assert run.agent.get_avail_actor_enabled()
    regressed=dict(failed,**{'avail/prior_accuracy':.98})
    for step in range(41,45):assert not w.update(regressed,step)
    with pytest.raises(RuntimeError,match='5 consecutive reports'):w.update(regressed,45)
    bad=copy.deepcopy(cfg);bad['action_mask_warmup']['post_add_recall']=.79
    with pytest.raises(ValueError,match='Gate policy'):gate_policy(bad,diagnostic=False)


@pytest.mark.parametrize('kind',['direct_full','after_bc_s3a'])
@pytest.mark.parametrize('source_bundle',[True],indirect=True)
def test_real_cross_threshold_ready_source_is_reset_without_other_changes(source_bundle,tmp_path,kind):
    b=source_bundle;cfg=copy.deepcopy(b['config']);cfg['agent']['avail_threshold']=.3
    spec=inspect_source(kind,catalog=b['catalog'],path=b['base'] if kind=='direct_full' else b['bc'],
        base_checkpoint=b['base'] if kind=='after_bc_s3a' else None,config=cfg)
    proof=spec['gate_transfer']
    assert proof['source_gate_threshold']==.5 and proof['target_gate_threshold']==.3
    assert proof['original_gate_state']['actor_enabled'] and proof['reset_required']
    control=transfer_control_state(spec)
    assert control['samples']==0 and control['aggregate']=={} and not control['gate']['actor_enabled']
    assert control['gate']['reports']==control['gate']['consecutive_passes']==0
    target=make_agent(cfg,tmp_path/kind,agent_cls=S3CAgent,bc_lambda=1.)
    receipt=apply_and_audit_transfer(target,spec)
    assert not target.get_avail_actor_enabled()
    assert receipt['gate_reset_changed_keys']==['avail_actor_gate/value']
    for key,before in receipt['original_source_param_digests'].items():
        if key!='avail_actor_gate/value':assert before==receipt['params_after'][key]
    assert not receipt['independent_bc_optimizer_imported']
    if kind=='after_bc_s3a':assert receipt['baseline_non_actor_optimizer_equality']=='PASS'
    for name in ('gate','samples','aggregate'):
        assert control[name]==transfer_control_state(spec)[name]


def test_same_threshold_transfer_preserves_authenticated_window_and_streak(source_bundle,tmp_path):
    b=source_bundle;root=tmp_path/'source_030'
    shutil.copytree(b['base'].parent.parent,root)
    node=root/'ckpt'/b['base'].name
    cfg=copy.deepcopy(b['config']);cfg['agent']['avail_threshold']=.3
    cfg['action_mask_warmup'].update(enabled=True)
    controller=__import__('embodied.run.action_mask_warmup',fromlist=['ActionMaskWarmup']).ActionMaskWarmup(
        elements.Config(cfg['action_mask_warmup']),False)
    state=controller.save();state.update(reports=7,consecutive_passes=4)
    agg=elements.Agg();agg.add('avail/post_add_positives',16,agg='sum')
    (node/'runtime.pkl').write_bytes(cloudpickle.dumps(dict(aggregates={'warmup':agg.reducers},samples=128)))
    (node/'action_mask_warmup.pkl').write_bytes(pickle.dumps(state))
    (node/'config.yaml').write_text(yaml.safe_dump(cfg))
    tag=read_manifest(root/'s3a_diagnostic_identity.json');identity=tag['identity']
    identity.update(agent_config=cfg['agent'],learning=plain(learning_spec(cfg)))
    ih=fingerprint(identity);write_json(root/'s3a_diagnostic_identity.json',signed(dict(identity=identity,identity_hash=ih)))
    meta=json.loads((node/'manifest.json').read_text());meta['identity']['identity_hash']=ih
    meta['files']={p.name:file_hash(p) for p in node.iterdir() if p.name not in ('manifest.json','done')}
    (node/'manifest.json').write_text(json.dumps(meta))
    spec=inspect_source('direct_full',catalog=b['catalog'],path=node,config=cfg)
    control=transfer_control_state(spec)
    assert not spec['gate_transfer']['reset_required'] and control['gate']==state
    assert control['samples']==128
    restored_agg=elements.Agg();restored_agg.reducers=control['aggregate']
    assert restored_agg.result()['avail/post_add_positives']==16
    target=make_agent(cfg,tmp_path/'target')
    apply_and_audit_transfer(target,spec)
    restored=make_gate_controller(cfg,spec['gate_policy'],control,target.get_avail_actor_enabled())
    assert restored.save()==state


def test_real_availability_head_raw_mask_and_metrics_share_threshold(source_bundle,tmp_path):
    b=source_bundle;cfg=copy.deepcopy(b['config']);cfg['agent']['avail_threshold']=.3
    a=make_agent(cfg,tmp_path/'head',agent_cls=S3CAgent,bc_lambda=1.)
    kernel=next(k for k in a.params if k.startswith('avail/') and k.endswith('/logit/kernel'))
    bias=next(k for k in a.params if k.startswith('avail/') and k.endswith('/logit/bias'))
    a.params[kernel]=jnp.zeros_like(a.params[kernel])
    a.params[bias]=jnp.log(jnp.array([.1,.4])/(1-jnp.array([.1,.4])))
    feat=dict(deter=jnp.ones((1,2,32)),stoch=jnp.ones((1,2,2,4)))
    def evaluate():
        dist=a.model.avail(a.model.feat2tensor(feat),2)
        prob,raw=availability_prediction(a.model._binary_logits(dist),.3)
        mask,fallback=a.model._predicted_action_mask_info(feat,2)
        return raw,mask,fallback
    _,(raw,mask,fallback)=nj.pure(evaluate)(a.params,seed=0,create=False,modify=False)
    np.testing.assert_array_equal(raw,np.broadcast_to([False,True],(1,2,2)))
    np.testing.assert_array_equal(mask,np.broadcast_to([True,True],(1,2,2)))
    assert not fallback.any()
