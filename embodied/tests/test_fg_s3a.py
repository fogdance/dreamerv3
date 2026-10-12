"""Real original neural Agent on CPU, full native data and exact BC resume."""
import copy
import json
import pickle
from pathlib import Path
import os
import sys

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import embodied.jax

from dreamerv3.agent import Agent as OriginalAgent
from dreamerv3.fg_s3a_actor import ActorBC,classification_metrics,parameter_digests
from embodied.run.fg_s3a_checkpoint import (audit_bc,diagnostic_source,load_source,restore_bc,save_bc,
                                           validate_bc,validate_source)
from embodied.run.fg_s3a_features import FeatureCache,build_features,teacher_step
from embodied.run.fg_s3a_train import Sampler,train,training_identity
from gym_trading_env.research.fg_replay import iter_replay_rows,read_manifest
from gym_trading_env.research.fg_s3a_labels import build_labels,load_labels

GYM=Path(__file__).resolve().parents[3]/'gym-trading-env'
sys.path.insert(0,str(GYM/'tests/entry_v1'))
from test_fg_s3a_labels import native_dataset


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root=tmp_path_factory.mktemp('s3a_neural')
    dataset,oracle,trace,manifest=native_dataset(root/'native')
    labels=root/'labels';lm=build_labels(dataset,oracle,labels)
    agent,source=diagnostic_source(lm,root/'source')
    snapshot=agent.save_exact()
    return dict(root=root,labels=labels,lm=lm,agent=agent,source=source,
                snapshot=snapshot,trace=trace,manifest=manifest)


@pytest.fixture
def original(bundle):
    a=bundle['agent'];a.load_exact(bundle['snapshot']);a.sync_policy_from_train(reset_counter=True)
    return a


def observed(bundle,limit=12):
    from itertools import islice
    return [step for _,step in islice(iter_replay_rows(bundle['trace'],bundle['manifest']),limit)]


def test_original_policy_parity_teacher_history_and_future_isolation(bundle,original):
    rows=observed(bundle);a=original
    # Raw original and new adapter share precisely the same variables and RNG.
    state=a.params
    init=nj.pure(a.model.init_policy);_,carry=init(state,1,seed=0,modify=False)
    obs={k:jax.device_put(np.asarray(rows[0][k])[None]) for k in a.obs_space}
    baseline=nj.pure(lambda c,o:OriginalAgent.policy(a.model,c,o,mode='eval'))
    adapter=nj.pure(lambda c,o:a.model.policy(c,o,mode='eval'))
    _,expected=baseline(state,carry,obs,seed=23,modify=False)
    _,actual=adapter(state,carry,obs,seed=23,modify=False)
    for x,y in zip(jax.tree.leaves(expected[:2]),jax.tree.leaves(actual[:2])):np.testing.assert_array_equal(x,y)
    np.testing.assert_allclose(expected[2]['p_add'],actual[2]['p_add'],rtol=0,atol=1e-7)
    blocked={k:jax.device_put(np.asarray(rows[1][k])[None]) for k in a.obs_space}
    _,result=baseline(state,carry,blocked,seed=23,modify=False)
    jax.block_until_ready(result)
    assert float(result[2]['p_add'][0]) == 0.
    def rollout(sequence):
        with a.n_actions.lock:a.n_actions.value=0
        carry=a.init_policy(1);features=[]
        for step in sequence:
            carry,x,_=teacher_step(a,carry,step);features.append(x)
        return np.stack(features)
    first=rollout(rows);second=rollout(rows);np.testing.assert_array_equal(first,second)
    changed=copy.deepcopy(rows)
    for step in changed[6:]:
        step['market_seq']=step['market_seq']+999;step['action']=np.int32(1-int(step['action']));step['reward']=np.float32(777)
    modified=rollout(changed);np.testing.assert_array_equal(first[:6],modified[:6])
    assert not np.array_equal(first[7:],modified[7:])
    forbidden=dict(rows[0],oracle_label=np.int32(1))
    with pytest.raises(ValueError,match='schema'):teacher_step(a,a.init_policy(1),forbidden)
    # Regression for the observed original diagnostic failure: forced WAIT must
    # return a safe ADD probability of zero without dropping the real mask.
    carry=a.init_policy(1);carry,_,_=teacher_step(a,carry,rows[0])
    _,_,out=teacher_step(a,carry,rows[1]);assert float(out['p_add'][0])==0.


def test_actor_gradient_equivalence_frozen_nonzero_state_and_inference(bundle,original):
    a=original
    for key,value in list(a.params.items()):
        if key.startswith(('opt/state/','retnorm/')):
            a.params[key]=jax.device_put(np.full(value.shape,3 if np.issubdtype(value.dtype,np.integer) else .125,dtype=value.dtype),value.sharding)
    rows=observed(bundle,1);_,feat,_=teacher_step(a,a.init_policy(1),rows[0])
    x=np.repeat(feat[None],8,axis=0);mask=np.ones((8,2),np.float32);labels=np.ones(8,np.int32)
    before=parameter_digests(a.params);bc=ActorBC(a)
    before_opt=jax.tree.map(np.asarray,bc.s3a_bc_opt)
    p0=bc.probabilities(x,mask)
    opt=embodied.jax.Optimizer([a.model.pol],a.model._make_opt(**bc.opt_config),diagnostics=False,name='s3a_bc_opt_reference')
    def reference_loss(x,m,y):
        d=a.model._masked_policy(a.model.pol(x,1),m,False)[a.model.action_key]
        return -d.logp(y).mean()
    reference=nj.pure(lambda x,m,y:opt(reference_loss,x,m,y))
    ref_state=nj.init(reference)(a.params,jax.device_put(x),jax.device_put(mask),jax.device_put(labels),seed=0)
    ref_state,_=reference(ref_state,jax.device_put(x),jax.device_put(mask),jax.device_put(labels),seed=0)
    metrics=bc.update(x,mask,labels)
    for key in bc.actor_keys:np.testing.assert_allclose(a.params[key],ref_state[key],rtol=1e-6,atol=1e-8)
    assert bc.probabilities(x,mask).mean()>p0.mean()
    audit=bc.freeze_audit();assert audit['frozen_keys']>100 and metrics['grad_norm']>0
    assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(before_opt),jax.tree.leaves(bc.s3a_bc_opt)))
    assert all(before[k]==parameter_digests(a.params)[k] for k in before if not k.startswith('pol/'))
    assert float(a.params['avail_actor_gate/value'])==0
    with a.n_actions.lock:a.n_actions.value=0
    _,_,policy_out=teacher_step(a,a.init_policy(1),rows[0])
    np.testing.assert_allclose(policy_out['p_add'],bc.probabilities(x[:1],mask[:1]),atol=1e-7)
    with pytest.raises(ValueError,match='legal'):bc.update(x,np.tile([1,0],(8,1)),labels)
    # WAIT gradient direction and a measurable decrease on the same tiny learnable batch.
    wait=np.zeros(8,np.int32);before_wait=bc.probabilities(x,mask).mean();losses=[]
    for _ in range(10):losses.append(bc.update(x,mask,wait)['loss'])
    assert bc.probabilities(x,mask).mean()<before_wait and losses[-1]<losses[0]
    assert int(a.n_updates)==int(a.n_batches)==0


def test_full_1024_features_checkpoint_resume_and_corruption(bundle,original,tmp_path):
    a=original
    m=build_features(a,bundle['source'],bundle['labels'],tmp_path/'features',shard_rows=137)
    assert m['complete'] and m['visited_rows']>1024 and m['max_original_probability_error']<=1e-6
    assert m['rows']==bundle['lm']['summary']['counts']['eligible']
    # Chunk-reset holding state was validated by the original collector and labels.
    a.load_exact(bundle['snapshot']);a.sync_policy_from_train(reset_counter=True)
    cache=FeatureCache(tmp_path/'features',bundle['source'],bundle['labels']);bc=ActorBC(a);sampler=Sampler(cache.rows,9)
    identity=training_identity(bundle['source'],cache,9,'resume',16,None)
    def update():
        x,index=cache.batch(sampler.next(16));return bc.update(x,index['mask'],index['label'])
    for _ in range(3):update()
    node=save_bc(tmp_path/'ckpt',bc,sampler,identity)
    assert audit_bc(node,identity)['freeze_audit']['status']=='PASS'
    probe,probe_index=cache.batch(np.arange(3))
    checkpoint_probability=bc.probabilities(probe,probe_index['mask'])
    for _ in range(3):update()
    continuous=parameter_digests(a.params);continuous_opt=jax.tree.map(np.asarray,bc.s3a_bc_opt);next_indices=sampler.next(31)
    # Restore requires the source first, then BC weights; source cannot overwrite resumed pol.
    a.load_exact(bundle['snapshot']);a.sync_policy_from_train(reset_counter=True);bc=ActorBC(a);sampler=Sampler(cache.rows,9)
    restore_bc(node,bc,sampler,identity)
    np.testing.assert_array_equal(bc.probabilities(probe,probe_index['mask']),checkpoint_probability)
    for _ in range(3):update()
    assert parameter_digests(a.params)==continuous
    for x,y in zip(jax.tree.leaves(bc.s3a_bc_opt),jax.tree.leaves(continuous_opt)):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(sampler.next(31),next_indices)
    with pytest.raises(ValueError,match='identity'):validate_bc(node,dict(identity,seed=10))
    # Interrupted package is never published over the latest valid checkpoint.
    (tmp_path/'ckpt/.pending_interrupted').mkdir();(tmp_path/'ckpt/.pending_interrupted/state.pkl').write_bytes(b'incomplete')
    assert (tmp_path/'ckpt/latest').read_text().strip()==node.name
    validate_bc(node,identity)
    (node/'state.pkl').write_bytes((node/'state.pkl').read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='changed'):validate_bc(node,identity)


def test_model_source_schema_and_cache_identity(bundle,original,tmp_path):
    loaded=load_source(bundle['source'],tmp_path/'loaded')
    assert parameter_digests(loaded.params)==parameter_digests(bundle['snapshot']['params'])
    m=build_features(original,bundle['source'],bundle['labels'],tmp_path/'features',max_rows=32)
    with pytest.raises(ValueError,match='identity'):
        FeatureCache(tmp_path/'features',dict(bundle['source'],manifest_sha256='bad'),bundle['labels'])
    shard=tmp_path/'features'/m['shards'][0]['features'];value=np.load(shard);value[0,0]+=1;np.save(shard,value)
    with pytest.raises(ValueError,match='changed'):FeatureCache(tmp_path/'features',bundle['source'],bundle['labels'])
    source8000=Path('/data/logdir/fg_single3/gate_margin_v2/C1_M1_1M/ckpt/step_000000008000')
    real_labels=Path(os.environ.get('FG_S3A_REAL_LABELS',GYM/'artifacts/fg_s3a_v1/labels_20261011_v2'))
    if source8000.exists() and real_labels.exists():
        real=load_labels(real_labels)
        with pytest.raises(ValueError,match='Completed protected'):
            validate_source(source8000,real)


def test_all_wait_prior_metrics_and_uniform_sampler():
    y=np.r_[np.ones(4,np.int32),np.zeros(96,np.int32)]
    scores=classification_metrics(y,np.zeros(100))
    assert scores['status']=='ALL_WAIT_COLLAPSE' and scores['accuracy']==.96 and scores['recall']==0
    assert scores['pr_auc']==.04 and scores['balanced_accuracy']==.5
    sampler=Sampler(100,0);assert sorted(sampler.next(100).tolist())==list(range(100))
    state=sampler.save();expected=sampler.next(107);sampler.load(state);np.testing.assert_array_equal(sampler.next(107),expected)
