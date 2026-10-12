"""Actual original neural Agent: weighted gradients, complete freeze, resume."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
import pytest

from dreamerv3.fg_s3a_actor import ActorBC,parameter_digests
from dreamerv3.fg_s3b_bc import WeightedActorBC
from embodied.run.fg_s3a_checkpoint import diagnostic_source,load_source
from embodied.run.fg_s3a_features import FeatureCache,build_features,teacher_step
from embodied.run.fg_s3a_train import Sampler
from embodied.run.fg_s3b_checkpoint import audit_bc,restore_bc,save_bc,validate_bc
from embodied.run.fg_s3b_train import TrainingRows,train,training_identity
from gym_trading_env.research.fg_replay import iter_replay_rows,read_manifest
from gym_trading_env.research.fg_s3a_labels import load_labels
from gym_trading_env.research.fg_s3b_advantage import build_values
from gym_trading_env.research.fg_s3b_weights import build_weight_index
from gym_trading_env.research.fg_s3b_normalization import build_normalization

GYM = Path(__file__).resolve().parents[3]/'gym-trading-env'
sys.path.insert(0,str(GYM/'tests/entry_v1'))
from test_fg_s3b_weights import certified_native_dataset


def tree_hash(tree):
    h = hashlib.sha256()
    for x in jax.tree.leaves(tree):
        a = np.asarray(x)
        h.update(str((a.shape,str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('s3b_neural')
    dataset,oracle,labels = certified_native_dataset(root/'native')
    values = root/'values'; build_values(dataset,oracle,labels,values)
    base_weights = root/'weights_v1'; build_weight_index(values,labels,base_weights)
    weights = root/'weights'; build_normalization(base_weights,weights)
    lm = load_labels(labels)
    a,source = diagnostic_source(lm,root/'source')
    snapshot = a.save_exact()
    features = root/'features'; build_features(a,source,labels,features,shard_rows=3)
    cache = FeatureCache(features,source,labels)
    return dict(root=root,labels=labels,weights=weights,source=source,agent=a,snapshot=snapshot,cache=cache)


@pytest.fixture
def original(bundle):
    a = bundle['agent']; a.load_exact(bundle['snapshot']); a.sync_policy_from_train(reset_counter=True)
    return a


def simple_batch(bundle):
    x,a = bundle['cache'].batch(np.array([0,0]))
    return x,np.ones((2,2),np.float32),np.array([0,1],np.int32)


def test_batch_one_preserves_threefold_raw_gradient_with_frozen_mean(bundle,original):
    x,mask,y = simple_batch(bundle)
    results = []
    for weight in (1.,3.):
        original.load_exact(bundle['snapshot']); original.sync_policy_from_train(reset_counter=True)
        bc = WeightedActorBC(original,global_mean_weight=1.)
        results.append(bc.update(x[:1],mask[:1],y[:1],[weight]))
    assert results[0]['grad_norm'] > 0
    np.testing.assert_allclose(results[1]['grad_norm']/results[0]['grad_norm'],3.,rtol=1e-6)
    np.testing.assert_allclose(results[1]['loss']/results[0]['loss'],3.,rtol=1e-6)


def test_batch_256_partition_gradient_mean_matches_full_population(bundle,original):
    count,batch = 4096,256
    order = np.arange(count) % bundle['cache'].rows
    x,row = bundle['cache'].batch(order)
    weights = np.where(np.arange(count) % 7 == 0,3.,.1).astype(np.float64)
    mean = float(weights.mean())
    bc = WeightedActorBC(original,global_mean_weight=mean)
    def gradients(indices):
        arrays = bc._batch(x[indices],row['mask'][indices],row['label'][indices],weights[indices])
        (_,p),g = bc._loss_grad(bc.actor_params,*arrays)
        return jax.tree.map(lambda v:np.asarray(v),g)
    full = gradients(np.arange(count))
    shuffled = np.random.default_rng(202).permutation(count)
    parts = [gradients(shuffled[start:start+batch]) for start in range(0,count,batch)]
    averages = jax.tree.map(lambda *g:np.stack(g).mean(axis=0),*parts)
    for key in full:
        np.testing.assert_allclose(averages[key],full[key],rtol=2e-5,atol=2e-6)


def test_global_mean_and_legacy_checkpoint_protocol_cannot_be_mixed(bundle,original,tmp_path):
    from gym_trading_env.research.fg_replay import signed,write_json
    rows = TrainingRows(bundle['cache'],bundle['weights'],'S3B_CERTIFIED',diagnostic=True)
    identity = training_identity(bundle['source'],bundle['cache'],rows,bundle['weights'],
        seed=0,run_id='normalization',batch_size=2,opt_config=None)
    mean = rows.normalization['global_mean_weight']
    bc = WeightedActorBC(original,global_mean_weight=mean)
    node = save_bc(tmp_path/'checkpoints',bc,Sampler(len(rows.indices),0),identity)
    wrong_bc = WeightedActorBC(original,global_mean_weight=mean*2)
    with pytest.raises(ValueError,match='global mean differs'):
        restore_bc(node,wrong_bc,Sampler(len(rows.indices),0),identity)
    with pytest.raises(ValueError,match='normalization differs'):
        save_bc(tmp_path/'wrong',wrong_bc,Sampler(len(rows.indices),0),identity)
    # Re-signing an incorrect declared constant cannot override the saved state.
    m = read_manifest(node/'manifest.json'); m.pop('content_hash')
    m['identity']['loss_normalization']['global_mean_weight'] *= 2
    write_json(node/'manifest.json',signed(m))
    with pytest.raises(ValueError,match='normalization differs'):
        audit_bc(node)
    m['protocol'] = 'FG_S3B_BC_V1'; write_json(node/'manifest.json',signed(m))
    with pytest.raises(ValueError,match='identity'):
        validate_bc(node)


def test_ones_byte_identical_to_s3a_and_analytic_weighted_gradient(bundle,original,monkeypatch):
    a = original; x,mask,y = simple_batch(bundle)
    def forbidden(*args,**kwargs): raise AssertionError('Original joint train was invoked')
    monkeypatch.setattr(a,'train',forbidden)
    base = ActorBC(a); base_metrics = base.update(x,mask,y)
    expected = parameter_digests(a.params); expected_opt = tree_hash(base.s3a_bc_opt)
    a.load_exact(bundle['snapshot']); a.sync_policy_from_train(reset_counter=True)
    bc = WeightedActorBC(a,global_mean_weight=1.); metrics = bc.update(x,mask,y,np.ones(2))
    assert parameter_digests(a.params) == expected and tree_hash(bc.s3b_bc_opt) == expected_opt
    assert metrics['loss'] == base_metrics['loss'] and metrics['grad_norm'] == base_metrics['grad_norm']
    # Nonzero frozen original optimizer/Norm/Gate states make accidental reset
    # or original joint update observable instead of merely testing zero state.
    a.load_exact(bundle['snapshot']); a.sync_policy_from_train(reset_counter=True)
    for k,v in list(a.params.items()):
        if k.startswith(('opt/state/','retnorm/')) or k == 'avail_actor_gate/value':
            a.params[k] = jax.device_put(np.full(v.shape,3 if np.issubdtype(v.dtype,np.integer) else .125,dtype=v.dtype),v.sharding)
    before = parameter_digests(a.params); bc = WeightedActorBC(a,global_mean_weight=1.)
    weights = np.array([1.,3.],np.float32)
    def reference(features,mask,labels):
        raw = a.model.pol(features,1)[a.model.action_key]
        dist = a.model._masked_policy({a.model.action_key:raw},mask,False)[a.model.action_key]
        return -(dist.logp(labels)*weights).sum()/len(weights),jax.nn.softmax(raw.logits)[:,1]
    pure = nj.pure(reference)
    def loss(params): return pure(params,jax.device_put(x),jax.device_put(mask),jax.device_put(y),seed=0,modify=False)[1][0]
    value,grads = jax.value_and_grad(loss)(bc.actor_params)
    _,(_,q) = pure(bc.actor_params,jax.device_put(x),jax.device_put(mask),jax.device_put(y),seed=0,modify=False)
    q = float(q[0]); u = a.model.config.policy.unimix
    p = (1-u)*q+u/2
    derivative = (.5/(1-p)-1.5/p)*(1-u)*q*(1-q)
    bias = [k for k in bc.actor_keys if k.endswith('/logits/bias')]
    assert len(bias) == 1
    np.testing.assert_allclose(grads[bias[0]],[-derivative,derivative],rtol=1e-5,atol=1e-6)
    changes,_ = bc.tx.update(grads,bc.s3b_bc_opt,bc.actor_params)
    expected_params = optax.apply_updates(bc.actor_params,changes)
    p0 = bc.probabilities(x,mask)
    opt0 = tree_hash(bc.s3b_bc_opt)
    metrics = bc.update(x,mask,y,weights)
    np.testing.assert_allclose(metrics['loss'],value,rtol=1e-6)
    for key in bc.actor_keys: np.testing.assert_allclose(a.params[key],expected_params[key],rtol=1e-5,atol=1e-8)
    assert bc.probabilities(x,mask).mean() > p0.mean()
    assert tree_hash(bc.s3b_bc_opt) != opt0
    assert all(before[k] == parameter_digests(a.params)[k] for k in before if not k.startswith('pol/'))
    assert bc.freeze_audit()['frozen_keys'] > 100


def test_zero_extreme_tiny_nonfinite_and_legal_masks(bundle,original):
    bc = WeightedActorBC(original,global_mean_weight=1e-300); x,mask,y = simple_batch(bundle)
    p0 = parameter_digests(original.params); opt0 = tree_hash(bc.s3b_bc_opt)
    result = bc.update(x,mask,y,np.zeros(2))
    assert result['status'] == 'ZERO_WEIGHT_SKIPPED' and bc.bc_updates == 0
    assert bc.bc_attempts == bc.skipped_batches == 1
    assert parameter_digests(original.params) == p0 and tree_hash(bc.s3b_bc_opt) == opt0
    tiny = bc.update(x,mask,np.ones(2,np.int32),np.array([1e-300,3e-300]))
    assert tiny['finite'] and tiny['single_label_batch'] and tiny['bc_updates'] == 1
    assert tiny['effective_weight_sum'] == 4e-300
    for weights in ([float('nan'),1.],[float('inf'),1.],[-1.,1.],[1e308,1e308]):
        with pytest.raises(ValueError): bc.update(x,mask,y,weights)
    with pytest.raises(FloatingPointError,match='float32 boundary'): bc.update(x,mask,y,[1e-300,1.])
    with pytest.raises(ValueError,match='legal'): bc.update(x,np.tile([1.,0.],(2,1)),y,[1.,3.])
    assert bc.bc_updates == 1 and bc.bc_attempts == 2


def test_three_interfaces_exact_resume_isolation_and_updated_policy(bundle,original,tmp_path):
    b = bundle; cache = b['cache']; a = original
    full = TrainingRows(cache,b['weights'],'S3A_FULL',diagnostic=True)
    matched = TrainingRows(cache,b['weights'],'S3A_MATCHED',diagnostic=True)
    certified = TrainingRows(cache,b['weights'],'S3B_CERTIFIED',diagnostic=True)
    assert np.array_equal(matched.indices,certified.indices) and len(full.indices) >= len(matched.indices)
    with pytest.raises(ValueError,match='subset indices'): matched.batch(np.array([-1]))
    one,two = Sampler(len(matched.indices),4),Sampler(len(certified.indices),4)
    for _ in range(3):
        left,right = one.next(5),two.next(5)
        np.testing.assert_array_equal(matched.batch(left)[3],certified.batch(right)[3])
        assert (matched.batch(left)[2] == 1).all()
    bc = WeightedActorBC(a,global_mean_weight=certified.normalization['global_mean_weight']); sampler = Sampler(len(certified.indices),4)
    identity = training_identity(b['source'],cache,certified,b['weights'],seed=4,run_id='resume',
        batch_size=5,opt_config=None,comparison_updates=6)
    def step():
        x,row,w,_ = certified.batch(sampler.next(5))
        return bc.update(x,row['mask'],row['label'],w)
    for _ in range(3): step()
    node = save_bc(tmp_path/'ckpt',bc,sampler,identity)
    assert audit_bc(node,identity)['freeze_audit']['status'] == 'PASS'
    x,row,_,_ = certified.batch(np.arange(min(2,len(certified.indices))))
    saved_p = bc.probabilities(x,row['mask'])
    for _ in range(3): step()
    params,opt,next_batch = parameter_digests(a.params),tree_hash(bc.s3b_bc_opt),sampler.next(17)
    a.load_exact(b['snapshot']); a.sync_policy_from_train(reset_counter=True)
    bc = WeightedActorBC(a,global_mean_weight=certified.normalization['global_mean_weight']); sampler = Sampler(len(certified.indices),4)
    restore_bc(node,bc,sampler,identity)
    np.testing.assert_array_equal(saved_p,bc.probabilities(x,row['mask']))
    for _ in range(3): step()
    assert parameter_digests(a.params) == params and tree_hash(bc.s3b_bc_opt) == opt
    np.testing.assert_array_equal(sampler.next(17),next_batch)
    # Genuine original Actor inference, using exactly the cached original RSSM
    # latent reconstructed again from the same first teacher observation.
    lm = load_labels(b['labels'],verify_sources=False)
    trace = Path(lm['source']['dataset'])/lm['traces'][0]['directory']
    first = next(iter_replay_rows(trace,read_manifest(trace/'manifest.json')))[1]
    with a.n_actions.lock: a.n_actions.value = 0
    _,feat,out = teacher_step(a,a.init_policy(1),first)
    np.testing.assert_allclose(out['p_add'],bc.probabilities(feat[None],np.ones((1,2),np.float32)),atol=1e-7)
    for name in ('mode','source_manifest_sha256','feature_manifest_sha256','weights_manifest_sha256','scale_sha256','seed','loss_normalization'):
        wrong = dict(identity); wrong[name] = 'foreign'
        with pytest.raises(ValueError,match='identity'): validate_bc(node,wrong)
    from embodied.run.fg_s3a_checkpoint import validate_bc as validate_s3a
    with pytest.raises(ValueError,match='identity'): validate_s3a(node)
    (tmp_path/'ckpt/.pending_interrupted').mkdir()
    assert (tmp_path/'ckpt/latest').read_text().strip() == node.name
    (node/'state.pkl').write_bytes((node/'state.pkl').read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='changed'): validate_bc(node,identity)


def test_real_native_end_to_end_pipeline_and_weight_feature_isolation(bundle,tmp_path):
    b = bundle
    common = dict(mode='S3B_CERTIFIED',seed=5,run_id='pipeline',batch_size=5,
                  comparison_updates=4,save_every=2)
    with pytest.raises(ValueError,match='overlaps'):
        train(b['source'],b['cache'],b['weights'],Path(b['source']['path']).parent.parent/'forbidden_s3b',
              updates=1,**common)
    train(b['source'],b['cache'],b['weights'],tmp_path/'resumed',updates=2,**common)
    resumed = train(b['source'],b['cache'],b['weights'],tmp_path/'resumed',updates=4,resume=True,**common)
    continuous = train(b['source'],b['cache'],b['weights'],tmp_path/'continuous',updates=4,**common)
    assert resumed['freeze_audit']['current_digests'] == continuous['freeze_audit']['current_digests']
    def state(receipt): return pickle.loads((Path(receipt['checkpoint'])/'state.pkl').read_bytes())
    assert tree_hash(state(resumed)['s3b_bc_opt']) == tree_hash(state(continuous)['s3b_bc_opt'])
    assert resumed['classification'] == continuous['classification']
    with pytest.raises(ValueError,match='identity'):
        train(b['source'],b['cache'],b['weights'],tmp_path/'resumed',updates=4,resume=True,
              **dict(common,mode='S3A_MATCHED'))
    before = [np.array(x) for x in b['cache'].features]
    # A future-dependent weight sidecar affects supervision alone. The cache
    # object passed to the ORIGINAL network contains only features and masks.
    matched = TrainingRows(b['cache'],b['weights'],'S3A_MATCHED',diagnostic=True)
    cert = TrainingRows(b['cache'],b['weights'],'S3B_CERTIFIED',diagnostic=True)
    x1,a1,_,_ = matched.batch(np.arange(len(matched.indices)))
    x2,a2,_,_ = cert.batch(np.arange(len(cert.indices)))
    np.testing.assert_array_equal(x1,x2); np.testing.assert_array_equal(a1,a2)
    for x,y in zip(before,b['cache'].features): np.testing.assert_array_equal(x,y)


def test_changed_future_ohlc_changes_evidence_but_preserves_past_rssm(bundle,original,tmp_path):
    from test_fg_single3_env import single_env
    from gym_trading_env.research.fg_single3_oracle import CandidateIndex
    from gym_trading_env.research.fg_s3b_advantage import RelaxedValues,build_value_evidence,funding_terms
    prefix = [(1000,1000,1000,1000)]*3
    histories,evidence = [],[]
    for name,last in [('TP',(1000,1008,1000,1007)),('CENSORED',(1000,1000,1000,1000))]:
        root = tmp_path/name; root.mkdir()
        env = single_env(root,prefix+[last],candidate='C1',fees=3)
        index = CandidateIndex(env)
        rows = [index.candidate(t) for t in range(index.flat_end)]
        dp = RelaxedValues(rows,index.flat_end)
        terms = funding_terms(env)
        evidence.append(build_value_evidence(rows[2],dp,300000,terms['required_cash']))
        env._initialized = False; env.reset(seed=0)
        with original.n_actions.lock: original.n_actions.value = 0
        carry = original.init_policy(1); features=[]; reward=0.
        for t in range(3):
            step = dict(env._get_obs(),action=np.int32(0),reward=np.float32(reward),
                is_first=np.bool_(t == 0),is_last=np.bool_(False),is_terminal=np.bool_(False),
                stepid=np.zeros(20,np.uint8))
            carry,feat,_ = teacher_step(original,carry,step)
            features.append(feat)
            if t < 2: _,reward,_,_,_ = env.step(0)
        histories.append(np.stack(features)); env.close()
    assert evidence[0]['certificate'] == 'ADD_BETTER'
    assert evidence[1]['proof'] == 'CENSORED' and evidence[1]['q_add'] is None
    np.testing.assert_array_equal(histories[0],histories[1])
