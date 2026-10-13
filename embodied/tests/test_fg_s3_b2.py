"""Small CPU B2 contracts over native rows collected from controlled FG Env."""
import copy
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import pandas as pd

import embodied
from embodied.jax.outs import Categorical, Binary
from embodied.run.fg_s3_source_transfer import make_agent
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3_b2_outcome import b2_extensions, DecisionAdapter, outcome_objective
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config, gpu_batch
from gym_trading_env.research.fg_s3_v2_collector import V2Collector

GYM = Path(__file__).resolve().parents[3]/'gym-trading-env'
sys.path.insert(0,str(GYM/'tests/entry_v1'))
from test_fg_single3_env import single_env
from test_fg_s3_b2_outcome import fixture_catalog


@pytest.fixture(scope='module')
def raw_batch(tmp_path_factory):
    folder = tmp_path_factory.mktemp('b2_original_env')
    prices = [(1000,1000,1000,1000),(1000,1007,1000,1007),
              (1000,1000,1000,1000),(1000,1000,965,980),(1000,1000,1000,1000)]*18
    env = single_env(folder, prices, candidate='C1',fees=3,chunk=90,
        times=pd.date_range('2026-01-05 13:31',periods=90,freq='min'),
        calendar=[dict(start='2026-01-05 13:30',end='2026-01-05 15:30')],
        liquidation='2026-01-05 15:30')
    catalog, query = fixture_catalog(env,folder)
    c = V2Collector(env,query,folder/'collector',GYM.parent/'dreamerv3',beta=1.,outcome_catalog=catalog)
    try:
        c.advance(75)
        stream = embodied.streams.Consec(embodied.streams.Stateless(lambda:c.sample(1)),
            length=64,consec=1,prefix=1,strict=True)
        raw = next(iter(stream))
        assert raw['outcome_event_valid'][:,1:].any() and raw['teacher_valid'][:,1:].any()
        yield raw
    finally:
        c.close()


def frozen(k):
    return k.startswith(('val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/')) or (
        k.startswith('opt/state/') and '/val/' in k)


def parameter_group(k, prefix):
    # Match the actual Ninjax Tree encoding used by persisted optimizer state.
    return k.startswith(prefix) or (k.startswith('opt/state/') and '/'+prefix.replace('_','') in k)


def model(folder, bc=1., **settings):
    return make_agent(small_config(),folder,agent_cls=S3V2Agent,device='cpu',bc_lambda=bc,
        extra_agent_config=b2_extensions(settings))


def test_outcome_masked_numerics_and_signed_bounded_adapter():
    labels = dict(outcome_event=jnp.array([[0,3,-1]]), outcome_event_valid=jnp.array([[1,1,0]],bool),
        outcome_early10=jnp.array([[0,1,-1]]), outcome_early_valid=jnp.array([[1,1,0]],bool))
    event_logits, early_logits = jnp.zeros((1,3,6)), jnp.zeros((1,3))
    def loss(e,a):
        el,al,_ = outcome_objective(Categorical(e),Binary(a),labels)
        return el+al
    np.testing.assert_allclose(loss(event_logits,early_logits),np.log(6)+np.log(2),rtol=1e-6)
    grads = jax.grad(loss,(0,1))(event_logits,early_logits)
    assert all(np.isfinite(np.asarray(g)).all() for g in grads)
    assert not np.asarray(grads[0][:,2]).any() and not np.asarray(grads[1][:,2]).any()
    empty = {**labels,'outcome_event_valid':jnp.zeros((1,3),bool),'outcome_early_valid':jnp.zeros((1,3),bool)}
    el,al,m = outcome_objective(Categorical(event_logits),Binary(early_logits),empty)
    assert el == al == m['b2/event_valid_count'] == m['b2/early_valid_count'] == 0
    adapter = DecisionAdapter(name='adapter')
    fn = nj.pure(lambda p,e:adapter(p,e)[0])
    state,_ = fn({},jnp.eye(6),jnp.zeros(6),seed=1,create=True,modify=True)
    _,d = fn(state,jnp.eye(6),jnp.zeros(6),seed=1)
    assert np.all(np.asarray(d[:3])>0) and d[3]<0 and d[4]==0 and d[5]<0
    _,risk = fn(state,jnp.eye(6),jnp.ones(6),seed=1)
    assert np.all(np.asarray(risk)<=np.asarray(d)) and np.max(np.abs(risk))<=2
    gradient = lambda p,e:nj.grad(lambda p,e:adapter(p,e)[0].sum(),[adapter])(p,e)
    _,(_,_,g) = nj.pure(gradient)(state,jnp.eye(6),jnp.zeros(6),seed=1,modify=True)
    assert any(np.asarray(v).any() for v in g.values())
    input_grad = jax.grad(lambda p:fn(state,p,jnp.zeros(6),seed=1)[1].sum())(jnp.eye(6))
    assert not np.asarray(input_grad).any()


def test_joint_updates_restore_and_frozen_critic_moments(raw_batch,tmp_path):
    a = model(tmp_path/'b2')
    for k,v in list(a.params.items()):
        if k.startswith('opt/state/') and '/val/' in k:
            a.params[k] = jax.device_put(np.full(v.shape,.125,v.dtype),v.sharding)
    carry = a.init_train(1)
    initial = parameter_digests(a.params)
    results = []
    for i,ready in enumerate([False,True,False,False,False,False,False,False]):
        a.set_avail_actor_enabled(ready)  # fixture cannot enable RL in B2.
        before = parameter_digests(a.params)
        step = int(np.asarray(a.params['opt/step/value']))
        carry,_,m = a.train(carry,gpu_batch(a,raw_batch,100+i))
        after = parameter_digests(a.params)
        assert int(np.asarray(a.params['opt/step/value'])) == step+1
        assert not m['v2/rl_active'] and m['opt/critic_grad_norm']==0
        assert all(before[k]==after[k] for k in before if frozen(k))
        assert m['opt/outcome_head_grad_norm']>0 and m['opt/rssm_grad_norm']>0 and m['opt/actor_grad_norm']>0
        assert int(np.asarray(a.params['b2_schedule/updates/value'])) == i+1
        if i<5:
            assert all(before[k]==after[k] for k in before if parameter_group(k,'b2_adapter/'))
        results.append(m)
    final = parameter_digests(a.params)
    for prefix in ('enc/','dyn/','pol/','b2_event/','b2_early/','b2_adapter/'):
        assert any(initial[k]!=final[k] for k in initial if k.startswith(prefix)),prefix
    assert results[-1]['b2/mix_value']>0
    saved = a.save_exact(); hashes = parameter_digests(a.params)
    a.load_exact(pickle.loads(pickle.dumps(saved))); a.sync_policy_from_train()
    assert hashes == parameter_digests(a.params)
    carry,_,m = a.train(carry,gpu_batch(a,raw_batch,109))
    assert m['opt/outcome_head_changed'] and m['opt/wm_changed']
    empty = copy.deepcopy(raw_batch)
    empty['teacher_valid'][:]=False
    empty['outcome_event_valid'][:]=False;empty['outcome_early_valid'][:]=False
    before = parameter_digests(a.params)
    carry,_,m = a.train(carry,gpu_batch(a,empty,110))
    after = parameter_digests(a.params)
    assert m['bc/nll']==m['b2/event_loss']==m['b2/early_loss']==0
    assert all(before[k]==after[k] for k in before if frozen(k) or any(parameter_group(k,p)
        for p in ('pol/','b2_adapter/','b2_event/','b2_early/')))


def test_bc_detaches_outcome_and_rssm_but_outcome_gradient_reaches_encoder(raw_batch,tmp_path):
    a = model(tmp_path/'gradient')
    for key,value in [('b2_schedule/updates/value',20),('b2_schedule/event_updates/value',4)]:
        a.params[key] = jax.device_put(np.array(value,np.int32),a.params[key].sharding)
    data = gpu_batch(a,raw_batch,2);data.pop('seed')
    def teaching(c,d,outcome):
        c,obs,prev,_ = a.model._apply_replay_context(c,d)
        _,(_,_,outs,_) = a.model.wm_only(c,obs,prev,False)
        _ = [m.values for m in a.model.modules]
        if not outcome:
            return a.model.b2_bc_loss(outs['repfeat'],obs['action_mask'],
                d['teacher_action'][:,1:],d['teacher_valid'][:,1:])[0]
        event,early = a.model.b2_predict(outs['repfeat'],2)
        labels = {k:d[k][:,1:] for k in d if k.startswith('outcome_')}
        el,al,_ = outcome_objective(event,early,labels)
        return el+al
    for outcome in (False,True):
        fn = jax.jit(nj.pure(lambda c,d:nj.grad(lambda c,d:teaching(c,d,outcome),
            a.model.modules)(c,d)),static_argnames=('create','modify'))
        _,(_,_,g) = fn(a.params,a.init_train(1),data,seed=2,create=False,modify=True)
        if outcome:
            for prefix in ('b2_event/','b2_early/','enc/','dyn/'):
                assert any(np.asarray(v).any() for k,v in g.items() if k.startswith(prefix)),prefix
            assert all(not np.asarray(v).any() for k,v in g.items() if k.startswith(('pol/','b2_adapter/','val/')))
        else:
            assert all(not np.asarray(v).any() for k,v in g.items() if not k.startswith(('pol/','b2_adapter/')))
            for prefix in ('pol/','b2_adapter/'):
                assert any(np.asarray(v).any() for k,v in g.items() if k.startswith(prefix))


def test_real_policy_mix_zero_equivalence_sync_and_mask(raw_batch,tmp_path):
    cfg = small_config()
    old = make_agent(cfg,tmp_path/'a2',agent_cls=S3V2Agent,device='cpu',bc_lambda=1.,
                     extra_agent_config={'s3_v2_mode':'a2'})
    a = model(tmp_path/'b2')
    groups = ('enc/','dyn/','dec/','pol/')
    a.params.update({k:v for k,v in old.params.items() if k.startswith(groups)})
    a.sync_policy_from_train();old.sync_policy_from_train()
    index = int(np.flatnonzero(raw_batch['action_mask'][0,:,1])[0])
    obs = {k:np.asarray(raw_batch[k][:,index]) for k in old.obs_space}
    obs['is_first'] = np.array([True]);obs['is_last'] = np.array([False])
    for mode in ('eval','train'):
        _,act,out = a.policy(a.init_policy(1),obs,mode=mode)
        _,base,bout = old.policy(old.init_policy(1),obs,mode=mode)
        np.testing.assert_array_equal(act['action'],base['action'])
        np.testing.assert_array_equal(out['p_add'],bout['p_add'])
        assert out['b2/mix_value'][0]==0
    for key,value in [('b2_schedule/updates/value',20),('b2_schedule/event_updates/value',4)]:
        a.params[key] = jax.device_put(np.array(value,np.int32),a.params[key].sharding)
    a.sync_policy_from_train()
    assert all(k in a.policy_keys for k in a.params if k.startswith('b2_'))
    _,_,out = a.policy(a.init_policy(1),obs,mode='eval')
    assert out['b2/mix_value'][0]==1 and abs(out['b2/delta_logit'][0])>0
    assert out['b2/p_add_final'][0]!=out['b2/p_add_base'][0]
    obs['action_mask'] = np.array([[1,0]],np.float32)
    _,act,out = a.policy(a.init_policy(1),obs,mode='train')
    assert act['action'][0]==out['p_add'][0]==0
    obs['outcome_event'] = np.zeros(1,np.int32)
    with pytest.raises(AssertionError): a.policy(a.init_policy(1),obs,mode='eval')


def test_lambda_zero_teacher_independence_and_frozen_actor(raw_batch,tmp_path):
    a = model(tmp_path/'lambda_zero',bc=0.)
    for k,v in list(a.params.items()):
        if k.startswith('opt/state/') and '/pol/' in k:
            a.params[k] = jax.device_put(np.full(v.shape,.125,v.dtype),v.sharding)
    saved = a.save_exact(); before = parameter_digests(a.params)
    _,_,m = a.train(a.init_train(1),gpu_batch(a,raw_batch,51))
    after = parameter_digests(a.params)
    assert m['bc/nll']==m['opt/actor_grad_norm']==0
    assert all(before[k]==after[k] for k in before if frozen(k) or parameter_group(k,'pol/'))
    a.load_exact(saved)
    changed = copy.deepcopy(raw_batch)
    changed['teacher_action'][:]=1-changed['teacher_action'];changed['teacher_valid'][:]=False
    a.train(a.init_train(1),gpu_batch(a,changed,51))
    assert after == parameter_digests(a.params)


def test_warm_extension_uses_real_ninjax_optimizer_names(tmp_path):
    """Serialized schema contract fixture, not real source-model acceptance."""
    import json
    from embodied.run.fg_s3_source_admission import prepare_transfer, PROTOCOL
    from embodied.run.fg_checkpoint import file_hash
    from gym_trading_env.research.fg_replay import signed
    node = tmp_path/'step_000000000002';node.mkdir()
    params = {'pol/kernel':np.zeros((2,2),np.float32),'opt/step/value':np.array(2,np.int32)}
    state = dict(params=params,counters=dict(updates=2,batches=2,actions=0),policy_params=None,
                 pending_sync=None,pending_outs=None,pending_mets=None)
    (node/'agent.pkl').write_bytes(pickle.dumps(state))
    (node/'step.pkl').write_bytes(pickle.dumps(2))
    (node/'manifest.json').write_text(json.dumps(dict(schema=1,protocol='FG_FULL_V1',step=2,identity={},
        files={'agent.pkl':file_hash(node/'agent.pkl'),'step.pkl':file_hash(node/'step.pkl')})))
    (node/'done').touch()
    spec = signed(dict(protocol=PROTOCOL,kind='warm_start',origin=dict(path=str(node),
        manifest_sha256=file_hash(node/'manifest.json'),step=2),agent_extension=b2_extensions({})))
    schema = {k:(v.shape,str(v.dtype)) for k,v in params.items()}
    nn = {'b2_event/kernel':np.zeros((2,6),np.float32),'b2_early/kernel':np.zeros((2,1),np.float32),
          'b2_adapter/raw_weights':np.zeros(6,np.float32)}
    schema.update({k:(v.shape,str(v.dtype)) for k,v in nn.items()})
    schema['b2_schedule/updates/value'] = ((), 'int32')
    mapping,_ = nj.flatten(nn)
    schema.update({'opt/state/1/1/'+k:(v.shape,str(v.dtype)) for k,v in mapping.items()})
    assert any('b2event/' in k for k in schema)
    assert prepare_transfer(spec,schema)['counters']['updates'] == 2
    with pytest.raises(ValueError,match='foreign state'):
        prepare_transfer(spec,{**schema,'opt/state/1/1/foreign/kernel':((2,6),'float32')})
    with pytest.raises(ValueError,match='original parameter schema'):
        prepare_transfer(spec,{**schema,'pol/kernel':((3,2),'float32')})
