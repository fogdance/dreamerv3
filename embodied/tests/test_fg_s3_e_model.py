"""Real E neural updates on CPU; controlled inputs, no public E unlock."""
import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied
from embodied.run.fg_s3_source_transfer import _make_agent
from embodied.run.fg_checkpoint import host, plain
from dreamerv3.fg_s3_v2_agent import _S3V2Model
from dreamerv3.fg_s3_composition import resolve_composition
from dreamerv3.fg_s3_e_config import e_target_config
from dreamerv3.fg_s3_tutor_loss import agent_extensions
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_s3_v2_gpu_check import small_config, gpu_batch

GYM=Path(__file__).resolve().parents[3]/'gym-trading-env'
sys.path.insert(0,str(GYM/'tests/entry_v1'))
from e_model_fixture import collector_fixture


def composition(mode):
    return resolve_composition({'composition':dict(encoder='e_multiscale_1m15m',
        teaching={'a2':'a2','tutor':'tutor_cd','b2':'b2'}[mode])})


def model(folder,mode='a2'):
    cfg=e_target_config(small_config())
    extension=agent_extensions(dict(mode=mode,tutor_hard_kappa=1.,
        b2_mix_warmup=2,b2_mix_ramp=2,b2_min_event_updates=2))
    return _make_agent(cfg,folder,agent_cls=_S3V2Model,device='cpu',bc_lambda=1.,
        extra_agent_config=extension,composition=composition(mode))


@pytest.fixture(scope='module')
def raw_e(tmp_path_factory):
    root=tmp_path_factory.mktemp('e_native')
    collector,provider=collector_fixture(root)
    try:
        collector.advance(75)
        stream=embodied.streams.Consec(embodied.streams.Stateless(lambda:collector.sample(1)),
            length=64,consec=1,prefix=1,strict=True)
        raw=next(iter(stream))
        assert raw['teacher_valid'][:,1:].any() and raw['outcome_event_valid'][:,1:].any()
        assert raw['market_context_15m'][...,6].any()
        yield raw
    finally:collector.close()


def neural_gradients(agent,raw,kind):
    def loss(c,d):
        c,obs,prev,_=agent.model._apply_replay_context(c,d)
        if kind=='wm':return agent.model.wm_only(c,obs,prev,False)[0]
        _,(_,_,out,_) = agent.model.wm_only(c,obs,prev,False)
        return agent.model.bc_loss(out['repfeat'],obs['action_mask'],
            d['teacher_action'][:,1:],d['teacher_valid'][:,1:])[0]
    mods=[m for m in agent.model.modules if m is not agent.model.val
          and (kind!='wm' or m is not agent.model.pol)]
    fn=jax.jit(nj.pure(lambda c,d:nj.grad(loss,mods)(c,d)),static_argnames=('create','modify'))
    data=train_batch(agent,raw,31);data.pop('seed')
    _,(_,_,grads)=fn(agent.params,agent.init_train(1),data,seed=31,create=False,modify=True)
    return {k:np.asarray(v) for k,v in grads.items()}


def train_batch(agent,raw,seed):
    return gpu_batch(agent,{k:v for k,v in raw.items() if k in agent.spaces},seed)


@pytest.mark.parametrize('mode',['a2','tutor','b2'])
def test_EM07_EM08_EM11_EM12_EM13_real_joint_updates_policy_sync(raw_e,tmp_path,mode):
    a=model(tmp_path/'model',mode)
    assert all(d.platform=='cpu' for d in jax.devices())
    before=parameter_digests(a.params)
    carry=a.init_train(1)
    metrics=[]
    for step in range(5):
        carry,_,m=a.train(carry,train_batch(a,raw_e,100+step))
        metrics.append({k:float(np.asarray(v)) for k,v in m.items() if np.asarray(v).size==1})
    after=parameter_digests(a.params)
    for prefix in ('enc/minute/','enc/context15/','enc/fusion/','pol/'):
        assert any(before[k]!=after[k] for k in before if k.startswith(prefix))
    assert all(before[k]==after[k] for k in before if k.startswith(
        ('val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/')))
    assert all(np.isfinite(v) for m in metrics for v in m.values())
    if mode=='tutor':
        assert any(k.startswith('tutor/') for k in metrics[-1])
    if mode=='b2':
        assert metrics[-1]['b2/mix_value']>0 and metrics[-1]['opt/outcome_head_grad_norm']>0
        for prefix in ('b2_event/','b2_early/','b2_adapter/'):
            assert any(before[k]!=after[k] for k in before if k.startswith(prefix))
        assert int(a.params['b2_schedule/updates/value'])==5
    a.sync_policy_from_train()
    state=a.save_exact();policy=parameter_digests(state['policy_params'])
    assert all(policy[k]==after[k] for k in policy)
    assert all(k in a.policy_keys for k in before if k.startswith('enc/'))
    receipt=dict(mode=mode,composition=composition(mode),scope='CPU_SYNTHETIC_CONTRACT_ONLY',
        actual_updates=5,optimizer_warmup=1000,parameter_changes=[k for k in before if before[k]!=after[k]],
        before=before,after=after,policy=policy,metrics=metrics)
    (tmp_path/'updates.json').write_text(json.dumps(receipt,indent=2))


def test_EM06_EM07_EM09_wm_branch_gradients_decoder_whitelist_and_bc_isolation(raw_e,tmp_path):
    a=model(tmp_path/'wm')
    grads=neural_gradients(a,raw_e,'wm')
    norms={p:sum(float(np.linalg.norm(v)) for k,v in grads.items() if k.startswith(p))
           for p in ('enc/minute/','enc/context15/','enc/fusion/')}
    assert all(np.isfinite(v).all() for v in grads.values()) and all(v>0 for v in norms.values())
    bc=neural_gradients(a,raw_e,'bc')
    assert any(np.count_nonzero(v) for k,v in bc.items() if k.startswith('pol/'))
    assert all(not np.count_nonzero(v) for k,v in bc.items() if not k.startswith('pol/'))

    def wm(c,d):
        c,obs,prev,ids=a.model._apply_replay_context(c,d)
        result=a.model.world_model(c,obs,prev,False)
        return c,obs,prev,ids,result[-2]
    data=train_batch(a,raw_e,40);data.pop('seed')
    fn=nj.pure(wm)
    _,(carry,obs,prev,ids,losses)=fn(a.params,a.init_train(1),data,seed=40,modify=True)
    assert set(losses)=={'dyn','rep','rew','con','avail_post','avail_prior','market_seq_e_shared','position_state'}
    assert all(np.isfinite(v).all() for v in losses.values())
    assert obs['market_context_15m'].shape==(1,64,24,7)
    np.testing.assert_array_equal(prev['action'],raw_e['action'][:,:-1])
    audit={**data,'market_seq':data['market_seq']+999.}
    _,out=fn(a.params,a.init_train(1),audit,seed=40,modify=True)
    for k in losses:np.testing.assert_array_equal(out[-1][k],losses[k])
    def direct_prefix(d):
        ec,dc,xc,action=a.model.init_train(1)
        prefix={k:d[k][:,:1] for k in a.obs_space}
        ec,_,tokens=a.model.enc(ec,prefix,prefix['is_first'],False)
        dc,_,feat=a.model.dyn.observe(dc,tokens,{k:v[:,None] for k,v in action.items()},
                                    prefix['is_first'],False)
        return dc,feat['logit'],tokens
    prefix=nj.pure(direct_prefix)
    _,(dc,logit,tokens)=prefix(a.params,data,seed=40,modify=True)
    for k in dc:np.testing.assert_array_equal(dc[k],carry[1][k])
    changed={**data,'market_context_15m':data['market_context_15m'].at[:,0,:,:6].set(10.).at[:,0,:,6].set(1.)}
    _,(_,changed_logit,changed_tokens)=prefix(a.params,changed,seed=40,modify=True)
    assert not np.array_equal(tokens,changed_tokens) and not np.array_equal(logit,changed_logit)
    (tmp_path/'gradients.json').write_text(json.dumps(dict(wm_branch_norms=norms,
        bc_encoder_norm=sum(float(np.linalg.norm(v)) for k,v in bc.items() if k.startswith('enc/')),
        decoder_targets=list(a.model.dec.obs_space),loss_keys=list(losses)),indent=2))


def test_EM04_actual_policy_context_changes_rssm_and_actor(raw_e,tmp_path):
    a=model(tmp_path/'policy')
    obs={k:raw_e[k][:,40].copy() for k in a.obs_space}
    obs['action_mask'][:]=1.
    a.sync_policy_from_train()
    snapshot=a.save_exact()
    c1,_,o1=a.policy(a.init_policy(1),obs,mode='eval')
    a.load_exact(snapshot)
    other=copy.deepcopy(obs)
    other['market_context_15m'][...,:6]=-100.
    other['market_context_15m'][...,6]=1.
    c2,_,o2=a.policy(a.init_policy(1),other,mode='eval')
    # A changed categorical posterior can produce the same single hard draw.
    # This also reproduces on the fixed WP03 baseline. First prove posterior
    # sensitivity, then actual causal Actor sensitivity over a fixed rollout.
    def posterior(raw):
        carry=a.model.init_policy(1)
        _,_,tokens=a.model.enc(carry[0],raw,raw['is_first'],False,single=True)
        _,_,feat=a.model.dyn.observe(carry[1],tokens,carry[3],raw['is_first'],False,single=True)
        return feat['logit']
    def logits(raw):
        return np.asarray(nj.pure(posterior)(a.params,jax.tree.map(jnp.asarray,raw),
                          seed=202,create=False,modify=False)[1])
    assert not np.array_equal(logits(obs),logits(other))
    def rollout(raw):
        a.load_exact(snapshot);carry=a.init_policy(1);out=[]
        for _ in range(8):
            carry,_,o=a.policy(carry,raw,mode='eval');out.append(np.asarray(o['p_add']))
        return np.stack(out)
    original_probs,changed_probs=rollout(obs),rollout(other)
    assert not np.array_equal(original_probs,changed_probs)
    (tmp_path/'policy_effect.json').write_text(json.dumps(dict(
        p_add_original=original_probs.tolist(),p_add_changed=changed_probs.tolist(),
        synthetic_context_magnitude=-100.,rng_identical=True,rollout_steps=8,
        posterior_logits_changed=True),indent=2))
