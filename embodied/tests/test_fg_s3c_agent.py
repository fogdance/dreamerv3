"""Real neural CPU tests; controlled Gate is diagnostic, never production admission."""
import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from embodied.tests.test_fg_s3_source_transfer import source_bundle
from embodied.run.fg_s3_source_admission import inspect_source
from embodied.run.fg_s3_source_transfer import make_agent,apply_and_audit_transfer
from embodied.run.fg_s3_teacher_adapter import attach_teacher
from dreamerv3.agent import Agent
from dreamerv3.fg_s3c_agent import S3CAgent
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_replay import iter_replay_rows


def batch_for(b,offset=1000,length=65):
    rows=[s for n,s in iter_replay_rows(b['trace'],b['tm']) if offset<=n<offset+length]
    assert len(rows)==length
    raw={k:np.stack([s[k] for s in rows])[None].repeat(2,0) for k in rows[0]}
    raw['consec']=np.zeros((2,length),np.int32)
    return attach_teacher(raw,b['catalog'],mode='train')


def config_for(b):
    c=copy.deepcopy(b['config']);c['batch_length']=64;c['report_length']=16
    return c


def on_device(agent,batch,seed=17):
    from embodied.jax import internal
    data=internal.device_put(copy.deepcopy(batch),agent.train_sharded)
    return {**data,'seed':agent._seeds(seed,agent.train_mirrored)}


@pytest.mark.parametrize('kind',['direct_init','direct_full','after_bc_s3a'])
@pytest.mark.parametrize('source_bundle',[True],indirect=True)
def test_lambda_zero_real_agent_equivalence_each_source(source_bundle,tmp_path,kind):
    b=source_bundle;cfg=config_for(b)
    if kind=='direct_init':
        cfg['agent']['avail_actor_enabled']=True  # Controlled READY equivalence fixture.
    kwargs=dict(catalog=b['catalog'])
    if kind=='direct_init':kwargs.update(seed=0,config=cfg)
    if kind=='direct_full':kwargs['path']=b['base']
    if kind=='after_bc_s3a':kwargs.update(path=b['bc'],base_checkpoint=b['base'])
    spec=inspect_source(kind,**kwargs)
    original=make_agent(cfg,tmp_path/'original')
    concurrent=make_agent(cfg,tmp_path/'s3c',agent_cls=S3CAgent,bc_lambda=0.)
    apply_and_audit_transfer(original,spec);apply_and_audit_transfer(concurrent,spec)
    assert original.get_avail_actor_enabled() and concurrent.get_avail_actor_enabled()
    assert parameter_digests(original.params)==parameter_digests(concurrent.params)
    assert original.model.scales==concurrent.model.scales
    oc,cc=original.init_train(2),concurrent.init_train(2)
    batch=batch_for(b)
    for step in range(2):
        if step:batch['consec'][:]=1
        plain={k:v for k,v in batch.items() if not k.startswith('teacher_')}
        oc,oo,om=original.train(oc,on_device(original,plain,step))
        cc,co,cm=concurrent.train(cc,on_device(concurrent,batch,step))
        assert parameter_digests(original.params)==parameter_digests(concurrent.params)
        for x,y in zip(jax.tree.leaves(oc),jax.tree.leaves(cc)):np.testing.assert_array_equal(x,y)
        assert oo.keys()==co.keys()
        for x,y in zip(jax.tree.leaves(oo),jax.tree.leaves(co)):np.testing.assert_array_equal(x,y)
        assert original.save_exact()['counters']==concurrent.save_exact()['counters']
        assert float(cm['opt/updates'])==float(om['opt/updates'])
    assert concurrent.model.loss.__func__ is Agent.loss
    assert concurrent.model.report.__func__ is Agent.report
    assert concurrent.model.policy.__func__ is Agent.policy


def test_posterior_bc_gradient_gate_masks_and_real_joint_update(source_bundle,tmp_path):
    b=source_bundle;cfg=config_for(b)
    agent=make_agent(cfg,tmp_path/'s3c',agent_cls=S3CAgent,bc_lambda=1.)
    spec=inspect_source('after_bc_s3a',catalog=b['catalog'],path=b['bc'],base_checkpoint=b['base'])
    apply_and_audit_transfer(agent,spec)
    model=agent.model;batch=batch_for(b);carry=agent.init_train(2)
    assert batch['teacher_valid'][:,1:].any() and batch['teacher_action'][:,1:].any()
    def only_bc(c,data):
        c,obs,prevact,ids=model._apply_replay_context(c,data)
        _,(_,_,outs,_)=Agent.loss(model,c,obs,prevact,False)
        loss,_=model.bc_loss(outs['repfeat'],obs['action_mask'],data['teacher_action'][:,1:],data['teacher_valid'][:,1:])
        return loss*model.avail_actor_gate.read()
    grad=jax.jit(nj.pure(lambda c,d:nj.grad(only_bc,model.modules)(c,d)),
                 static_argnames=('create','modify'))
    data={k:v for k,v in on_device(agent,batch).items() if k!='seed'}
    for gate in (False,True):
        agent.set_avail_actor_enabled(gate)  # explicit test fixture only.
        _,(loss,params,grads)=grad(agent.params,carry,data,seed=7,create=False,modify=True)
        assert all(np.count_nonzero(np.asarray(v))==0 for k,v in grads.items() if not k.startswith('pol/'))
        if gate:assert any(np.count_nonzero(np.asarray(v)) for k,v in grads.items() if k.startswith('pol/'))
        else:assert all(np.count_nonzero(np.asarray(v))==0 for v in grads.values())
    before=parameter_digests(agent.params)
    old_opt_step=int(agent.params['opt/step/value'])
    for seed in (0,1):carry,outs,metrics=agent.train(carry,on_device(agent,batch,seed))
    assert metrics['bc/active']==1 and np.isfinite(metrics['bc/nll'])
    assert int(agent.n_updates)==spec['origin']['step']+2
    assert int(agent.params['opt/step/value'])==old_opt_step+2
    assert any(parameter_digests(agent.params)[k]!=before[k] for k in before if k.startswith('pol/'))
    assert any(parameter_digests(agent.params)[k]!=before[k] for k in before if k.startswith(('enc/','dyn/')))
    empty=copy.deepcopy(batch);empty['teacher_valid'][:]=False;empty['teacher_action'][:]=1
    carry,_,zero=agent.train(carry,on_device(agent,empty,2))
    assert zero['bc/nll']==0 and zero['bc/active']==0 and np.isfinite(zero['bc/total_loss'])
    # Validate effective IDs/prevact in both carry branches against original path.
    pure=nj.pure(model._apply_replay_context)
    for consec in (0,1):
        test=copy.deepcopy(batch);test['consec'][:]=consec
        td={k:v for k,v in on_device(agent,test).items() if k!='seed'}
        _,(_,obs,prevact,ids)=pure(agent.params,carry,td,seed=3,create=False,modify=True)
        np.testing.assert_array_equal(ids,test['stepid'][:,1:])
        np.testing.assert_array_equal(prevact['action'][:,1:],test['action'][:,1:-1])
    # Teacher values are training side inputs; original inference rejects them.
    obs={k:np.asarray(batch[k][:,0]) for k in agent.obs_space}
    with pytest.raises(AssertionError):
        agent.policy(agent.init_policy(2),{**obs,'teacher_action':np.ones(2,np.int32)},mode='eval')
    for shift in (-1,1):
        bad=copy.deepcopy(batch)
        bad['stepid']=np.roll(bad['stepid'],shift,axis=1)
        bad={k:v for k,v in bad.items() if not k.startswith('teacher_')}
        with pytest.raises(ValueError,match='MISMATCH'):
            attach_teacher(bad,b['catalog'],mode='train')


def test_masked_nll_uses_actual_distribution_and_safe_invalid_rows(source_bundle,tmp_path):
    b=source_bundle;a=make_agent(config_for(b),tmp_path/'s3c',agent_cls=S3CAgent,bc_lambda=1.)
    model=a.model
    feat=dict(deter=jnp.ones((1,2,32)),stoch=jnp.ones((1,2,2,4)))
    mask=jnp.ones((1,2,2));y=jnp.array([[1,0]],jnp.int32);valid=jnp.ones((1,2),bool)
    # Original categorical Head emits logits; original _masked_policy applies
    # unimix once. Set its existing final Linear so the actual P(ADD) is 0.3.
    u=float(model.config.policy.unimix);raw=(.3-.5*u)/(1-u)
    kernel=next(k for k in a.params if k.startswith('pol/') and k.endswith('/logits/kernel'))
    bias=next(k for k in a.params if k.startswith('pol/') and k.endswith('/logits/bias'))
    a.params[kernel]=jax.device_put(np.zeros_like(a.params[kernel]),a.params[kernel].sharding)
    a.params[bias]=jax.device_put(np.log([1-raw,raw]).astype(np.float32),a.params[bias].sharding)
    def compute():
        d=model._masked_policy(model.pol(jax.lax.stop_gradient(model.feat2tensor(feat)),2),mask,False)[model.action_key]
        return model.bc_loss(feat,mask,y,valid)[0],-d.logp(y).mean(),jax.nn.softmax(d.logits,-1)
    _,(loss,expected,prob)=nj.pure(compute)(a.params,seed=0,create=False,modify=False)
    np.testing.assert_allclose(loss,expected,rtol=1e-6)
    np.testing.assert_allclose(prob[...,1],.3,rtol=1e-6)
    np.testing.assert_allclose(loss,-np.log(np.array([prob[0,0,1],prob[0,1,0]])).mean(),rtol=1e-6)
    # Empty invalid rows may carry masked-out ADD; safe sentinel is applied first.
    _,(loss,mets)=nj.pure(lambda:model.bc_loss(feat,jnp.zeros_like(mask),y,jnp.zeros_like(valid)))(a.params,seed=0,create=False,modify=False)
    assert loss==0 and mets['bc/valid_count']==0
    for v,expected in ((jnp.array([[True,False]]),-np.log(.3)),(jnp.array([[False,True]]),-np.log(.7))):
        _,(value,_)=nj.pure(lambda:model.bc_loss(feat,mask,y,v))(a.params,seed=0,create=False,modify=False)
        np.testing.assert_allclose(value,expected,rtol=1e-6)
    pure=nj.pure(lambda:nj.grad(lambda:model.bc_loss(feat,jnp.zeros_like(mask),y,jnp.zeros_like(valid))[0],model.pol)())
    _,(_,_,grads)=pure(a.params,seed=0,create=False,modify=True)
    assert all(np.count_nonzero(np.asarray(g))==0 for g in grads.values())
