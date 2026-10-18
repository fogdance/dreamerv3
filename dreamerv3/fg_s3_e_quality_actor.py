"""F3 permissions over the original complete E+Quality model."""
from __future__ import annotations
import math
from .fg_s3_e_quality_course_contract import ALL_GROUPS, expected_stage_contract

PROTOCOL='FG_S3_E_QUALITY_ACTOR_SUPERVISED_V1'
STAGE='ACTOR_SUPERVISED'
SAMPLING_STAGE='ACTOR_STRICT'
GROUPS=('quality_actor/',)
LABELS=('teacher_action','teacher_valid','quality_owner','quality_score',
        'quality_score_valid','quality_outcome_valid','f3_natural')
LOSS_DEFAULTS=dict(natural=1.,balanced=0.,strict_pair=0.,quality_pair=0.,pair_tau=1.)


def validate_loss(raw):
    if type(raw) is not dict or set(raw)-LOSS_DEFAULTS.keys():raise ValueError('F3_LOSS_CONFIGURATION')
    result={**LOSS_DEFAULTS,**raw}
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in result.values()):
        raise ValueError('F3_LOSS_WEIGHT_FINITE_NONNEGATIVE_REQUIRED')
    if result['natural']<=0 or result['pair_tau']<=0:raise ValueError('F3_NATURAL_ANCHOR_AND_TEMPERATURE_REQUIRED')
    return result


def execution_contract():
    upper=expected_stage_contract(STAGE)['parameter_group_policy']
    if upper['quality_actor/']!='TRAIN':raise ValueError('F3_COURSE_UPPER_BOUND')
    return dict(protocol=PROTOCOL,stage=STAGE,effective_trainable_groups=list(GROUPS),
        parameter_group_policy={p:'TRAIN_WITH_VALID_L2' if p in GROUPS else 'FROZEN' for p in ALL_GROUPS},
        prediction_admitted=False,q_admitted=False,decision_admitted=False)


def dimensions(target,burnin):
    from .fg_s3_e_quality_foundation import dimensions as original
    return original(target,burnin)


def model_extensions(target,burnin,loss=None):
    from .fg_s3_quality_v2 import quality_extensions
    from gym_trading_env.research.fg_s3_e_quality_supervision import PROTOCOL as SUPERVISION
    loss=validate_loss(loss or {})
    return quality_extensions(dict(quality_stage=STAGE,quality_f3_protocol=PROTOCOL,
        quality_f3_target_steps=target,quality_f3_burnin_steps=burnin,
        quality_f3_natural_lambda=loss['natural'],quality_f3_balanced_lambda=loss['balanced'],
        quality_strict_pair_lambda=loss['strict_pair'],quality_pair_lambda=loss['quality_pair'],
        quality_pair_tau=loss['pair_tau'],quality_supervision_protocol=SUPERVISION))


def quality_configuration(settings,defaults):
    from gym_trading_env.research.fg_s3_e_quality_supervision import PROTOCOL as SUPERVISION
    if (settings.get('quality_f3_protocol')!=PROTOCOL or settings.get('quality_stage')!=STAGE
            or settings.get('quality_supervision_protocol')!=SUPERVISION):raise ValueError('F3_EXECUTION_PROTOCOL_REQUIRED')
    dimensions(settings.get('quality_f3_target_steps'),settings.get('quality_f3_burnin_steps'))
    off=('quality_prediction_admitted','quality_decision_admitted','quality_q_admitted',
         'quality_fuse_outcome','quality_fuse_score','quality_weighting')
    allowed=('quality_strict_pair_lambda','quality_pair_lambda')
    forbidden=tuple(k for k in defaults if k.endswith('_lambda') and k not in allowed)+('quality_repr_grad_scale',)
    if any(settings.get(k,False) is not False for k in off):raise ValueError('F3_DECISION_FUSION_FORBIDDEN')
    if any(type(settings.get(k,0.)) not in (int,float) or settings.get(k,0.)!=0 for k in forbidden):
        raise ValueError('F3_PREDICTION_Q_SOFT_RL_FORBIDDEN')
    known=set(defaults)|set(off)|{'mode','s3_v2_mode','quality_variant','quality_stage','quality_f3_protocol',
        'quality_f3_target_steps','quality_f3_burnin_steps','quality_f3_natural_lambda','quality_f3_balanced_lambda',
        'quality_supervision_protocol','quality_gradient_every_updates'}
    if set(settings)-known or settings.get('quality_variant','FULL')!='FULL' or settings.get('quality_gradient_every_updates',0)!=0:
        raise ValueError('F3_UNKNOWN_CONFIGURATION')
    loss=validate_loss(dict(natural=settings.get('quality_f3_natural_lambda',1.),
        balanced=settings.get('quality_f3_balanced_lambda',0.),strict_pair=settings.get('quality_strict_pair_lambda',0.),
        quality_pair=settings.get('quality_pair_lambda',0.),pair_tau=settings.get('quality_pair_tau',1.)))
    return {**defaults,**settings,**dict.fromkeys(off,False),**dict.fromkeys(forbidden,0.),
        's3_v2_mode':'quality_v2','quality_variant':'FULL','quality_gradient_every_updates':0,
        'quality_f3_natural_lambda':loss['natural'],'quality_f3_balanced_lambda':loss['balanced'],
        'quality_strict_pair_lambda':loss['strict_pair'],'quality_pair_lambda':loss['quality_pair']}


def validate_model(config):
    if (config.get('quality_f3_protocol')!=PROTOCOL or config.quality_stage!=STAGE
            or config.enc.typ!='e_multiscale_1m15m' or config.s3_v2_mode!='quality_v2' or config.bc_lambda!=0
            or config.replay_context!=1 or config.replay_context_mode!='recompute_prefix'
            or config.batch_length!=dimensions(config.quality_f3_target_steps,config.quality_f3_burnin_steps)):
        raise ValueError('F3_MODEL_STAGE_OR_TEMPORAL_CONTRACT_MISMATCH')


def register_state_schema(model,params):
    from .fg_s3_e_quality_foundation import state_owners
    model.f3_state_owner=state_owners(model,params,GROUPS)
    model.f3_state_schema=dict(model.f1_state_schema)
    return model.f3_state_owner


def features(model,data,*,diagnostics=False):
    from .fg_s3_e_quality_foundation import frozen_features
    return frozen_features(model,data,model.config.quality_f3_burnin_steps,model.config.quality_f3_target_steps,diagnostics=diagnostics)


def distribution(model,feat,mask,bdims):
    """One original quality_actor, one native mask/unimix, no predictions."""
    import jax
    import embodied
    raw=model.quality_actor(jax.lax.stop_gradient(feat),bdims)[model.action_key].logits
    policy=model._masked_policy({model.action_key:embodied.jax.outs.Categorical(raw)},mask,False)
    return policy,raw


def actor_objective(dist,labels,mask,first,last,terminal,config):
    """L2 truth; each optional component has its own actual denominator."""
    import jax
    import jax.numpy as jnp
    from .fg_s3_quality_v2 import mean,pair_identity
    y=labels['teacher_action']
    valid=labels['teacher_valid'] & ((y==0)|(y==1)) & (mask==1).all(-1) & ~first & ~last & ~terminal
    natural=labels['f3_natural'];auxiliary=jnp.where((~natural).any(),~natural,jnp.ones_like(natural))
    nv=valid & natural;av=valid & auxiliary
    safe_y=jnp.where(valid,y,0)
    nll=-dist.logp(safe_y);z=jnp.where(valid,dist.logits[...,1]-dist.logits[...,0],0.)
    pos=av & (y==1);neg=av & (y==0)
    nat=mean(nll,nv);bal=jnp.where(pos.any() & neg.any(),.5*(mean(nll,pos)+mean(nll,neg)),mean(nll,av))
    identity=pair_identity(labels,first)
    strict=identity & pos[:,:,None] & neg[:,None,:]
    gap=z[:,:,None]-z[:,None,:]
    pairloss=jax.nn.softplus(-gap/config['quality_pair_tau'])
    strict_loss=mean(pairloss,strict)
    known=labels['quality_score_valid'] & labels['quality_outcome_valid'] & av
    score=jnp.where(known,labels['quality_score'],0.)
    candidates=identity & known[:,:,None] & known[:,None,:] & (score[:,:,None]-score[:,None,:]>=15)
    consistent=candidates & strict
    same=candidates & (y[:,:,None]==y[:,None,:])
    reverse=candidates & (y==0)[:,:,None] & (y==1)[:,None,:]
    quality_loss=mean(pairloss,consistent)
    loss=(config['quality_f3_natural_lambda']*nat+config['quality_f3_balanced_lambda']*bal+
          config['quality_strict_pair_lambda']*strict_loss+config['quality_pair_lambda']*quality_loss)
    active=(config['quality_f3_natural_lambda']>0)&nv.any()
    active|=(config['quality_f3_balanced_lambda']>0)&av.any()
    active|=(config['quality_strict_pair_lambda']>0)&strict.any()
    active|=(config['quality_pair_lambda']>0)&consistent.any()
    padd=dist.prob(jnp.ones_like(safe_y));pred=dist.pred()
    m={'f3/active':active,'f3/valid_count':valid.sum(),'f3/natural_count':nv.sum(),
       'f3/natural_add':(nv & (y==1)).sum(),'f3/natural_wait':(nv & (y==0)).sum(),
       'f3/auxiliary_count':(valid & ~natural).sum(),
       'f3/auxiliary_add':(valid & ~natural & (y==1)).sum(),'f3/auxiliary_wait':(valid & ~natural & (y==0)).sum(),
       'f3/balanced_count':av.sum(),
       'f3/natural_loss':nat,'f3/balanced_loss':bal,'f3/strict_pair_loss':strict_loss,
       'f3/quality_pair_loss':quality_loss,'f3/strict_pair_count':strict.sum(),
       'f3/quality_pair_count':consistent.sum(),'f3/quality_pair_candidates':candidates.sum(),
       'f3/quality_pair_conflicts':(candidates & ~strict).sum(),
       'f3/quality_pair_same_action_filtered':same.sum(),'f3/quality_pair_reversed_filtered':reverse.sum(),
       'f3/no_pair':~strict.any(),'f3/accuracy':mean((pred==y).astype(jnp.float32),nv),
       'f3/add_accuracy':mean((pred==y).astype(jnp.float32),nv & (y==1)),
       'f3/wait_accuracy':mean((pred==y).astype(jnp.float32),nv & (y==0)),
       'f3/p_add_mean':mean(padd,nv),'f3/brier':mean((padd-safe_y)**2,nv),
       'f3/strict_pair_accuracy':mean((gap>0).astype(jnp.float32),strict),
       'f3/soft_valid_count':jnp.int32(0),'f3/q_valid_count':jnp.int32(0)}
    return loss,m


def objective(model,feat,labels,mask,first,last,terminal):
    for module in model.modules:_=module.values
    policy,raw=distribution(model,feat,mask,2)
    loss,metrics=actor_objective(policy[model.action_key],labels,mask,first,last,terminal,model.quality)
    return loss,(raw,metrics)


def train(model,data):
    import jax.numpy as jnp
    import ninjax as nj
    from functools import partial
    from .fg_s3_v2_agent import masked_joint_optimizer
    actual={k:(tuple(v.shape),str(v.dtype)) for k,v in nj.context().items()}
    if actual!=getattr(model,'f3_state_schema',None):raise ValueError('F3_UNKNOWN_OR_CHANGED_STATE_SCHEMA')
    before=dict(nj.context());feat,carry,ids=features(model,data)
    start=1+model.config.quality_f3_burnin_steps
    labels={key:data[key][:,start:] for key in LABELS}
    args=[data[k][:,start:] for k in ('action_mask','is_first','is_last','is_terminal')]
    # Read-only activity uses the same objective masks, never an executed action.
    _,(_,preview)=objective(model,feat,labels,*args)
    active=preview['f3/active']
    with masked_joint_optimizer(model.opt,False,False,{GROUPS[0]:active},allowed_prefixes=GROUPS,stage=STAGE):
        metrics,(_,extra)=model.opt(partial(objective,model),feat,labels,*args,has_aux=True)
    finite=jnp.isfinite(metrics['opt/loss']) & jnp.isfinite(metrics['opt/grad_norm'])
    finite &= jnp.stack([jnp.isfinite(v).all() for v in nj.context().values()]).all()
    for key,old in before.items():
        enabled=active if model.f3_state_owner[key] in (GROUPS[0],'CLOCK') else jnp.bool_(False)
        nj.context()[key]=jnp.where(finite & enabled,nj.context()[key],old)
    metrics.update(extra);metrics['opt/updates']=model.opt.step.read()
    metrics.update({'f3/finite':finite,'f3/updated':finite & active,'f3/skip_no_valid':~extra['f3/valid_count'].astype(bool),
        'f3/target_positions':jnp.int32(feat.shape[0]*feat.shape[1]),'f3/burnin_positions':jnp.int32(feat.shape[0]*(start-1)),
        'f3/target_market_minutes':data['market_advance'][:,start:].sum(),
        'f3/burnin_market_minutes':data['market_advance'][:,1:start].sum(),'f3/optimizer_updates':model.opt.step.read()})
    return (*carry,{k:data[k][:,-1] for k in model.act_space}),{'effective_stepid':ids},metrics


def read_only_forward(agent,data,*,seed_counter=202):
    import numpy as np
    import jax,jax.numpy as jnp,ninjax as nj
    if not hasattr(agent,'_f3_read_forward'):
        def fn(raw):
            feat,_,ids,diagnostics=features(agent.model,raw,diagnostics=True)
            start=1+agent.config.quality_f3_burnin_steps
            labels={k:raw[k][:,start:] for k in LABELS}
            args=[raw[k][:,start:] for k in ('action_mask','is_first','is_last','is_terminal')]
            value,(logits,mets)=objective(agent.model,feat,labels,*args)
            policy,_=distribution(agent.model,feat,args[0],2)
            return dict(loss=value,features=feat,stepid=ids,logits=logits,
                p_add=policy[agent.model.action_key].prob(jnp.ones(logits.shape[:-1],jnp.int32)),metrics=mets,**diagnostics)
        agent._f3_read_forward=jax.jit(nj.pure(fn),static_argnames=('create','modify'))
    # Match the original Agent.train seed protocol, rather than treating the
    # counter as a raw Ninjax key (which changes stochastic RSSM history).
    seed=agent._seeds(seed_counter,agent.train_mirrored)
    _,out=agent._f3_read_forward(agent.params,{k:jnp.asarray(v) for k,v in data.items()},seed=seed,create=False,modify=True)
    return jax.tree.map(np.asarray,out)
