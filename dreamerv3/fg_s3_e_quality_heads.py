"""F2 execution permissions; the immutable V2 course remains the upper bound."""
from __future__ import annotations
import math
from .fg_s3_e_quality_course_contract import ALL_GROUPS, expected_stage_contract

PROTOCOL='FG_S3_E_QUALITY_HEADS_V1'
STAGE='QUALITY_HEADS'
SAMPLING_STAGE='HEADS_VALID'
GROUPS=('quality_event/','quality_touch/','quality_adverse/','quality_score/')
TARGETS=('event','touch','adverse','score')
LABELS=tuple('quality_'+n+s for n in TARGETS for s in ('','_valid'))

def execution_contract():
    upper=expected_stage_contract(STAGE)['parameter_group_policy']
    assert all(upper[p]=='TRAIN_WITH_VALID_LABEL' for p in GROUPS)
    return dict(protocol=PROTOCOL,stage=STAGE,effective_trainable_groups=list(GROUPS),
        parameter_group_policy={p:'TRAIN_WITH_VALID_LABEL' if p in GROUPS else 'FROZEN' for p in ALL_GROUPS},
        prediction_admitted=False,q_admitted=False,decision_admitted=False)

def dimensions(target,burnin):
    from .fg_s3_e_quality_foundation import dimensions as original
    return original(target,burnin)

def model_extensions(target,burnin,weights=None):
    from .fg_s3_quality_v2 import quality_extensions
    from gym_trading_env.research.fg_s3_e_quality_supervision import PROTOCOL as SUPERVISION
    return quality_extensions(dict(quality_stage=STAGE,quality_f2_protocol=PROTOCOL,
        quality_f2_target_steps=target,quality_f2_burnin_steps=burnin,
        quality_supervision_protocol=SUPERVISION,**(weights or {})))

def quality_configuration(settings,defaults):
    from gym_trading_env.research.fg_s3_e_quality_supervision import PROTOCOL as SUPERVISION
    if (settings.get('quality_f2_protocol')!=PROTOCOL or settings.get('quality_stage')!=STAGE
            or settings.get('quality_supervision_protocol')!=SUPERVISION):
        raise ValueError('F2_EXECUTION_PROTOCOL_REQUIRED')
    dimensions(settings.get('quality_f2_target_steps'),settings.get('quality_f2_burnin_steps'))
    allowed=('quality_outcome_lambda','quality_score_lambda','quality_mono_lambda')
    off=('quality_prediction_admitted','quality_decision_admitted','quality_q_admitted',
         'quality_fuse_outcome','quality_fuse_score','quality_weighting')
    forbidden=tuple(k for k in defaults if k.endswith('_lambda') and k not in allowed)+('quality_repr_grad_scale',)
    if any(settings.get(k,False) is not False for k in off):raise ValueError('F2_DECISION_FORBIDDEN')
    if any(type(settings.get(k,0.)) not in (int,float) or settings.get(k,0.)!=0. for k in forbidden):
        raise ValueError('F2_Q_SOFT_ACTOR_FORBIDDEN')
    known=set(defaults)|set(off)|{'mode','s3_v2_mode','quality_variant','quality_stage','quality_f2_protocol',
        'quality_f2_target_steps','quality_f2_burnin_steps','quality_supervision_protocol',
        'quality_gradient_every_updates'}
    if set(settings)-known or settings.get('quality_variant','FULL')!='FULL' or settings.get('quality_gradient_every_updates',0)!=0:
        raise ValueError('F2_UNKNOWN_CONFIGURATION')
    result={**defaults,**settings,**dict.fromkeys(off,False),**dict.fromkeys(forbidden,0.)}
    for k in allowed:
        v=result[k]
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('F2_INVALID_WEIGHT')
    if not any(result[k]>0 for k in allowed):raise ValueError('F2_NO_ENABLED_OBJECTIVE')
    result.update(s3_v2_mode='quality_v2',quality_variant='FULL',quality_gradient_every_updates=0)
    result.pop('mode',None)
    return result

def validate_model(config):
    if (config.get('quality_f2_protocol')!=PROTOCOL or config.quality_stage!=STAGE
            or config.get('quality_f1_protocol') is not None or config.enc.typ!='e_multiscale_1m15m'
            or config.s3_v2_mode!='quality_v2' or config.bc_lambda!=0
            or config.replay_context!=1 or config.replay_context_mode!='recompute_prefix'
            or config.batch_length!=dimensions(config.quality_f2_target_steps,config.quality_f2_burnin_steps)):
        raise ValueError('F2_MODEL_STAGE_OR_TEMPORAL_CONTRACT_MISMATCH')

def register_state_schema(model,params):
    from .fg_s3_e_quality_foundation import state_owners
    owner=state_owners(model,params,GROUPS)
    model.f2_state_schema=dict(model.f1_state_schema)
    model.f2_state_owner=owner
    return owner

def features(model,data,*,diagnostics=False):
    import jax
    from .fg_s3_e_quality_foundation import causal_context
    carry,obs,prevact,ids=causal_context(model,data,model.config.quality_f2_burnin_steps,model.config.quality_f2_target_steps)
    _,(carry,_,out,metrics)=model.wm_only(carry,obs,prevact,False,diagnostics=diagnostics)
    result=(jax.lax.stop_gradient(model.feat2tensor(out['repfeat'])),carry,ids)
    return (*result,dict(encoder_tokens=metrics['f1_raw/encoder_tokens'],posterior_logits=out['repfeat']['logit'])) if diagnostics else result

def active_heads(labels,config):
    import jax.numpy as jnp
    v=labels['quality_event_valid'][...,:5]
    pairs=(jnp.arange(5)[:,None]<jnp.arange(5)[None,:]) & v[..., :,None] & v[...,None,:]
    return {'quality_event/':((config['quality_outcome_lambda']>0)&labels['quality_event_valid'].any()) | ((config['quality_mono_lambda']>0)&pairs.any()),
        'quality_touch/':(config['quality_outcome_lambda']>0)&labels['quality_touch_valid'].any(),
        'quality_adverse/':(config['quality_outcome_lambda']>0)&labels['quality_adverse_valid'].any(),
        'quality_score/':(config['quality_score_lambda']>0)&labels['quality_score_valid'].any()}

def objective(model,feat,labels):
    import jax.numpy as jnp
    from .fg_s3_quality_v2 import prediction_terms
    for module in model.modules:_=module.values
    pred={name:getattr(model,'quality_'+name)(feat,2).pred().astype(jnp.float32) for name in TARGETS}
    value,metrics=prediction_terms(pred,labels,model.quality)
    return value,(pred,metrics)

def train(model,data):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from functools import partial
    from .fg_s3_v2_agent import masked_joint_optimizer
    actual={k:(tuple(v.shape),str(v.dtype)) for k,v in nj.context().items()}
    if actual!=getattr(model,'f2_state_schema',None):raise ValueError('F2_UNKNOWN_OR_CHANGED_STATE_SCHEMA')
    before=dict(nj.context())
    feat,carry,ids=features(model,data)
    start=1+model.config.quality_f2_burnin_steps
    labels={key:data[key][:,start:] for key in LABELS}
    active=active_heads(labels,model.quality);any_active=jnp.stack(list(active.values())).any()
    with masked_joint_optimizer(model.opt,False,False,active,allowed_prefixes=GROUPS,stage=STAGE):
        metrics,(pred,extra)=model.opt(partial(objective,model),feat,labels,has_aux=True)
    finite=jnp.isfinite(metrics['opt/loss']) & jnp.isfinite(metrics['opt/grad_norm'])
    finite &= jnp.stack([jnp.isfinite(v).all() for v in nj.context().values()]).all()
    for key,old in before.items():
        owner=model.f2_state_owner[key]
        enabled=any_active if owner=='CLOCK' else active.get(owner,jnp.bool_(False))
        nj.context()[key]=jnp.where(finite & enabled,nj.context()[key],old)
    metrics.update(extra)
    metrics['opt/updates']=model.opt.step.read()
    metrics.update({'f2/finite':finite,'f2/updated':finite & any_active,
        'f2/skip_no_valid':~jnp.stack([labels['quality_'+n+'_valid'].any() for n in TARGETS]).any(),
        'f2/target_positions':jnp.int32(feat.shape[0]*feat.shape[1]),
        'f2/burnin_positions':jnp.int32(feat.shape[0]*(start-1)),
        'f2/target_market_minutes':data['market_advance'][:,start:].sum(),
        'f2/burnin_market_minutes':data['market_advance'][:,1:start].sum(),
        'f2/optimizer_updates':model.opt.step.read()})
    metrics.update({'f2/'+p[:-1]+'_active':v for p,v in active.items()})
    return (*carry,{k:data[k][:,-1] for k in model.act_space}),{'effective_stepid':ids},metrics


def read_only_forward(agent,data):
    import numpy as np
    import jax,jax.numpy as jnp,ninjax as nj
    from .fg_s3_e_quality_heads import features,objective,LABELS
    if not hasattr(agent,'_f2_read_forward'):
        def fn(raw):
            feat,_,ids,diagnostics=features(agent.model,raw,diagnostics=True)
            start=1+agent.config.quality_f2_burnin_steps
            value,(pred,mets)=objective(agent.model,feat,{k:raw[k][:,start:] for k in LABELS})
            return dict(loss=value,features=feat,stepid=ids,predictions=pred,metrics=mets,**diagnostics)
        agent._f2_read_forward=jax.jit(nj.pure(fn),static_argnames=('create','modify'))
    _,out=agent._f2_read_forward(agent.params,{k:jnp.asarray(v) for k,v in data.items()},seed=202,create=False,modify=True)
    return jax.tree.map(np.asarray,out)
