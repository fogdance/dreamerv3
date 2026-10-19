"""Versioned online execution over the existing E+Quality course/model."""
from __future__ import annotations

import copy
import math

PROTOCOL = 'FG_S3_E_QUALITY_F4_EXECUTION_V1'
MODES = ('SHADOW', 'ONLINE_SUPERVISED', 'ONLINE_RL_CONTROLLED')
RL_STAGE = 'ONLINE_RL_CONTROLLED'
RL_GROUPS = ('quality_actor/', 'val/')


def execution_contract(mode):
    from .fg_s3_e_quality_course_contract import expected_stage_contract, ALL_GROUPS
    if mode not in MODES:
        raise ValueError('ONLINE_EXECUTION_MODE_INVALID')
    groups = (() if mode == 'SHADOW' else ('quality_actor/',)
              if mode == 'ONLINE_SUPERVISED' else ('quality_actor/', 'val/', 'slowval/'))
    upper = expected_stage_contract('ONLINE_ADAPT')['parameter_group_policy']
    if any(upper[g] == 'FROZEN' for g in groups):
        raise ValueError('ONLINE_PERMISSION_EXCEEDS_COURSE')
    return dict(protocol=PROTOCOL, source_stage='ACTOR_SUPERVISED', execution_mode=mode,
        execution_kind='F3_ONLINE_CONTINUATION' if mode != MODES[2] else 'CONTROLLED_F4_EXECUTION',
        effective_trainable_groups=list(groups),
        parameter_group_policy={g: 'CONDITIONAL' if g in groups else 'FROZEN' for g in ALL_GROUPS},
        teacher_controls_action=False, prediction_fusion=False, q_fusion=False,
        formal_online_adapt_admitted=False)


def validate_execution(raw):
    from gym_trading_env.research.fg_s3_e_quality_course_preflight import _local_path
    required = {'protocol','source_stage','execution_mode','fold','source_run','source_checkpoint',
        'run_dir','model_seed','env_seed','replay_seed','env_steps','collect_every',
        'updates_per_collect','max_consumed_batches','save_every','replay_capacity',
        'exploration_rate','history_ratio'}
    if type(raw) is not dict or set(raw) != required:
        raise ValueError('ONLINE_EXPLICIT_EXECUTION_FIELDS_REQUIRED')
    if raw['protocol'] != PROTOCOL or raw['source_stage'] != 'ACTOR_SUPERVISED' or raw['fold'] != 'TRAIN':
        raise ValueError('ONLINE_F3_TRAIN_SOURCE_REQUIRED')
    execution_contract(raw['execution_mode'])
    result = copy.deepcopy(raw)
    for k in ('model_seed','env_seed','replay_seed','updates_per_collect','max_consumed_batches'):
        if type(raw[k]) is not int or raw[k] < 0:
            raise ValueError('ONLINE_NONNEGATIVE_INTEGER_REQUIRED: '+k)
    for k in ('env_steps','collect_every','save_every','replay_capacity'):
        if type(raw[k]) is not int or raw[k] < 1:
            raise ValueError('ONLINE_POSITIVE_INTEGER_REQUIRED: '+k)
    for k in ('exploration_rate','history_ratio'):
        if type(raw[k]) not in (int,float) or not math.isfinite(raw[k]) or not 0 <= raw[k] <= 1:
            raise ValueError('ONLINE_RATIO_INVALID: '+k)
    shadow = raw['execution_mode'] == 'SHADOW'
    if shadow and (raw['updates_per_collect'] or raw['max_consumed_batches']):
        raise ValueError('SHADOW_REQUIRES_ZERO_UPDATES')
    if not shadow and (not raw['updates_per_collect'] or not raw['max_consumed_batches']
                       or not 0 < raw['history_ratio'] < 1):
        raise ValueError('ONLINE_EXPLICIT_HISTORY_AND_ONLINE_BUDGET_REQUIRED')
    if raw['collect_every']>raw['env_steps'] or raw['save_every']%raw['collect_every']:
        raise ValueError('ONLINE_COLLECTION_AND_CHECKPOINT_BOUNDARIES_MUST_ALIGN')
    for k in ('source_run','source_checkpoint','run_dir'):
        result[k] = str(_local_path(raw[k]))
    from pathlib import Path
    src, node, out = (Path(result[k]) for k in ('source_run','source_checkpoint','run_dir'))
    if not node.is_relative_to(src/'ckpt') or out.is_relative_to(src) or src.is_relative_to(out):
        raise ValueError('ONLINE_SOURCE_OR_OUTPUT_PATH_INVALID')
    return result


def model_extensions(target,burnin,mode,loss=None):
    from .fg_s3_e_quality_actor import model_extensions as original
    execution_contract(mode)
    return {**original(target,burnin,loss), 'quality_online_protocol':PROTOCOL,'quality_online_mode':mode}


def validate_model_settings(settings):
    from .fg_s3_e_quality_actor import PROTOCOL as F3
    if settings.get('quality_online_protocol')!=PROTOCOL or settings.get('quality_f3_protocol')!=F3:
        raise ValueError('ONLINE_EXPLICIT_F3_MODEL_PROTOCOL_REQUIRED')
    execution_contract(settings.get('quality_online_mode'))


def train(model,data):
    mode=model.quality['quality_online_mode']
    if mode=='SHADOW':raise ValueError('SHADOW_OPTIMIZER_CALL_FORBIDDEN')
    if mode=='ONLINE_SUPERVISED':
        from .fg_s3_e_quality_actor import train as original
        return original(model,data)
    return train_rl(model,data)


def rl_objective(model,feat,labels,mask,first,last,terminal,dyn_entries,dyn_carry,repfeat,enabled):
    import jax.numpy as jnp
    import ninjax as nj
    from .fg_s3_e_quality_actor import objective
    bc,(_,metrics)=objective(model,feat,labels,mask,first,last,terminal)
    def on():
        losses,_,m=model.imagination_loss(dyn_entries,dyn_carry,repfeat,True)
        policy=losses['policy'].mean();value=losses['value'].mean()
        return policy*model.scales['policy']+value*model.scales['value'], dict(
            policy_loss=policy,value_loss=value,imag_invalid=m['avail/img_action_invalid_rate'],
            imag_fallback=m['avail/img_fallback_rate'],value_mean=m['val'])
    def off():
        z=jnp.float32(0)
        return z,dict(policy_loss=z,value_loss=z,imag_invalid=z,imag_fallback=z,value_mean=z)
    rl,extra=nj.cond(enabled,on,off)
    metrics.update({'f4/'+k:v for k,v in extra.items()})
    metrics['f4/rl_active']=enabled
    return bc+rl,(metrics,)


def train_rl(model,data):
    """One ORIGINAL optimizer invocation; original imagination, frozen WM."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from functools import partial
    from .fg_s3_e_quality_foundation import causal_context,state_owners
    from .fg_s3_e_quality_actor import LABELS,objective
    from .fg_s3_v2_agent import masked_joint_optimizer
    actual={k:(tuple(v.shape),str(v.dtype)) for k,v in nj.context().items()}
    if actual!=model.f3_state_schema:raise ValueError('ONLINE_UNKNOWN_STATE_SCHEMA')
    before=dict(nj.context())
    owner=state_owners(model,before,RL_GROUPS)
    k=model.config.quality_f3_burnin_steps;t=model.config.quality_f3_target_steps;start=1+k
    carry,obs,prevact,ids=causal_context(model,data,k,t)
    wm=model.world_model(carry,obs,prevact,False)
    dyn_carry,dyn_entries,repfeat=wm[1],wm[4],wm[7]
    dyn_carry,dyn_entries,repfeat=jax.tree.map(jax.lax.stop_gradient,(dyn_carry,dyn_entries,repfeat))
    feat=jax.lax.stop_gradient(model.feat2tensor(repfeat))
    labels={key:data[key][:,start:] for key in LABELS}
    args=[data[key][:,start:] for key in ('action_mask','is_first','is_last','is_terminal')]
    _,(_,preview)=objective(model,feat,labels,*args)
    enabled=model.avail_actor_gate.read()>0
    actor_active=preview['f3/active']|enabled
    with masked_joint_optimizer(model.opt,False,False,
            dict(zip(RL_GROUPS,(actor_active,enabled))),allowed_prefixes=RL_GROUPS,stage=RL_STAGE):
        metrics,(extra,)=model.opt(partial(rl_objective,model),feat,labels,*args,
            dyn_entries,dyn_carry,repfeat,enabled,has_aux=True)
    def slow_on():model.slowval.update();return jnp.int32(0)
    nj.cond(enabled,slow_on,lambda:jnp.int32(0))
    finite=jnp.isfinite(metrics['opt/loss']) & jnp.isfinite(metrics['opt/grad_norm'])
    finite &= jnp.stack([jnp.isfinite(v).all() for v in nj.context().values()]).all()
    for key,old in before.items():
        active=(actor_active if owner[key] in ('quality_actor/','CLOCK') else
                enabled if owner[key]=='val/' or key.startswith(
                    ('slowval/','slowval_count/','retnorm/','valnorm/','advnorm/')) else jnp.bool_(False))
        nj.context()[key]=jnp.where(finite & active,nj.context()[key],old)
    metrics.update(extra)
    metrics.update({'f3/finite':finite,'f3/updated':finite & actor_active,
        'f3/skip_no_valid':~actor_active,'opt/updates':model.opt.step.read(),
        'f3/optimizer_updates':model.opt.step.read(),
        'f3/target_positions':jnp.int32(feat.shape[0]*feat.shape[1]),
        'f3/burnin_positions':jnp.int32(feat.shape[0]*k),
        'f3/target_market_minutes':data['market_advance'][:,start:].sum(),
        'f3/burnin_market_minutes':data['market_advance'][:,1:start].sum()})
    final=(wm[0],wm[1],wm[2])
    return (*final,{key:data[key][:,-1] for key in model.act_space}),{'effective_stepid':ids},metrics
