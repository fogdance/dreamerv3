"""F1 execution contract over the original complete E+Quality model.

This contract selects a loss and a freeze policy, never source authority.
"""
from __future__ import annotations

from .fg_s3_e_quality_course_contract import WM_GROUPS, ALL_GROUPS, expected_stage_contract

PROTOCOL = 'FG_S3_E_QUALITY_FOUNDATION_WM_V1'
STAGE = 'FOUNDATION_WM'
SAMPLING_STAGE = 'WM_NATURAL'
FROZEN_GROUPS = tuple(p for p in ALL_GROUPS if p not in WM_GROUPS)


def register_state_schema(model, params):
    """Freeze the actually initialized schema, including original optax leaves."""
    parameter_layout(params)
    known=(*ALL_GROUPS,'avail_actor_gate/','retnorm/','valnorm/','advnorm/','slowval_count/')
    if any(not k.startswith(('opt/',*known)) for k in params):
        raise ValueError('F1_UNKNOWN_INITIAL_STATE_PREFIX')
    slot_parameters={k.replace('_','') for k in params if not k.startswith('opt/')}
    clocks = {'opt/step/value','opt/state/1/0','opt/state/2/0',
              'opt/state/'+str(4 if model.config.opt.wd else 3)+'/count'}
    if not clocks <= params.keys():
        raise ValueError('F1_UNRECOGNIZED_OPTIMIZER_CLOCK_SCHEMA')
    mutable = {}
    for key in params:
        if key.startswith('opt/'):
            if key in clocks:
                mutable[key] = True
            elif key.startswith(('opt/state/1/1/','opt/state/2/1/')):
                # Ninjax Tree sanitizes '_' in nested dict names. The six WM
                # prefixes have no underscores and remain unambiguous.
                leaf=key.split('/',4)[4]
                if leaf not in slot_parameters:
                    raise ValueError('F1_UNKNOWN_OPTIMIZER_PARAMETER_SLOT: '+key)
                mutable[key] = leaf.startswith(WM_GROUPS)
            else:
                raise ValueError('F1_UNKNOWN_OPTIMIZER_SLOT: '+key)
        else:
            mutable[key] = key.startswith(WM_GROUPS)
    model.f1_state_schema = {k:(tuple(v.shape),str(v.dtype)) for k,v in params.items()}
    model.f1_mutable_keys = mutable
    model.f1_shared_clocks = tuple(sorted(clocks))
    return dict(mutable=mutable,shared_clocks=sorted(clocks))


def state_owners(model,params,groups):
    """Shared complete native-state ownership for a stage execution subset."""
    register_state_schema(model,params)
    names={k.replace('_',''):k for k in params if not k.startswith('opt/')}
    if len(names)!=sum(not k.startswith('opt/') for k in params):raise ValueError('QUALITY_AMBIGUOUS_SLOT_KEY')
    owner={}
    for key in params:
        if key in model.f1_shared_clocks:owner[key]='CLOCK'
        else:
            parameter=names[key.split('/',4)[4]] if key.startswith(('opt/state/1/1/','opt/state/2/1/')) else key
            owner[key]=next((p for p in groups if parameter.startswith(p)),None)
    return owner


def check_state_schema(model, params):
    actual = {k:(tuple(v.shape),str(v.dtype)) for k,v in params.items()}
    if actual != getattr(model,'f1_state_schema',None):
        raise ValueError('F1_UNKNOWN_OR_CHANGED_STATE_SCHEMA')


def context(model, data):
    """One original prefix plus K detached original WM steps, never old carry."""
    return causal_context(model,data,model.config.quality_f1_burnin_steps,model.config.quality_f1_target_steps)


def causal_context(model,data,k,t):
    """Shared original WM causal history, parameterized only by window geometry."""
    import jax
    import jax.numpy as jnp
    if data['stepid'].shape[1:] != (1+k+t,20):
        raise ValueError('F1_WINDOW_LENGTH_OR_STEPID_MISMATCH')
    carry,obs,prevact,ids = model._apply_replay_context(model.init_train(len(data['is_first'])),data)
    prefix = lambda tree:jax.tree.map(lambda x:x[:,:k],tree)
    _,(carry,_,_,_) = model.wm_only(carry,prefix(obs),prefix(prevact),training=False)
    carry = jax.tree.map(jax.lax.stop_gradient,carry)
    target = lambda tree:jax.tree.map(lambda x:x[:,k:],tree)
    return carry,target(obs),target(prevact),ids[:,k:]


def loss(model, carry, obs, prevact, training, *, diagnostics=False):
    # Original Ninjax requires every selected optimizer parameter to be read.
    # Read-only .values keeps its complete joint schema without evaluating
    # reference/Quality heads, inventing targets, or adding zero-weight losses.
    for module in model.modules:
        _ = module.values
    return model.wm_only(carry,obs,prevact,training,diagnostics=diagnostics)


def train(model, data):
    """One original joint optimizer invocation with a positive WM allowlist."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from .fg_s3_v2_agent import masked_joint_optimizer
    check_state_schema(model,nj.context())
    before = dict(nj.context())
    carry,obs,prevact,ids = context(model,data)
    with masked_joint_optimizer(model.opt,False,False,allowed_prefixes=WM_GROUPS):
        from functools import partial
        metrics,(carry,_,_,extra) = model.opt(partial(loss,model),carry,obs,prevact,training=True,has_aux=True)
    check_state_schema(model,nj.context())
    finite = jnp.isfinite(metrics['opt/loss']) & jnp.isfinite(metrics['opt/grad_norm'])
    finite &= jnp.stack([jnp.isfinite(v).all() for v in nj.context().values()]).all()
    # Preserve original values directly: adding a zero update can change -0.0.
    # A rejected numerical candidate restores the entire old functional state.
    nj.context().update({key:jnp.where(finite,nj.context()[key],old)
        if model.f1_mutable_keys[key] else old for key,old in before.items()})
    metrics.update(extra)
    metrics.update({'f1/finite':finite,'f1/target_positions':jnp.int32(obs['reward'].size),
        'f1/target_market_minutes':obs['market_advance'].sum(),
        'f1/burnin_positions':jnp.int32(len(data['is_first'])*model.config.quality_f1_burnin_steps),
        'f1/burnin_market_minutes':data['market_advance'][:,1:1+model.config.quality_f1_burnin_steps].sum(),
        'f1/optimizer_updates':model.opt.step.read(),
        'f1/teacher_updates':jnp.int32(0),'f1/actor_updates':jnp.int32(0)})
    carry = (*carry,{key:data[key][:,-1] for key in model.act_space})
    return carry,{'effective_stepid':ids},metrics


def dimensions(target_steps, burnin_steps):
    if (type(target_steps) is not int or target_steps < 1 or
            type(burnin_steps) is not int or burnin_steps < 1):
        raise ValueError('F1_EXPLICIT_POSITIVE_TARGET_AND_BURNIN_REQUIRED')
    return target_steps + burnin_steps


def model_extensions(target_steps, burnin_steps):
    from .fg_s3_quality_v2 import DEFAULTS, quality_extensions
    from gym_trading_env.research.fg_s3_e_quality_supervision import PROTOCOL as SUPERVISION
    dimensions(target_steps, burnin_steps)
    settings = dict(quality_stage=STAGE, quality_f1_protocol=PROTOCOL,
        quality_f1_target_steps=target_steps, quality_f1_burnin_steps=burnin_steps,
        quality_supervision_protocol=SUPERVISION,
        quality_prediction_admitted=False, quality_decision_admitted=False,
        quality_fuse_outcome=False, quality_fuse_score=False, quality_q_admitted=False,
        quality_weighting=False, quality_gradient_every_updates=0,
        quality_repr_grad_scale=0.)
    settings.update({k:0. for k in DEFAULTS if k.endswith('_lambda')})
    return quality_extensions(settings)


def quality_configuration(settings, defaults):
    from gym_trading_env.research.fg_s3_e_quality_supervision import PROTOCOL as SUPERVISION
    if (settings.get('quality_f1_protocol') != PROTOCOL or settings.get('quality_stage') != STAGE
            or settings.get('quality_supervision_protocol') != SUPERVISION):
        raise ValueError('F1_EXECUTION_PROTOCOL_REQUIRED')
    dimensions(settings.get('quality_f1_target_steps'), settings.get('quality_f1_burnin_steps'))
    off = ('quality_prediction_admitted', 'quality_decision_admitted', 'quality_fuse_outcome',
           'quality_fuse_score', 'quality_q_admitted', 'quality_weighting')
    if any(settings.get(k, False) is not False for k in off):
        raise ValueError('F1_PREDICTION_Q_DECISION_FUSION_FORBIDDEN')
    forbidden = tuple(k for k in defaults if k.endswith('_lambda')) + ('quality_repr_grad_scale',)
    if (any(type(settings.get(k, 0.)) not in (int,float) or settings.get(k,0.) != 0. for k in forbidden)
            or settings.get('quality_gradient_every_updates', 0) != 0):
        raise ValueError('F1_TEACHER_ACTOR_LOSSES_FORBIDDEN')
    known = set(defaults) | set(off) | {'mode','s3_v2_mode','quality_variant','quality_stage',
        'quality_f1_protocol','quality_f1_target_steps','quality_f1_burnin_steps',
        'quality_supervision_protocol','quality_gradient_every_updates'}
    if set(settings)-known or settings.get('quality_variant','FULL') != 'FULL':
        raise ValueError('F1_UNKNOWN_OR_NON_FULL_CONFIGURATION')
    result = dict(defaults)
    result.update(settings)
    result.update({k:0. for k in forbidden})
    result.update({k:False for k in off})
    result.update(s3_v2_mode='quality_v2', quality_variant='FULL', quality_gradient_every_updates=0)
    result.pop('mode',None)
    assert expected_stage_contract(STAGE)['parameter_groups_trainable'] == list(WM_GROUPS)
    return result


def validate_model(config):
    if (config.get('quality_stage') != STAGE or config.get('quality_f1_protocol') != PROTOCOL
            or config.enc.typ != 'e_multiscale_1m15m' or config.s3_v2_mode != 'quality_v2'
            or config.bc_lambda != 0 or config.replay_context != 1
            or config.replay_context_mode != 'recompute_prefix'
            or config.batch_length != dimensions(config.quality_f1_target_steps, config.quality_f1_burnin_steps)):
        raise ValueError('F1_MODEL_STAGE_OR_TEMPORAL_CONTRACT_MISMATCH')


def parameter_layout(params):
    """Audit actual complete model/optimizer keys, not just an architecture claim."""
    import numpy as np
    missing = [p for p in ALL_GROUPS if not any(k.startswith(p) for k in params)]
    if missing:
        raise ValueError('F1_INCOMPLETE_FRESH_MODEL: ' + ','.join(missing))
    if not any(k.startswith('opt/') for k in params):
        raise ValueError('F1_ORIGINAL_OPTIMIZER_MISSING')
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in params.items()
            if hasattr(v,'shape') and np.issubdtype(v.dtype, np.number)}
