"""F1 execution contract over the original complete E+Quality model.

This contract selects a loss and a freeze policy, never source authority.
"""
from __future__ import annotations

from .fg_s3_e_quality_course_contract import WM_GROUPS, ALL_GROUPS, expected_stage_contract

PROTOCOL = 'FG_S3_E_QUALITY_FOUNDATION_WM_V1'
STAGE = 'FOUNDATION_WM'
SAMPLING_STAGE = 'WM_NATURAL'
FROZEN_GROUPS = tuple(p for p in ALL_GROUPS if p not in WM_GROUPS)


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
