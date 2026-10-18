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
