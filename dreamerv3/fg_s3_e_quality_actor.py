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


def train(model,data):
    raise NotImplementedError('F3_UPDATE_IMPLEMENTATION_PENDING')
