"""Versioned online execution over the existing E+Quality course/model."""
from __future__ import annotations

import copy
import math

PROTOCOL = 'FG_S3_E_QUALITY_F4_EXECUTION_V1'
MODES = ('SHADOW', 'ONLINE_SUPERVISED', 'ONLINE_RL_CONTROLLED')


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
    for k in ('source_run','source_checkpoint','run_dir'):
        result[k] = str(_local_path(raw[k]))
    from pathlib import Path
    src, node, out = (Path(result[k]) for k in ('source_run','source_checkpoint','run_dir'))
    if not node.is_relative_to(src/'ckpt') or out.is_relative_to(src) or src.is_relative_to(out):
        raise ValueError('ONLINE_SOURCE_OR_OUTPUT_PATH_INVALID')
    return result
