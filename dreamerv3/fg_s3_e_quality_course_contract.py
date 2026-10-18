"""WP01 design contract. Standard library only; no I/O or execution admission.

Identity payloads are hashed by the Gym adapter's existing fingerprint helper.
These stage names are deliberately separate from runtime quality_stage values.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import math
import re


COURSE_SCHEMA = 'FG_S3_E_QUALITY_COURSE_V2'
COURSE_STAGES = ('FOUNDATION_WM', 'QUALITY_HEADS', 'ACTOR_SUPERVISED', 'ONLINE_ADAPT')
REWARD_SCHEMA = 'FG_SINGLE3_REWARD_MARKET_MINUTE_TP_QUALITY_V1'
CLOCK = 'S3_MARKET_MINUTE'
COMPOSITION = dict(encoder='e_multiscale_1m15m', teaching='quality_v2')
REWARD = dict(schema=REWARD_SCHEMA, candidate='C1', beta=.5, rho=.4, L=3., clock_protocol=CLOCK)
POLICY_FIELDS = ['market_seq_e_shared', 'position_state', 'action_mask', 'market_context_15m', 'market_macd_5m']
MACD = {'schema': 'FG_S3_E_MACD5M_CAUSAL_V3', 'periods': [12, 26, 9], 'histogram': 'DIF-Signal', 'initialization': 'first_complete_close; first_DIF_and_signal_zero', 'warmup': '35_complete_buckets_strictly_before_current_bucket', 'scale': '1000*raw/current_visible_1m_Close', 'invalid': 'all_zero', 'bucket': 'five_exact_calendar_minutes_in_one_dated_session', 'partial': 'derive_from_last_committed_EMAs; commit_on_fifth_visible_minute_once', 'incomplete_tail': 'discard_uncommitted_tail_only_at_authenticated_dated_interval_end', 'interval_end_marker': 'visible_zero_duration_end_row_closes_interval_without_price_or_EMA_update', 'continuity': 'carry_EMAs_across_sessions_and_official_days; reset_on_contract', 'errors': 'reject_missing_duplicate_out_of_order_or_late_minutes; never_backfill'}
OBSERVATION = dict(macd_5m=MACD, schema='FG_S3_E_OBSERVATION_V2',
    fields={k: dict(shape=s, dtype='float32') for k, s in (
        ('market_seq_e_shared', [60, 18]), ('position_state', [9]),
        ('action_mask', [2]), ('market_context_15m', [24, 7]), ('market_macd_5m', [4]))},
    policy_observation_fields=POLICY_FIELDS,
    context_fields=['rel_open', 'rel_high', 'rel_low', 'rel_close', 'vol_relative', 'oi_change', 'valid'],
    visibility=dict(data_schema='S3_E_DATA_V1',
        availability='max(bucket_end, all constituent arrival times) <= visible_cutoff',
        cutoff='visible_cutoff <= decision_time', provenance_location='HOST_ONLY',
        provenance_fields=['source_contract', 'visible_cutoff', 'decision_time', 'data_identity'],
        invalid_context='zeros_with_valid_0'))
TRADING = dict(actions=dict(WAIT=0, ADD=1), lots=3, max_batches=1,
    take_profit_points=5, stop_loss_points=30, action_mask_source='NATIVE_ENV',
    account_state_source='NATIVE_ENV')
SIDECAR_LAYERS = dict(
    strict_action=dict(valid_fields=['teacher_valid'], source='quality_tp_strict_suffix'),
    score=dict(valid_fields=['quality_score_valid'], source='native_trade_outcome'),
    outcome=dict(valid_fields=['quality_outcome_valid', 'quality_event_valid',
        'quality_touch_valid', 'quality_adverse_valid'], source='native_trade_outcome'),
    soft=dict(valid_fields=['quality_soft_valid'], source='quality_soft_source',
        sources=['INVALID', 'HEURISTIC_V2', 'Q_V2_VERIFIED']),
    q=dict(valid_fields=['quality_q_c1_valid', 'quality_q_v2_valid'], source='native_certified_q'))
TEACHER = dict(kind='quality_v2', fold='TRAIN', score_policy='FG_S3_QUALITY_TP_MARKET_DAY_V1',
    reward_schema=REWARD_SCHEMA, label_schema='FG_S3_E_QUALITY_TEACHER_V2',
    label_location='SIDECAR', layers=SIDECAR_LAYERS,
    strict_objective=dict(reward_schema=REWARD_SCHEMA,gamma_rational=[332,333],
        tie_break='WAIT',scope='TRAIN_NATIVE_COMPLETE_SUFFIX_ONLY'))
AUTHORIZATION = dict(scope='CONTRACT_ONLY', training_authorization='EXTERNAL_REQUIRED')
BEHAVIORS = ('always_wait', 'always_add', 'probabilistic', 'single_delay', 'quality_oracle_schedule')
WM_GROUPS = ('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'avail/')
REFERENCE_GROUPS = ('pol/', 'val/', 'slowval/')
HEAD_GROUPS = ('quality_event/', 'quality_touch/', 'quality_adverse/', 'quality_score/',
               'quality_q_c1/', 'quality_q_v2/')
ALL_GROUPS = (*WM_GROUPS, *REFERENCE_GROUPS, *HEAD_GROUPS, 'quality_actor/')


def _fail(path, message):
    raise ValueError(f'COURSE_INVALID {path}: {message}')


def _object(value, keys, path):
    if type(value) is not dict or set(value) != set(keys):
        _fail(path, 'requires exactly ' + ', '.join(sorted(keys)))


def _same(value, expected, path):
    """Compare constants with exact JSON types (True is not the action 1)."""
    if type(value) is not type(expected):
        _fail(path, 'type differs from the frozen contract')
    if isinstance(expected, dict):
        _object(value, expected, path)
        for key in expected:
            _same(value[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        if len(value) != len(expected):
            _fail(path, 'length differs from the frozen contract')
        for i, (actual, wanted) in enumerate(zip(value, expected)):
            _same(actual, wanted, f'{path}[{i}]')
    elif value != expected:
        _fail(path, 'differs from the frozen contract')


def _text(value, path):
    if type(value) is not str or not value.strip() or value != value.strip() or '\x00' in value:
        _fail(path, 'requires a nonempty string without surrounding whitespace or NUL')


def _integer(value, path):
    if type(value) is not int or value < 0:
        _fail(path, 'requires a nonnegative integer')


def _sha(value, path, draft):
    if value is None and draft:
        return
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        _fail(path, 'requires a lowercase SHA256; only explicit draft permits null')


def _reference(value, path, draft):
    if value is None and draft:
        return
    _object(value, ('path', 'sha256'), path)
    _text(value['path'], path + '.path')
    # Validate path syntax without importing filesystem helpers or performing I/O.
    parts = value['path'].replace('\\', '/').split('/')
    if '..' in parts or '\\' in value['path'] or '://' in value['path']:
        _fail(path, 'requires a local path without parent traversal')
    _sha(value['sha256'], path + '.sha256', False)


def _contracts(value, path):
    if (type(value) is not list or not value or
            any(type(x) is not str or not re.fullmatch(r'CZCE\.FG\d{3,4}', x) for x in value) or
            len(value) != len(set(value))):
        _fail(path, 'requires unique explicit FG contracts')


def _bounds(value, path):
    _object(value, ('start', 'end'), path)
    times = []
    for key in ('start', 'end'):
        _text(value[key], path + '.' + key)
        try:
            time = datetime.fromisoformat(value[key])
        except ValueError:
            _fail(path, 'requires ISO8601 boundaries')
        if time.utcoffset() != timedelta(hours=8):
            _fail(path, 'boundaries must have explicit Asia/Shanghai +08:00 offset')
        times.append(time)
    if times[0] >= times[1]:
        _fail(path, 'requires start < end (end exclusive)')
    return times


def expected_stage_contract(stage: str) -> dict:
    """Declare future parameter permissions; never enable an optimizer or head."""
    if type(stage) is not str or stage not in COURSE_STAGES:
        _fail('stage', 'unknown course stage')
    policy = dict.fromkeys(ALL_GROUPS, 'FROZEN')
    if stage == 'FOUNDATION_WM':
        policy.update(dict.fromkeys(WM_GROUPS, 'TRAIN'))
    elif stage == 'QUALITY_HEADS':
        policy.update(dict.fromkeys(HEAD_GROUPS, 'TRAIN_WITH_VALID_LABEL'))
    elif stage == 'ACTOR_SUPERVISED':
        policy['quality_actor/'] = 'TRAIN'
    else:
        policy.update(dict.fromkeys(ALL_GROUPS, 'REQUIRES_WP07_REVIEW_AND_ADMISSION'))
        policy['quality_actor/'] = 'TRAIN'
    return dict(stage=stage, status='CONTRACT_ONLY', admission_status='NOT_ADMITTED',
        training_enabled=False, parameter_group_policy=policy,
        parameter_groups_trainable=[k for k, v in policy.items() if v.startswith('TRAIN')],
        selected_policy_head='quality_actor' if stage in ('ACTOR_SUPERVISED', 'ONLINE_ADAPT') else None,
        requires_prediction_admission=stage == 'ONLINE_ADAPT',
        requires_q_admission=stage == 'ONLINE_ADAPT',
        requires_decision_admission=stage == 'ONLINE_ADAPT',
        admission_scope='prediction_q_decision_influence_only',
        rl_gate_required=stage == 'ONLINE_ADAPT',
        logits_correction=dict(outcome=0., score=0., q=0.),
        prerequisite_stages=([] if stage == 'FOUNDATION_WM' else
            ['ACTOR_SUPERVISED'] if stage == 'ONLINE_ADAPT' else ['FOUNDATION_WM']))


def validate_course_spec(raw: dict) -> dict:
    """Strictly normalize a design specification; ValueError on invalid input.

    spec_mode is mandatory. Only explicit draft permits null production identity.
    Derived stage_contracts are output metadata and cannot be supplied as input.
    """
    _object(raw, ('schema', 'spec_mode', 'composition', 'initialization', 'reward',
        'observation', 'trading', 'teacher', 'curriculum', 'data_identity', 'authorization'), 'spec')
    _same(raw['schema'], COURSE_SCHEMA, 'schema')
    if type(raw['spec_mode']) is not str or raw['spec_mode'] not in ('draft', 'formal'):
        _fail('spec_mode', 'explicit draft or formal required')
    draft = raw['spec_mode'] == 'draft'
    for key, expected in (('composition', COMPOSITION), ('observation', OBSERVATION),
                          ('trading', TRADING), ('teacher', TEACHER), ('authorization', AUTHORIZATION)):
        _same(raw[key], expected, key)
    init = raw['initialization']
    _object(init, ('kind', 'seed'), 'initialization')
    _same(init['kind'], 'direct_init', 'initialization.kind')
    _integer(init['seed'], 'initialization.seed')
    reward = copy.deepcopy(raw['reward'])
    _object(reward, REWARD, 'reward')
    for key in ('beta', 'rho', 'L'):
        if type(reward[key]) not in (int, float) or not math.isfinite(reward[key]):
            _fail('reward.' + key, 'requires a finite number')
        reward[key] = float(reward[key])
    _same(reward, REWARD, 'reward')
    data = raw['data_identity']
    _object(data, ('fold', 'contracts', 'time_bounds', 'timezone', 'official_day_source',
        'source_manifest', 'market_sources', 'e_profile_schema_sha256', 'teacher_manifest'), 'data_identity')
    _same(data['fold'], 'TRAIN', 'data_identity.fold')
    _same(data['timezone'], 'Asia/Shanghai', 'data_identity.timezone')
    _same(data['official_day_source'], 'native_trading_date_or_trading_day', 'data_identity.official_day_source')
    _contracts(data['contracts'], 'data_identity.contracts')
    start, end = _bounds(data['time_bounds'], 'data_identity.time_bounds')
    for key in ('source_manifest', 'teacher_manifest'):
        _reference(data[key], 'data_identity.' + key, draft)
    _sha(data['e_profile_schema_sha256'], 'data_identity.e_profile_schema_sha256', draft)
    sources = data['market_sources']
    if sources is not None or not draft:
        if type(sources) is not list or not sources:
            _fail('data_identity.market_sources', 'requires nonempty sources; only explicit draft permits null')
        contracts = []
        for row in sources:
            _object(row, ('contract', 'source_sha256', 'env_config_sha256', 'metadata_sha256'), 'market_source')
            contracts.append(row['contract'])
            for key in ('source_sha256', 'env_config_sha256', 'metadata_sha256'):
                _sha(row[key], 'market_source.' + key, False)
        if contracts != data['contracts']:
            _fail('market_sources', 'must match the declared ordered contracts exactly')
    curriculum = raw['curriculum']
    _object(curriculum, ('behaviors', 'risk_selected'), 'curriculum')
    _same(curriculum['risk_selected'], 'sampling_label_from_real_trajectories_only', 'curriculum.risk_selected')
    if type(curriculum['behaviors']) is not list or not curriculum['behaviors']:
        _fail('curriculum.behaviors', 'requires explicit behaviors')
    ids = set()
    for behavior in curriculum['behaviors']:
        _object(behavior, ('behavior_id', 'kind', 'seed', 'parameters', 'boundary'), 'behavior')
        name, kind, params, boundary = (behavior[k] for k in ('behavior_id', 'kind', 'parameters', 'boundary'))
        if type(name) is not str or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in ids:
            _fail('behavior_id', 'requires a unique portable identifier')
        ids.add(name)
        if type(kind) is not str or kind not in BEHAVIORS:
            _fail('behavior.kind', 'unknown behavior')
        _integer(behavior['seed'], 'behavior.seed')
        keys = {'probabilistic': ('probability',), 'single_delay': ('wait_market_minutes',),
                'quality_oracle_schedule': ('source',)}.get(kind, ())
        _object(params, keys, 'behavior.parameters')
        if kind == 'probabilistic':
            p = params['probability']
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                _fail('behavior.probability', 'requires probability in [0, 1]')
        elif kind == 'single_delay':
            _integer(params['wait_market_minutes'], 'behavior.wait_market_minutes')
        elif kind == 'quality_oracle_schedule':
            _reference(params['source'], 'behavior.oracle_source', draft)
        _object(boundary, ('contracts', 'time_bounds', 'initial_equity'), 'behavior.boundary')
        _contracts(boundary['contracts'], 'behavior.boundary.contracts')
        a, b = _bounds(boundary['time_bounds'], 'behavior.boundary.time_bounds')
        if not set(boundary['contracts']) <= set(data['contracts']) or not start <= a < b <= end:
            _fail('behavior.boundary', 'outside declared TRAIN contracts/time bounds')
        _text(boundary['initial_equity'], 'behavior.boundary.initial_equity')
        try:
            equity = Decimal(boundary['initial_equity'])
        except InvalidOperation:
            _fail('behavior.boundary.initial_equity', 'requires positive finite decimal funds')
        if not equity.is_finite() or equity <= 0:
            _fail('behavior.boundary.initial_equity', 'requires positive finite decimal funds')
    result = copy.deepcopy(raw)
    result['reward'] = reward
    for behavior in result['curriculum']['behaviors']:
        if behavior['kind'] == 'probabilistic':
            behavior['parameters']['probability'] = float(behavior['parameters']['probability'])
    result['stage_contracts'] = {s: expected_stage_contract(s) for s in COURSE_STAGES}
    return result


def _revalidate(validated):
    if type(validated) is not dict:
        _fail('validated', 'requires validated specification')
    body = {k: v for k, v in validated.items() if k != 'stage_contracts'}
    expected = validate_course_spec(body)
    _same(validated, expected, 'validated')
    return expected


def course_identity_payload(validated: dict) -> dict:
    """Canonical research identity, excluding only the diagnostic spec_mode.

    Mapping key order is irrelevant to existing fingerprint; list order matters.
    Callers hash this with the existing native helper, never a new wire format.
    """
    value = _revalidate(validated)
    return {k: v for k, v in value.items() if k != 'spec_mode'}


def inspect_course_readiness(validated: dict) -> dict:
    value = _revalidate(validated)
    blockers = ['E_QUALITY_EXECUTION_NOT_IMPLEMENTED', 'COURSE_STAGES_NOT_IMPLEMENTED',
        'TRAIN_SOURCE_BYTES_NOT_VERIFIED', 'QUALITY_TP_CORPUS_NOT_CERTIFIED',
        'QUALITY_TP_TEACHER_NOT_CERTIFIED', 'STAGE_HANDOFF_NOT_IMPLEMENTED',
        'TRAINING_AUTHORIZATION_NOT_GRANTED']
    for key in ('source_manifest', 'market_sources', 'e_profile_schema_sha256', 'teacher_manifest'):
        if value['data_identity'][key] is None:
            blockers.append('MISSING_' + key.upper())
    for behavior in value['curriculum']['behaviors']:
        if behavior['kind'] == 'quality_oracle_schedule':
            blockers.append('QUALITY_ORACLE_SCHEDULE_NOT_AUTHENTICATED_BY_PREFLIGHT:' + behavior['behavior_id'])
            if behavior['parameters']['source'] is None:
                blockers.append('MISSING_QUALITY_ORACLE_SOURCE:' + behavior['behavior_id'])
    return dict(status='CONTRACT_ONLY', admission_status='NOT_ADMITTED',
        authenticated=False, execution_admitted=False, blocking_reasons=blockers,
        stage_contracts=value['stage_contracts'])
