"""WP01 pure behavior contracts; no Gym, JAX, market data or training."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dreamerv3.fg_s3_e_quality_course_contract import (
    AUTHORIZATION, COMPOSITION, COURSE_SCHEMA, COURSE_STAGES, OBSERVATION, REWARD,
    TEACHER, TRADING, course_identity_payload, expected_stage_contract,
    inspect_course_readiness, validate_course_spec)


def draft_spec():
    """Explicit pending identities, never blank strings or fake certifications."""
    bounds = dict(start='2020-01-02T09:00:00+08:00', end='2020-01-03T15:00:00+08:00')
    kinds = [('P0', 'always_wait', {}), ('P1', 'always_add', {}),
        ('P2_low', 'probabilistic', dict(probability=.1)),
        ('P2_mid', 'probabilistic', dict(probability=.5)),
        ('P3', 'single_delay', dict(wait_market_minutes=15)),
        ('P4', 'quality_oracle_schedule', dict(source=None))]
    return copy.deepcopy(dict(schema=COURSE_SCHEMA, spec_mode='draft', composition=COMPOSITION,
        initialization=dict(kind='direct_init', seed=17), reward=REWARD,
        observation=OBSERVATION, trading=TRADING, teacher=TEACHER, authorization=AUTHORIZATION,
        curriculum=dict(risk_selected='sampling_label_from_real_trajectories_only',
            behaviors=[dict(behavior_id=name, kind=kind, seed=i, parameters=params,
                boundary=dict(contracts=['CZCE.FG201'], time_bounds=bounds, initial_equity='300000'))
                for i, (name, kind, params) in enumerate(kinds)]),
        data_identity=dict(fold='TRAIN', contracts=['CZCE.FG201'], time_bounds=bounds,
            timezone='Asia/Shanghai', official_day_source='native_trading_date_or_trading_day',
            source_manifest=None, market_sources=None, e_profile_schema_sha256=None, teacher_manifest=None)))


def changed(path, value):
    spec = draft_spec()
    target = spec
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return spec


def test_WP01_01_valid_design_has_all_stages_but_never_admits():
    raw = draft_spec()
    original = copy.deepcopy(raw)
    parsed = validate_course_spec(raw)
    assert raw == original
    ready = inspect_course_readiness(parsed)
    assert ready['status'] == 'CONTRACT_ONLY' and ready['admission_status'] == 'NOT_ADMITTED'
    assert ready['execution_admitted'] is ready['authenticated'] is False
    assert tuple(ready['stage_contracts']) == COURSE_STAGES
    assert len(ready['blocking_reasons']) == 13
    wm, heads, actor, online = [expected_stage_contract(s) for s in COURSE_STAGES]
    assert wm['parameter_groups_trainable'] == ['enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'avail/']
    assert set(heads['parameter_groups_trainable']) == {
        'quality_event/', 'quality_touch/', 'quality_adverse/', 'quality_score/', 'quality_q_c1/', 'quality_q_v2/'}
    assert set(heads['parameter_group_policy'][k] for k in heads['parameter_groups_trainable']) == {'TRAIN_WITH_VALID_LABEL'}
    assert actor['parameter_groups_trainable'] == ['quality_actor/']
    assert actor['selected_policy_head'] == online['selected_policy_head'] == 'quality_actor'
    assert wm['selected_policy_head'] is heads['selected_policy_head'] is None
    assert actor['prerequisite_stages'] == ['FOUNDATION_WM']  # F2 admission cannot block pure-label F3.
    for stage in (wm, heads, actor):
        assert not stage['requires_prediction_admission'] and not stage['requires_decision_admission']
        assert stage['logits_correction'] == dict(outcome=0., score=0., q=0.)
        assert stage['parameter_group_policy']['pol/'] == 'FROZEN'
    assert online['requires_prediction_admission'] and online['requires_q_admission']
    assert online['requires_decision_admission'] and online['rl_gate_required']
    assert online['parameter_group_policy']['avail/'] == 'REQUIRES_WP07_REVIEW_AND_ADMISSION'
    assert all(s['training_enabled'] is False for s in (wm, heads, actor, online))


@pytest.mark.parametrize('path,value', [
    (('composition', 'encoder'), 'legacy_1m'),
    (('composition', 'teaching'), 'tutor_cd'), (('composition', 'teaching'), 'b2'),
    (('initialization', 'kind'), 'warm_start'),
])
def test_WP01_02_no_other_student_or_old_parameter_source(path, value):
    with pytest.raises(ValueError):
        validate_course_spec(changed(path, value))


@pytest.mark.parametrize('key,value', [('schema', 'FG_SINGLE3_REWARD_MARKET_MINUTE_V2'),
    ('candidate', 'C2'), ('beta', .6), ('rho', .5), ('L', 6), ('beta', True), ('L', float('nan'))])
def test_WP01_03_reward_must_be_complete_quality_tp_c1(key, value):
    with pytest.raises(ValueError):
        validate_course_spec(changed(('reward', key), value))
    raw = draft_spec(); del raw['reward']['clock_protocol']
    with pytest.raises(ValueError): validate_course_spec(raw)


@pytest.mark.parametrize('mutation', ['minute', 'context', 'valid', 'cutoff'])
def test_WP01_04_native_e_shapes_valid_and_cutoff_are_required(mutation):
    spec = draft_spec(); obs = spec['observation']
    if mutation == 'minute': obs['fields']['market_seq_e_shared']['shape'] = [60, 17]
    elif mutation == 'context': obs['fields']['market_context_15m']['shape'] = [24, 6]
    elif mutation == 'valid': obs['context_fields'].pop()
    else: del obs['visibility']['cutoff']
    with pytest.raises(ValueError): validate_course_spec(spec)


@pytest.mark.parametrize('path,value', [
    (('trading', 'actions', 'WAIT'), 1), (('trading', 'actions', 'ADD'), True),
    (('trading', 'lots'), 1), (('trading', 'take_profit_points'), 6),
    (('trading', 'stop_loss_points'), 29), (('reward', 'clock_protocol'), 'WALL_CLOCK'),
])
def test_WP01_05_financial_and_clock_assertions_cannot_change(path, value):
    with pytest.raises(ValueError): validate_course_spec(changed(path, value))


@pytest.mark.parametrize('section,fold', [('teacher', 'DEV'), ('teacher', 'TEST'),
                                        ('data_identity', 'DEV'), ('data_identity', 'TEST')])
def test_WP01_06_teacher_and_sources_are_train_only(section, fold):
    with pytest.raises(ValueError): validate_course_spec(changed((section, 'fold'), fold))


@pytest.mark.parametrize('field', ['teacher_valid', 'strict_suffix', 'quality_score',
                                 'outcome_event', 'q_future', 'visible_cutoff'])
def test_WP01_07_future_and_host_fields_cannot_be_policy_inputs(field):
    raw = draft_spec(); raw['observation']['policy_observation_fields'].append(field)
    with pytest.raises(ValueError): validate_course_spec(raw)
    assert validate_course_spec(draft_spec())['teacher']['label_location'] == 'SIDECAR'


def test_WP01_08_oracle_null_is_a_blocker_but_certified_boolean_is_rejected():
    raw = draft_spec()
    assert 'MISSING_QUALITY_ORACLE_SOURCE:P4' in inspect_course_readiness(validate_course_spec(raw))['blocking_reasons']
    raw['curriculum']['behaviors'][-1]['parameters']['source'] = dict(certified=True)
    with pytest.raises(ValueError): validate_course_spec(raw)


@pytest.mark.parametrize('key', ['ready', 'approved', 'admitted', 'support_state',
                               'resolved_composition', 'stage_contracts', 'mode'])
def test_WP01_09_derived_state_and_approvals_are_not_input(key):
    raw = draft_spec(); raw[key] = True
    with pytest.raises(ValueError): validate_course_spec(raw)


@pytest.mark.parametrize('mutation', ['duplicate', 'probability', 'seed', 'delay', 'unknown', 'boundary'])
def test_WP01_10_behavior_parameters_and_identity_fail_closed(mutation):
    raw = draft_spec(); rows = raw['curriculum']['behaviors']
    if mutation == 'duplicate': rows[-1]['behavior_id'] = rows[0]['behavior_id']
    elif mutation == 'probability': rows[2]['parameters']['probability'] = 1.01
    elif mutation == 'seed': del rows[0]['seed']
    elif mutation == 'delay': rows[-2]['parameters']['wait_market_minutes'] = -1
    elif mutation == 'unknown': rows[0]['kind'] = 'risk_selected'
    else: rows[0]['boundary']['contracts'] = ['CZCE.FG301']
    with pytest.raises(ValueError): validate_course_spec(raw)


@pytest.mark.parametrize('stage,field,value', [
    ('FOUNDATION_WM', 'parameter_groups_trainable', ['quality_actor/']),
    ('ACTOR_SUPERVISED', 'selected_policy_head', 'pol'),
    ('QUALITY_HEADS', 'parameter_groups_trainable', ['quality_actor/']),
])
def test_WP01_11_derived_stage_matrix_cannot_be_forged(stage, field, value):
    parsed = validate_course_spec(draft_spec()); parsed['stage_contracts'][stage][field] = value
    with pytest.raises(ValueError): inspect_course_readiness(parsed)
    with pytest.raises(ValueError): course_identity_payload(parsed)


@pytest.mark.parametrize('mutation', ['missing_mode', 'blank_mode', 'implicit', 'formal', 'blank_source', 'empty_sources'])
def test_WP01_12_only_explicit_structurally_valid_draft_permits_null_identity(mutation):
    raw = draft_spec()
    if mutation == 'missing_mode': del raw['spec_mode']
    elif mutation == 'blank_mode': raw['spec_mode'] = ''
    elif mutation == 'implicit': del raw['data_identity']['source_manifest']
    elif mutation == 'formal': raw['spec_mode'] = 'formal'
    elif mutation == 'blank_source': raw['data_identity']['source_manifest'] = ''
    else: raw['data_identity']['market_sources'] = []
    with pytest.raises(ValueError): validate_course_spec(raw)


def test_WP01_13_identity_payload_preserves_order_source_and_configuration():
    raw = draft_spec()
    identity = course_identity_payload(validate_course_spec(raw))
    assert course_identity_payload(validate_course_spec(json.loads(json.dumps(raw, sort_keys=True)))) == identity
    changed_seed = copy.deepcopy(raw); changed_seed['initialization']['seed'] += 1
    changed_order = copy.deepcopy(raw); changed_order['curriculum']['behaviors'].reverse()
    changed_source = copy.deepcopy(raw)
    changed_source['data_identity']['source_manifest'] = dict(path='pending.json', sha256='a' * 64)
    assert all(course_identity_payload(validate_course_spec(s)) != identity
               for s in (changed_seed, changed_order, changed_source))
    changed_source['data_identity']['source_manifest']['sha256'] = 'b' * 64
    assert course_identity_payload(validate_course_spec(changed_source))['data_identity']['source_manifest']['sha256'] == 'b' * 64
    identity['stage_contracts']['FOUNDATION_WM']['parameter_groups_trainable'].clear()
    assert expected_stage_contract('FOUNDATION_WM')['parameter_groups_trainable']


def test_WP01_16_import_and_validation_without_site_packages_or_io(tmp_path):
    path = tmp_path / 'draft.json'; path.write_text(json.dumps(draft_spec()))
    script = '''import builtins, json, sys
from dreamerv3.fg_s3_e_quality_course_contract import validate_course_spec, inspect_course_readiness
raw = json.load(open(sys.argv[1]))
def denied(*args, **kwargs): raise AssertionError('contract performed I/O')
builtins.open = denied
assert inspect_course_readiness(validate_course_spec(raw))['execution_admitted'] is False
assert not {'jax', 'embodied', 'gym_trading_env', 'numpy', 'pandas'} & set(sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    subprocess.run([sys.executable, '-S', '-B', '-c', script, str(path)], env=env, check=True)
