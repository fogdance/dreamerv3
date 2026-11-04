"""Thinking Student V1 S2: demonstrated behavior prior (L3), formal pi0, the
behavior policy mu, autonomous Native collection with resume, frozen pi0.

Synthetic native market, TS teaching material and batches (engineering checks).
Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.thinking_student import brain
from dreamerv3.thinking_student import corpus as TC
from dreamerv3.thinking_student import labels as L
from dreamerv3.thinking_student import market
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student.student import GROUPS, make_agent

T = 8
# Device of the TS tests (user 2026-11-04: on the GPU): cuda runs the formal
# path (platform cuda, bfloat16); TS_TEST_DEVICE=cpu the float32 CPU one.
DEVICE = os.environ.get('TS_TEST_DEVICE', 'cuda')
INFERENCE = ({'platform': 'cpu', 'compute_dtype': 'float32'} if DEVICE == 'cpu' else
             {'platform': 'cuda', 'compute_dtype': 'bfloat16'})


def small(folder, **ts):
  updates = {f'agent.ts.{k}': v for k, v in ts.items()}
  return make_agent(folder, preset='small', device=DEVICE, batch_size=1, batch_length=T,
                    updates=updates)


def group_of(key):
  head = key.split('/', 1)[0]
  return next((g for g, names in GROUPS.items() if head in names), head)


def snapshot(agent):
  return {k: np.asarray(v).copy() for k, v in agent.params.items()}


def update(agent, data, n=2):
  from embodied.jax import internal
  for i in range(n):  # the first update runs at lr 0 (linear warmup start)
    batch = internal.device_put(data, agent.train_sharded)
    batch['seed'] = agent._seeds(i, agent.train_mirrored)
    _, _, mets = agent.train(agent.init_train(1), batch)
  return mets


def flat_states(agent, seed=0):
  """Flat, ADD-legal synthetic decision states."""
  rng = np.random.default_rng(seed)
  data = {k: np.zeros((1, T, *s.shape), s.dtype) for k, s in agent.spaces.items()}
  for k in market.MARKET_KEYS:
    data[k] = rng.normal(size=data[k].shape).astype(np.float32)
  data['is_first'][:, 0] = True
  data['action_mask'][...] = 1.
  data['account_valid'][...] = 1.
  data['account_native'][..., 0] = 1.
  return data


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
  from wp02_fixture import ts_corpus
  folder, _ = ts_corpus(tmp_path_factory.mktemp('ts'))
  return folder


@pytest.fixture(scope='module')
def factory(corpus):
  return lambda: ts_run.corpus_env(corpus)


def test_formal_action_is_masked_argmax_with_wait_on_ties():
  z = jnp.array([1e-6, 0., -1e-6, 2., 2.])
  mask = jnp.array([[1, 1], [1, 1], [1, 1], [1, 0], [1, 1]], jnp.float32)
  assert brain.formal_action(z, mask).tolist() == [1, 0, 0, 0, 1]


def test_demonstrations_and_class_weights():
  # Graded GOOD, FAIR, none, POOR, EXCELLENT: FAIR and an ungraded cursor
  # demonstrate nothing (gym ts_v1_grades).
  grade = np.array([2, 1, -1, 0, 3], np.int8)
  action, valid = L.demo_labels(dict(valid=(grade >= 2) | (grade == 0),
                                     action=(grade >= 2).astype(np.int8), grade=grade))
  assert action.tolist() == [1, 0, 0, 0, 1] and valid.tolist() == [True, False, False, True, True]
  action, valid = np.array([0.] * 99 + [1.]), np.ones(100, bool)
  w_wait, w_add = L.class_weights(action, valid)
  assert np.isclose(99 * w_wait, w_add), 'both actions carry the same total weight'
  assert np.isclose((99 * w_wait + w_add) / 100, 1.)
  assert L.class_weights(action, valid, 0.) == (1., 1.)
  with pytest.raises(ValueError):
    L.class_weights(np.zeros(5), np.ones(5, bool))


def test_l3_update_moves_only_decision_and_keeps_q_context_neutral(tmp_path):
  agent = small(tmp_path, stage='L3')
  data = flat_states(agent)
  data['demo_action'][0] = np.arange(T) % 2
  data['demo_weight'][0] = np.where(data['demo_action'][0] == 1, 1.5, 0.75)
  before = snapshot(agent)
  mets = update(agent, data)
  after = snapshot(agent)
  changed = {k for k in before if not np.array_equal(before[k], after[k])}
  moved = {group_of(k) for k in changed if not k.startswith('opt_')}
  assert moved == {'decision'}, moved
  assert {k.split('/', 1)[0] for k in changed if k.startswith('opt_')} == {'opt_l3'}
  # Q is untrained (Q_ADD == Q_WAIT == 0): its projections get no gradient.
  for key in ('decision/reason_q/kernel', 'decision/gate_q/kernel'):
    assert not after[key].any(), key
  assert mets['policy/n_demo'] == T - 2, 'two burn-in rows after the reset'
  assert mets['policy/n_add'] == 3 and np.isfinite(mets['loss/policy_demo'])
  assert mets['brain/delta_q_abs'] == 0.


def test_q_moves_the_action_only_through_the_explicit_anchor(tmp_path):
  agent = small(tmp_path / 'a')
  obs = {k: v[:, 3] for k, v in flat_states(agent).items() if k in agent.obs_space}

  def answer(a):
    a.n_actions.value = 0  # same policy seeds: same sampled market latent
    _, _, out = a.policy(a.init_policy(1), obs, mode='policy')
    return {k: float(v[0]) for k, v in out.items() if k.startswith('answer/')}

  neutral = answer(agent)
  data = agent.save()
  key = 'value/q/kernel'
  data['params'][key] = np.random.default_rng(0).normal(
      size=data['params'][key].shape).astype(data['params'][key].dtype)
  agent.load(data)
  trained = answer(agent)
  assert trained['answer/delta_q'] != 0.
  for k in ('answer/z', 'answer/alpha', 'answer/correction', 'answer/reason_pref'):
    assert trained[k] == neutral[k], f'{k}: Q reached the behavior prior'
  anchored = small(tmp_path / 'b', **{'decision.beta_q': 0.5})
  anchored.load(data)
  moved = answer(anchored)
  np.testing.assert_allclose(
      moved['answer/z'] - neutral['answer/z'],
      0.5 * np.clip(trained['answer/delta_q'], -3., 3.), rtol=1e-4, atol=1e-6)


def test_autonomous_runs_only_on_the_ts_env_without_chunk_end(tmp_path):
  from wp02_fixture import native_factory
  from gym_trading_env.research.ts_v1_account import StudentEnv
  make, _, _ = native_factory(tmp_path / 'chunked', minutes=210, chunk=64)
  with pytest.raises(ValueError, match='without CHUNK_END'):
    ts_run.Autonomous(small(tmp_path / 'a'), StudentEnv(make()))


def test_collection_resumes_exactly_where_it_stopped(corpus, factory, tmp_path):
  agent = small(tmp_path / 'a', explore_eps=0.3)
  init = agent.save()
  quiet = dict(log=lambda *_: None)
  ts_run.collect(agent, factory(), tmp_path / 'full', 150, 'explore', chunk=50, **quiet)
  first = small(tmp_path / 'b', explore_eps=0.3)
  first.load(init)
  ts_run.collect(first, factory(), tmp_path / 'split', 100, 'explore', chunk=50, **quiet)
  later = small(tmp_path / 'c', explore_eps=0.3)  # a new process: new model and env
  later.load(init)
  ts_run.collect(later, factory(), tmp_path / 'split', 150, 'explore', chunk=50, **quiet)
  full, split = tmp_path / 'full', tmp_path / 'split'
  names = sorted(p.name for p in (full / 'steps').iterdir())
  assert names == sorted(p.name for p in (split / 'steps').iterdir()) and len(names) == 3
  for name in names:
    assert (full / 'steps' / name).read_text() == (split / 'steps' / name).read_text(), name
    a = np.load(full / 'replay' / name.replace('.jsonl', '.npz'))
    b = np.load(split / 'replay' / name.replace('.jsonl', '.npz'))
    for k in a.files:
      np.testing.assert_array_equal(a[k], b[k], err_msg=f'{name} {k}')
  meta = json.loads((split / 'meta.json').read_text())
  assert meta['steps'] == 150 and meta['resumed'] == 1 and meta['explored'] > 0
  rows = [json.loads(line) for name in names for line in (split / 'steps' / name).open()]
  assert rows[0]['first'] and not any(r['first'] for r in rows[1:]), \
      'market memory runs on from the first minute (no CHUNK_END)'
  report = ts_run.check_collection(split, corpus)
  assert report['adds'] > 0 and report['closed'] > 0
  for key in ('rule_violations', 'formal_violations', 'masked_adds', 'point_violations'):
    assert report[key] == [], key
  # Every student ADD from a funded flat account trades exactly its cursor's fact.
  assert report['facts']['disagree'] == [] and report['facts']['agree'] > 0


def test_frozen_policy_records_and_reloads_what_decides_actions(corpus, tmp_path):
  agent = small(tmp_path / 'a', **{'decision.beta_q': 0.25})
  run = tmp_path / 'run'
  ts_run.save(agent, run)
  folder = ts_run.freeze(agent, run, 'pi0', 'small', [corpus])
  meta = json.loads((folder / 'policy.json').read_text())
  source = json.loads((corpus / 'source.json').read_text())
  assert meta['action_selection'] == 'DETERMINISTIC_MASKED_ARGMAX'
  assert meta['ts']['decision.beta_q'] == 0.25 and meta['preset'] == 'small'
  assert meta['inference'] == INFERENCE
  assert meta['outcome']['horizon_minutes'] == 2760 and len(meta['outcome']['classes']) == 34
  assert 'no CHUNK_END' in meta['rssm_initialization']
  assert meta['reward_version'] == source['inputs']['reward_schema']
  assert meta['corpora'][0]['funds_floor'] == source['funds_floor']
  loaded, _ = ts_run.load_policy(folder)
  assert loaded.config.ts.decision.beta_q == 0.25
  for k, v in agent.params.items():
    np.testing.assert_array_equal(np.asarray(v), np.asarray(loaded.params[k]), err_msg=k)
  obs = {k: v[:, 3] for k, v in flat_states(agent).items() if k in agent.obs_space}
  outs = []
  for a in (agent, loaded):
    a.n_actions.value = 0
    outs.append(a.policy(a.init_policy(1), obs, mode='policy')[2])
  assert outs[0]['proposed'][0] == outs[1]['proposed'][0]
  np.testing.assert_allclose(outs[0]['answer/z'], outs[1]['answer/z'], rtol=1e-6)


def test_corpus_demonstrations_are_class_weighted_on_legal_flat_states(corpus):
  tc = TC.TSCorpus(corpus)
  kw = dict(weight='demo', burnin=2, holdout=64, accounts=[TC.LABEL_ACCOUNT])
  train = TC.CorpusWindows(tc, split='train', **kw)
  held = TC.CorpusWindows(tc, split='eval', demo_weights=train.demo_weights, **kw)
  assert train.demo_counts['add'] > 0 and train.demo_counts['wait'] > train.demo_counts['add']
  w_wait, w_add = train.demo_weights
  assert np.isclose(w_wait * train.demo_counts['wait'], w_add * train.demo_counts['add'])
  assert held.demo_weights == train.demo_weights
  data = train.sample(4, 16)
  demo = data['demo_weight'] > 0
  assert demo[:, 2:].any(-1).all(), 'windows are drawn by their demonstrations'
  assert (data['action_mask'][..., 1][demo] > 0.5).all()
  assert (data['position_state'][..., 1][demo] == 0).all()
  assert np.isin(data['demo_weight'][demo], np.float32([w_wait, w_add])).all()
