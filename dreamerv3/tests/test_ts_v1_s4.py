"""Thinking Student V1 S4: the refreshed behavior prior (L_policy_demo on the
neutral-anchor teaching view + L_reason), the stopped Q context entering
Reasoning and the gate once Q is supervised, and one improvement round
pi_k -> Q^{pi_k} -> prior refresh -> pi_{k+1}.

Synthetic native market, TS teaching material and a small real student
(engineering checks). Run from gym-trading-env with
PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import json
import pickle

import numpy as np
import pytest

from dreamerv3.tests.test_ts_v1_s2 import T, flat_states, group_of, small, snapshot, update
from dreamerv3.tests.test_ts_v1_s3 import HOLDOUT, QUIET, world  # noqa: F401 (fixture)
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student import value as V

REASONING = ('decision/reason_in/', 'decision/reason_q/', 'decision/reason_norm/',
             'decision/reason/', 'decision/reason_pref/')
Q_CONTEXT = ('decision/reason_q/kernel', 'decision/gate_q/kernel')


def supervised_q(agent, seed=0):
  """A Q head with Q_ADD != Q_WAIT, as after L4; returns the saved state."""
  data = agent.save()
  key = 'value/q/kernel'
  data['params'][key] = np.random.default_rng(seed).normal(
      size=data['params'][key].shape).astype(data['params'][key].dtype)
  agent.load(data)
  return data


def demonstrated(agent):
  data = flat_states(agent)
  data['demo_action'][0] = np.arange(T) % 2
  data['demo_weight'][0] = np.where(data['demo_action'][0] == 1, 1.5, 0.75)
  return data


def changed(before, after):
  return {k for k in before if not np.array_equal(before[k], after[k])}


def test_l_reason_trains_only_the_reasoning_path(tmp_path):
  agent = small(tmp_path, stage='L3', **{'loss_scales.policy_demo': 0.})
  supervised_q(agent)
  before = snapshot(agent)
  mets = update(agent, demonstrated(agent))
  moved = {k for k in changed(before, snapshot(agent)) if not k.startswith('opt_')}
  assert moved and all(k.startswith(REASONING) for k in moved), moved
  assert {'decision/reason_pref/kernel', 'decision/reason_q/kernel'} <= moved
  assert mets['loss/reason'] > 0 and mets['policy/n_demo'] == T - 2


def test_the_prior_refresh_does_not_see_the_value_anchor(tmp_path):
  neutral = small(tmp_path / 'a', stage='L3')
  state = supervised_q(neutral)
  anchored = small(tmp_path / 'b', stage='L3', **{'decision.beta_q': 1.})
  anchored.load(state)
  data = demonstrated(neutral)
  m0, m1 = update(neutral, data), update(anchored, data)
  assert m1['brain/delta_q_abs'] > 0, 'the anchor of beta_Q 1 is not zero'
  for key in ('loss/policy_demo', 'loss/reason', 'policy/add_rate'):
    assert m0[key] == m1[key], key
  assert m0['policy/formal_add_rate'] == m0['policy/add_rate']
  after0, after1 = snapshot(neutral), snapshot(anchored)
  for key in after0:
    np.testing.assert_array_equal(after0[key], after1[key], err_msg=key)


def test_supervised_q_context_starts_learning_from_demonstrations(tmp_path):
  agent = small(tmp_path, stage='L3')
  supervised_q(agent)
  before = snapshot(agent)
  mets = update(agent, demonstrated(agent))
  after = snapshot(agent)
  moved = changed(before, after)
  assert {group_of(k) for k in moved if not k.startswith('opt_')} == {'decision'}
  assert {k.split('/', 1)[0] for k in moved if k.startswith('opt_')} == {'opt_l3'}
  for key in Q_CONTEXT:
    assert not before[key].any() and after[key].any(), key
  assert mets['brain/delta_q_abs'] > 0


def test_one_improvement_round_freezes_the_next_complete_policy(world, tmp_path):
  cf = tmp_path / 'cf'
  V.run_cf(world['policy'], world['corpus'], world['collection'], cf, workers=1, **QUIET)
  args = ['--preset', 'small', '--device', 'cpu', '--batch-size', '1',
          '--batch-length', str(T), '--holdout', str(HOLDOUT)]
  run = tmp_path / 'run'
  ts_run.main(['train-value', '--run', str(run), '--init-from', str(world['policy']),
               '--cf', str(cf), '--updates', '3', *args])
  valued = ts_run.latest(run)
  ts_run.main(['train-policy', '--run', str(run), '--corpus', str(world['corpus']),
               '--updates', '3', *args])
  ts_run.main(['freeze', '--run', str(run), '--name', 'pik1', '--beta-q', '1',
               '--kappa-q', 'auto', '--cf', str(cf), '--corpus', str(world['corpus']), *args])
  policy = run / 'policies' / 'pik1'
  frozen = json.loads((policy / 'policy.json').read_text())

  # The record keeps both stages of the round, read from their own checkpoints.
  teaching = frozen['teaching']
  assert set(teaching) == {'train_value', 'train_policy', 'value_anchor'}
  value = json.loads((valued / 'train_value.json').read_text())
  assert teaching['train_value']['final_eval'] == value['eval'][-1]
  final = teaching['train_policy']['final_eval']
  for key in ('loss/policy_demo', 'loss/reason', 'reason/acc_add', 'reason/acc_wait',
              'policy/formal_add_rate'):
    assert key in final, key
  assert final['constant/reason'] == final['constant/policy_demo']
  assert 'reason_pref' in frozen['reasoning'] and 'untrained' not in frozen

  # The refresh moved only the decision group on top of Q^{pi_k}; the Q
  # context now feeds Reasoning and the gate.
  load = lambda folder: pickle.loads((folder / 'agent.pkl').read_bytes())['params']
  before, after = load(valued), load(policy)
  moved = {k for k in before if not k.startswith('opt_')
           and not np.array_equal(np.asarray(before[k]), np.asarray(after[k]))}
  assert {group_of(k) for k in moved} == {'decision'}
  for key in Q_CONTEXT:
    assert np.asarray(after[key]).any(), key

  # The formal pi_{k+1}: z is the new anchor plus the refreshed correction.
  agent, _ = ts_run.load_policy(policy, **QUIET)
  _, rows = ts_run.rollout(agent, ts_run.corpus_env(world['corpus']), 40)
  assert any(r['anchor'] != 0 for r in rows)
  for r in rows:
    assert r['z'] == pytest.approx(r['anchor'] + r['correction'], rel=1e-5, abs=1e-6)
