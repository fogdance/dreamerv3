"""Thinking Student V1 S0: one policy, Native account, exogenous market, L1, resume.

Synthetic native Env only (engineering checks, not REAL_TRAIN evidence).
Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
from pathlib import Path

import jax
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.thinking_student import labels
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student import transfer
from dreamerv3.thinking_student.student import GROUPS, make_agent

F1 = Path('/data/logdir/fg_single3/s3_v2/E_QUALITY_F1_FG809_12M_U10000_20261021/ckpt/step_000000010000')
T = 8


def small(folder):
  return make_agent(folder, preset='small', batch_size=1, batch_length=T)


@pytest.fixture(scope='module')
def factory(tmp_path_factory):
  from wp02_fixture import ts_corpus
  folder, _ = ts_corpus(tmp_path_factory.mktemp('native'))
  return lambda: ts_run.corpus_env(folder)


@pytest.fixture(scope='module')
def agent(tmp_path_factory):
  return small(tmp_path_factory.mktemp('agent'))


def group_of(key):
  head = key.split('/', 1)[0]
  return next((g for g, names in GROUPS.items() if head in names), None)


def collect(env, actions):
  """Step a Native env with a fixed action rule; return per-step student obs."""
  obs, _ = env.reset(seed=0)
  out = []
  for t, rule in enumerate(actions):
    out.append(ts_run.student_obs(obs, 0., t == 0, False, False))
    action = int(rule and obs['action_mask'][1] > 0.5)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
      break
  return out


def stack(rows):
  return {k: np.stack([r[k] for r in rows], 1) for k in rows[0]}


def pure(agent, fn):
  return jax.jit(nj.pure(fn), static_argnames=('create', 'modify'))


def test_native_rollout_single_policy(agent, factory, tmp_path):
  # The behavior policy mu (half of the steps a uniform legal action) makes
  # the untrained student trade, so the Native ledger closes TP/SL trades.
  explorer = make_agent(tmp_path, preset='small', batch_size=1, batch_length=T,
                        updates={'agent.ts.explore_eps': 0.5})
  _, rows = ts_run.rollout(explorer, factory(), 200, mode='explore')
  assert len(rows) == 200
  assert all(r['mask'][1] > 0.5 for r in rows if r['executed'] == 1)
  assert any(r['executed'] == 1 for r in rows), 'the explored student never tried ADD'
  # Q starts neutral (A6 L3 bootstrap): the value anchor contributes nothing.
  assert all(r['delta_q'] == 0. and r['anchor'] == 0. for r in rows)
  # Outcome exists only for a legal ADD from flat (C1-07).
  for r in rows:
    if r['mask'][1] < 0.5:
      assert r['outcome_valid'] == 0. and r['p_tp_h'] == 0.
  assert any(r['outcome_valid'] == 1. for r in rows)
  assert any(r['reward'] != 0. for r in rows), 'no native TP/SL reward observed'
  # mu is recorded around the formal action of the same policy forward.
  for r in rows:
    legal = r['mask'][1] > 0.5
    assert r['proposed'] == int(legal and r['z'] > 0)
    assert np.isclose(r['mu_prob'], 0.5 * (r['executed'] == r['proposed']) + 0.5 / (1 + legal))
  assert any(r['explored'] for r in rows) and any(r['executed'] != r['proposed'] for r in rows)
  # The formal pi executes its own proposal: masked argmax of [0, z].
  _, formal = ts_run.rollout(agent, factory(), 20, mode='policy')
  for r in formal:
    assert r['executed'] == r['proposed'] == int(r['mask'][1] > 0.5 and r['z'] > 0)
    assert r['mu_prob'] == 1. and not r['explored']


def test_market_brain_ignores_student_actions(agent, factory):
  waits = collect(factory(), [0] * 60)
  adds = collect(factory(), [1] * 60)
  n = min(len(waits), len(adds))
  a, b = stack(waits[:n]), stack(adds[:n])
  assert not np.array_equal(a['position_state'], b['position_state'])
  fn = pure(agent, lambda c, o: agent.model.observe_market(c, o, False, False))
  carry = {'market': agent.model.market_dyn.initial(1)}
  _, (_, fa) = fn(agent.params, carry, a, seed=7, create=False, modify=True)
  _, (_, fb) = fn(agent.params, carry, b, seed=7, create=False, modify=True)
  np.testing.assert_array_equal(np.asarray(fa['deter']), np.asarray(fb['deter']))
  np.testing.assert_array_equal(np.asarray(fa['stoch']), np.asarray(fb['stoch']))


def test_policy_loss_does_not_reach_outcome_or_value(agent, factory):
  data = stack(collect(factory(), [0, 1] * 20)[:T])
  model = agent.model

  def loss(c, o):
    c, feat = model.observe_market(c, o, False, False)
    t = model.think(feat, o, np.ones(o['is_first'].shape, np.float32))
    logp = jax.nn.log_softmax(t['decision']['logits'], -1)[..., 1]
    return -(logp * t['legal']).sum()

  mods = model.group_modules['outcome'] + model.group_modules['value'] + \
      model.group_modules['decision']
  fn = pure(agent, lambda c, o: nj.grad(loss, mods)(c, o))
  carry = {'market': model.market_dyn.initial(1)}
  _, (_, _, grads) = fn(agent.params, carry, data, seed=3, create=False, modify=True)
  assert data['action_mask'][..., 1].any()
  by_group = {}
  for k, v in grads.items():
    by_group.setdefault(group_of(k), []).append(float(np.abs(np.asarray(v)).max()))
  assert max(by_group['outcome']) == 0. and max(by_group['value']) == 0.
  assert max(by_group['decision']) > 0.


def test_l1_update_save_resume(agent, factory, tmp_path):
  rows = collect(factory(), [0, 1, 0] * 10)[:T]
  data = stack(rows)
  data['action'] = np.zeros((1, T), np.int32)
  data['account_valid'] = np.ones((1, T), np.float32)
  data.update(labels.empty_labels((1, T)))
  before = {k: np.asarray(v).copy() for k, v in agent.params.items()}
  from embodied.jax import internal
  for i in range(2):  # the first update runs at lr 0 (linear warmup start)
    batch = internal.device_put(data, agent.train_sharded)
    batch['seed'] = agent._seeds(i, agent.train_mirrored)
    _, _, mets = agent.train(agent.init_train(1), batch)
  assert np.isfinite(mets['opt_l1/loss'])
  after = {k: np.asarray(v) for k, v in agent.params.items()}
  moved = {group_of(k) for k in before if not k.startswith('opt')
           and not np.array_equal(before[k], after[k])}
  assert moved == {'market', 'account'}, moved

  run = tmp_path / 'run'
  ts_run.save(agent, run)
  other = small(tmp_path / 'other')
  assert ts_run.resume(other, run) is not None
  for k, v in agent.params.items():
    np.testing.assert_array_equal(np.asarray(v), np.asarray(other.params[k]), err_msg=k)
  # With jax.sync_train_metrics the embodied agent hands trained parameters
  # to its policy only on load; frozen snapshots and collections always load.
  agent.load(agent.save())
  obs = rows[0]
  _, a1, o1 = agent.policy(agent.init_policy(1), obs, mode='policy')
  _, a2, o2 = other.policy(other.init_policy(1), obs, mode='policy')
  assert int(a1['action'][0]) == int(a2['action'][0])
  np.testing.assert_array_equal(o1['answer/p_add'], o2['answer/p_add'])


@pytest.mark.skipif(not F1.exists(), reason='formal F1 checkpoint not present')
def test_f1_market_weights_inherited(tmp_path):
  student = make_agent(tmp_path, preset='size12m', batch_size=1, batch_length=T)
  lists = transfer.inherit_f1(student, F1, log=lambda *_: None)
  source = transfer.load_params(F1)
  params = {k: np.asarray(v) for k, v in student.params.items()}
  np.testing.assert_array_equal(
      params['market_enc/minute/mlp0/kernel'],
      source['enc/minute/mlp0/kernel'][2:1082])
  for k, v in source.items():
    if k.startswith('dyn/'):
      np.testing.assert_array_equal(params['market_' + k], v)
  assert lists['sliced'] == ['market_enc/minute/mlp0/kernel']
  assert any(k.startswith('decision/') for k in lists['initialized'])
  assert not any(k.startswith(('market_enc/', 'market_dyn/')) for k in lists['initialized'])
