"""Thinking Student V1 S1: Outcome labels, L_outcome, opt_l2, corpus labels.

Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.thinking_student import brain
from dreamerv3.thinking_student import labels as L
from dreamerv3.thinking_student import market
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student import transfer
from dreamerv3.thinking_student.student import GROUPS, make_agent

T = 8
CORPUS = Path('/data/s3_corpus/train_full_main_optimized_20261020_cpu20/batches/TRAIN01_FG809_FULL_MAIN')


def facts(kind, minutes, **kw):
  closed = kind in ('TP5', 'SL30', 'OTHER_EXIT')
  base = dict(
      immediate_outcome_verified=True, exit_type=kind, held_market_minutes=minutes,
      early10_5=False, early10_5_valid=True, early10_15=False, early10_15_valid=True,
      mae_15=3.0, mae_15_complete=True, mae_60=6.0,
      mae_window_complete=[True, closed or minutes >= 60, closed or minutes >= 240],
      mae_full=9.0, path_complete=closed,
      configured_reward={'TP5': 0.9, 'SL30': -3.0, 'OTHER_EXIT': -0.93}.get(kind, 0.4))
  base.update(kw)
  return base


def allowed(label):
  return set(np.flatnonzero(label['outcome_allowed']).tolist())


def test_tp_sl_are_competing_first_terminations_on_the_minute_grid():
  assert L.time_bin(1) == 0 and L.time_bin(5) == 0 and L.time_bin(6) == 1
  assert L.time_bin(2760) == L.NBINS - 1 and L.time_bin(2761) == L.NBINS
  tp = L.outcome_label(facts('TP5', 12))
  assert allowed(tp) == {2}  # (10, 15]
  sl = L.outcome_label(facts('SL30', 400))
  assert allowed(sl) == {L.NBINS + 8}  # (345, 690]
  late = L.outcome_label(facts('TP5', 3000))
  assert allowed(late) == {L.SURVIVE}  # first termination after the last edge
  for label in (tp, sl, late):
    assert label['outcome_valid'] == 1 and label['outcome_quality_valid'] == 1
  assert tp['outcome_quality'] == np.float32(0.9) and sl['outcome_quality'] == -3


def test_censoring_only_removes_bins_known_to_be_event_free():
  cut = L.outcome_label(facts('RIGHT_CENSORED', 100))  # 60 < 100 <= 120
  m = 5  # bins (..5], (..10], (..15], (..30], (..60] are known empty
  assert allowed(cut) == {*range(m, L.NBINS), *range(L.NBINS + m, 2 * L.NBINS), L.SURVIVE}
  assert cut['outcome_quality_valid'] == 0, 'open trade has no final reward'
  assert cut['outcome_mae_valid'].tolist() == [1, 1, 0], 'full-path MAE unknown'
  short = L.outcome_label(facts('RIGHT_CENSORED', 3))
  assert allowed(short) == set(range(L.OUTCOME_CLASSES)), 'no information yet'
  forced = L.outcome_label(facts('OTHER_EXIT', 1034))
  assert allowed(forced) == {9, 10, L.NBINS + 9, L.NBINS + 10, L.SURVIVE}
  assert forced['outcome_quality_valid'] == 1, 'a forced exit still has its native reward'


def test_invalid_teacher_states_produce_no_outcome_label():
  for l1 in (None, {}, dict(immediate_outcome_verified=False, invalid_reason='FORCED_WAIT_POSITION'),
             dict(immediate_outcome_verified=True, exit_type='REJECTED')):
    label = L.outcome_label(l1)
    assert all(not v.any() for v in label.values())
  risky = L.outcome_label(facts('SL30', 50, early10_5=True, early10_15_valid=False))
  assert risky['outcome_risk'].tolist() == [1, 0]
  assert risky['outcome_risk_valid'].tolist() == [1, 0]
  assert np.allclose(risky['outcome_mae'], [3 / 30, 6 / 30, 9 / 30])
  with pytest.raises(ValueError):
    L.allowed_classes('TP5', 0)


def batch(*labels):
  return {k: np.stack([label[k] for label in labels]) for k in L.LABEL_SPACE}


def head(logits, valid):
  n = len(logits)
  zeros = jnp.zeros(n)
  return dict(event_logits=jnp.asarray(logits), risk_logits=jnp.zeros((n, 2)),
              mae=jnp.zeros((n, 3)), quality=zeros, p_tp=zeros, p_tp_15=zeros,
              p_sl=zeros, valid=jnp.asarray(valid, jnp.float32))


def test_outcome_loss_is_the_likelihood_of_what_the_trace_observed():
  logits = np.random.default_rng(0).normal(size=(5, L.OUTCOME_CLASSES)).astype(np.float32)
  data = batch(L.outcome_label(facts('TP5', 12)),
               L.outcome_label(facts('RIGHT_CENSORED', 100)),
               L.outcome_label(facts('RIGHT_CENSORED', 3)),
               L.empty_labels(),
               L.outcome_label(facts('SL30', 40)))
  legal = [1, 1, 1, 1, 0]  # the SL label sits on a state where ADD was masked
  losses, mets = brain.outcome_loss(head(logits, legal), data, np.ones(5, np.float32))
  logp = jax.nn.log_softmax(logits, -1)
  cut = data['outcome_allowed'][1] > 0.5
  nll = [-logp[0, 2], -jax.nn.logsumexp(logp[1][cut]), 0.]
  assert np.isclose(losses['outcome_event'], np.mean(nll), atol=1e-6)
  assert mets['outcome/labels'] == 3 and mets['outcome/label_not_legal'] == 1
  assert np.isclose(losses['outcome_quality'], 0.9 ** 2), 'only the closed TP trade has a reward'
  mask = data['outcome_mae_valid'][:3]
  mae = (np.square(data['outcome_mae'][:3]) * mask).sum() / mask.sum()
  assert np.isclose(losses['outcome_mae'], mae)
  assert np.isclose(losses['outcome_risk'], np.log(2.))


def test_rows_without_information_give_zero_loss_and_zero_gradient():
  def total(logits, data, weight, term=None):
    losses, _ = brain.outcome_loss(head(logits, [1] * len(logits)), data, weight)
    return losses[term] if term else sum(losses.values())
  logits = jnp.asarray(np.random.default_rng(1).normal(size=(3, L.OUTCOME_CLASSES)), jnp.float32)
  empty = batch(*[L.empty_labels()] * 3)
  labelled = batch(*[L.outcome_label(facts('SL30', 40))] * 3)
  for data, weight in ((empty, np.ones(3, np.float32)), (labelled, np.zeros(3, np.float32))):
    value, grad = jax.value_and_grad(total)(logits, data, weight)
    assert value == 0 and not np.asarray(grad).any()
  short = batch(L.outcome_label(facts('RIGHT_CENSORED', 3)))
  value, grad = jax.value_and_grad(total)(logits[:1], short, np.ones(1, np.float32), 'outcome_event')
  assert abs(value) < 1e-6 and np.abs(grad).max() < 1e-6, 'censored before the first edge'


def group_of(key):
  head = key.split('/', 1)[0]
  return next((g for g, names in GROUPS.items() if head in names), head)


def small(folder, **ts):
  updates = {f'agent.ts.{k}': v for k, v in ts.items()}
  return make_agent(folder, preset='small', batch_size=1, batch_length=T, updates=updates)


def labelled_batch(agent, seed=0):
  """Flat, ADD-legal synthetic states carrying TP, SL and censored Native facts."""
  rng = np.random.default_rng(seed)
  data = {k: np.zeros((1, T, *s.shape), s.dtype) for k, s in agent.spaces.items()}
  for k in market.MARKET_KEYS:
    data[k] = rng.normal(size=data[k].shape).astype(np.float32)
  data['is_first'][:, 0] = True
  data['action_mask'][...] = 1.
  data['account_valid'][...] = 1.
  data['account_native'][..., 0] = 1.
  cases = [facts('TP5', 12), facts('SL30', 400), facts('RIGHT_CENSORED', 100), facts('TP5', 3)]
  for t in range(T):
    for k, v in L.outcome_label(cases[t % len(cases)]).items():
      data[k][0, t] = v
  return data


def update(agent, data, n=2):
  from embodied.jax import internal
  for i in range(n):  # the first update runs at lr 0 (linear warmup start)
    batch = internal.device_put(data, agent.train_sharded)
    batch['seed'] = agent._seeds(i, agent.train_mirrored)
    _, _, mets = agent.train(agent.init_train(1), batch)
  return mets


def snapshot(agent):
  return {k: np.asarray(v).copy() for k, v in agent.params.items()}


@pytest.mark.parametrize('fusion', [True, False])
def test_l2_update_moves_only_outcome_and_declared_fusion(tmp_path, fusion):
  agent = small(tmp_path, stage='L2', l2_fusion=fusion)
  before = snapshot(agent)
  mets = update(agent, labelled_batch(agent))
  after = snapshot(agent)
  changed = {k for k in before if not np.array_equal(before[k], after[k])}
  moved = {group_of(k) for k in changed if not k.startswith('opt_')}
  assert moved == ({'outcome', 'fusion'} if fusion else {'outcome'}), moved
  assert {k.split('/', 1)[0] for k in changed if k.startswith('opt_')} == {'opt_l2'}
  assert np.isfinite(mets['opt_l2/loss']) and mets['opt_l2/loss'] > 0
  assert mets['outcome/labels'] == T - 2, 'two burn-in rows after the reset'
  assert mets['outcome/label_not_legal'] == 0


@pytest.mark.skipif(not CORPUS.exists(), reason='FG809 TRAIN corpus not present')
def test_corpus_windows_carry_aligned_outcome_labels(tmp_path):
  from dreamerv3.thinking_student.corpus import CorpusWindows
  train = CorpusWindows(CORPUS, policies=['P3'], cache=tmp_path, split='train')
  held = CorpusWindows(CORPUS, policies=['P3'], cache=tmp_path, split='eval')
  assert (tmp_path / 'P3.l1_facts.pkl').exists()
  assert not {c[1] for c in train.chunks} & {c[1] for c in held.chunks}
  count = train.summary()['P3']
  assert count['labels'] > 0
  assert count['tp'] + count['sl'] + count['survive'] + count['censored'] == count['labels']
  agent = small(tmp_path / 'agent', stage='L2')
  data = next(d for d in (train.sample(1, T) for _ in range(200))
              if d['outcome_valid'][:, 2:].any())
  assert sorted(data) == sorted(agent.spaces)
  labelled = data['outcome_valid'] > 0
  flat = data['position_state'][..., 1] == 0
  assert (data['action_mask'][..., 1][labelled] > 0.5).all() and flat[labelled].all()
  assert np.array_equal(data['account_valid'] > 0, flat), 'account_native is exact on flat rows'
  assert (data['account_native'][..., 0][flat] == 1).all()
  mets = update(agent, data)
  assert np.isfinite(mets['opt_l2/loss']) and mets['outcome/labels'] > 0


def test_checkpoint_carries_trained_outcome_and_lists_structure_changes(tmp_path):
  agent = small(tmp_path / 'a', stage='L2')
  fresh = snapshot(agent)
  data = labelled_batch(agent)
  update(agent, data)
  trained = snapshot(agent)
  run = tmp_path / 'run'
  ts_run.save(agent, run)

  # Same structure: strict resume carries the trained Outcome and keeps thinking.
  other = ts_run.build(run, 'small', 'cpu', batch_size=1, batch_length=T,
                       updates={'agent.ts.stage': 'L2'})
  outcome = [k for k in trained if k.startswith('outcome/')]
  assert any(not np.array_equal(fresh[k], trained[k]) for k in outcome)
  for k in trained:
    np.testing.assert_array_equal(trained[k], np.asarray(other.params[k]), err_msg=k)
  obs = {k: data[k][:, T - 1] for k in agent.obs_space}
  _, _, o1 = agent.policy(agent.init_policy(1), obs, mode='eval')
  _, _, o2 = other.policy(other.init_policy(1), obs, mode='eval')
  assert o2['answer/outcome_valid'][0] == 1 and 0 < o2['answer/p_tp'][0] < 1
  np.testing.assert_allclose(o1['answer/p_tp'], o2['answer/p_tp'], rtol=1e-5)

  # Changed structure (an S0-era checkpoint): copy what fits, list the rest.
  old = agent.save()
  params = dict(old['params'])
  for k in ('outcome/event/kernel', 'outcome/event/bias'):
    params[k] = params[k][..., :15]
  missing = sorted(k for k in params if k.startswith('opt_l2/'))
  for k in missing:
    del params[k]
  params['retired/kernel'] = np.zeros(3, np.float32)
  folder = tmp_path / 'old' / 'ckpt' / 'x'
  folder.mkdir(parents=True)
  with (folder / 'agent.pkl').open('wb') as f:
    pickle.dump({**old, 'params': params}, f)
  (tmp_path / 'old' / 'ckpt' / 'latest').write_text('x')
  third = small(tmp_path / 'c', stage='L2')
  init = snapshot(third)
  lists = transfer.inherit_student(third, tmp_path / 'old', log=lambda *_: None)
  assert [s.split(':')[0] for s in lists['reshaped']] == ['outcome/event/bias', 'outcome/event/kernel']
  assert lists['initialized'] == missing and lists['dropped'] == ['retired/kernel']
  now = snapshot(third)
  for k in lists['reused']:
    np.testing.assert_array_equal(now[k], params[k], err_msg=k)
  for k in ('outcome/event/kernel', *missing):
    np.testing.assert_array_equal(now[k], init[k], err_msg=k)
  assert int(third.n_updates) == old['counters']['updates']
