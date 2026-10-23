"""Thinking Student V1 S1: Outcome labels, L_outcome, opt_l2, corpus labels.

Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import pickle
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gym_trading_env.research import ts_v1_corpus as C
from dreamerv3.thinking_student import brain
from dreamerv3.thinking_student import corpus as TC
from dreamerv3.thinking_student import labels as L
from dreamerv3.thinking_student import market
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student import transfer
from dreamerv3.thinking_student.student import GROUPS, make_agent

T = 8
OTHER = 2 * L.NBINS  # first OTHER_EXIT class


def label(kind, minutes, filled=True, **kw):
  """Outcome label of one corpus facts row (gym ts_v1_corpus.FACTS)."""
  ended = kind != 'RIGHT_CENSORED'
  row = dict(
      filled=filled, kind=C.KINDS.index(kind), held_minutes=minutes,
      early10_5=False, early10_5_valid=True, early10_15=False, early10_15_valid=True,
      mae_15=3.0, mae_15_complete=True, mae_60=6.0, mae_60_complete=ended or minutes >= 60,
      mae_full=9.0, path_complete=ended,
      reward={'TP5': 0.9, 'SL30': -3.0, 'OTHER_EXIT': -0.93}.get(kind, np.nan))
  row.update(kw)
  labels = L.outcome_labels({k: np.array([v]) for k, v in row.items()}, C.KINDS)
  return {k: v[0] for k, v in labels.items()}


def allowed(label):
  return set(np.flatnonzero(label['outcome_allowed']).tolist())


def test_tp_sl_and_forced_exits_are_competing_first_endings_on_the_minute_grid():
  assert L.time_bin(1) == 0 and L.time_bin(5) == 0 and L.time_bin(6) == 1
  assert L.time_bin(2760) == L.NBINS - 1 and L.time_bin(2761) == L.NBINS
  assert L.OUTCOME_CLASSES == 34 and L.OPEN_AT_H == 3 * L.NBINS
  tp = label('TP5', 12)
  assert allowed(tp) == {2}  # (10, 15]
  sl = label('SL30', 400)
  assert allowed(sl) == {L.NBINS + 8}  # (345, 690]
  forced = label('OTHER_EXIT', 1034)
  assert allowed(forced) == {OTHER + 9}, 'a main-roll exit ends the trade in (690, 1380]'
  late = label('TP5', 3000)
  assert allowed(late) == {L.OPEN_AT_H}, 'still open at H, whatever came later'
  for lab in (tp, sl, forced, late):
    assert lab['outcome_valid'] == 1 and lab['outcome_quality_valid'] == 1
  assert tp['outcome_quality'] == np.float32(0.9) and sl['outcome_quality'] == -3
  assert forced['outcome_quality'] == np.float32(-0.93), 'a forced exit has its native reward'


def test_censoring_only_removes_bins_known_to_be_ending_free():
  cut = label('RIGHT_CENSORED', 100)  # 60 < 100 <= 120
  m = 5  # bins (..5], (..10], (..15], (..30], (..60] are known empty
  assert allowed(cut) == {*range(m, L.NBINS), *range(L.NBINS + m, OTHER),
                          *range(OTHER + m, L.OPEN_AT_H), L.OPEN_AT_H}
  assert cut['outcome_quality_valid'] == 0, 'an open trade has no final reward'
  assert cut['outcome_mae_valid'].tolist() == [1, 1, 0], 'full-path MAE unknown'
  for minutes in (0, 3):
    assert allowed(label('RIGHT_CENSORED', minutes)) == set(range(L.OUTCOME_CLASSES))
  assert allowed(label('RIGHT_CENSORED', 2760)) == {L.OPEN_AT_H}


def test_unfilled_adds_produce_no_outcome_label():
  assert all(not v.any() for v in label('TP5', 12, filled=False).values())
  risky = label('SL30', 50, early10_5=True, early10_15_valid=False)
  assert risky['outcome_risk'].tolist() == [1, 0]
  assert risky['outcome_risk_valid'].tolist() == [1, 0]
  assert np.allclose(risky['outcome_mae'], [3 / 30, 6 / 30, 9 / 30])
  for kind in L.EVENTS:
    with pytest.raises(ValueError):
      L.allowed_classes(kind, 0)
  with pytest.raises(ValueError):
    L.allowed_classes('REJECTED', 5)


def batch(*labels):
  return {k: np.stack([label[k] for label in labels]) for k in L.LABEL_SPACE}


def head(logits, valid):
  n = len(logits)
  zeros = jnp.zeros(n)
  return dict(event_logits=jnp.asarray(logits), risk_logits=jnp.zeros((n, 2)),
              mae=jnp.zeros((n, 3)), quality=zeros, p_tp_h=zeros, p_tp_15=zeros,
              p_sl_h=zeros, p_other_h=zeros, p_open_h=zeros,
              valid=jnp.asarray(valid, jnp.float32))


def test_outcome_loss_is_the_likelihood_of_what_the_trace_observed():
  logits = np.random.default_rng(0).normal(size=(5, L.OUTCOME_CLASSES)).astype(np.float32)
  data = batch(label('TP5', 12), label('RIGHT_CENSORED', 100),
               label('RIGHT_CENSORED', 3), L.empty_labels(), label('SL30', 40))
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
  labelled = batch(*[label('SL30', 40)] * 3)
  for data, weight in ((empty, np.ones(3, np.float32)), (labelled, np.zeros(3, np.float32))):
    value, grad = jax.value_and_grad(total)(logits, data, weight)
    assert value == 0 and not np.asarray(grad).any()
  short = batch(label('RIGHT_CENSORED', 3))
  value, grad = jax.value_and_grad(total)(logits[:1], short, np.ones(1, np.float32), 'outcome_event')
  assert abs(value) < 1e-6 and np.abs(grad).max() < 1e-6, 'censored before the first edge'


def group_of(key):
  head = key.split('/', 1)[0]
  return next((g for g, names in GROUPS.items() if head in names), head)


def small(folder, **ts):
  updates = {f'agent.ts.{k}': v for k, v in ts.items()}
  return make_agent(folder, preset='small', batch_size=1, batch_length=T, updates=updates)


def labelled_batch(agent, seed=0):
  """Flat, ADD-legal synthetic states carrying TP, SL, forced-exit and
  censored Native facts."""
  rng = np.random.default_rng(seed)
  data = {k: np.zeros((1, T, *s.shape), s.dtype) for k, s in agent.spaces.items()}
  for k in market.MARKET_KEYS:
    data[k] = rng.normal(size=data[k].shape).astype(np.float32)
  data['is_first'][:, 0] = True
  data['action_mask'][...] = 1.
  data['account_valid'][...] = 1.
  data['account_native'][..., 0] = 1.
  cases = [label('TP5', 12), label('SL30', 400), label('RIGHT_CENSORED', 100),
           label('OTHER_EXIT', 3)]
  for t in range(T):
    for k, v in cases[t % len(cases)].items():
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


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
  from wp02_fixture import ts_corpus
  folder, _ = ts_corpus(tmp_path_factory.mktemp('ts'))
  return folder


def test_corpus_windows_put_cursor_labels_on_funded_flat_legal_rows(corpus, tmp_path):
  tc = TC.TSCorpus(corpus)
  facts = C.load_arrays(corpus / 'facts')
  expected = L.outcome_labels(facts, C.KINDS)
  floor = tc.source['funds_floor']
  held_rows = 0
  for name, b in tc.behaviors.items():
    d = tc.window(name, 0, tc.rows)
    cursor = b['cursor']
    on = ((b['position_state'][:, 1] == 0) & (b['action_mask'][:, 1] > 0.5)
          & (b['equity'] >= floor) & ~b['is_last'])
    np.testing.assert_array_equal(d['outcome_valid'] > 0, on & facts['filled'][cursor])
    np.testing.assert_array_equal(d['outcome_allowed'][on], expected['outcome_allowed'][cursor[on]])
    assert not d['outcome_allowed'][~on].any()
    for k in market.MARKET_KEYS:
      np.testing.assert_array_equal(d[k], tc.market[k][cursor])
    # account_native is the account's own Native snapshot on flat and held rows.
    np.testing.assert_array_equal(d['account_native'], b['account_native'])
    held_rows += int((b['position_state'][:, 1] != 0).sum())
  assert held_rows > 0
  kw = dict(weight='outcome', burnin=2, holdout=64, accounts=[TC.LABEL_ACCOUNT])
  train = TC.CorpusWindows(tc, split='train', **kw)
  held = TC.CorpusWindows(tc, split='eval', **kw)
  assert train.hi == held.lo == tc.rows - 64
  (count,) = train.summary().values()
  assert count['tp5'] + count['sl30'] + count['other_exit'] + count['open_at_h'] + \
      count['censored'] == count['labels']
  # The label account carries every filled cursor's label exactly once.
  assert count['labels'] == int(facts['filled'][:train.hi].sum())
  agent = small(tmp_path / 'agent', stage='L2')
  many = train.sample(16, T)
  assert sorted(many) == sorted(agent.spaces)
  assert all(many[k].dtype == s.dtype for k, s in agent.spaces.items()), 'space dtypes'
  assert (many['outcome_valid'][:, 2:] > 0).any(-1).all(), 'windows are drawn by their labels'
  mets = update(agent, train.sample(1, T))
  assert np.isfinite(mets['opt_l2/loss']) and mets['outcome/labels'] > 0


def test_corpus_refuses_facts_that_are_not_the_accounts_own(corpus, tmp_path):
  broken = tmp_path / 'broken'
  shutil.copytree(corpus, broken)
  eligible = np.load(broken / 'facts' / 'eligible.npy')
  t = int(np.flatnonzero(eligible)[0])
  eligible[t] = False
  np.save(broken / 'facts' / 'eligible.npy', eligible)
  with pytest.raises(ValueError, match='ADD legality differs from the facts'):
    TC.TSCorpus(broken)
  short = tmp_path / 'short'
  shutil.copytree(corpus, short)
  native = np.load(short / 'behaviors' / 'P2_mid' / 'account_native.npy')
  np.save(short / 'behaviors' / 'P2_mid' / 'account_native.npy', native[:-1])
  with pytest.raises(ValueError, match='rows for'):
    TC.TSCorpus(short)


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
  _, _, o1 = agent.policy(agent.init_policy(1), obs, mode='policy')
  _, _, o2 = other.policy(other.init_policy(1), obs, mode='policy')
  assert o2['answer/outcome_valid'][0] == 1 and 0 < o2['answer/p_tp_h'][0] < 1
  np.testing.assert_allclose(o1['answer/p_tp_h'], o2['answer/p_tp_h'], rtol=1e-5)

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
  assert lists['remapped'] == [], 'no layout migration was named'
  now = snapshot(third)
  for k in lists['reused']:
    np.testing.assert_array_equal(now[k], params[k], err_msg=k)
  for k in ('outcome/event/kernel', *missing):
    np.testing.assert_array_equal(now[k], init[k], err_msg=k)
  assert int(third.n_updates) == old['counters']['updates']


def test_r2r3_migration_places_the_s2_student_in_the_new_layout(tmp_path):
  agent = small(tmp_path / 'a', stage='L2')
  update(agent, labelled_batch(agent))
  new = snapshot(agent)
  units = new['decision/gate/kernel'].shape[0] - transfer.EVIDENCE
  # The same student in the S2 layout: 10 account fields, TP | SL | SURVIVE
  # events and an Outcome evidence with a 4-wide latent.
  rng = np.random.default_rng(0)
  old = {k: v for k, v in new.items() if not k.startswith('opt_')}
  account = new[transfer.ACCOUNT_KERNEL]
  old[transfer.ACCOUNT_KERNEL] = np.concatenate([account[:19], account[-3:]])
  for k in ('outcome/event/kernel', 'outcome/event/bias'):
    old[k] = np.concatenate([new[k][..., :OTHER], new[k][..., L.OPEN_AT_H:]], -1)
  for k in transfer.EVIDENCE_KERNELS:
    latent = rng.normal(size=(4, new[k].shape[1])).astype(new[k].dtype)
    old[k] = np.concatenate([new[k][:units + OTHER], new[k][units + L.OPEN_AT_H:], latent])
  old['outcome/latent/kernel'] = rng.normal(size=(units, 4)).astype(np.float32)
  old['outcome/latent/bias'] = np.zeros(4, np.float32)
  folder = tmp_path / 's2' / 'ckpt' / 'x'
  folder.mkdir(parents=True)
  with (folder / 'agent.pkl').open('wb') as f:
    pickle.dump({**agent.save(), 'params': old}, f)
  fresh = small(tmp_path / 'b', stage='L2')
  init = snapshot(fresh)
  lists = transfer.inherit_student(fresh, folder / 'agent.pkl', 'r2r3', log=lambda *_: None)
  now = snapshot(fresh)
  assert sorted(c.split(':')[0] for c in lists['remapped']) == sorted([
      transfer.ACCOUNT_KERNEL, 'outcome/event/bias', 'outcome/event/kernel',
      *transfer.EVIDENCE_KERNELS])
  assert lists['reshaped'] == []
  assert lists['dropped'] == ['outcome/latent/bias', 'outcome/latent/kernel']
  # The four new account fields enter through zero rows.
  expect = new[transfer.ACCOUNT_KERNEL].copy()
  expect[19:23] = 0
  np.testing.assert_array_equal(now[transfer.ACCOUNT_KERNEL], expect)
  # TP, SL and SURVIVE -> OPEN_AT_H carry over; OTHER_EXIT bins start fresh.
  for k in ('outcome/event/kernel', 'outcome/event/bias'):
    keep = np.r_[0:OTHER, L.OPEN_AT_H]
    np.testing.assert_array_equal(now[k][..., keep], new[k][..., keep])
    np.testing.assert_array_equal(now[k][..., OTHER:L.OPEN_AT_H], init[k][..., OTHER:L.OPEN_AT_H])
  # Evidence rows: OTHER_EXIT probabilities enter through zero rows.
  for k in transfer.EVIDENCE_KERNELS:
    expect = new[k].copy()
    expect[units + OTHER:units + L.OPEN_AT_H] = 0
    np.testing.assert_array_equal(now[k], expect)
  for k in lists['reused']:
    np.testing.assert_array_equal(now[k], old[k], err_msg=k)
