"""Thinking Student V1 S3: Q^{pi_k} teaching material of a frozen student
(exact counterfactual branches, gym ts_v1_cf), the L4 Value stage and the A5
value anchor of the next snapshot.

Synthetic native market, TS teaching material and a small real student
(engineering checks). Run from gym-trading-env with
PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import json
import pickle

import numpy as np
import pytest

from dreamerv3.tests.test_ts_v1_s2 import T, flat_states, group_of, small, snapshot, update
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student import value as V

HOLDOUT = 64
QUIET = dict(log=lambda *_: None)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
  """A frozen small student pi_k and its formal collection to the true
  terminal of the synthetic market."""
  from wp02_fixture import ts_corpus
  root = tmp_path_factory.mktemp('ts_s3')
  corpus, _ = ts_corpus(root / 'corpus')
  agent = small(root / 'agent')
  run = root / 'run'
  ts_run.save(agent, run)
  policy = ts_run.freeze(agent, run, 'pik', 'small', corpus)
  # A snapshot frozen before L4 existed has no opt_l4 state.
  with (policy / 'agent.pkl').open('rb') as f:
    data = pickle.load(f)
  data['params'] = {k: v for k, v in data['params'].items() if not k.startswith('opt_l4/')}
  (policy / 'agent.pkl').write_bytes(pickle.dumps(data))
  started = []
  loaded, _ = ts_run.load_policy(policy, log=started.append)
  assert started and 'fresh optimizer state' in started[0]
  collection = root / 'collect'
  ts_run.collect(loaded, ts_run.corpus_env(corpus), collection, 10 ** 6, chunk=64,
                 policy='pik', **QUIET)
  return dict(root=root, corpus=corpus, run=run, policy=policy, collection=collection)


def job(world, forks, max_steps=8192):
  return dict(index=0, policy=str(world['policy']), corpus=str(world['corpus']),
              collection=str(world['collection']), forks=forks, max_steps=max_steps,
              log_every=0)


def test_l4_update_moves_only_the_value_brain(tmp_path):
  agent = small(tmp_path, stage='L4')
  data = flat_states(agent)
  rng = np.random.default_rng(1)
  data['action'][0] = np.arange(T) % 2
  data['q_real'][0] = rng.normal(size=T)
  data['q_real_valid'][0] = 1.
  data['q_cf'][0] = rng.normal(size=T)
  data['q_cf_valid'][0, 4:] = 1.
  before = snapshot(agent)
  mets = update(agent, data)
  after = snapshot(agent)
  changed = {k for k in before if not np.array_equal(before[k], after[k])}
  assert {group_of(k) for k in changed if not k.startswith('opt_')} == {'value'}
  assert {k.split('/', 1)[0] for k in changed if k.startswith('opt_')} == {'opt_l4'}
  burnin = agent.config.ts.q_burnin
  assert mets['q/n_real'] == T - burnin and mets['q/n_cf'] == T - max(4, burnin)
  for key in ('loss/q_real', 'loss/q_cf', 'loss/q_pair'):
    assert np.isfinite(mets[key]) and mets[key] > 0, key
  # The pair target is G_ADD - G_WAIT whichever action pi_k executed.
  rows = slice(max(4, burnin), T)
  a, real, cf = data['action'][0, rows], data['q_real'][0, rows], data['q_cf'][0, rows]
  np.testing.assert_allclose(mets['q/delta_g'], np.where(a == 1, real - cf, cf - real).mean(),
                             rtol=1e-5)


def test_q_targets_need_the_complete_formal_trajectory(world, tmp_path):
  agent, _ = ts_run.load_policy(world['policy'], **QUIET)
  env = ts_run.corpus_env(world['corpus'])
  ts_run.collect(agent, env, tmp_path / 'explore', 20, 'explore', chunk=20, **QUIET)
  with pytest.raises(ValueError, match='formal pi_k'):
    V.load_collection(tmp_path / 'explore')
  agent, _ = ts_run.load_policy(world['policy'], **QUIET)
  ts_run.collect(agent, ts_run.corpus_env(world['corpus']), tmp_path / 'short', 20, chunk=20, **QUIET)
  reference, _ = V.build_reference(tmp_path / 'short', env)
  assert not reference.complete
  with pytest.raises(ValueError, match='true terminal'):
    V.run_cf(world['policy'], world['corpus'], tmp_path / 'short', tmp_path / 'cf', workers=1, **QUIET)


def test_cf_branches_of_the_frozen_student_are_exact(world, monkeypatch):
  reference, _ = V.build_reference(world['collection'], ts_run.corpus_env(world['corpus']))
  assert reference.complete and reference.n > HOLDOUT
  forks = V.choose_forks(reference, forks=12, seed=0)
  own = reference.action[np.asarray(forks) - reference.start]
  assert (own == V.ADD).any() and (own == V.WAIT).sum() == 12
  # Every worker replays the collection first; on the recorded CPU float32
  # the replayed answers are the recorded ones.
  coupled = V._cf_job(job(world, forks))
  assert coupled['replay_steps'] == reference.n and coupled['z_differs'] == 0
  # Never coupling runs every branch with the same student to the true terminal.
  monkeypatch.setattr(V.CF, 'coupling_floor', lambda env: np.inf)
  full = {r['cursor']: r for r in V._cf_job(job(world, forks))['results']}
  monkeypatch.undo()
  reasons = set()
  for r in coupled['results']:
    f = full[r['cursor']]
    assert r['status'] == f['status'] == 'VALID_COMPLETE' and f['reason'] == 'TERMINAL'
    assert f['steps'] == reference.n - reference.row(r['cursor'])
    assert r['g'] == pytest.approx(f['g'], rel=1e-9, abs=1e-9)
    assert r['first_action'] == 1 - own[forks.index(r['cursor'])]
    reasons.add(r['reason'])
  assert 'COUPLED' in reasons and {r['first_action'] for r in coupled['results']} == {0, 1}
  long = [r for r in coupled['results'] if r['reason'] == 'COUPLED' and r['trades']]
  assert long, 'a coupled branch that traded on its own'
  # Without coupling or the true terminal inside the cap: no numeric label.
  capped = V._cf_job(job(world, forks, max_steps=1))['results']
  capped = [r for r in capped if r['status'] == 'NOT_COMPARABLE']
  assert capped and all(r['reason'] == 'CAP' and r['g'] is None and r['delta_g'] is None
                        for r in capped)


def test_cf_targets_train_the_value_brain_and_freeze_the_anchor(world, tmp_path):
  cf = tmp_path / 'cf'
  meta = V.run_cf(world['policy'], world['corpus'], world['collection'], cf, workers=2, **QUIET)
  pairs = [json.loads(line) for line in (cf / 'pairs.jsonl').open()]
  reference, _ = V.build_reference(world['collection'], ts_run.corpus_env(world['corpus']))
  assert len(pairs) == meta['forks']['chosen'] == len(reference.forks())
  assert meta['valid_pairs'] == sum(p['delta_g'] is not None for p in pairs) > 0
  assert meta['replay']['z_differs'] == [0, 0] and meta['inference']['platform'] == 'cpu'
  targets = dict(np.load(cf / 'targets.npz'))
  np.testing.assert_array_equal(targets['q_real'], reference.g.astype(np.float32))
  for p in pairs:
    i = reference.row(p['cursor'])
    assert p[('g_wait', 'g_add')[p['reference_action']]] == reference.g[i]
    assert targets['cf_action'][i] == 1 - p['reference_action']
    assert targets['q_cf_valid'][i] == (p['delta_g'] is not None)
    if p['delta_g'] is not None:
      assert targets['q_cf'][i] == np.float32(p['g'])
  assert targets['q_cf_valid'][targets['cf_action'] < 0].sum() == 0

  burnin = 2
  train = V.QWindows(world['collection'], cf / 'targets.npz', 'train', holdout=HOLDOUT,
                     burnin=burnin)
  held = V.QWindows(world['collection'], cf / 'targets.npz', 'eval', holdout=HOLDOUT)
  assert (train.hi, held.lo, held.hi) == (reference.n - HOLDOUT,) * 2 + (reference.n,)
  data = train.sample(16, T)
  assert data['is_first'][:, 0].all() and not data['is_first'][:, 1:].any()
  assert (data['q_real_valid'][:, burnin:] > 0).any(-1).all()

  # L4 in a new run from the frozen pi_k: only the Value Brain learns.
  args = ['--preset', 'small', '--device', 'cpu', '--batch-size', '1',
          '--batch-length', str(T), '--holdout', str(HOLDOUT)]
  run = tmp_path / 'run'
  ts_run.main(['train-value', '--run', str(run), '--init-from', str(world['policy']),
               '--cf', str(cf), '--updates', '3', *args])
  folder = ts_run.latest(run)
  report = json.loads((folder / 'train_value.json').read_text())
  final = report['eval'][-1]
  for key in ('loss/q_real', 'constant/q_real', 'q/n_real'):
    assert key in final, key
  with (world['policy'] / 'agent.pkl').open('rb') as f:
    source = pickle.load(f)['params']
  with (folder / 'agent.pkl').open('rb') as f:
    trained = pickle.load(f)['params']
  changed = {k for k in source if not np.array_equal(np.asarray(source[k]), np.asarray(trained[k]))}
  assert {group_of(k) for k in changed if not k.startswith('opt_')} == {'value'}
  assert not any(k.startswith('opt_') and not k.startswith('opt_l4/') for k in changed)

  # The next snapshot reads Q only through beta_Q * clip(dQ / kappa_Q).
  ts_run.main(['freeze', '--run', str(run), '--name', 'pik1', '--beta-q', '1',
               '--kappa-q', 'auto', '--cf', str(cf), '--corpus', str(world['corpus']), *args])
  frozen = json.loads((run / 'policies' / 'pik1' / 'policy.json').read_text())
  kappa, n = V.QWindows(world['collection'], cf / 'targets.npz', 'train', holdout=HOLDOUT).kappa()
  assert frozen['ts']['decision.beta_q'] == 1.0 and n > 1
  assert frozen['ts']['decision.kappa_q'] == pytest.approx(kappa)
  assert frozen['teaching']['value_anchor']['kappa_source']['pairs'] == n
  assert frozen['teaching']['train_value']['final_eval'] == final
  agent, _ = ts_run.load_policy(run / 'policies' / 'pik1', **QUIET)
  _, rows = ts_run.rollout(agent, ts_run.corpus_env(world['corpus']), 40)
  for r in rows:
    legal = r['mask'][1] > 0.5 and r['delta_q'] != 0
    expect = np.clip(r['delta_q'] / kappa, -3., 3.) if legal else 0.
    assert r['anchor'] == pytest.approx(expect, rel=1e-5, abs=1e-7)
  with pytest.raises(ValueError, match='positive'):
    ts_run.main(['freeze', '--run', str(run), '--name', 'bad', '--beta-q', '1',
                 '--kappa-q', '0', *args])
