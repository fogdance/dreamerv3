"""Thinking Student V1 S3: Q^{pi_k} teaching material of a frozen student
(exact counterfactual branches, gym ts_v1_cf), the L4 Value stage and the A5
value anchor of the next snapshot.

Synthetic native market, TS teaching material and a small real student
(engineering checks). Run from gym-trading-env with
PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import itertools
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
  policy = ts_run.freeze(agent, run, 'pik', 'small', [corpus])
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


def job(world, forks, max_steps=8192, floor=None, probe=(), probe_steps=None):
  if floor is None:
    floor = V.CF.coupling_floor(ts_run.corpus_env(world['corpus']))
  return dict(index=0, policy=str(world['policy']), corpus=str(world['corpus']),
              collection=str(world['collection']), floor=floor, forks=forks,
              probe=list(probe), probe_steps=probe_steps, max_steps=max_steps, log_every=0)


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
  # Without the FUNDS floor only EXACT couples (the Native account and the
  # student's own memory equal to the reference's, the witness checks both).
  exact = V._cf_job(job(world, forks, floor=np.inf))['results']
  # Never coupling runs every branch with the same student to the true terminal.
  monkeypatch.setattr(V.Stepper, 'native', None)
  full = {r['cursor']: r for r in V._cf_job(job(world, forks, floor=np.inf))['results']}
  monkeypatch.undo()
  reasons, channels = set(), {}
  for r in coupled['results'] + exact:
    f = full[r['cursor']]
    assert r['status'] == f['status'] == 'VALID_COMPLETE' and f['reason'] == 'TERMINAL'
    assert f['steps'] == reference.n - reference.row(r['cursor']) and f['coupling'] is None
    assert r['g'] == pytest.approx(f['g'], rel=1e-9, abs=1e-9)
    assert r['first_action'] == 1 - own[forks.index(r['cursor'])]
    reasons.add(r['reason'])
    channels[r['coupling']] = channels.get(r['coupling'], 0) + 1
  assert 'COUPLED' in reasons and {r['first_action'] for r in coupled['results']} == {0, 1}
  assert {r['coupling'] for r in exact} <= {'EXACT', None}
  assert sum(r['coupling'] == 'EXACT' for r in exact) > 0, channels
  long = [r for r in coupled['results'] if r['reason'] == 'COUPLED' and r['trades']]
  assert long, 'a coupled branch that traded on its own'
  # A policy whose own memory differs at the coupling cursor is refused.
  calls = itertools.count()
  monkeypatch.setattr(V, 'policy_memory', lambda agent, carry: next(calls))
  with pytest.raises(RuntimeError, match='differs from the reference'):
    V._cf_job(job(world, forks, floor=np.inf))
  monkeypatch.undo()
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
  # The state-free sign answer is scored on the same held-out pairs.
  const, agree, signed = report['constant']['delta_g'], 0, 0
  for data in held.batches(1, T):
    dg = np.where(data['action'] == 1, data['q_real'] - data['q_cf'],
                  data['q_cf'] - data['q_real'])
    s = (data['q_cf_valid'] > 0) & (dg != 0)
    s[:, :burnin] = False
    signed, agree = signed + s.sum(), agree + (s & (np.sign(dg) == np.sign(const))).sum()
  assert final['q/n_sign'] == signed
  assert final['constant/sign_agree'] == round(agree / max(signed, 1), 5)
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


def test_a_cf_budget_probes_the_forks_it_does_not_run_to_the_cap(world, tmp_path, monkeypatch):
  """Forks the reference cannot decide (no FUNDS coupling, no true terminal
  in the cap): a stratified sample runs to the cap, every other fork runs a
  probe, where an EXACT coupling still gives its label; a probe still open
  is UNAVAILABLE_BUDGET and never a Q label (user decisions 2026-10-24,
  2026-10-27). The unreachable floor of this process is the CF's: the
  workers get it with their job."""
  env = ts_run.corpus_env(world['corpus'])
  monkeypatch.setattr(V.CF, 'coupling_floor', lambda env: np.inf)
  reference, _ = V.build_reference(world['collection'], env)
  cap, probe, end = 8, 3, reference.start + reference.n
  forks = V.choose_forks(reference)
  undecided = [t for t in forks if not reference.decidable(t, cap)]
  assert undecided == [t for t in forks if end - t > cap] and len(undecided) > 16
  close = env._raw.Close.to_numpy()
  sample, strata = V.undecided_sample(reference, undecided, 8, close, seed=3)
  assert len(sample) == 8 and set(sample) <= set(undecided)
  assert V.undecided_sample(reference, undecided, 8, close, seed=3)[0] == sample
  sizes = strata['strata'].values()
  assert sum(s['forks'] for s in sizes) == len(undecided)
  for s in sizes:  # a systematic draw gives each stratum its share
    share = s['forks'] * 8 / len(undecided)
    assert np.floor(share) <= s['simulated'] <= np.ceil(share)
  assert V.undecided_sample(reference, undecided, 10 ** 6, close)[0] == undecided

  cf = tmp_path / 'cf'
  with pytest.raises(ValueError, match='probe'):
    V.run_cf(world['policy'], world['corpus'], world['collection'], cf, max_steps=cap,
             budget=8, probe_steps=cap, **QUIET)
  meta = V.run_cf(world['policy'], world['corpus'], world['collection'], cf, workers=2,
                  max_steps=cap, seed=3, budget=8, probe_steps=probe, **QUIET)
  pairs = {p['cursor']: p for p in map(json.loads, (cf / 'pairs.jsonl').open())}
  assert sorted(pairs) == forks and meta['coupling_floor'] == np.inf
  probed = set(undecided) - set(sample)
  off = [p for p in pairs.values() if p['status'] == 'UNAVAILABLE_BUDGET']
  assert {p['cursor'] for p in off} <= probed
  for p in off:
    assert (p['reason'], p['steps'], p['g'], p['delta_g'], p['coupling']) == (
        'OPEN_AFTER_PROBE', probe, None, None, None)
    assert p['branch_rewards'] is None and p['end_status'] == 'ACTIVE'
  for t in probed - {p['cursor'] for p in off}:  # closed within the probe
    assert pairs[t]['status'] == 'VALID_COMPLETE' and pairs[t]['steps'] <= probe
    assert pairs[t]['coupling'] == 'EXACT' or pairs[t]['reason'] == 'TERMINAL'
  for t in sample:
    assert pairs[t]['steps'] <= cap and pairs[t]['status'] != 'UNAVAILABLE_BUDGET'
    assert pairs[t]['reason'] in ('CAP', 'COUPLED', 'TERMINAL')
  assert all(p['coupling'] != 'FUNDS' for p in pairs.values())
  assert meta['counts'].get('UNAVAILABLE_BUDGET/OPEN_AFTER_PROBE', 0) == len(off) > 0
  budget = meta['budget']
  assert (budget['forks'], budget['full'], budget['probed'], budget['probe_steps']) == (
      8, 8, len(probed), probe)
  assert budget['unavailable'] == len(off) and budget['probe_closed'] == len(probed) - len(off)
  assert meta['coupled']['EXACT'] == sum(p['coupling'] == 'EXACT' for p in pairs.values())
  assert meta['forks']['undecided'] == len(undecided) and meta['couple_from'] == end
  assert sum(meta['end_status'].values()) == len(pairs)
  assert meta['branch_steps']['total'] == sum(p['steps'] for p in pairs.values())
  targets = dict(np.load(cf / 'targets.npz'))
  assert not targets['q_cf_valid'][[reference.row(p['cursor']) for p in off]].any()
  np.testing.assert_array_equal(targets['q_real'], reference.g.astype(np.float32))

  # A round that skips L4 keeps the value anchor of the policy it inherits.
  args = ['--preset', 'small', '--device', 'cpu', '--batch-size', '1', '--batch-length', str(T),
          '--holdout', str(HOLDOUT)]
  ts_run.main(['freeze', '--run', str(world['run']), '--name', 'anchored', '--out',
               str(tmp_path / 'anchored'), '--beta-q', '1', '--kappa-q', '0.7', '--cf', str(cf),
               *args])
  ts_run.main(['freeze', '--run', str(world['run']), '--name', 'kept', '--out', str(tmp_path / 'kept'),
               '--value-anchor-of', str(tmp_path / 'anchored'), *args])
  anchored, kept = (json.loads((tmp_path / n / 'policy.json').read_text()) for n in ('anchored', 'kept'))
  for key in ('decision.beta_q', 'decision.kappa_q'):
    assert kept['ts'][key] == anchored['ts'][key]
  anchor = kept['teaching']['value_anchor']
  assert anchor['status'] == 'L4_SKIPPED_ANCHOR_KEPT' and anchor['kept'] == anchored['teaching'][
      'value_anchor']
  assert kept['lineage']['value_kept_from'] == str(tmp_path / 'anchored') and 'cf' not in kept['lineage']
  with pytest.raises(SystemExit):
    ts_run.main(['freeze', '--run', str(world['run']), '--name', 'both', '--out',
                 str(tmp_path / 'both'), '--value-anchor-of', str(tmp_path / 'anchored'),
                 '--beta-q', '1', *args])
