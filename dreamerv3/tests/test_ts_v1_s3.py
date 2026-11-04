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

from dreamerv3.tests.test_ts_v1_s2 import (DEVICE, INFERENCE, T, flat_states, group_of, small, snapshot,
                                          update)
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


def worker(world, work, forks, max_steps=8192):
  work.mkdir()
  V._cf_init(str(world['policy']), str(world['corpus']), str(work))
  ref, rows = V.build_reference(world['collection'], V._CF_WORKER['env'])
  replay = V._cf_prepare(dict(work=str(work), reference=ref, recorded=rows, forks=forks))
  job = dict(work=str(work), forks=forks, max_steps=max_steps, probe_steps=64, log_every=0)
  probe = V._cf_job(dict(job, phase='probe'))['results']
  pending = [r['cursor'] for r in probe if r['status'] == 'UNAVAILABLE_BUDGET']
  rows = {r['cursor']: r for r in probe}
  rows.update((r['cursor'], r) for r in V._cf_job(dict(job, phase='continuation', forks=pending))['results'])
  return dict(results=list(rows.values()), replay_steps=replay['native_steps'],
              z_differs=replay['z_differs'])


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


def test_cf_branches_of_the_frozen_student_are_exact(world, tmp_path, monkeypatch):
  reference, _ = V.build_reference(world['collection'], ts_run.corpus_env(world['corpus']))
  assert reference.complete and reference.n > HOLDOUT
  forks = sorted([int(t) for a in (V.ADD, V.WAIT) for t in reference.forks()[
      reference.action[reference.forks() - reference.start] == a][:4]])
  own = reference.action[np.asarray(forks) - reference.start]
  assert set(own) == {V.ADD, V.WAIT}
  # The shared reference replay on the recorded CPU float32
  # the replayed answers are the recorded ones.
  coupled = worker(world, tmp_path / 'coupled', forks)
  assert coupled['replay_steps'] == reference.n and coupled['z_differs'] == 0
  # Never coupling runs every branch with the same student to the true terminal.
  monkeypatch.setattr(V.CF.Reference, 'coupling_kind', lambda self, state: None)
  full = {r['cursor']: r for r in worker(world, tmp_path / 'full', forks)['results']}
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
  capped = worker(world, tmp_path / 'cap', forks, max_steps=1)['results']
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
  assert meta['replay']['z_differs'] == [0] and meta['inference'] == INFERENCE
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
  args = ['--preset', 'small', '--device', DEVICE, '--batch-size', '1',
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


def test_a_cf_budget_probes_every_fork_and_resumes_only_the_sample(world, tmp_path):
  env = ts_run.corpus_env(world['corpus'])
  reference, _ = V.build_reference(world['collection'], env)
  forks = V.choose_forks(reference)
  sample, strata = V.undecided_sample(reference, forks, 8, env._raw.Close.to_numpy(), seed=3)
  assert len(sample) == 8 and set(sample) <= set(forks)
  assert V.undecided_sample(reference, forks, 8, env._raw.Close.to_numpy(), seed=3)[0] == sample
  assert sum(s['forks'] for s in strata['strata'].values()) == len(forks)
  cf = tmp_path / 'cf'
  meta = V.run_cf(world['policy'], world['corpus'], world['collection'], cf, workers=2,
                  max_steps=50, probe_steps=1, seed=3, budget=8, **QUIET)
  pairs = [json.loads(line) for line in (cf / 'pairs.jsonl').open()]
  off = [p for p in pairs if p['status'] == 'UNAVAILABLE_BUDGET']
  assert len(pairs) == len(forks) and off
  for p in off:
    assert (p['reason'], p['steps'], p['g'], p['delta_g']) == ('PROBE_UNFINISHED_BUDGET', 1, None, None)
    assert p['end_status'] == 'ACTIVE' and not p['native_terminal']
  assert meta['probe']['candidates'] == len(forks) and meta['probe']['native_steps'] == len(forks)
  assert meta['continuation']['branches'] == 8
  assert sum(p['continuation_steps'] > 0 for p in pairs) == 8
  assert meta['branch_steps']['total'] == sum(p['steps'] for p in pairs) <= meta['branch_step_bound']
  assert all(p['steps'] <= 50 and p['cost']['native_steps'] == p['steps'] for p in pairs)
  assert meta['replay']['steps'] == [reference.n]
  targets = dict(np.load(cf / 'targets.npz'))
  assert not targets['q_cf_valid'][[reference.row(p['cursor']) for p in off]].any()
  np.testing.assert_array_equal(targets['q_real'], reference.g.astype(np.float32))
  # Once the output is complete only the unfinished probes keep a snapshot.
  snapshots = sorted(int(p.stem) for p in (cf / 'work').glob('*.pkl') if p.stem.isdigit())
  assert snapshots == sorted(p['cursor'] for p in off) and (cf / 'work' / 'reference.pkl').exists()
  assert (meta['snapshots']['removed'], meta['snapshots']['kept']) == (len(pairs) - len(off), len(off))
  # Recovery reuses completed probes and results; no branch or reference repeats.
  again = V.run_cf(world['policy'], world['corpus'], world['collection'], cf, workers=1,
                   max_steps=50, probe_steps=1, seed=3, budget=8, **QUIET)
  assert again['branch_steps'] == meta['branch_steps'] and again['snapshots']['removed'] == 0
  assert again['compute']['executed_native_steps'] == again['compute']['executed_policy_calls'] == 0
  assert [json.loads(line) for line in (cf / 'pairs.jsonl').open()] == pairs
  # A local supplement cannot replace existing complete Q with a shorter CAP.
  good = next(p for p in pairs if p['status'] == 'VALID_COMPLETE' and p['steps'] > 1)
  added = tmp_path / 'supplement'
  supplement = V.run_cf(world['policy'], world['corpus'], world['collection'], added,
                         workers=1, max_steps=1, cursors=[good['cursor']], base_cf=cf, **QUIET)
  merged = {p['cursor']: p for p in map(json.loads, (added / 'pairs.jsonl').open())}
  assert merged[good['cursor']]['status'] == 'VALID_COMPLETE'
  rejected = json.loads((added / 'work' / f'{good["cursor"]}.json').read_text())
  assert rejected['reason'] == 'CAP' and rejected['g'] is None
  assert merged[good['cursor']]['g'] == pytest.approx(good['g'], abs=1e-12)
  assert supplement['valid_pairs'] >= meta['valid_pairs']

  # A round that skips L4 keeps the value anchor of the policy it inherits.
  args = ['--preset', 'small', '--device', DEVICE, '--batch-size', '1', '--batch-length', str(T),
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
