"""Thinking Student V1 C4: formal training over several TRAIN contracts --
mixed corpus windows, staged training with saves and resume, immutable
policy snapshots, Q targets of one frozen policy on several contracts, the
orchestrator and Teacher-free exams.

Two synthetic markets of the same Native rules (engineering checks). Run
from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import importlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from dreamerv3.tests.test_ts_v1_s2 import T, group_of, small
from dreamerv3.thinking_student import config as TSC
from dreamerv3.thinking_student import corpus as TC
from dreamerv3.thinking_student import exam as E
from dreamerv3.thinking_student import formal_train as FT
from dreamerv3.thinking_student import labels as L
from dreamerv3.thinking_student import q_return as QR
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student import value as V

QUIET = dict(log=lambda *_: None)
CONTRACTS = ['CZCE.FG2701', 'CZCE.FG2702']


@pytest.fixture(scope='module')
def two(tmp_path_factory):
  """Teaching material of two synthetic TRAIN contracts: the same Native
  rules on two different price paths."""
  from wp02_fixture import ts_corpus
  root = tmp_path_factory.mktemp('ts_c4')
  a, _ = ts_corpus(root / 'a')
  b, _ = ts_corpus(root / 'b', minutes=240, shift=5)
  source = json.loads((b / 'source.json').read_text())
  source['contract'] = CONTRACTS[1]
  (b / 'source.json').write_text(json.dumps(source))
  return a, b


@pytest.fixture(scope='module')
def pik(two, tmp_path_factory):
  """A frozen small student over both contracts, its formal collection to
  the true terminal of each and the same-policy CF of each."""
  root = tmp_path_factory.mktemp('ts_c4_pik')
  agent = small(root / 'agent')
  ts_run.save(agent, root / 'run')
  policy = ts_run.freeze(agent, root / 'run', 'pik', 'small', list(two), out=root / 'pik')
  out = dict(policy=policy, run=root / 'run', collect={}, cf={})
  for corpus, name in zip(two, CONTRACTS):
    loaded, _ = ts_run.load_policy(policy, **QUIET)
    out['collect'][name], out['cf'][name] = root / 'collect' / name, root / 'cf' / name
    ts_run.collect(loaded, ts_run.corpus_env(corpus), out['collect'][name], 10 ** 6, chunk=64,
                   policy='pik', **QUIET)
    V.run_cf(policy, corpus, out['collect'][name], out['cf'][name], forks=4, workers=1, **QUIET)
  return out


@pytest.fixture(scope='module')
def stores(pik, two, tmp_path_factory):
  """MarketCarry stores of both contracts, computed by the frozen pik."""
  root = tmp_path_factory.mktemp('ts_c4_carry')
  ts_run.main(['market-carry', '--policy', str(pik['policy']), '--corpus', str(two[0]),
               '--corpus', str(two[1]), '--out', str(root), '--device', 'cpu'])
  return root


def in_process(calls, fail=None, argvs=None):
  """formal_train._call without a subprocess; the first call of `fail` (a
  run.py command or the exam module) fails. `argvs` collects the arguments."""
  failed = set()

  def call(module, argv, log):
    argv = [str(a) for a in argv]
    name = argv[0] if module == 'run' else module
    calls.append(name)
    if argvs is not None:
      argvs.append(argv)
    if name == fail and fail not in failed:
      failed.add(fail)
      raise RuntimeError(f'{fail} failed')
    importlib.import_module(f'dreamerv3.thinking_student.{module}').main(argv)
  return call


def files(root):
  return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(root).rglob('*') if p.is_file()}


def params(folder):
  with (folder / 'agent.pkl').open('rb') as f:
    return pickle.load(f)['params']


def changed_groups(before, after):
  return {group_of(k) for k in before if not k.startswith('opt_')
          and not np.array_equal(np.asarray(before[k]), np.asarray(after[k]))}


def test_discovery_admits_train_corpora_of_one_reward_only(two, tmp_path):
  root = tmp_path / 'root'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  (root / 'logs').mkdir()  # no source.json: not teaching material
  assert [p.name for p in TC.discover(root)] == CONTRACTS
  assert [p.name for p in TC.discover(root, CONTRACTS[1:])] == CONTRACTS[1:]
  bad = tmp_path / 'bad'
  shutil.copytree(two[1], bad)
  source = json.loads((bad / 'source.json').read_text())
  for key, value, message in (('fold', 'DEV', 'not TRAIN'), ('gamma', [1, 2], 'differs'),
                              ('contract', CONTRACTS[0], 'twice')):
    (bad / 'source.json').write_text(json.dumps({**source, key: value}))
    with pytest.raises(ValueError, match=message):
      TC.corpus_sources([two[0], bad])


def test_windows_mix_contracts_and_never_join_them(two):
  corpora = [TC.TSCorpus(c, behaviors=[TC.LABEL_ACCOUNT]) for c in two]
  market = [c.market['market_seq_e_shared'] for c in corpora]
  assert isinstance(market[0], np.memmap), 'the market stays on disk, mapped'
  assert not np.array_equal(market[0][:200], market[1][:200])
  kw = dict(weight='demo', burnin=2, accounts=[TC.LABEL_ACCOUNT])
  windows = TC.MultiCorpusWindows(corpora, holdout=0, seed=0, **kw)
  assert [(p.lo, p.hi) for p in windows.parts] == [(0, c.rows) for c in corpora]
  with pytest.raises(ValueError, match='Holdout'):
    TC.MultiCorpusWindows(corpora, split='eval', holdout=0, **kw)
  # One pair of class weights from the demonstrations of both contracts.
  action, valid = (np.concatenate(x) for x in zip(*(p.demos() for p in windows.parts)))
  assert windows.demo_weights == L.class_weights(action, valid, 1.0)
  assert {p.demo_weights for p in windows.parts} == {windows.demo_weights}
  mixed = False
  for _ in range(5):
    data = windows.sample(8, T)
    assert data['is_first'][:, 0].all() and not data['is_first'][:, 1:].any()
    for b, (contract, name, start) in enumerate(windows.last):
      own = windows.parts[windows.contracts.index(contract)]._stack([(name, start)], T)
      for k, v in own.items():
        np.testing.assert_array_equal(data[k][b], v[0], err_msg=k)
    mixed |= len({c for c, _, _ in windows.last}) == 2
  assert mixed, 'a batch holds windows of both contracts'
  assert sum(windows.exposure.values()) == 40 and min(windows.exposure.values()) > 0


def test_stages_on_two_contracts_save_resume_and_freeze_pi0(two, tmp_path):
  base = ['--preset', 'small', '--device', 'cpu', '--batch-size', '2', '--batch-length', str(T),
          '--holdout', '0', '--save-every', '2', '--corpus', str(two[0]), '--corpus', str(two[1])]
  source = tmp_path / 'source'
  ts_run.save(small(tmp_path / 'init'), source)  # an earlier student to inherit from
  run, ckpt = tmp_path / 'run', tmp_path / 'run' / 'ckpt'
  l1 = ['train-market', '--run', str(run), '--init-from', str(source), *base]
  ts_run.main([*l1, '--updates', '3'])
  assert sorted(p.name for p in ckpt.iterdir() if p.is_dir()) == ['000000000002', '000000000003']
  saved = json.loads((ckpt / '000000000002' / 'progress.json').read_text())
  assert saved['updates'] == {'L1': 2, 'L2': 0, 'L3': 0, 'L4': 0}
  report = json.loads((ckpt / '000000000003' / 'train_market.json').read_text())
  assert report['corpora'] == CONTRACTS and report['diagnostic'] is None
  assert sum(report['exposure'].values()) == 3 * 2
  start = params(ts_run.latest(source))
  assert changed_groups(start, params(ckpt / '000000000003')) <= {'market', 'account'}

  # The process died after its save at update 2: the run resumes there,
  # never from --init-from, and trains the remaining update.
  shutil.rmtree(ckpt / '000000000003')
  (ckpt / 'latest').write_text('000000000002')
  agent = ts_run.build(run, 'small', 'cpu', init_from=source, batch_size=2, batch_length=T,
                       updates={'agent.ts.stage': 'L1'})
  for k, v in params(ckpt / '000000000002').items():
    np.testing.assert_array_equal(np.asarray(agent.params[k]), np.asarray(v), err_msg=k)
  remaining = 3 - ts_run.progress(run)['L1']
  ts_run.main([*l1, '--updates', str(remaining)])
  assert ts_run.latest(run).name == '000000000003' and ts_run.progress(run)['L1'] == 3

  # L2 with a TRAIN-internal diagnostic tail of each contract, then L3.
  before = params(ts_run.latest(run))
  ts_run.main(['train-outcome', '--run', str(run), '--updates', '2', *base, '--holdout', '64'])
  report = json.loads((ts_run.latest(run) / 'train_outcome.json').read_text())
  assert report['diagnostic'] == 'TRAIN_INTERNAL_DIAGNOSTIC' and len(report['eval']) == 2
  assert set(report['eval_labels']) == set(CONTRACTS) and report['eval'][-1]['outcome/labels'] > 0
  after = params(ts_run.latest(run))
  assert changed_groups(before, after) <= {'outcome', 'fusion'}
  ts_run.main(['train-policy', '--run', str(run), '--updates', '2', *base])
  report = json.loads((ts_run.latest(run) / 'train_policy.json').read_text())
  wait, add = report['train_demos']['wait'], report['train_demos']['add']
  assert report['demo_weights'] == pytest.approx([(wait + add) / (2 * wait), (wait + add) / (2 * add)])
  assert report['eval'] == []
  assert set(report['train_demos']['contracts']) == set(CONTRACTS)
  assert changed_groups(after, params(ts_run.latest(run))) == {'decision'}
  assert ts_run.progress(run) == {'L1': 3, 'L2': 2, 'L3': 2, 'L4': 0}

  # pi_0: an immutable snapshot that names both contracts and its source.
  out = tmp_path / 'policies' / 'pi_0000'
  freeze = ['freeze', '--run', str(run), '--name', 'pi_0000', '--out', str(out), *base]
  ts_run.main(freeze)
  meta = json.loads((out / 'policy.json').read_text())
  assert [c['contract'] for c in meta['corpora']] == CONTRACTS
  assert set(meta['teaching']) == {'train_market', 'train_outcome', 'train_policy'}
  assert meta['teaching']['train_policy']['final_eval'] is None
  assert meta['lineage']['inherited_from'].startswith(str(source))
  with pytest.raises(FileExistsError):
    ts_run.main(freeze)
  # The one formal student.policy trades both contracts.
  agent, _ = ts_run.load_policy(out)
  for corpus in two:
    _, rows = ts_run.rollout(agent, ts_run.corpus_env(corpus), 30)
    assert len(rows) == 30
    assert all(r['proposed'] == int(r['mask'][1] > 0.5 and r['z'] > 0) for r in rows)


def test_q_targets_of_one_policy_on_two_contracts(pik, two, tmp_path):
  cf = [pik['cf'][c] for c in CONTRACTS]
  train = V.MultiQWindows(cf, 'train', holdout=0, burnin=2)
  assert train.contracts == CONTRACTS and train.policy == str(pik['policy'].resolve())
  data = train.sample(8, T)
  assert data['is_first'][:, 0].all() and not data['is_first'][:, 1:].any()
  assert min(train.exposure.values()) > 0
  # kappa_Q: one std of Delta G over the pairs of both contracts, each
  # contract weighing the same.
  dgs = [part.pairs() for part in train.parts]
  assert all(len(d) for d in dgs)
  x = np.concatenate(dgs)
  w = np.concatenate([np.full(len(d), 1 / len(d)) for d in dgs])
  mean = (w * x).sum() / w.sum()
  kappa, source = train.kappa()
  assert kappa == pytest.approx(np.sqrt((w * (x - mean) ** 2).sum() / w.sum()))
  assert source['pairs'] == len(x) and not source['floored']
  assert train.pair_counts() == dict(total=len(x), **{c: len(d) for c, d in zip(CONTRACTS, dgs)})

  # Q targets of another policy, or one contract twice, never mix.
  other = tmp_path / 'other'
  shutil.copytree(cf[1], other)
  meta = json.loads((other / 'meta.json').read_text())
  (other / 'meta.json').write_text(json.dumps({**meta, 'policy': str(tmp_path / 'pi_other')}))
  with pytest.raises(ValueError, match='different policies'):
    V.MultiQWindows([cf[0], other], 'train', holdout=0)
  with pytest.raises(ValueError, match='twice'):
    V.MultiQWindows([cf[0], cf[0]], 'train', holdout=0)

  # A branch that cannot be compared is no Q label, and a floor never
  # stands in for missing pairs (C4 review R2).
  capped = tmp_path / 'capped'
  V.run_cf(pik['policy'], two[0], pik['collect'][CONTRACTS[0]], capped, forks=4, workers=1,
           max_steps=1, **QUIET)
  targets = dict(np.load(capped / 'targets.npz'))
  pairs = [json.loads(line) for line in (capped / 'pairs.jsonl').open()]
  idle = [p for p in pairs if p['status'] == 'NOT_COMPARABLE']
  assert idle and all(targets['q_cf_valid'][p['cursor'] - targets['cursor'][0]] == 0 for p in idle)
  np.savez(capped / 'targets.npz', **dict(targets, q_cf_valid=np.zeros_like(targets['q_cf_valid'])))
  for floor in (None, 0.05):
    with pytest.raises(ValueError, match='two valid'):
      V.MultiQWindows([capped], 'train', holdout=0).kappa(floor)
  # Real pairs of one Delta G: the configured floor replaces their zero std.
  flat = tmp_path / 'flat'
  shutil.copytree(cf[0], flat)
  targets = dict(np.load(flat / 'targets.npz'))
  add = train.parts[0].data['action'] == V.ADD
  np.savez(flat / 'targets.npz', **dict(targets, q_real=np.zeros_like(targets['q_real']),
                                         q_cf=np.where(add, -.5, .5).astype(np.float32)))
  with pytest.raises(ValueError, match='floor'):
    V.MultiQWindows([flat], 'train', holdout=0).kappa()
  floored, source = V.MultiQWindows([flat], 'train', holdout=0).kappa(0.05)
  assert floored == 0.05 and source['floored'] and source['std'] == 0.
  assert source['pairs'] == len(dgs[0]) >= 2

  # L4 on both contracts' targets moves only the Value Brain.
  args = ['--preset', 'small', '--device', 'cpu', '--batch-size', '2', '--batch-length', str(T),
          '--holdout', '0']
  q = [a for c in cf for a in ('--cf', str(c))]
  run = tmp_path / 'run'
  ts_run.main(['train-value', '--run', str(run), '--init-from', str(pik['policy']), *q,
               '--updates', '2', *args])
  report = json.loads((ts_run.latest(run) / 'train_value.json').read_text())
  assert report['contracts'] == CONTRACTS and report['diagnostic'] is None
  assert report['train_pairs'] == train.pair_counts()
  assert changed_groups(params(pik['policy']), params(ts_run.latest(run))) == {'value'}

  # pi_k+1 reads Q through the pooled anchor.
  out = tmp_path / 'pik1'
  ts_run.main(['freeze', '--run', str(run), '--name', 'pik1', '--out', str(out), '--beta-q', '1',
               '--kappa-q', 'auto', *q, '--corpus', str(two[0]), '--corpus', str(two[1]), *args])
  frozen = json.loads((out / 'policy.json').read_text())
  assert frozen['ts']['decision.beta_q'] == 1.0
  assert frozen['ts']['decision.kappa_q'] == pytest.approx(kappa)
  assert frozen['teaching']['value_anchor']['kappa_source']['pairs'] == len(x)
  assert frozen['teaching']['value_anchor']['pairs'] == train.pair_counts()
  assert frozen['lineage']['q_teacher_policy'] == [str(pik['policy'])]
  agent, _ = ts_run.load_policy(out, **QUIET)
  _, rows = ts_run.rollout(agent, ts_run.corpus_env(two[1]), 40)
  assert any(r['anchor'] != 0 for r in rows)
  for r in rows:
    legal = r['mask'][1] > 0.5 and r['delta_q'] != 0
    expect = np.clip(r['delta_q'] / kappa, -3., 3.) if legal else 0.
    assert r['anchor'] == pytest.approx(expect, rel=1e-5, abs=1e-7)

  # Without two valid pairs the anchor stays off, floor or not (R2).
  neutral = tmp_path / 'pik1_neutral'
  ts_run.main(['freeze', '--run', str(run), '--name', 'pik1_neutral', '--out', str(neutral),
               '--beta-q', '1', '--kappa-q', 'auto', '--kappa-floor', '0.05', '--cf', str(capped),
               '--corpus', str(two[0]), *args])
  frozen = json.loads((neutral / 'policy.json').read_text())
  anchor = frozen['teaching']['value_anchor']
  assert frozen['ts']['decision.beta_q'] == 0. and anchor['kappa_q'] is None
  assert (anchor['status'], anchor['requested_beta_q'], anchor['pairs']['total']) == (
      'Q_ANCHOR_NOT_READY', 1., 0)
  agent, _ = ts_run.load_policy(neutral, **QUIET)
  _, rows = ts_run.rollout(agent, ts_run.corpus_env(two[0]), 40)
  assert all(r['anchor'] == 0 for r in rows)
  with pytest.raises(ValueError, match='--cf targets'):
    ts_run.main(['freeze', '--run', str(run), '--name', 'bad', '--out', str(tmp_path / 'bad'),
                 '--beta-q', '1', '--kappa-q', '0.5', '--corpus', str(two[0]), *args])


def test_state_check_counts_action_flips_on_legal_rows_only():
  """path_agreement on hand-made paths: two windows of four rows, burn-in 1,
  answers (delta_q, z); the formal action is ADD iff legal and z > 0."""
  keys = ('delta_q', 'z')
  z_cont = np.array([[5., -2., .02, 3.], [0., 2., 0., -.02]])
  z_short = np.array([[-5., 1., -.01, -3.], [0., 2.5, 0., -.03]])
  legal = np.array([[1, 1, 1, 0], [0, 1, 0, 1]], bool)
  cont, short = np.zeros((2, 4, 2)), np.zeros((2, 4, 2))
  cont[..., 1], short[..., 1] = z_cont, z_short

  def act(z):
    return (legal & (z > 0)).astype(np.int8)

  r = V.path_agreement(short, cont, act(z_short), act(z_cont), legal, burnin=1, keys=keys)
  # After the burn-in, four legal rows: a WAIT -> ADD flip far from the
  # boundary (z -2), an ADD -> WAIT flip next to it (z 0.02), none at z 2
  # and -0.02. The masked row (z 3 / -3) is WAIT on both paths and no row
  # of any rate.
  assert r['actions'] == dict(legal_rows=4, flips=2, flip_rate=.5, continuous_adds=2,
                              window_adds=2, add_to_wait=1, wait_to_add=1, masked_flips=0)
  z_std = np.std([-2., .02, 2., -.02])
  assert r['boundary']['z_std'] == pytest.approx(z_std)
  bins = r['boundary']['bins']
  assert bins['0-0.05'] == bins['1-inf'] == dict(legal_rows=2, flips=1, flip_rate=.5)
  assert all(bins[b]['legal_rows'] == 0 for b in ('0.05-0.1', '0.1-0.25', '0.25-0.5', '0.5-1'))
  assert r['boundary']['flip_distance']['max'] == pytest.approx(2 / z_std)
  # One position bin of four rows; it holds the flip before the burn-in.
  assert (r['by_position']['0-4']['legal_rows'], r['by_position']['0-4']['flips']) == (5, 3)
  diff = np.abs(z_short - z_cont)[:, 1:]
  assert r['after_burnin']['z']['in_std'] == pytest.approx(diff.mean() / z_cont[:, 1:].std())
  # The signed bias counts every row, masked ones too.
  bias = (z_short - z_cont)[:, 1:].mean()
  assert r['after_burnin']['z']['mean_diff'] == pytest.approx(bias)
  assert r['after_burnin']['z']['mean_diff_in_std'] == pytest.approx(bias / z_cont[:, 1:].std())
  assert r['by_position']['0-4']['mean_diff']['z'] == pytest.approx((z_short - z_cont).mean())
  assert r['after_burnin']['delta_q']['in_std'] is None
  assert V.position_bins(64) == [(0, 16), (16, 32), (32, 48), (48, 64)]
  assert V.position_bins(1024)[3:] == [(48, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
  assert V.position_bins(4) == [(0, 4)]


def test_state_check_from_the_continuous_carry_repeats_the_path(pik):
  """Windows that start from the continuous path's own market carry, with
  its policy seeds, repeat the path exactly; from the zero state they do not
  (C4 review R3, item 4 of the user decision)."""
  collection = pik['collect'][CONTRACTS[0]]
  check = dict(windows=4, length=T, device='cpu', **QUIET)
  exact = V.window_check(pik['policy'], collection, start='carry', **check)
  assert (exact['start'], exact['seed_offset']) == ('carry', 0)
  assert exact['control_max_abs_diff'] == exact['control_action_flips'] == 0
  assert all(v['max_abs_diff'] == 0 for v in exact['after_burnin'].values())
  assert all(b['flips'] == 0 for b in exact['by_position'].values())
  zero = V.window_check(pik['policy'], collection, **check)
  assert max(v['max_abs_diff'] for v in zero['after_burnin'].values()) > 0
  other = V.window_check(pik['policy'], collection, start='carry', seed_offset=1000, **check)
  assert other['seed_offset'] == 1000 and other['windows'] == 4


def test_market_carry_store_is_the_deployed_market_memory(pik, two, stores, tmp_path):
  """The store of a contract holds, before each cursor, the market carry of
  the formal collection: windows that start from it with the path's policy
  seeds repeat the path exactly and need no burn-in. The student of a run
  (its latest checkpoint) writes the same store and leaves the run as it was."""
  store = TC.MarketCarry(stores / CONTRACTS[0])
  corpus = TC.TSCorpus(two[0], behaviors=[TC.LABEL_ACCOUNT])
  assert store.meta['rows'] == corpus.rows == len(store.arrays['deter'])
  assert store.meta['source'] == dict(policy=str(Path(pik['policy']).resolve()))
  agent, _ = ts_run.load_policy(pik['policy'], **QUIET)
  zero = agent.init_policy(1)['market']
  for k in TC.CARRY_KEYS:
    np.testing.assert_array_equal(store.arrays[k][0], np.asarray(zero[k][0]))
  # Both contracts of the one market-carry call: each starts from the zero
  # state and the policy's own seed counter, like its collection.
  check = dict(windows=4, length=T, device='cpu', start='carry', **QUIET)
  for name in CONTRACTS:
    exact = V.window_check(pik['policy'], pik['collect'][name], store=stores / name, **check)
    assert exact['store_vs_path_max_abs_diff'] == 0 and exact['burnin'] == 0, name
    assert exact['control_max_abs_diff'] == exact['actions']['flips'] == 0, name
    assert all(v['max_abs_diff'] == 0 for v in exact['after_burnin'].values()), name
    assert TC.MarketCarry(stores / name).meta['seed_counter'] == 0
  # Other policy seeds write another causal memory, not the deployed one.
  ts_run.market_carry(agent, TC.TSCorpus(two[1], behaviors=[TC.LABEL_ACCOUNT]), tmp_path / 'seeds',
                      {}, counter=5, **QUIET)
  off = V.window_check(pik['policy'], pik['collect'][CONTRACTS[1]],
                       store=tmp_path / 'seeds' / CONTRACTS[1], **check)
  assert off['store_vs_path_max_abs_diff'] > 0
  collection = pik['collect'][CONTRACTS[0]]
  with pytest.raises(ValueError, match='store'):
    V.window_check(pik['policy'], collection, store=stores / CONTRACTS[0], windows=4, length=T,
                   device='cpu', **QUIET)
  run, before = pik['run'], files(pik['run'])
  ts_run.main(['market-carry', '--run', str(run), '--preset', 'small', '--device', 'cpu',
               '--corpus', str(two[0]), '--out', str(tmp_path)])
  assert files(run) == before
  again = TC.MarketCarry(tmp_path / CONTRACTS[0])
  assert again.meta['source']['run'] == str(run.resolve())
  for k in TC.CARRY_KEYS:
    np.testing.assert_array_equal(again.arrays[k], store.arrays[k])


def test_windows_start_from_the_stored_carry(pik, two, stores):
  """Corpus and Q windows bring the stored carry of their first cursor;
  is_first marks only the first market minute."""
  corpora = [TC.TSCorpus(c, behaviors=[TC.LABEL_ACCOUNT], carry_root=stores) for c in two]
  windows = TC.MultiCorpusWindows(corpora, holdout=0, seed=0, weight='demo', burnin=0,
                                  accounts=[TC.LABEL_ACCOUNT])
  part = windows.parts[0]
  data = part._stack([(TC.LABEL_ACCOUNT, 0), (TC.LABEL_ACCOUNT, 5)], T)
  store = corpora[0].carry
  cursors = corpora[0].behaviors[TC.LABEL_ACCOUNT]['cursor'][[0, 5]]
  for k in TC.CARRY_KEYS:
    np.testing.assert_array_equal(data[f'carry/{k}'], store.arrays[k][cursors])
  assert data['is_first'][:, 0].tolist() == [True, False] and not data['is_first'][:, 1:].any()
  data = windows.sample(4, T)
  for b, (contract, name, start) in enumerate(windows.last):
    corpus = corpora[CONTRACTS.index(contract)]
    cursor = corpus.behaviors[name]['cursor'][start]
    np.testing.assert_array_equal(data['carry/deter'][b], corpus.carry.arrays['deter'][cursor])
    assert data['is_first'][b, 0] == (cursor == 0)
  q = V.MultiQWindows([pik['cf'][c] for c in CONTRACTS], 'train', holdout=0, carry_root=stores)
  part = q.parts[1]
  data = part._stack([0, 3], T)
  for k in TC.CARRY_KEYS:
    np.testing.assert_array_equal(data[f'carry/{k}'], part.carry.arrays[k][part.cursor[[0, 3]]])
  assert data['is_first'][:, 0].tolist() == [True, False] and not data['is_first'][:, 1:].any()
  carry = ts_run.pop_carry(data)
  assert set(carry) == set(TC.CARRY_KEYS) and not any(k.startswith('carry/') for k in data)


def test_carry_windows_train_only_their_stage(pik, two, stores, tmp_path):
  args = ['--preset', 'small', '--device', 'cpu', '--batch-size', '2', '--batch-length', str(T),
          '--holdout', '0', '--window-start', 'carry', '--carry-root', str(stores)]
  corpora = ['--corpus', str(two[0]), '--corpus', str(two[1])]
  run = tmp_path / 'run'
  ts_run.main(['train-outcome', '--run', str(run), '--init-from', str(pik['policy']), *corpora,
               '--updates', '2', *args])
  report = json.loads((ts_run.latest(run) / 'train_outcome.json').read_text())
  assert (report['window_start'], report['carry_root']) == ('carry', str(stores))
  changed = changed_groups(params(pik['policy']), params(ts_run.latest(run)))
  assert 'outcome' in changed and changed <= {'outcome', 'fusion'}
  before = params(ts_run.latest(run))
  ts_run.main(['train-policy', '--run', str(run), *corpora, '--updates', '2', *args])
  report = json.loads((ts_run.latest(run) / 'train_policy.json').read_text())
  assert (report['window_start'], report['carry_root']) == ('carry', str(stores))
  assert changed_groups(before, params(ts_run.latest(run))) == {'decision'}
  before = params(ts_run.latest(run))
  q = [a for c in CONTRACTS for a in ('--cf', str(pik['cf'][c]))]
  ts_run.main(['train-value', '--run', str(run), *q, '--updates', '2', *args])
  report = json.loads((ts_run.latest(run) / 'train_value.json').read_text())
  assert report['window_start'] == 'carry'
  assert changed_groups(before, params(ts_run.latest(run))) == {'value'}
  for command, extra in (('train-market', []), ('train-policy', ['--carry-root', ''])):
    with pytest.raises(SystemExit):
      ts_run.main([command, '--run', str(tmp_path / 'no'), *corpora, *args, *extra])


def test_orchestrator_rounds_resume_from_the_disk(two, tmp_path, monkeypatch):
  root = tmp_path / 'corpora'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  config = tmp_path / 'smoke.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run'),
      student=dict(preset='small', device='cpu', seed=1, batch_size=2, batch_length=T, log_every=1),
      foundation=dict(market_updates=2, outcome_updates=2, policy_updates=2, save_every=1),
      iteration=dict(max_rounds=1, cf_wait_forks=2, value_updates=2, prior_refresh_updates=2,
                     save_every=1, kappa_floor=0.05, state_check_windows=4),
      evaluation=dict(random_seeds=2, smoke_steps=40))))
  lay = FT.Layout(tmp_path / 'run')
  calls = []
  monkeypatch.setattr(FT, '_call', in_process(calls))
  with pytest.raises(SystemExit):
    FT.main(['iterate', '--config', str(config), '--until', 'L2'])
  # L1 and L2 first; the next pretrain trains only L3 and freezes pi_0.
  FT.main(['pretrain', '--config', str(config), '--until', 'L2'])
  assert calls == ['train-market', 'train-outcome'] and not lay.policy(0).exists()
  assert json.loads(lay.state.read_text())['step'] == 'L2_done'
  FT.main(['pretrain', '--config', str(config)])
  assert ts_run.progress(lay.foundation) == {'L1': 2, 'L2': 2, 'L3': 2, 'L4': 0}
  assert calls[:4] == ['train-market', 'train-outcome', 'train-policy', 'freeze']
  for name in CONTRACTS:
    report = json.loads((lay.exam(lay.policy(0), 'TRAIN_SMOKE', name) / 'exam.json').read_text())
    assert report['label'] == 'TRAIN_SMOKE' and report['student']['minutes'] == 40
  assert json.loads((lay.root / 'eval' / 'pi_0000' / 'DEV_NOT_CONFIGURED.json').read_text())
  # More L1 after L2, L3 and pi_0 stand on it is refused, before any call.
  raised = tmp_path / 'raised.yaml'
  raised.write_text(config.read_text().replace('market_updates: 2', 'market_updates: 3'))
  n = len(calls)
  with pytest.raises(ValueError, match='L2, L3, pi_0000 already stands on it'):
    FT.main(['pretrain', '--config', str(raised)])
  assert len(calls) == n

  # --until cf stops the round after its CF and state-check, before L4.
  with pytest.raises(SystemExit):
    FT.main(['pretrain', '--config', str(config), '--until', 'cf'])
  calls.clear()
  FT.main(['iterate', '--config', str(config), '--until', 'cf'])
  assert calls == ['collect', 'collect', 'check', 'check', 'cf', 'cf', 'state-check']
  state = json.loads(lay.state.read_text())
  assert (state['round'], state['step']) == (1, 'cf_done') and not lay.learner(1).exists()
  assert all((lay.experience(lay.policy(0), 'cf', c) / 'meta.json').exists() for c in CONTRACTS)
  check = json.loads((lay.experience(lay.policy(0), 'collect', CONTRACTS[0])
                      / 'state_check.json').read_text())
  assert check['control_max_abs_diff'] == check['control_action_flips'] == 0
  assert check['replay_vs_record_max_abs_diff'] == check['replay_vs_record_action_flips'] == 0
  assert check['actions']['masked_flips'] == 0

  # The rerun keeps pi_0's collections and CF and stops at L4; the next one
  # freezes pi_1 and stops at its first exam.
  experience = files(lay.root / 'experience')
  calls.clear()
  monkeypatch.setattr(FT, '_call', in_process(calls, fail='train-value'))
  with pytest.raises(RuntimeError, match='train-value failed'):
    FT.main(['iterate', '--config', str(config)])
  assert calls == ['train-value']
  state = json.loads(lay.state.read_text())
  assert (state['round'], state['step']) == (1, 'L4')
  assert state['rounds']['1']['collected'] == CONTRACTS
  assert files(lay.root / 'experience') == experience
  calls.clear()
  monkeypatch.setattr(FT, '_call', in_process(calls, fail='exam'))
  with pytest.raises(RuntimeError, match='exam failed'):
    FT.main(['iterate', '--config', str(config)])
  assert calls == ['train-value', 'train-policy', 'freeze', 'exam']
  assert {p: v for p, v in files(lay.root / 'experience').items() if p in experience} == experience
  state = json.loads(lay.state.read_text())
  record = state['rounds']['1']
  assert (record['L5'], record['L6'], record['updates']) == (
      'S4_PRIOR_REFRESH', 'L6_SKIPPED_BY_CONFIG', {'L4': 2, 'L5': 2})
  # The round records the anchor pi_1 was frozen with.
  pairs = V.MultiQWindows([lay.experience(lay.policy(0), 'cf', c) for c in CONTRACTS], 'train',
                          holdout=0).pair_counts()['total']
  frozen = json.loads((lay.policy(1) / 'policy.json').read_text())
  assert record['value_anchor']['pairs'] == pairs
  assert record['value_anchor']['beta_q'] == frozen['ts']['decision.beta_q'] == float(pairs >= 2)
  assert state['step'] == 'exam' and state['latest_candidate'] == str(lay.policy(1))
  assert frozen['lineage']['q_teacher_policy'] == [str(lay.policy(0))]
  assert (lay.experience(lay.policy(0), 'collect', CONTRACTS[0]) / 'state_check.json').exists()

  # pi_1 is frozen and its exam is not done: the rerun takes only its
  # missing exams and no new round (C4 review R1).
  calls.clear()
  monkeypatch.setattr(FT, '_call', in_process(calls))
  FT.main(['iterate', '--config', str(config)])
  assert calls == ['exam', 'exam']
  assert all((lay.exam(lay.policy(1), 'TRAIN_SMOKE', c) / 'exam.json').exists() for c in CONTRACTS)
  state = json.loads(lay.state.read_text())
  assert state['step'] == 'done' and state['best_dev'] is None

  # The configured rounds are done; status only reads.
  calls.clear()
  FT.main(['iterate', '--config', str(config)])
  assert calls == []
  before = files(lay.root)
  report = FT.status(FT.load_config(config), lay)
  assert files(lay.root) == before
  assert report['policies'] == {'pi_0000': True, 'pi_0001': True}
  assert report['learners']['pi_0001'] == {'L1': 0, 'L2': 0, 'L3': 2, 'L4': 2}


def test_orchestrator_starts_windows_from_the_market_carry(pik, two, tmp_path, monkeypatch):
  """student.window_start carry (user decision 2026-10-24, item 4) with L1
  and L2 inherited: the stores of every contract come before L3, L3 and the
  round's L4 read them, and the round's state-check starts from them. The
  stores of the foundation are pi_0000's deployed market memory."""
  root = tmp_path / 'corpora'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  config = tmp_path / 'carry.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run', init_from=str(pik['run'])),
      student=dict(preset='small', device='cpu', seed=1, batch_size=2, batch_length=T,
                   log_every=1, window_start='carry'),
      foundation=dict(policy_updates=1, save_every=1),
      iteration=dict(max_rounds=1, cf_wait_forks=1, value_updates=1, state_check_windows=2,
                     state_check_seed_offset=7),
      evaluation=dict(random_seeds=1, smoke_steps=20))))
  lay = FT.Layout(tmp_path / 'run')
  calls, argvs = [], []
  monkeypatch.setattr(FT, '_call', in_process(calls, argvs=argvs))
  FT.main(['pretrain', '--config', str(config)])
  assert calls == ['market-carry', 'market-carry', 'train-policy', 'freeze', 'exam', 'exam']
  assert ['--window-start', 'carry', '--carry-root', str(lay.carry)] == [
      a for a in argvs[2] if a in ('--window-start', 'carry', '--carry-root', str(lay.carry))]
  assert sorted(p.parent.name for p in lay.carry.glob('*/meta.json')) == CONTRACTS
  report = json.loads((ts_run.latest(lay.foundation) / 'train_policy.json').read_text())
  assert report['window_start'] == 'carry' and report['carry_root'] == str(lay.carry)
  assert FT.status(FT.load_config(config), lay)['market_carry'] == CONTRACTS
  # A store stands on L1 like the later stages do.
  raised = tmp_path / 'raised.yaml'
  raised.write_text(config.read_text().replace('policy_updates: 1', 'policy_updates: 1\n  '
                                               'market_updates: 1'))
  with pytest.raises(ValueError, match='market_carry already stands on it'):
    FT.main(['pretrain', '--config', str(raised)])

  calls.clear()
  FT.main(['iterate', '--config', str(config), '--until', 'cf'])
  assert calls == ['collect', 'collect', 'check', 'check', 'cf', 'cf', 'state-check']
  check = json.loads((lay.experience(lay.policy(0), 'collect', CONTRACTS[0])
                      / 'state_check.json').read_text())
  assert (check['start'], check['seed_offset'], check['burnin']) == ('carry', 7, 0)
  assert check['store'] == str(lay.carry / CONTRACTS[0])
  assert check['store_vs_path_max_abs_diff'] == 0
  assert check['replay_vs_record_max_abs_diff'] == 0
  calls.clear()
  monkeypatch.setattr(FT, '_call', in_process(calls, fail='train-value', argvs=argvs))
  with pytest.raises(RuntimeError, match='train-value failed'):
    FT.main(['iterate', '--config', str(config)])
  assert calls == ['train-value']
  assert argvs[-1][argvs[-1].index('--window-start') + 1] == 'carry'
  assert argvs[-1][argvs[-1].index('--carry-root') + 1] == str(lay.carry)


def test_net_pnl_return_counts_every_trade_open_in_the_window():
  """q_return NET_PNL: undiscounted net PnL / 1818 of the trades open within
  H market minutes of t; one still open at the window end counts its
  realized PnL, one entered after the window counts nothing."""
  class Clock:  # one market minute per row, advance(a, b) = prefix[b + 1] - prefix[a + 1]
    prefix = np.arange(12.)
  trades = [dict(entry_cursor=2, exit_cursor=4, net_trade_pnl=-1818.),   # decisions 1 -> 3
            dict(entry_cursor=5, exit_cursor=10, net_trade_pnl=909.)]    # decisions 4 -> 9
  g = QR.window_returns(Clock, 0, 10, trades, horizon=3)
  # t = 0, 1: the first trade (entry within 3 minutes); the second opens at 4
  # only from t = 2; it closes at 9, after the window of t = 2..5, yet counts.
  np.testing.assert_allclose(g, [-1, -1, -.5, -.5, .5, .5, .5, .5, .5, .5])
  assert QR.window_returns(Clock, 0, 10, trades, horizon=1)[0] == 0


def test_orchestrator_relabels_the_cf_with_the_net_pnl_return(pik, two, tmp_path, monkeypatch):
  """iteration.q_return net_pnl_h2760 (E2 R2, user 2026-10-30): the CF of pi_k
  goes to cf_native/, q_return writes the relabeled cf/ that L4 and the
  freeze read, and the round keeps the label statistics."""
  root = tmp_path / 'corpora'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  config = tmp_path / 'r2.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run', init_from=str(pik['run'])),
      student=dict(preset='small', device='cpu', seed=1, batch_size=2, batch_length=T, log_every=1),
      foundation=dict(policy_updates=1, save_every=1),
      iteration=dict(max_rounds=1, cf_wait_forks=2, value_updates=1, kappa_floor=0.05,
                     q_return='net_pnl_h2760'),
      evaluation=dict(random_seeds=1, smoke_steps=20))))
  lay = FT.Layout(tmp_path / 'run')
  calls, argvs = [], []
  monkeypatch.setattr(FT, '_call', in_process(calls, argvs=argvs))
  FT.main(['pretrain', '--config', str(config)])
  calls.clear(), argvs.clear()
  FT.main(['iterate', '--config', str(config)])
  assert calls[:8] == ['collect', 'collect', 'check', 'check', 'cf', 'cf', 'q_return', 'q_return']
  pi0 = lay.policy(0)
  for c in CONTRACTS:
    native, cf = lay.experience(pi0, 'cf_native', c), lay.experience(pi0, 'cf', c)
    meta = json.loads((cf / 'meta.json').read_text())
    assert meta['q_return']['name'] == 'net_pnl_h2760' and meta['q_return']['horizon'] == 2760
    assert meta['q_return']['relabeled_from'] == str(native.resolve())
    assert meta['policy'] == json.loads((native / 'meta.json').read_text())['policy']
  for name in ('train-value', 'freeze'):
    argv = argvs[calls.index(name)]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == '--cf'] == [
        str(lay.experience(pi0, 'cf', c)) for c in CONTRACTS]
  rec = json.loads(lay.state.read_text())['rounds']['1']
  assert rec['q_return']['name'] == 'net_pnl_h2760' and set(rec['q_return']['stats']) == set(CONTRACTS)
  bad = tmp_path / 'bad.yaml'
  bad.write_text(config.read_text().replace('net_pnl_h2760', 'net_pnl'))
  with pytest.raises(ValueError, match='q_return'):
    FT.load_config(bad)


def test_orchestrator_keeps_the_value_brain_when_the_cf_is_not_representative(
    pik, two, tmp_path, monkeypatch):
  """iteration.cf_budget and cf_min_coverage: every fork gets one probe step;
  unfinished probes get no continuation and remain UNAVAILABLE_BUDGET.
  The valid pairs then cover too few forks: the
  round skips L4, and pi_1 keeps the value brain and anchor of pi_0."""
  root = tmp_path / 'corpora'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  config = tmp_path / 'thin.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run', init_from=str(pik['run'])),
      student=dict(preset='small', device='cpu', seed=1, batch_size=2, batch_length=T,
                   log_every=1),
      foundation=dict(policy_updates=1, save_every=1),
      iteration=dict(max_rounds=1, cf_wait_forks=1, cf_max_steps=2, cf_probe_steps=1, cf_budget=0,
                     cf_min_coverage=0.8, value_updates=1, prior_refresh_updates=1, save_every=1,
                     kappa_floor=0.05),
      evaluation=dict(random_seeds=1, smoke_steps=20))))
  lay = FT.Layout(tmp_path / 'run')
  calls, argvs = [], []
  monkeypatch.setattr(FT, '_call', in_process(calls, argvs=argvs))
  FT.main(['pretrain', '--config', str(config)])
  monkeypatch.setattr(V.CF, 'coupling_floor', lambda env: np.inf)
  calls.clear()
  start = len(argvs)
  FT.main(['iterate', '--config', str(config)])
  assert calls[:6] == ['collect', 'collect', 'check', 'check', 'cf', 'cf']
  assert 'train-value' not in calls and calls[6:8] == ['train-policy', 'freeze']
  rec = json.loads(lay.state.read_text())['rounds']['1']
  assert rec['L4'] == 'L4_SKIPPED_CF_NOT_REPRESENTATIVE'
  assert rec['L4_coverage']['min_coverage'] == 0.8 and set(rec['L4_coverage']['below']) == set(CONTRACTS)
  for c in CONTRACTS:
    budget = rec['cf'][c]['budget']
    assert budget['simulated'] == 0 and budget['unavailable'] == budget['undecided'] > 0
    assert rec['cf'][c]['coverage'] == rec['L4_coverage']['below'][c] < 0.8
  freeze = next(a for a in argvs[start:] if a[0] == 'freeze')
  assert freeze[freeze.index('--value-anchor-of') + 1] == str(lay.policy(0))
  assert '--cf' not in freeze and '--beta-q' not in freeze
  assert ts_run.progress(lay.learner(1))['L4'] == 0
  assert 'value' not in changed_groups(params(lay.policy(0)), params(lay.policy(1)))
  pi0, pi1 = (json.loads((lay.policy(k) / 'policy.json').read_text()) for k in (0, 1))
  for key in ('decision.beta_q', 'decision.kappa_q'):
    assert pi1['ts'][key] == pi0['ts'][key]
  assert pi1['teaching']['value_anchor']['status'] == rec['value_anchor']['status'] == (
      'L4_SKIPPED_ANCHOR_KEPT')
  assert pi1['lineage']['value_kept_from'] == str(lay.policy(0))

  # An explicit exploration config, in a new run that goes on from pi_0 of
  # this one (its policy, exams and experience linked and only read): the
  # same pairs below the threshold now teach L4, and the round says so.
  before = files(lay.root)
  blay = FT.Layout(tmp_path / 'branch')
  links = [(blay.policy(0), lay.policy(0)), (blay.root / 'eval' / 'pi_0000', lay.root / 'eval' / 'pi_0000')]
  links += [(blay.experience(blay.policy(0), kind, c), lay.experience(lay.policy(0), kind, c))
            for kind in ('collect', 'cf') for c in CONTRACTS]
  for link, target in links:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
  data = yaml.safe_load(config.read_text())
  data['paths']['run_root'] = 'branch'
  data['iteration']['cf_coverage'] = 'exploratory'
  # pi_0 never trained L4 here: its L4 optimizer warms up from a zero rate,
  # so the first update moves no weight and the second one does.
  data['iteration']['value_updates'] = 2
  explore = tmp_path / 'explore.yaml'
  explore.write_text(yaml.safe_dump(data))
  calls.clear()
  start = len(argvs)
  FT.main(['iterate', '--config', str(explore)])
  assert calls[:3] == ['train-value', 'train-policy', 'freeze']
  assert not {'collect', 'check', 'cf'} & set(calls)
  branch = json.loads(blay.state.read_text())['rounds']['1']
  assert branch['L4'] == 'S3_Q_TARGETS_EXPLORATORY' and branch['cf'] == rec['cf']
  assert branch['L4_coverage'] == dict(rec['L4_coverage'], cf_coverage='exploratory')
  freeze = next(a for a in argvs[start:] if a[0] == 'freeze')
  assert '--value-anchor-of' not in freeze and [freeze[i + 1] for i, a in enumerate(freeze)
                                                if a == '--cf'] == [
      str(blay.experience(blay.policy(0), 'cf', c)) for c in CONTRACTS]
  assert ts_run.progress(blay.learner(1))['L4'] == 2
  assert 'value' in changed_groups(params(lay.policy(0)), params(blay.policy(1)))
  assert files(lay.root) == before
  data['iteration']['cf_coverage'] = 'yes'
  explore.write_text(yaml.safe_dump(data))
  with pytest.raises(ValueError, match='cf_coverage is representative or exploratory'):
    FT.load_config(explore)


def test_orchestrator_trains_l2_on_the_carry_of_an_l1_continued_by_hand(pik, two, stores,
                                                                        tmp_path, monkeypatch):
  """formal_v1_l1_l2_10x_continue.yaml: a copied foundation checkpoint whose
  L1 went on by hand after its L2. With cumulative budgets the orchestrator
  leaves L1 alone, builds the stores from the continued L1 brain and trains
  the remaining L2 and then L3 on them. L2 and L3 leave the market brain
  alone, so the stores are pi_0000's deployed market memory."""
  base = ['--preset', 'small', '--device', 'cpu', '--batch-size', '2', '--batch-length', str(T),
          '--holdout', '0', '--corpus', str(two[0]), '--corpus', str(two[1])]
  source = tmp_path / 'source'
  ts_run.main(['train-market', '--run', str(source), '--init-from', str(pik['run']), *base,
               '--updates', '1'])
  ts_run.main(['train-outcome', '--run', str(source), *base, '--updates', '1'])
  root = tmp_path / 'corpora'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  lay = FT.Layout(tmp_path / 'run')
  shutil.copytree(source, lay.foundation)
  ts_run.main(['train-market', '--run', str(lay.foundation), '--init-from', str(source), *base,
               '--updates', '2'])
  assert ts_run.progress(lay.foundation) == {'L1': 3, 'L2': 1, 'L3': 0, 'L4': 0}
  config = tmp_path / 'continue.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run', init_from=str(ts_run.latest(source))),
      student=dict(preset='small', device='cpu', seed=1, batch_size=2, batch_length=T,
                   log_every=1, window_start='carry'),
      foundation=dict(market_updates=3, outcome_updates=2, policy_updates=1, save_every=1),
      evaluation=dict(random_seeds=1, smoke_steps=20))))
  calls, argvs = [], []
  monkeypatch.setattr(FT, '_call', in_process(calls, argvs=argvs))
  FT.main(['pretrain', '--config', str(config)])
  assert calls == ['market-carry', 'market-carry', 'train-outcome', 'train-policy', 'freeze',
                   'exam', 'exam']
  for argv in argvs[2:4]:
    assert argv[argv.index('--updates') + 1] == '1'
    assert argv[argv.index('--window-start') + 1] == 'carry'
    assert argv[argv.index('--carry-root') + 1] == str(lay.carry)
  assert ts_run.progress(lay.foundation) == {'L1': 3, 'L2': 2, 'L3': 1, 'L4': 0}
  report = json.loads(sorted(lay.foundation.glob('ckpt/*/train_outcome.json'))[-1].read_text())
  assert (report['window_start'], report['carry_root']) == ('carry', str(lay.carry))
  deployed = tmp_path / 'deployed'
  ts_run.main(['market-carry', '--policy', str(lay.policy(0)), '--corpus', str(two[0]),
               '--corpus', str(two[1]), '--out', str(deployed), '--device', 'cpu'])
  for name in CONTRACTS:
    built = TC.MarketCarry(lay.carry / name).arrays
    for k in TC.CARRY_KEYS:
      np.testing.assert_array_equal(built[k], TC.MarketCarry(deployed / name).arrays[k],
                                    err_msg=f'{name} {k}')
  # Not the memory of the brain before L1 went on.
  assert not np.array_equal(built['deter'], TC.MarketCarry(stores / CONTRACTS[1]).arrays['deter'])


def test_orchestrator_finishes_the_exam_of_a_frozen_candidate_first(pik, two, tmp_path,
                                                                    monkeypatch):
  """C4 review R1: pi_1 is frozen and one of its two DEV exams is done. The
  rerun takes the other one only and no round 2, though max_rounds allows
  it: finishing round 1 is the round of this call."""
  from wp02_fixture import native_factory
  rows = []
  for shift, name in ((3, 'DEV1'), (4, 'DEV2')):
    _, env, _ = native_factory(tmp_path / name, shift=shift)
    rows.append(dict(contract=name, role='validation', eligible_for_training=False,
                     path=str(env.parent / 'market.csv'), env_config=str(env)))
  (tmp_path / 'pool.json').write_text(json.dumps(dict(files=rows)))
  (tmp_path / 'corpora').mkdir()
  for folder, name in zip(two, CONTRACTS):
    (tmp_path / 'corpora' / name).symlink_to(folder)
  config = tmp_path / 'dev.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(markets='pool.json', corpus_root='corpora', run_root='run'),
      dev=dict(contracts=['DEV1', 'DEV2']), student=dict(preset='small', device='cpu'),
      iteration=dict(max_rounds=2), evaluation=dict(random_seeds=2))))
  lay = FT.Layout(tmp_path / 'run')
  for k in (0, 1):
    shutil.copytree(pik['policy'], lay.policy(k))
  first = lay.exam(lay.policy(1), 'DEV', 'DEV1')
  E.main(['--policy', str(lay.policy(1)), '--out', str(first), '--pool', str(tmp_path / 'pool.json'),
          '--contract', 'DEV1', '--random-seeds', '2'])
  done = files(first)
  calls = []
  monkeypatch.setattr(FT, '_call', in_process(calls))
  FT.main(['iterate', '--config', str(config), '--rounds', '1'])
  assert calls == ['exam'] and files(first) == done
  report = json.loads((lay.exam(lay.policy(1), 'DEV', 'DEV2') / 'exam.json').read_text())
  assert report['label'] == 'DEV' and report['complete']
  state = json.loads(lay.state.read_text())
  assert state['step'] == 'done' and state['best_dev']['policy'] == str(lay.policy(1))
  assert not lay.policy(2).exists() and not (lay.root / 'experience').exists()


def test_exam_is_teacher_free_and_names_its_market(pik, two, tmp_path):
  from wp02_fixture import native_factory
  policy = str(pik['policy'])
  # TRAIN smoke: the first minutes of a TRAIN corpus, never complete.
  E.main(['--policy', policy, '--out', str(tmp_path / 'smoke'), '--corpus', str(two[0]),
          '--steps', '60', '--random-seeds', '2'])
  report = json.loads((tmp_path / 'smoke' / 'exam.json').read_text())
  assert report['label'] == 'TRAIN_SMOKE' and report['teacher_free'] and not report['trained']
  assert not report['complete'] and report['source']['fold'] == 'TRAIN'
  student, random = report['student'], report['random']
  assert student['minutes'] == 60 and [x['minutes'] for x in random['per_seed']] == [60, 60]
  assert random['probability'] == pytest.approx(student['adds'] / max(student['legal'], 1))
  again = E._random_job(dict(config=str(two[0] / 'inputs' / 'env.yaml'),
                             probability=random['probability'], seed=1, steps=60))
  assert again == random['per_seed'][1]

  # DEV: only a VALIDATION row of the pool, run to its true terminal.
  _, env, _ = native_factory(tmp_path / 'dev_market', shift=3)
  pool = tmp_path / 'pool.json'
  pool.write_text(json.dumps(dict(files=[
      dict(contract=name, role=role, eligible_for_training=role == 'train',
           path=str(env.parent / 'market.csv'), env_config=str(env))
      for name, role in (('DEV', 'validation'), ('TRN', 'train'), ('TST', 'test'))])))
  for name in ('TRN', 'TST'):
    with pytest.raises(ValueError, match='not a VALIDATION row'):
      E.main(['--policy', policy, '--out', str(tmp_path / name), '--pool', str(pool),
              '--contract', name])
  E.main(['--policy', policy, '--out', str(tmp_path / 'dev'), '--pool', str(pool),
          '--contract', 'DEV', '--random-seeds', '2'])
  report = json.loads((tmp_path / 'dev' / 'exam.json').read_text())
  assert report['label'] == 'DEV' and report['complete']
  assert report['source']['row']['role'] == 'validation'
  assert [x['minutes'] for x in report['random']['per_seed']] == [report['student']['minutes']] * 2
  assert set(report['advantage']) == set(E.SCORES)


def test_paper_trading_runs_paper_rows_only_and_chains_them(pik, tmp_path):
  from wp02_fixture import native_factory
  from dreamerv3.thinking_student import paper as P
  policy = str(pik['policy'])
  rows = []
  for shift, name, role, eligible in ((3, 'P1', 'paper', False), (4, 'P2', 'paper', False),
                                      (5, 'TRN', 'train', True), (6, 'DEV', 'validation', False),
                                      (7, 'PTR', 'paper', True)):
    _, env, _ = native_factory(tmp_path / name, shift=shift)
    rows.append(dict(contract=name, role=role, eligible_for_training=eligible,
                     path=str(env.parent / 'market.csv'), env_config=str(env),
                     first_timestamp=f'2030-0{shift}-01'))
  pool = tmp_path / 'paper_pool.json'
  pool.write_text(json.dumps(dict(files=rows)))
  for name in ('TRN', 'DEV', 'PTR'):
    with pytest.raises(ValueError, match='not a PAPER row'):
      E.main(['--policy', policy, '--out', str(tmp_path / 'x' / name), '--paper-pool', str(pool),
              '--contract', name])
  with pytest.raises(SystemExit):  # one market source per exam
    E.main(['--policy', policy, '--out', str(tmp_path / 'x' / 'both'), '--paper-pool', str(pool),
            '--pool', str(pool), '--contract', 'P1'])
  with pytest.raises(SystemExit):  # other TP / SL distances are for PAPER only
    E.main(['--policy', policy, '--out', str(tmp_path / 'x' / 'dev'), '--pool', str(pool),
            '--contract', 'DEV', '--stop-loss-points', '5'])
  for name in ('P1', 'P2'):
    E.main(['--policy', policy, '--out', str(tmp_path / 'run' / 'pi' / name), '--paper-pool', str(pool),
            '--contract', name, '--random-seeds', '2'])
  exams = [json.loads((tmp_path / 'run' / 'pi' / n / 'exam.json').read_text()) for n in ('P1', 'P2')]
  assert all(e['label'] == 'PAPER' and e['complete'] and e['teacher_free'] for e in exams)
  assert 'stop_loss_points' not in yaml.safe_load(
      (tmp_path / 'run' / 'pi' / 'P1' / 'inputs' / 'env.yaml').read_text())['entry']
  # Paper rules TP 6 / SL 5 (user 2026-10-30): the env, the student and the random accounts.
  rules = ['--take-profit-points', '6', '--stop-loss-points', '5']
  for name in ('P1', 'P2'):
    E.main(['--policy', policy, '--out', str(tmp_path / 'run' / 'sl5' / name), '--paper-pool', str(pool),
            '--contract', name, '--random-seeds', '2', *rules])
  out = tmp_path / 'run' / 'sl5' / 'P2'
  entry = yaml.safe_load((out / 'inputs' / 'env.yaml').read_text())['entry']
  assert (entry['take_profit_points'], entry['stop_loss_points']) == (6, 5)
  assert json.loads((out / 'exam.json').read_text())['source']['rules'] == dict(
      take_profit_points=6, stop_loss_points=5)
  from gym_trading_env.research import ts_v1_corpus as GC
  native = GC.native_env(out / 'inputs' / 'env.yaml', market=False)
  assert (native.take_profit_points, native.stop_loss_points) == (6, 5)
  assert native.reward_function.scorer.config.take_profit_points == 6.
  assert native._compute_stop_loss_price(100, 'long') == 95
  assert native._compute_take_profit_price(100, None, 'short') == 94
  for name in ('P1', 'P2'):
    collect = tmp_path / 'run' / 'sl5' / name / 'collect'
    assert not ts_run.check_collection(
        collect, inputs=tmp_path / 'run' / 'sl5' / name / 'inputs' / 'env.yaml')['point_violations']
    with (collect / 'state.pkl').open('rb') as f:
      for t in pickle.load(f)['run']['env']['attributions']:
        move = (1 if t['side'] == 'LONG' else -1) * (float(t['exit_price']) - float(t['entry_price']))
        assert t['reason'] != 'TAKE_PROFIT' or move >= 6
        assert t['reason'] != 'STOP_LOSS' or move <= -5
  shutil.copytree(tmp_path / 'run' / 'pi' / 'P1', tmp_path / 'mixed' / 'pi' / 'P1')
  shutil.copytree(out, tmp_path / 'mixed' / 'pi' / 'P2')
  # Only the paper rows are chained, in time order.
  paper_only = tmp_path / 'paper_only.json'
  paper_only.write_text(json.dumps(dict(files=rows[1::-1])))
  with pytest.raises(ValueError, match='different TP / SL rules'):
    P.main(['--run', str(tmp_path / 'mixed'), '--pool', str(paper_only), 'pi'])
  P.main(['--run', str(tmp_path / 'run'), '--pool', str(paper_only), 'pi', 'sl5'])
  report = json.loads((tmp_path / 'run' / 'paper_report.json').read_text())
  assert report['contracts'] == ['P1', 'P2']
  whole, parts = report['policies']['pi']['whole'], report['policies']['pi']['contracts']
  pnl = [e['student']['net_pnl'] for e in exams]
  assert whole['net_pnl'] == pytest.approx(sum(pnl))
  assert sum(whole['monthly'].values()) == pytest.approx(sum(pnl))
  assert whole['minutes'] == sum(e['student']['minutes'] for e in exams)
  seeds = [sum(e['random']['per_seed'][s]['net_pnl'] for e in exams) for s in (0, 1)]
  assert whole['random_net_pnl']['mean'] == pytest.approx(np.mean(seeds))
  assert [p['contract'] for p in parts] == ['P1', 'P2'] and 'equity' not in parts[0]
  assert report['policies']['pi']['rules'] == dict(take_profit_points=5, stop_loss_points=30)
  assert report['policies']['sl5']['rules'] == dict(take_profit_points=6, stop_loss_points=5)
  assert (tmp_path / 'run' / 'paper_report.md').read_text().startswith('# Paper trading')


def test_exam_numbers_have_defined_denominators():
  rows = [dict(cursor=c, legal=legal, executed=a, reward=r, equity=e) for c, legal, a, r, e in (
      (0, True, 1, 0., 100.), (1, False, 0, 0., 90.), (2, False, 0, 2., 120.), (3, True, 0, -1., 110.))]
  trades = [dict(reason='TAKE_PROFIT', entry_cursor=0, exit_cursor=2)]
  m = E.metrics(rows, trades, [], {0: 'd1', 1: 'd1', 2: 'd2', 3: 'd2'}, 100.)
  assert (m['minutes'], m['trading_days'], m['reward']) == (4, 2, 1.)
  assert (m['reward_per_minute'], m['reward_per_day']) == (0.25, 0.5)
  assert (m['legal'], m['adds'], m['add_rate'], m['tp'], m['sl']) == (2, 1, 0.5, 1, 0)
  assert m['occupancy'] == 0.5 and m['tp_hold_minutes']['mean'] == 2.
  assert (m['max_drawdown'], m['net_pnl']) == (10., 10.)


def test_thinking_student_configs_live_in_one_place(tmp_path):
  root = Path(FT.__file__).parents[2] / 'configs' / 'thinking_student'
  assert TSC.STUDENT == root / 'student.yaml'
  for name in ('formal_v1.yaml', 'c4_smoke_2c.yaml'):
    cfg = FT.load_config(root / name)
    assert cfg['student']['preset'] == 'size50m' and cfg['dev']['contracts'] == ['CZCE.FG409']
    assert cfg['paths']['markets'] == str(root / 'markets' / 'pool.json')
    assert cfg['paths']['init_f1'] and not cfg['paths']['init_from']
  pool = json.loads((root / 'markets' / 'pool.json').read_text())['files']
  roles = [(r['role'], r['eligible_for_training']) for r in pool]
  assert (roles.count(('train', True)), roles.count(('validation', False)), len(roles)) == (18, 1, 19)
  paper = json.loads((root / 'markets' / 'paper_pool.json').read_text())['files']
  assert [(r['role'], r['eligible_for_training']) for r in paper] == [('paper', False)] * 5
  assert not {r['contract'] for r in paper} & {r['contract'] for r in pool}
  for row in pool + paper:
    env = yaml.safe_load((root / 'markets' / row['env_config']).read_text())
    assert env['entry']['contract'] == row['contract']
    assert Path(env['trading']['data_path']) == Path(row['path'])
    for key in ('contract_config', 'strategy_config'):
      assert not Path(env['entry'][key]).is_absolute(), (row['contract'], key)
      assert (root / 'markets' / 'envs' / env['entry'][key]).exists()
  bad = tmp_path / 'bad.yaml'
  bad.write_text(yaml.safe_dump(dict(paths=dict(corpus_root='c', run_root='r'),
                                     iteration=dict(rounds=3))))
  with pytest.raises(ValueError, match='unknown keys'):
    FT.load_config(bad)


def test_f1_from_before_the_macd_branch_is_widened_exactly():
  from dreamerv3.thinking_student import transfer as TR
  rng = np.random.default_rng(0)
  source = {'enc/fusion/kernel': rng.normal(size=(6, 3)).astype(np.float32),
            'enc/context15/bias': np.ones(4, np.float32)}
  target = {'market_enc/fusion/kernel': np.zeros((8, 3), np.float32),
            'market_enc/context15/bias': np.zeros(4, np.float32),
            'decision/final/bias': np.zeros(2, np.float32)}
  values, reused, sliced, widened, initialized = TR.plan(source, target)
  assert (reused, sliced, widened) == (['market_enc/context15/bias'], [], ['market_enc/fusion/kernel'])
  kernel = values['market_enc/fusion/kernel']
  np.testing.assert_array_equal(kernel[:6], source['enc/fusion/kernel'])
  assert not kernel[6:].any() and initialized == ['decision/final/bias']
