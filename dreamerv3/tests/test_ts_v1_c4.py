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
  out = dict(policy=policy, collect={}, cf={})
  for corpus, name in zip(two, CONTRACTS):
    loaded, _ = ts_run.load_policy(policy, **QUIET)
    out['collect'][name], out['cf'][name] = root / 'collect' / name, root / 'cf' / name
    ts_run.collect(loaded, ts_run.corpus_env(corpus), out['collect'][name], 10 ** 6, chunk=64,
                   policy='pik', **QUIET)
    V.run_cf(policy, corpus, out['collect'][name], out['cf'][name], forks=4, workers=1, **QUIET)
  return out


def in_process(calls, fail=None):
  """formal_train._call without a subprocess; the first call of `fail` (a
  run.py command or the exam module) fails."""
  failed = set()

  def call(module, argv, log):
    argv = [str(a) for a in argv]
    name = argv[0] if module == 'run' else module
    calls.append(name)
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

  # The round stops at L4; the rerun keeps pi_0's collections and CF,
  # freezes pi_1 and stops at its first exam.
  monkeypatch.setattr(FT, '_call', in_process(calls, fail='train-value'))
  with pytest.raises(RuntimeError, match='train-value failed'):
    FT.main(['iterate', '--config', str(config)])
  state = json.loads(lay.state.read_text())
  assert (state['round'], state['step']) == (1, 'L4')
  assert state['rounds']['1']['collected'] == CONTRACTS
  experience = files(lay.root / 'experience')
  assert all((lay.experience(lay.policy(0), 'cf', c) / 'meta.json').exists() for c in CONTRACTS)
  calls.clear()
  monkeypatch.setattr(FT, '_call', in_process(calls, fail='exam'))
  with pytest.raises(RuntimeError, match='exam failed'):
    FT.main(['iterate', '--config', str(config)])
  assert calls == ['train-value', 'train-policy', 'freeze', 'exam']
  assert {p: v for p, v in files(lay.root / 'experience').items() if p in experience} == experience
  state = json.loads(lay.state.read_text())
  record = state['rounds']['1']
  assert (record['L5'], record['L6'], record['updates']) == (
      'S4_PRIOR_REFRESH', 'L6_NOT_IMPLEMENTED', {'L4': 2, 'L5': 2})
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
  for row in pool:
    env = yaml.safe_load((root / 'markets' / row['env_config']).read_text())
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
