"""Thinking Student V1 S5: teaching items with dated feedback (B1), the
OFFLINE_TRAIN release, the correct and remedial windows (B2) and L6 in the
formal orchestrator -- one small closed loop on the two synthetic TRAIN
contracts of the C4 tests.

Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dreamerv3.tests.test_ts_v1_c4 import (CONTRACTS, changed_groups, in_process, params,
                                           pik, stores, two)  # noqa: F401 (fixtures)
from dreamerv3.tests.test_ts_v1_s2 import DEVICE, T
from dreamerv3.thinking_student import corpus as TC
from dreamerv3.thinking_student import feedback as F
from dreamerv3.thinking_student import formal_train as FT
from dreamerv3.thinking_student import labels as L
from dreamerv3.thinking_student import run as ts_run
from dreamerv3.thinking_student import value as V


def steps(collection):
  return [json.loads(line) for f in sorted((Path(collection) / 'steps').glob('*.jsonl'))
          for line in f.open()]


def flags(argv, name):
  """The value after every occurrence of `name`."""
  return [argv[i + 1] for i, a in enumerate(argv) if a == name]


def test_teaching_items_join_the_answers_with_dated_feedback(pik, two):
  from gym_trading_env.research import ts_v1_corpus as C
  name, corpus = CONTRACTS[0], two[0]
  collection, cf = pik['collect'][name], pik['cf'][name]
  kappa = V.cf_kappa([pik['cf'][c] for c in CONTRACTS], floor=0.05)
  items, summary = F.build(collection, corpus, cf, kappa, outcome_error_prob=0.5,
                           decision_error_margin=0., outcome_correct_prob=0.5)
  rows, facts = steps(collection), C.load_arrays(corpus / 'facts')
  graded = C.load_arrays(corpus / 'graded')
  source = json.loads((corpus / 'source.json').read_text())
  t, ffl, end = items['cursor'], items['flat_funded_legal'], source['flat_end']
  assert len(t) == len(rows) and t[-1] + 1 == end and summary['complete']
  for k in ('p_tp_h', 'p_open_h', 'z', 'delta_q', 'reason_pref'):
    np.testing.assert_allclose(items[f'answer_{k}'], [r[k] for r in rows], rtol=1e-6, err_msg=k)
  # TEACHER_ADD_FACT: the fact of the cursor at every flat funded legal
  # decision whose ADD fills, known once the trade exits or is censored.
  assert ffl.any() and facts['eligible'][t[ffl]].all()
  np.testing.assert_array_equal(items['fact_valid'], ffl & facts['filled'][t])
  on = items['fact_valid']
  np.testing.assert_array_equal(items['fact_ready_at'][on], facts['observed_until'][t[on]])
  assert (items['fact_ready_at'][on] > t[on]).all() and (items['fact_ready_at'][~on] == -1).all()
  # The realized first ending is the one class the label allows; a trade
  # censored before H abstains.
  allowed = L.outcome_labels(facts, C.KINDS, graded)['outcome_allowed'][t]
  for i in np.flatnonzero(on):
    e = items['realized'][i]
    if e < 0:
      assert C.KINDS[facts['kind'][t[i]]] == 'RIGHT_CENSORED' and allowed[i].sum() > 1
    else:
      only = int(np.flatnonzero(allowed[i])[0])
      assert allowed[i].sum() == 1 and (only == L.OPEN_AT_H if e == 3 else only // L.NBINS == e)
      assert items['p_realized'][i] == items[f'answer_{F.ANSWER_P[e]}'][i]
  # VERIFIED_CF_NATIVE: Delta G known at the coupling (a TERMINAL pair at
  # the end), the returns at the end.
  pairs = [json.loads(line) for line in (cf / 'pairs.jsonl').open()]
  assert items['cf_fork'].sum() == len(pairs) and pairs
  for p in pairs:
    i = p['cursor']
    assert items['cf_valid'][i] == (p['delta_g'] is not None)
    if p['delta_g'] is not None:
      assert items['cf_ready_at'][i] == (end if p['coupled_at'] is None else p['coupled_at'])
      assert items['cf_return_ready_at'][i] == end and items['cf_delta_g'][i] == p['delta_g']
  # STUDENT_EXECUTED_NATIVE: the student's own trade agrees with the fact.
  adds = np.flatnonzero((items['executed'] == 1) & items['fact_valid'])
  assert len(adds) and (items['executed_valid'][adds] | items['executed_open'][adds]).all()
  for i in adds:
    if items['executed_valid'][i]:
      kind = ts_run.FACT_KINDS.get(str(items['executed_reason'][i]), 'OTHER_EXIT')
      assert kind == C.KINDS[facts['kind'][t[i]]]
      assert items['executed_ready_at'][i] == facts['exit_cursor'][t[i]]
  # HINDSIGHT_GRADED_TEACHING: the graded demonstrations (ADD for GOOD or
  # better, WAIT for POOR), known at the exit or the end of the room window.
  demo = items['demo_valid']
  np.testing.assert_array_equal(demo, ffl & graded['valid'][t])
  np.testing.assert_array_equal(items['demo_action'][demo], graded['action'][t[demo]])
  np.testing.assert_array_equal(items['demo_ready_at'][demo], graded['ready_at'][t[demo]])
  assert demo.any() and (items['demo_ready_at'][demo] > t[demo]).all()
  assert summary['sources']['demo']['source'] == 'HINDSIGHT_GRADED_TEACHING'
  # Errors and correct answers never meet; with one probability threshold
  # they split the known facts.
  known = items['realized'] >= 0
  np.testing.assert_array_equal(items['outcome_error'] ^ items['outcome_correct'], known)
  assert not (items['decision_error'] & items['decision_correct']).any()
  for name, found in (('errors', F.errors(items)), ('correct', F.correct(items))):
    assert {k: summary[name][k] for k in found} == {k: len(v) for k, v in found.items()}
  assert summary['sources']['fact'] == dict(source='TEACHER_ADD_FACT', items=int(on.sum()),
                                            cursors=int(on.sum()))
  assert (summary['teaching_order'], summary['time_ordered']) == (
      'OFFLINE_TRAIN', 'TIME_ORDERED_NOT_SUPPORTED')
  assert summary['thresholds']['kappa_q'] == pytest.approx(kappa[0])


def test_only_offline_train_releases_feedback(pik, two, tmp_path):
  name = CONTRACTS[1]
  args = ['--collection', str(pik['collect'][name]), '--corpus', str(two[1]),
          '--cf', str(pik['cf'][name])]
  out = tmp_path / 'items'
  F.main([*args, '--out', str(out)])
  items, summary = F.load(out)
  released = F.release(items, 'OFFLINE_TRAIN')
  for key in F.SOURCES:
    np.testing.assert_array_equal(released[key], items[f'{key}_valid'])
    # Every released item keeps the market cursor it is known from.
    ready = items[f'{key}_ready_at'] if key != 'cf' else items['cf_ready_at']
    assert (ready[released[key]] >= 0).all()
  # OFFLINE_TRAIN releases feedback known only later than the decision.
  assert (items['fact_ready_at'][released['fact']] > items['cursor'][released['fact']]).all()
  with pytest.raises(NotImplementedError, match='TIME_ORDERED_NOT_SUPPORTED'):
    F.release(items, 'TIME_ORDERED')
  with pytest.raises(ValueError, match='Unknown teaching order'):
    F.release(items, 'ONLINE')
  with pytest.raises(NotImplementedError, match='TIME_ORDERED_NOT_SUPPORTED'):
    F.main([*args, '--out', str(tmp_path / 'timed'), '--order', 'TIME_ORDERED'])
  assert not (tmp_path / 'timed').exists()
  config = tmp_path / 'timed.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run'),
      iteration=dict(l6=dict(from_round=1, order='TIME_ORDERED')))))
  with pytest.raises(ValueError, match='TIME_ORDERED_NOT_SUPPORTED'):
    FT.load_config(config)
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run'), iteration=dict(l6=dict(begin=1)))))
  with pytest.raises(ValueError, match=r'unknown keys \[.begin.\] in iteration.l6'):
    FT.load_config(config)
  # Teaching items come from TRAIN material only.
  dev = tmp_path / 'dev'
  shutil.copytree(two[1], dev)
  source = json.loads((dev / 'source.json').read_text())
  (dev / 'source.json').write_text(json.dumps({**source, 'fold': 'VALIDATION'}))
  with pytest.raises(ValueError, match='not TRAIN material'):
    F.build(pik['collect'][name], dev)
  # Existing summaries are regenerated for current CF and thresholds.
  before = (out / 'summary.json').read_text()
  with pytest.raises(ValueError, match='apart'):  # errors above the correct threshold
    F.main([*args, '--out', str(out), '--outcome-error-prob', '0.9'])
  assert (out / 'summary.json').read_text() == before
  F.main([*args, '--out', str(out), '--outcome-error-prob', '0.9', '--outcome-correct-prob', '0.95'])
  assert (out / 'summary.json').read_text() != before
  thresholds = F.load(out)[1]['thresholds']
  assert (thresholds['outcome_error_prob'], thresholds['outcome_correct_prob']) == (0.9, 0.95)
  # Replacing a CF pair with an unfinished result must remove its old error /
  # pair supervision, even though this feedback folder already has a summary.
  changed = tmp_path / 'changed_cf'
  changed.mkdir()
  rows = [json.loads(line) for line in (pik['cf'][name] / 'pairs.jsonl').open()]
  victim = next(r for r in rows if r['delta_g'] is not None)
  victim.update(status='NOT_COMPARABLE', reason='CAP', g=None, delta_g=None)
  victim['g_wait' if victim['reference_action'] else 'g_add'] = None
  (changed / 'pairs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
  meta = json.loads((pik['cf'][name] / 'meta.json').read_text())
  meta['valid_pairs'] -= 1
  (changed / 'meta.json').write_text(json.dumps(meta))
  new_args = [str(changed) if a == str(pik['cf'][name]) else a for a in args]
  F.main([*new_args, '--out', str(out), '--kappa-cf', str(pik['cf'][name])])
  updated, summary = F.load(out)
  assert not updated['cf_valid'][victim['cursor']]
  assert not updated['decision_error'][victim['cursor']]
  assert summary['sources']['cf']['items'] == meta['valid_pairs']


def test_error_tags_follow_the_answers_and_the_pairs():
  """Hand-made facts, answers and pairs of five decisions."""
  from gym_trading_env.research import ts_v1_corpus as C
  kind = {k: i for i, k in enumerate(C.KINDS)}
  facts = dict(filled=np.array([1, 1, 1, 1, 1, 0], bool),
               kind=np.array([kind['TP'], kind['SL'], kind['OTHER_EXIT'],
                              kind['RIGHT_CENSORED'], kind['TP'], -1], np.int8),
               held_minutes=np.array([10, 20, 30, 100, L.HORIZON + 1, 0], np.int32),
               exit_cursor=np.array([11, 22, 33, -1, 3000, -1], np.int32),
               observed_until=np.array([11, 22, 33, 104, 3000, -1], np.int32))
  answers = {f'answer_{k}': np.array(v, np.float32) for k, v in dict(
      p_tp_h=[.05, .3, .3, .3, .3, .3], p_sl_h=[.3, .5, .3, .01, .3, .3],
      p_other_h=[.3, .1, .09, .3, .3, .3], p_open_h=[.35, .1, .31, .39, .2, .1]).items()}
  ffl = np.ones(6, bool)
  out = F.fact_items(facts, C.KINDS, np.arange(6), ffl, answers, 0.1, 0.5)
  assert out['realized'].tolist() == [0, 1, 2, -1, 3, -1]  # OPEN_AT_H after H; censored abstains
  assert out['outcome_error'].tolist() == [True, False, True, False, False, False]
  assert out['outcome_correct'].tolist() == [False, True, False, False, False, False]
  assert np.isnan(out['p_realized'][[3, 5]]).all() and out['p_realized'][4] == np.float32(.2)
  low = F.fact_items(facts, C.KINDS, np.arange(6), ffl, answers, 0.3, 0.2)
  assert low['outcome_error'][4] and low['outcome_correct'][[1, 4]].all()

  items = dict(cursor=np.arange(6), executed=np.array([1, 0, 1, 0, 0, 1], np.int8),
               demo_valid=np.array([1, 1, 1, 1, 0, 1], bool),
               demo_action=np.array([0, 1, 0, 0, -1, 1], np.int8))
  pairs = [
      dict(cursor=0, reference_action=1, status='VALID_COMPLETE', reason='COUPLED',
           coupled_at=7, g_add=-1., g_wait=0., delta_g=-1.),     # ADD worse by 1
      dict(cursor=1, reference_action=0, status='VALID_COMPLETE', reason='COUPLED',
           coupled_at=5, g_add=.3, g_wait=0., delta_g=.3),       # WAIT worse by 0.3
      dict(cursor=2, reference_action=1, status='NOT_COMPARABLE', reason='CAP',
           coupled_at=None, g_add=None, g_wait=None, delta_g=None),
      dict(cursor=3, reference_action=0, status='VALID_COMPLETE', reason='COUPLED',
           coupled_at=6, g_add=.2, g_wait=.2, delta_g=1e-17),    # equal branches
      dict(cursor=4, reference_action=0, status='VALID_COMPLETE', reason='TERMINAL',
           coupled_at=None, g_add=2., g_wait=0., delta_g=2.),    # WAIT worse by 2
      dict(cursor=5, reference_action=1, status='VALID_COMPLETE', reason='COUPLED',
           coupled_at=8, g_add=1.5, g_wait=0., delta_g=1.5)]     # ADD better by 1.5
  out = F.pair_items(pairs, items, end=10, kappa=0.5, margin=1.0, correct_margin=1.0)
  assert out['cf_valid'].tolist() == [True, True, False, True, True, True]
  assert out['cf_ready_at'].tolist() == [7, 5, -1, 6, 10, 8]
  assert out['cf_return_ready_at'].tolist() == [10, 10, -1, 10, 10, 10]
  np.testing.assert_allclose(out['cf_regret'][[0, 1, 3, 4, 5]], [1., .3, 1e-17, 2., -1.5])
  assert out['decision_error'].tolist() == [True, False, False, False, True, False]
  assert out['decision_correct'].tolist() == [False, False, False, False, False, True]
  # The graded teacher demonstrates the better action at 0 (WAIT), 1 and 5
  # (ADD); an equal pair has no better action; 4 has no demonstration.
  assert out['teachers_agree'].tolist() == [True, True, False, False, False, True]
  untagged = F.pair_items(pairs, items, 10, None, 1.0, 1.0)
  assert not untagged['decision_error'].any() and not untagged['decision_correct'].any()
  assert F.pair_items(pairs, items, 10, .5, 1., 4.)['decision_correct'].sum() == 0
  invalid = dict(pairs[2], g_add=2., g_wait=0., delta_g=2.)
  assert not F.pair_items([invalid], items, 10, .5, 1., 1.)['cf_valid'].any()
  with pytest.raises(ValueError, match='forks from action'):
    F.pair_items([dict(pairs[0], reference_action=0)], items, 10, .5, 1., 1.)
  full = dict(items, **out, fact_valid=np.zeros(6, bool), outcome_error=np.zeros(6, bool),
              outcome_correct=np.zeros(6, bool), executed_valid=np.zeros(6, bool))
  e, c = F.errors(full), F.correct(full)
  assert e['decision'].tolist() == [0, 4] and e['demo'].tolist() == [0]
  assert c['decision'].tolist() == [5] and c['demo'].tolist() == [5]


def test_correct_and_remedial_windows_hold_their_rows(two):
  corpora = [TC.TSCorpus(c, behaviors=[TC.LABEL_ACCOUNT]) for c in two]
  kw = dict(weight='outcome', burnin=2, accounts=[TC.LABEL_ACCOUNT])
  errors, correct = {}, {}
  for name, corpus in zip(CONTRACTS, corpora):
    b = corpus.behaviors[TC.LABEL_ACCOUNT]
    on = np.flatnonzero(corpus.labelled[TC.LABEL_ACCOUNT]
                        & (corpus.outcome['outcome_valid'][b['cursor']] > 0))
    on = on[on >= 2]  # after the burn-in of the first window
    errors[name] = b['cursor'][on[[0, len(on) // 2, -1]]]
    correct[name] = b['cursor'][on[[1, len(on) // 3]]]
  windows = TC.MultiCorpusWindows(corpora, holdout=0, seed=0, remedial=errors,
                                  remedial_fraction=0.25, correct=correct, correct_fraction=0.25,
                                  **kw)
  for _ in range(5):
    data = windows.sample(8, T)
    for i, (contract, name, start) in enumerate(windows.last):
      part = windows.parts[windows.contracts.index(contract)]
      shown = part.corpus.behaviors[name]['cursor'][start + 2:start + T]
      # round(0.25 x 8) remedial, then round(0.25 x 8) correct windows lead the batch.
      if i < 4:
        assert np.isin(shown, (errors if i < 2 else correct)[contract]).any()
      # A remedial or correct window is the natural window of its rows.
      natural = part.corpus.window(name, start, start + T)
      for k, v in natural.items():
        if k not in ('demo_valid', 'is_first'):
          np.testing.assert_array_equal(data[k][i], v, err_msg=k)
      assert data['is_first'][i, 0] and not data['is_first'][i, 1:].any()
  report = windows.curriculum()
  assert (report['windows'], report['remedial_windows'], report['remedial_share']) == (40, 10, .25)
  assert (report['correct_windows'], report['correct_share']) == (10, .25)
  assert report['remedial_rows'] == 6 and 0 < report['remedial_rows_seen'] <= 6
  assert report['correct_rows'] == 4 and 0 < report['correct_rows_seen'] <= 4
  assert report['unique_label_rows'] <= report['label_rows']
  # A cursor of either set must carry a label of the sampling weight.
  b = corpora[0].behaviors[TC.LABEL_ACCOUNT]
  bare = b['cursor'][np.flatnonzero(~(corpora[0].labelled[TC.LABEL_ACCOUNT]
                                      & (corpora[0].outcome['outcome_valid'][b['cursor']] > 0)))[0]]
  for kind in ('remedial', 'correct'):
    with pytest.raises(ValueError, match=f'{kind} cursors carry no outcome label'):
      TC.MultiCorpusWindows(corpora, holdout=0, **{kind: {CONTRACTS[0]: [bare]}},
                            **{f'{kind}_fraction': .25}, **kw)
  with pytest.raises(ValueError, match='exceed the batch'):
    TC.MultiCorpusWindows(corpora, holdout=0, remedial=errors, remedial_fraction=.6,
                          correct=correct, correct_fraction=.5, **kw)
  # Without rows of either set the batches are the natural ones.
  plain = TC.MultiCorpusWindows(corpora, holdout=0, seed=3, **kw)
  empty = TC.MultiCorpusWindows(corpora, holdout=0, seed=3, remedial={c: [] for c in CONTRACTS},
                                remedial_fraction=.5, correct={c: [] for c in CONTRACTS},
                                correct_fraction=.5, **kw)
  for _ in range(3):
    a, b = plain.sample(8, T), empty.sample(8, T)
    assert plain.last == empty.last
    for k in a:
      np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_q_windows_boost_correct_and_wrong_decision_rows(pik):
  cf = [pik['cf'][c] for c in CONTRACTS]
  plain = V.MultiQWindows(cf, 'train', holdout=0, burnin=2)
  kappa, source = V.cf_kappa(cf)
  assert kappa == pytest.approx(plain.kappa()[0])
  assert source['pairs'] == plain.pair_counts()['total']
  part = plain.parts[0]
  fork = np.flatnonzero(part.data['q_cf_valid'] > 0)
  boosted = V.MultiQWindows(cf, 'train', holdout=0, burnin=2,
                            remedial={CONTRACTS[0]: part.cursor[fork[:1]]}, remedial_boost=4.,
                            correct={CONTRACTS[0]: part.cursor[fork[1:2]]}, correct_boost=2.)
  w0, w1 = (np.diff(np.concatenate([[0.], q.parts[0]._starts(T)])) for q in (plain, boosted))
  start = np.arange(len(w0))
  covers = lambda row: (start + 2 <= row) & (row < start + T)
  np.testing.assert_allclose(w1 - w0, np.where(covers(fork[0]), 3 * (1 + 16), 0)
                             + np.where(covers(fork[1]), 1 * (1 + 16), 0), rtol=1e-6)
  np.testing.assert_array_equal(boosted.parts[1]._starts(T), plain.parts[1]._starts(T))
  idle = np.flatnonzero(part.data['q_cf_valid'] == 0)[0]
  for kind in ('remedial', 'correct'):
    with pytest.raises(ValueError, match='valid ADD/WAIT pair'):
      V.MultiQWindows(cf, 'train', holdout=0, **{kind: {CONTRACTS[0]: part.cursor[[idle]]}})
  with pytest.raises(ValueError, match='both an error and correct'):
    V.MultiQWindows(cf, 'train', holdout=0, remedial={CONTRACTS[0]: part.cursor[fork[:1]]},
                    correct={CONTRACTS[0]: part.cursor[fork[:1]]})
  boosted.sample(8, T)
  report = boosted.curriculum()
  assert report['windows'] == 8 and report['remedial_rows'] == report['correct_rows'] == 1
  assert (report['remedial_boost'], report['correct_boost']) == (4., 2.)
  assert report['contracts'][CONTRACTS[1]]['remedial_rows'] == 0


def test_l6_rounds_teach_offline_train_feedback(pik, two, tmp_path, monkeypatch):
  """Round 1 without L6, round 2 with it: teaching items of pi_1, the L2
  review before L4, correct and remedial windows in L4 and L5,
  OFFLINE_TRAIN in the round record and the lineage; a rerun keeps the
  items; the exam reads no teacher material."""
  from gym_trading_env.research import ts_v1_corpus as C
  root = tmp_path / 'corpora'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  config = tmp_path / 's5.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run', init_from=str(pik['run'])),
      student=dict(preset='small', device=DEVICE, seed=1, batch_size=2, batch_length=T,
                   log_every=1, window_start='carry'),
      foundation=dict(policy_updates=1, save_every=1),
      iteration=dict(max_rounds=2, cf_wait_forks=1, value_updates=1, prior_refresh_updates=1,
                     save_every=1, kappa_floor=0.05,
                     l6=dict(from_round=2, outcome_review_updates=2, correct_fraction=.5,
                             remedial_fraction=.5, remedial_boost=4., outcome_error_prob=.5,
                             decision_error_margin=0.)),
      evaluation=dict(random_seeds=1, smoke_steps=20))))
  lay = FT.Layout(tmp_path / 'run')
  calls, argvs = [], []

  def refuse(*args, **kwargs):
    raise AssertionError(f'The exam read teacher material: {args}')

  def guarded(inner):
    def call(module, argv, log):
      if module != 'exam':
        return inner(module, argv, log)
      with monkeypatch.context() as m:
        for owner, attr in ((C, 'load_arrays'), (V, 'load_collection'), (F, 'load')):
          m.setattr(owner, attr, refuse)
        return inner(module, argv, log)
    return call

  monkeypatch.setattr(FT, '_call', guarded(in_process(calls, argvs=argvs)))
  FT.main(['pretrain', '--config', str(config)])
  FT.main(['iterate', '--config', str(config)])
  state = json.loads(lay.state.read_text())
  assert state['rounds']['1']['L6'] == 'L6_SKIPPED_BY_CONFIG' and 'l6' not in state['rounds']['1']
  assert not (lay.root / 'experience' / 'pi_0000' / 'feedback').exists()

  # Round 2 stops in its L2 review; the rerun refreshes local teaching items.
  calls.clear()
  monkeypatch.setattr(FT, '_call', guarded(in_process(calls, fail='train-outcome', argvs=argvs)))
  with pytest.raises(RuntimeError, match='train-outcome failed'):
    FT.main(['iterate', '--config', str(config)])
  assert calls == ['collect', 'collect', 'check', 'check', 'cf', 'cf', 'feedback', 'feedback',
                   'train-outcome']
  pi1 = lay.policy(1)
  items = {c: lay.experience(pi1, 'feedback', c) for c in CONTRACTS}
  kept = {c: F.load(f)[0] for c, f in items.items()}
  calls.clear()
  start = len(argvs)
  monkeypatch.setattr(FT, '_call', guarded(in_process(calls, argvs=argvs)))
  # Below the learner's checkpoints are read after its freeze, so they stay
  # here (the pruning of a run: test_ts_v1_c4).
  monkeypatch.setattr(FT, '_prune', lambda lay: None)
  FT.main(['iterate', '--config', str(config)])
  assert calls == ['feedback', 'feedback', 'train-outcome', 'train-value', 'train-policy', 'freeze', 'exam', 'exam']
  for c, folder in items.items():
    for key, value in F.load(folder)[0].items():
      np.testing.assert_array_equal(value, kept[c][key])
  review, value, policy, freeze = argvs[start + 2:start + 6]
  for argv in (review, value, policy, freeze):
    assert flags(argv, '--feedback') == [str(items[c]) for c in CONTRACTS]
  assert flags(review, '--remedial-fraction') == flags(policy, '--remedial-fraction') == ['0.5']
  assert flags(review, '--correct-fraction') == flags(policy, '--correct-fraction') == ['0.5']
  assert flags(value, '--remedial-boost') == flags(value, '--correct-boost') == ['4.0']
  assert flags(review, '--updates') == ['2']

  state = json.loads(lay.state.read_text())
  record = state['rounds']['2']
  assert record['check'] == {c: 'ok' for c in CONTRACTS}
  assert record['L6'] == 'S5_FEEDBACK_CURRICULUM'
  assert (record['l6']['order'], record['l6']['time_ordered']) == (
      'OFFLINE_TRAIN', 'TIME_ORDERED_NOT_SUPPORTED')
  assert record['l6']['L2_review_updates'] == 2 and set(record['feedback']) == set(CONTRACTS)
  learner = lay.learner(2)
  assert ts_run.progress(learner) == {'L1': 0, 'L2': 2, 'L3': 1, 'L4': 1}
  # The learner inherits pi_1, and its L2 review moves only the L2 groups.
  inherit = json.loads((learner / 'inherit.json').read_text())
  assert Path(inherit['source']).resolve() == (pi1 / 'agent.pkl').resolve()
  reviewed = next(f for f in sorted((learner / 'ckpt').iterdir())
                  if (f / 'progress.json').exists()
                  and json.loads((f / 'progress.json').read_text())['updates'] == {
                      'L1': 0, 'L2': 2, 'L3': 0, 'L4': 0})
  assert changed_groups(params(pi1), params(reviewed)) <= {'outcome', 'fusion'}
  report = json.loads((reviewed / 'train_outcome.json').read_text())
  assert report['feedback']['kind'] == 'outcome'
  assert report['feedback']['teaching_order'] == 'OFFLINE_TRAIN'
  # round(0.5 x 2) windows per update around each set that has rows.
  rows = report['feedback']['cursors']
  for kind in ('remedial', 'correct'):
    has = sum(rows[kind].values()) > 0
    assert report['curriculum'][f'{kind}_windows'] == (2 if has else 0), kind
  assert sum(rows['remedial'].values()) > 0
  assert (report['window_start'], report['carry_root']) == ('carry', str(lay.carry))
  frozen = json.loads((lay.policy(2) / 'policy.json').read_text())
  assert frozen['lineage']['feedback'] == [str(items[c]) for c in CONTRACTS]
  assert (frozen['lineage']['teaching_order'], frozen['lineage']['time_ordered']) == (
      'OFFLINE_TRAIN', 'TIME_ORDERED_NOT_SUPPORTED')
  assert frozen['teaching']['train_outcome']['feedback']['policy'] == pi1.name
  # The market brain is untouched from pi_1 to pi_2: the stores stay valid.
  assert changed_groups(params(pi1), params(lay.policy(2))) <= {
      'outcome', 'fusion', 'value', 'decision'}
  assert FT.status(FT.load_config(config), lay)['experience']['pi_0001']['feedback'] == {
      c: True for c in CONTRACTS}

  # Round 3 without representative pairs (cf_budget 0 and cf_min_coverage;
  # one-step probes with no continuation quota): the L2
  # review still runs on the outcome errors of the facts, L4 is skipped, L5
  # takes no decision remediation from the same CF, and pi_3 keeps the value
  # brain and anchor of pi_2.
  data = yaml.safe_load(config.read_text())
  data['iteration'].update(max_rounds=3, cf_max_steps=2, cf_probe_steps=1, cf_budget=0, cf_min_coverage=0.8)
  config.write_text(yaml.safe_dump(data))
  monkeypatch.setattr(V.CF, 'coupling_floor', lambda env: np.inf)
  calls.clear()
  start = len(argvs)
  FT.main(['iterate', '--config', str(config)])
  assert calls[:11] == ['collect', 'collect', 'check', 'check', 'cf', 'cf', 'feedback', 'feedback',
                        'train-outcome', 'train-policy', 'freeze']
  assert 'train-value' not in calls
  record = json.loads(lay.state.read_text())['rounds']['3']
  assert (record['L4'], record['L6']) == (
      'L4_SKIPPED_CF_NOT_REPRESENTATIVE', 'S5_FEEDBACK_CURRICULUM')
  assert record['l6']['decision_feedback'] == 'SKIPPED_CF_NOT_REPRESENTATIVE'
  assert set(record['l6']['feedback_windows']) == {'train_outcome'}
  pi2 = lay.policy(2)
  items = {c: lay.experience(pi2, 'feedback', c) for c in CONTRACTS}
  for c in CONTRACTS:
    summary = json.loads((items[c] / 'summary.json').read_text())
    budget = record['cf'][c]['budget']
    assert budget['simulated'] == 0 and budget['unavailable'] == budget['undecided'] > 0
    assert summary['abstained']['cf_unavailable_budget'] == budget['unavailable']
    assert summary['abstained']['cf_not_comparable'] == 0
    assert summary['cf_coverage'] == record['cf'][c]['coverage'] < 0.8
    assert summary['sources']['cf']['items'] == record['cf'][c]['valid_pairs']
  review, policy, freeze = (next(a for a in argvs[start:] if a[0] == name)
                            for name in ('train-outcome', 'train-policy', 'freeze'))
  assert flags(review, '--feedback') == flags(freeze, '--feedback') == [
      str(items[c]) for c in CONTRACTS]
  assert '--feedback' not in policy and '--remedial-fraction' not in policy
  assert '--correct-fraction' not in policy
  assert freeze[freeze.index('--value-anchor-of') + 1] == str(pi2) and '--cf' not in freeze
  pi3 = json.loads((lay.policy(3) / 'policy.json').read_text())
  assert pi3['teaching']['value_anchor']['status'] == 'L4_SKIPPED_ANCHOR_KEPT'
  assert (pi3['lineage']['value_kept_from'], pi3['lineage']['teaching_order']) == (
      str(pi2), 'OFFLINE_TRAIN')
  assert 'value' not in changed_groups(params(pi2), params(lay.policy(3)))


def test_l6_uses_the_correct_set_by_default_and_errors_on_request(tmp_path, two, pik):
  l6 = FT.DEFAULTS['iteration']['l6']
  assert FT._feedback_args(l6) == (['--correct-fraction', .25], ['--correct-boost', 4.])
  both = dict(l6, remedial_fraction=.25, remedial_boost=2.)
  assert FT._feedback_args(both) == (['--correct-fraction', .25, '--remedial-fraction', .25],
                                     ['--correct-boost', 4., '--remedial-boost', 2.])
  assert FT._feedback_args(dict(l6, correct_fraction=0., correct_boost=1.)) == ([], [])
  config = tmp_path / 'l6.yaml'
  for bad in (dict(correct_fraction=.6, remedial_fraction=.5), dict(correct_boost=0.),
              dict(outcome_error_prob=.6, outcome_correct_prob=.5),
              dict(decision_correct_margin=-1.)):
    config.write_text(yaml.safe_dump(dict(paths=dict(corpus_root='c', run_root='r'),
                                          iteration=dict(l6=bad))))
    with pytest.raises(ValueError, match='iteration.l6'):
      FT.load_config(config)
  # Errors and correct answers never meet.
  F.check_thresholds(.5, 0., .5, 0.)
  for bad in ((.6, 1., .5, 1.), (.1, -1., .5, 1.), (.1, 1., 1.5, 1.)):
    with pytest.raises(ValueError, match='apart|at least 0'):
      F.check_thresholds(*bad)
  # run.py: either set needs --feedback, and the two shares fit one batch.
  base = ['train-outcome', '--run', str(tmp_path / 'run'), '--corpus', str(two[0])]
  for extra in (['--correct-fraction', '.25'], ['--correct-boost', '2'],
                ['--feedback', str(tmp_path), '--correct-fraction', '.6',
                 '--remedial-fraction', '.5']):
    with pytest.raises(SystemExit):
      ts_run.main([*base, *extra])
  # A set that is off passes no cursors.
  folders = []
  for name in CONTRACTS:
    out = tmp_path / 'feedback' / name
    F.main(['--collection', str(pik['collect'][name]), '--corpus',
            str(two[CONTRACTS.index(name)]), '--cf', str(pik['cf'][name]), '--out', str(out)])
    folders.append(out)
  args = SimpleNamespace(feedback=folders, correct_fraction=.25, remedial_fraction=0.,
                         correct_boost=1., remedial_boost=1.)
  sets, report = ts_run._feedback(args, 'outcome')
  assert sets['remedial'] is None and set(sets['correct']) == set(CONTRACTS)
  assert report['on'] == dict(remedial=False, correct=True)
  assert set(report['cursors']['remedial']) == set(CONTRACTS)
  sets, report = ts_run._feedback(args, 'decision')
  assert sets == dict(remedial=None, correct=None)


def test_round_learners_draw_their_own_windows(pik, two, stores, tmp_path, monkeypatch):
  """Every iteration round trains a new learner that inherits pi_k, so its
  stage progress starts at 0 again. Learners of different rounds must still
  draw their own L2 review and L5 windows; the same inheritance draws the
  same windows."""
  drawn = []

  class Recording(TC.MultiCorpusWindows):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      drawn.append([])

    def sample(self, batch, length):
      data = super().sample(batch, length)
      drawn[-1].append(list(self.last))
      return data

  monkeypatch.setattr(ts_run, 'MultiCorpusWindows', Recording)
  args = ['--preset', 'small', '--device', DEVICE, '--batch-size', '2', '--batch-length', str(T),
          '--holdout', '0', '--window-start', 'carry', '--carry-root', str(stores),
          '--corpus', str(two[0]), '--corpus', str(two[1]), '--updates', '2']
  for command in ('train-outcome', 'train-policy'):
    drawn.clear()
    runs = {k: tmp_path / command / k for k in ('a', 'b', 'c')}
    ts_run.main([command, '--run', str(runs['a']), '--init-from', str(pik['policy']), *args])
    ts_run.main([command, '--run', str(runs['b']), '--init-from', str(pik['policy']), *args])
    # pi_1: the learner of round a, frozen; round c inherits it.
    agent = ts_run.build(runs['a'], 'small', DEVICE)
    pi1 = ts_run.freeze(agent, runs['a'], 'pi1', 'small', list(two), out=tmp_path / command / 'pi1')
    ts_run.main([command, '--run', str(runs['c']), '--init-from', str(pi1), *args])
    first, again, next_round = drawn
    assert len(first) == 2 and first == again
    assert next_round != first
