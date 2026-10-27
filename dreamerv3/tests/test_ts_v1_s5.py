"""Thinking Student V1 S5: teaching items with dated feedback (B1), the
OFFLINE_TRAIN release, remediation windows (B2) and L6 in the formal
orchestrator -- one small closed loop on the two synthetic TRAIN contracts
of the C4 tests.

Run from gym-trading-env with PYTHONPATH=.:src:../dreamerv3:tests/entry_v1.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from dreamerv3.tests.test_ts_v1_c4 import (CONTRACTS, changed_groups, in_process, params,
                                           pik, two)  # noqa: F401 (fixtures)
from dreamerv3.tests.test_ts_v1_s2 import T
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
                           decision_error_margin=0.)
  rows, facts = steps(collection), C.load_arrays(corpus / 'facts')
  strict = C.load_arrays(corpus / 'strict')
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
  allowed = L.outcome_labels(facts, C.KINDS)['outcome_allowed'][t]
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
  # HINDSIGHT_ORACLE_TEACHING: valid, untied strict choices, known at the end.
  np.testing.assert_array_equal(items['strict_valid'], ffl & strict['valid'][t] & ~strict['tie'][t])
  assert (items['strict_ready_at'][items['strict_valid']] == end).all()
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
  # A built folder is kept as it is.
  before = (out / 'summary.json').read_text()
  F.main([*args, '--out', str(out), '--outcome-error-prob', '0.9'])
  assert (out / 'summary.json').read_text() == before


def test_error_tags_follow_the_answers_and_the_pairs():
  """Hand-made facts, answers and pairs of five decisions."""
  from gym_trading_env.research import ts_v1_corpus as C
  kind = {k: i for i, k in enumerate(C.KINDS)}
  facts = dict(filled=np.array([1, 1, 1, 1, 1, 0], bool),
               kind=np.array([kind['TP5'], kind['SL30'], kind['OTHER_EXIT'],
                              kind['RIGHT_CENSORED'], kind['TP5'], -1], np.int8),
               held_minutes=np.array([10, 20, 30, 100, L.HORIZON + 1, 0], np.int32),
               exit_cursor=np.array([11, 22, 33, -1, 3000, -1], np.int32),
               observed_until=np.array([11, 22, 33, 104, 3000, -1], np.int32))
  answers = {f'answer_{k}': np.array(v, np.float32) for k, v in dict(
      p_tp_h=[.05, .3, .3, .3, .3, .3], p_sl_h=[.3, .5, .3, .01, .3, .3],
      p_other_h=[.3, .1, .09, .3, .3, .3], p_open_h=[.35, .1, .31, .39, .2, .1]).items()}
  ffl = np.ones(6, bool)
  out = F.fact_items(facts, C.KINDS, np.arange(6), ffl, answers, 0.1)
  assert out['realized'].tolist() == [0, 1, 2, -1, 3, -1]  # OPEN_AT_H after H; censored abstains
  assert out['outcome_error'].tolist() == [True, False, True, False, False, False]
  assert np.isnan(out['p_realized'][[3, 5]]).all() and out['p_realized'][4] == np.float32(.2)
  assert F.fact_items(facts, C.KINDS, np.arange(6), ffl, answers, 0.3)['outcome_error'][4]

  items = dict(cursor=np.arange(5), executed=np.array([1, 0, 1, 0, 0], np.int8),
               strict_valid=np.array([1, 1, 1, 1, 0], bool),
               strict_action=np.array([0, 1, 0, 0, -1], np.int8))
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
           coupled_at=None, g_add=2., g_wait=0., delta_g=2.)]    # WAIT worse by 2
  out = F.pair_items(pairs, items, end=10, kappa=0.5, margin=1.0)
  assert out['cf_valid'].tolist() == [True, True, False, True, True]
  assert out['cf_ready_at'].tolist() == [7, 5, -1, 6, 10]
  assert out['cf_return_ready_at'].tolist() == [10, 10, -1, 10, 10]
  np.testing.assert_allclose(out['cf_regret'][[0, 1, 3, 4]], [1., .3, 1e-17, 2.])
  assert out['decision_error'].tolist() == [True, False, False, False, True]
  # Strict agrees with the better action at 0 (WAIT) and 1 (ADD); an equal
  # pair has no better action; 4 has no valid strict choice.
  assert out['teachers_agree'].tolist() == [True, True, False, False, False]
  assert not F.pair_items(pairs, items, 10, None, 1.0)['decision_error'].any()
  with pytest.raises(ValueError, match='forks from action'):
    F.pair_items([dict(pairs[0], reference_action=0)], items, 10, .5, 1.)
  full = dict(items, **out, fact_valid=np.zeros(5, bool), outcome_error=np.zeros(5, bool),
              executed_valid=np.zeros(5, bool))
  e = F.errors(full)
  assert e['decision'].tolist() == [0, 4] and e['demo'].tolist() == [0]


def test_remedial_windows_hold_their_error_rows(two):
  corpora = [TC.TSCorpus(c, behaviors=[TC.LABEL_ACCOUNT]) for c in two]
  kw = dict(weight='outcome', burnin=2, accounts=[TC.LABEL_ACCOUNT])
  errors = {}
  for name, corpus in zip(CONTRACTS, corpora):
    b = corpus.behaviors[TC.LABEL_ACCOUNT]
    on = np.flatnonzero(corpus.labelled[TC.LABEL_ACCOUNT]
                        & (corpus.outcome['outcome_valid'][b['cursor']] > 0))
    on = on[on >= 2]  # after the burn-in of the first window
    errors[name] = b['cursor'][on[[0, len(on) // 2, -1]]]
  windows = TC.MultiCorpusWindows(corpora, holdout=0, seed=0, remedial=errors,
                                  remedial_fraction=0.25, **kw)
  for _ in range(5):
    data = windows.sample(8, T)
    for i, (contract, name, start) in enumerate(windows.last):
      part = windows.parts[windows.contracts.index(contract)]
      shown = part.corpus.behaviors[name]['cursor'][start + 2:start + T]
      if i < 2:  # round(0.25 x 8) remedial windows lead the batch
        assert np.isin(shown, errors[contract]).any()
      # A remedial window is the natural window of its rows.
      natural = part.corpus.window(name, start, start + T)
      for k, v in natural.items():
        if k not in ('demo_valid', 'is_first'):
          np.testing.assert_array_equal(data[k][i], v, err_msg=k)
      assert data['is_first'][i, 0] and not data['is_first'][i, 1:].any()
  report = windows.curriculum()
  assert (report['windows'], report['remedial_windows'], report['remedial_share']) == (40, 10, .25)
  assert report['remedial_rows'] == 6 and 0 < report['remedial_rows_seen'] <= 6
  assert report['unique_label_rows'] <= report['label_rows']
  # An error cursor must carry a label of the sampling weight.
  b = corpora[0].behaviors[TC.LABEL_ACCOUNT]
  bare = b['cursor'][np.flatnonzero(~(corpora[0].labelled[TC.LABEL_ACCOUNT]
                                      & (corpora[0].outcome['outcome_valid'][b['cursor']] > 0)))[0]]
  with pytest.raises(ValueError, match='carry no outcome label'):
    TC.MultiCorpusWindows(corpora, holdout=0, remedial={CONTRACTS[0]: [bare]},
                          remedial_fraction=.25, **kw)
  # Without error rows the batches are the natural ones.
  plain = TC.MultiCorpusWindows(corpora, holdout=0, seed=3, **kw)
  empty = TC.MultiCorpusWindows(corpora, holdout=0, seed=3, remedial={c: [] for c in CONTRACTS},
                                remedial_fraction=.5, **kw)
  for _ in range(3):
    a, b = plain.sample(8, T), empty.sample(8, T)
    assert plain.last == empty.last
    for k in a:
      np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_q_remediation_boosts_decision_error_rows(pik):
  cf = [pik['cf'][c] for c in CONTRACTS]
  plain = V.MultiQWindows(cf, 'train', holdout=0, burnin=2)
  kappa, source = V.cf_kappa(cf)
  assert kappa == pytest.approx(plain.kappa()[0])
  assert source['pairs'] == plain.pair_counts()['total']
  part = plain.parts[0]
  fork = np.flatnonzero(part.data['q_cf_valid'] > 0)
  boosted = V.MultiQWindows(cf, 'train', holdout=0, burnin=2,
                            remedial={CONTRACTS[0]: part.cursor[fork[:1]]}, remedial_boost=4.)
  w0, w1 = (np.diff(np.concatenate([[0.], q.parts[0]._starts(T)])) for q in (plain, boosted))
  start, row = np.arange(len(w0)), fork[0]
  covers = (start + 2 <= row) & (row < start + T)
  np.testing.assert_allclose(w1 - w0, np.where(covers, 3 * (1 + 16), 0), rtol=1e-6)
  np.testing.assert_array_equal(boosted.parts[1]._starts(T), plain.parts[1]._starts(T))
  idle = np.flatnonzero(part.data['q_cf_valid'] == 0)[0]
  with pytest.raises(ValueError, match='valid ADD/WAIT pair'):
    V.MultiQWindows(cf, 'train', holdout=0, remedial={CONTRACTS[0]: part.cursor[[idle]]})
  boosted.sample(8, T)
  report = boosted.curriculum()
  assert report['windows'] == 8 and report['remedial_rows'] == 1
  assert report['contracts'][CONTRACTS[1]]['remedial_rows'] == 0


def test_l6_rounds_teach_offline_train_feedback(pik, two, tmp_path, monkeypatch):
  """Round 1 without L6, round 2 with it: teaching items of pi_1, the L2
  review before L4, remediation in L4 and L5, OFFLINE_TRAIN in the round
  record and the lineage; a rerun keeps the items; the exam reads no
  teacher material."""
  from gym_trading_env.research import ts_v1_corpus as C
  root = tmp_path / 'corpora'
  root.mkdir()
  for folder, name in zip(two, CONTRACTS):
    (root / name).symlink_to(folder)
  config = tmp_path / 's5.yaml'
  config.write_text(yaml.safe_dump(dict(
      paths=dict(corpus_root='corpora', run_root='run', init_from=str(pik['run'])),
      student=dict(preset='small', device='cpu', seed=1, batch_size=2, batch_length=T,
                   log_every=1, window_start='carry'),
      foundation=dict(policy_updates=1, save_every=1),
      iteration=dict(max_rounds=2, cf_wait_forks=1, value_updates=1, prior_refresh_updates=1,
                     save_every=1, kappa_floor=0.05,
                     l6=dict(from_round=2, outcome_review_updates=2, remedial_fraction=.5,
                             remedial_boost=4., outcome_error_prob=.5, decision_error_margin=0.)),
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

  # Round 2 stops in its L2 review; the rerun keeps the teaching items.
  calls.clear()
  monkeypatch.setattr(FT, '_call', guarded(in_process(calls, fail='train-outcome', argvs=argvs)))
  with pytest.raises(RuntimeError, match='train-outcome failed'):
    FT.main(['iterate', '--config', str(config)])
  assert calls == ['collect', 'collect', 'check', 'check', 'cf', 'cf', 'feedback', 'feedback',
                   'train-outcome']
  pi1 = lay.policy(1)
  items = {c: lay.experience(pi1, 'feedback', c) for c in CONTRACTS}
  kept = {c: (f / 'items.npz').read_bytes() for c, f in items.items()}
  calls.clear()
  start = len(argvs)
  monkeypatch.setattr(FT, '_call', guarded(in_process(calls, argvs=argvs)))
  FT.main(['iterate', '--config', str(config)])
  assert calls == ['train-outcome', 'train-value', 'train-policy', 'freeze', 'exam', 'exam']
  assert {c: (f / 'items.npz').read_bytes() for c, f in items.items()} == kept
  review, value, policy, freeze = argvs[start:start + 4]
  for argv in (review, value, policy, freeze):
    assert flags(argv, '--feedback') == [str(items[c]) for c in CONTRACTS]
  assert flags(review, '--remedial-fraction') == flags(policy, '--remedial-fraction') == ['0.5']
  assert flags(value, '--remedial-boost') == ['4.0'] and flags(review, '--updates') == ['2']

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
  assert report['curriculum']['remedial_windows'] == 2  # round(0.5 x 2) per update
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

  # Round 3 without representative pairs (cf_budget 0 with one-step probes
  # and cf_min_coverage; no cursor may couple by FUNDS, an unreachable
  # floor, the CF's): the L2
  # review still runs on the outcome errors of the facts, L4 is skipped, L5
  # takes no decision remediation from the same CF, and pi_3 keeps the value
  # brain and anchor of pi_2.
  data = yaml.safe_load(config.read_text())
  data['iteration'].update(max_rounds=3, cf_max_steps=2, cf_budget=0, cf_probe_steps=1,
                           cf_min_coverage=0.8)
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
  assert record['l6']['decision_remediation'] == 'SKIPPED_CF_NOT_REPRESENTATIVE'
  assert set(record['l6']['remediation']) == {'train_outcome'}
  pi2 = lay.policy(2)
  items = {c: lay.experience(pi2, 'feedback', c) for c in CONTRACTS}
  for c in CONTRACTS:
    summary = json.loads((items[c] / 'summary.json').read_text())
    budget = record['cf'][c]['budget']
    assert budget['full'] == 0 and budget['probed'] == budget['undecided'] > 0
    assert budget['unavailable'] + budget['probe_closed'] == budget['probed']
    assert summary['abstained']['cf_unavailable_budget'] == budget['unavailable'] > 0
    assert summary['abstained']['cf_not_comparable'] == 0
    assert summary['cf_coverage'] == record['cf'][c]['coverage'] < 0.8
    assert summary['thresholds']['decision_errors'] == 'DECISION_ERRORS_NOT_TAGGED'
  review, policy, freeze = (next(a for a in argvs[start:] if a[0] == name)
                            for name in ('train-outcome', 'train-policy', 'freeze'))
  assert flags(review, '--feedback') == flags(freeze, '--feedback') == [
      str(items[c]) for c in CONTRACTS]
  assert '--feedback' not in policy and '--remedial-fraction' not in policy
  assert freeze[freeze.index('--value-anchor-of') + 1] == str(pi2) and '--cf' not in freeze
  pi3 = json.loads((lay.policy(3) / 'policy.json').read_text())
  assert pi3['teaching']['value_anchor']['status'] == 'L4_SKIPPED_ANCHOR_KEPT'
  assert (pi3['lineage']['value_kept_from'], pi3['lineage']['teaching_order']) == (
      str(pi2), 'OFFLINE_TRAIN')
  assert 'value' not in changed_groups(params(pi2), params(lay.policy(3)))
