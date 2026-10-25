"""Thinking Student V1 teaching items (S5 B1): one item per decision of a
frozen pi_k's formal collection on one TRAIN contract, joining its answer
sheet with every feedback known about that decision. Each feedback keeps its
source, the market cursor from which it is known (ready_at, -1 if never) and
whether it is valid.

  python -m dreamerv3.thinking_student.feedback --collection DIR --corpus DIR \\
      --cf DIR [--kappa-cf DIR ...] --out DIR [--order OFFLINE_TRAIN]

TEACHER_ADD_FACT           the Native fact of "ADD now, then hold" from a flat
                           funded account (corpus facts/), at every decision
                           where the student was flat, funded and could ADD;
                           ready_at = facts.observed_until (exit or censor).
STUDENT_EXECUTED_NATIVE    the student's own Native trade of an executed,
                           filled ADD (the collection's attributions);
                           ready_at = its exit cursor; a trade still open at
                           the end of the collection is not ready.
VERIFIED_CF_NATIVE         the same-state ADD/WAIT pair of a fork (cf/): Delta G
                           is known at coupled_at (a TERMINAL pair: at the end),
                           the returns q_real / q_cf only at the end;
                           NOT_COMPARABLE and UNAVAILABLE_BUDGET (a fork the
                           CF budget left unsimulated) abstain.
HINDSIGHT_ORACLE_TEACHING  the strict TRAIN oracle's WAIT/ADD (corpus strict/)
                           where it is valid and not tied; ready_at = the end,
                           since it reads the whole TRAIN suffix.

The end of a contract is its flat_end, the market cursor after its last
decision. Errors (A6 L6, concept 8) come from valid feedback only:
- outcome error: the answer gave the realized first ending within H (TP,
  SL or another exit), or "open after H", a probability below
  `outcome_error_prob`; a trade censored before H abstains;
- decision error: the executed action's return falls below the other
  action's by more than `decision_error_margin` x kappa_Q, the std of Delta G
  pooled over the round's CF (value.cf_kappa, the statistic of the value
  anchor). teachers_agree marks a pair whose better action the strict oracle
  demonstrates; an equal pair (|Delta G| <= TIE) has no better action.
Chance cannot be told apart item by item; only the bounded remediation
share limits its weight (S5 B2). No market (F1) error is tagged: L6 never
trains L1.

Release (C3 5.3, 5.6): OFFLINE_TRAIN releases every valid feedback -- TRAIN
teaching material, whose teachers may read the whole TRAIN suffix. ready_at
is kept with every item but gates nothing. TIME_ORDERED (release only what
is ready by the student's simulated market time) is not implemented: it is
refused with TIME_ORDERED_NOT_SUPPORTED, and nothing learned from
OFFLINE_TRAIN items is a strict time-ordered online learning result. Only
TRAIN corpora are read (corpus_sources); DEV and TEST never enter.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import labels as L

OFFLINE_TRAIN = 'OFFLINE_TRAIN'
TIME_ORDERED = 'TIME_ORDERED'
TIME_ORDERED_NOT_SUPPORTED = 'TIME_ORDERED_NOT_SUPPORTED'
ORDERS = (OFFLINE_TRAIN, TIME_ORDERED)
SOURCES = dict(fact='TEACHER_ADD_FACT', executed='STUDENT_EXECUTED_NATIVE',
               cf='VERIFIED_CF_NATIVE', strict='HINDSIGHT_ORACLE_TEACHING')
# The realized first ending of a filled fact: an event within H, or open after H.
REALIZED = (*L.EVENTS, 'OPEN_AT_H')
ANSWER_P = ('p_tp_h', 'p_sl_h', 'p_other_h', 'p_open_h')
TIE = 1e-9  # |Delta G| of equal branches up to float rounding
# Round 1 of FORMAL_V1_CARRY_20261024 (pi_0000 on FG001/FG005/FG009): below
# 0.2 nearly every SL fact is an error (1011 of 1185), a class balance rather
# than a mistake; below 0.1 398 of 18502 (SL, OPEN_AT_H and FG005 TP). kappa_Q
# 0.559 over 614 pairs, 101 executed actions worse by more than kappa_Q.
DEFAULTS = dict(outcome_error_prob=0.1, decision_error_margin=1.0)
ITEMS, SUMMARY = 'items.npz', 'summary.json'


def check_order(order):
  """The teaching order of a curriculum: OFFLINE_TRAIN only."""
  if order == TIME_ORDERED:
    raise NotImplementedError(
        f'{TIME_ORDERED_NOT_SUPPORTED}: releasing feedback by its ready_at before the '
        "student's market time is not implemented; only OFFLINE_TRAIN is")
  if order != OFFLINE_TRAIN:
    raise ValueError(f'Unknown teaching order {order!r}, one of {ORDERS}')
  return order


def release(items, order=OFFLINE_TRAIN):
  """{source key: items whose feedback the teaching order releases}; under
  OFFLINE_TRAIN every valid one, whatever its ready_at."""
  check_order(order)
  return {name: np.asarray(items[f'{name}_valid'], bool) for name in SOURCES}


def errors(items, order=OFFLINE_TRAIN):
  """Cursors of the released errors: outcome errors (L2 review), decision
  errors (L4) and the decision errors whose better action the strict
  teacher demonstrates (L5)."""
  on, cursor = release(items, order), np.asarray(items['cursor'])
  decision = on['cf'] & items['decision_error']
  return dict(outcome=cursor[on['fact'] & items['outcome_error']], decision=cursor[decision],
              demo=cursor[decision & on['strict'] & items['teachers_agree']])


def load(folder):
  """(items, summary) of one feedback folder."""
  folder = Path(folder)
  summary = json.loads((folder / SUMMARY).read_text())
  with np.load(folder / ITEMS) as data:
    return {k: data[k] for k in data.files}, summary


def summaries(folders):
  """summary.json of feedback folders of one frozen policy on distinct
  contracts; anything else is refused."""
  out = [json.loads((Path(f) / SUMMARY).read_text()) for f in folders]
  contracts = [s['contract'] for s in out]
  if len(set(contracts)) != len(contracts):
    raise ValueError(f'One contract has two feedback folders: {contracts}')
  policies = {(s['policy'], s['cf_policy']) for s in out}
  if len(policies) != 1:
    raise ValueError(f'Feedback of one frozen policy is needed, got {sorted(policies, key=str)}')
  return out


def remedial(folders, kind):
  """{contract: cursors} of one error kind ('outcome', 'decision' or
  'demo') over feedback folders of one frozen policy, each released by the
  order it was built for."""
  summaries(folders)
  out = {}
  for folder in folders:
    items, summary = load(folder)
    out[summary['contract']] = errors(items, summary['teaching_order'])[kind]
  return out


def _collection(folder):
  """Step rows, the replay arrays the items need (not the market) and the
  run state of a formal pi_k collection."""
  folder = Path(folder)
  rows = [json.loads(line) for f in sorted((folder / 'steps').glob('*.jsonl'))
          for line in f.open()]
  parts = []
  for f in sorted((folder / 'replay').glob('*.npz')):
    with np.load(f) as z:
      parts.append({k: z[k] for k in z.files
                    if k in ('position_state', 'terminal', 'mu_prob') or k.startswith('answer_')})
  arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
  if len(rows) != len(arrays['terminal']):
    raise ValueError(f'{folder}: {len(rows)} step rows for {len(arrays["terminal"])} replay rows')
  if any(r['mode'] != 'policy' or r['forced'] or r['explored'] for r in rows):
    raise ValueError(f'{folder}: teaching items need the formal pi_k trajectory')
  with (folder / 'state.pkl').open('rb') as f:
    run = pickle.load(f)['run']
  return rows, arrays, run, json.loads((folder / 'meta.json').read_text())


def build(collection, corpus, cf=None, kappa=None, outcome_error_prob=0.1,
          decision_error_margin=1.0):
  """(items, summary): columns over the decision rows of the collection.
  `kappa` is (kappa_Q, its source) or None, which tags no decision error
  (DECISION_ERRORS_NOT_TAGGED)."""
  from gym_trading_env.research import ts_v1_corpus as C
  from .corpus import corpus_sources
  source = corpus_sources([corpus])[0]
  rows, arrays, run, meta = _collection(collection)
  n, end = len(rows), int(source['flat_end'])
  cursor = np.array([r['cursor'] for r in rows], np.int64)
  if not np.array_equal(cursor, np.arange(n)) or rows[0]['time'] != source['first_time']:
    raise ValueError(f'{collection} is no collection of {source["contract"]} '
                     'from its first market minute')
  facts = C.load_arrays(Path(corpus) / 'facts')
  strict = C.load_arrays(Path(corpus) / 'strict')
  items = dict(cursor=cursor, step=np.array([r['step'] for r in rows], np.int64),
               executed=np.array([r['executed'] for r in rows], np.int8),
               proposed=np.array([r['proposed'] for r in rows], np.int8),
               mu_prob=arrays['mu_prob'].astype(np.float32),
               **{k: v.astype(np.float32) for k, v in arrays.items() if k.startswith('answer_')})
  equity = np.array([r['equity'] for r in rows])
  legal = np.array([r['mask'][1] > 0.5 for r in rows])
  # As the corpus labels: flat, funds floor kept, ADD legal, not the terminal row.
  ffl = ((arrays['position_state'][:, 1] == 0) & (equity >= source['funds_floor'])
         & legal & ~arrays['terminal'])
  bad = np.flatnonzero(ffl & ~facts['eligible'][cursor])
  if len(bad):
    raise ValueError(f'{source["contract"]}: ADD legal at {len(bad)} flat funded decisions '
                     f'without an eligible fact, first cursor {cursor[bad[0]]}')
  items['flat_funded_legal'] = ffl
  items.update(fact_items(facts, C.KINDS, cursor, ffl, items, outcome_error_prob))
  items.update(executed_items(rows, run['env']))
  items.update(strict_items(strict, cursor, ffl, end))
  cf_meta, pairs = None, []
  if cf is not None:
    cf_meta = json.loads((Path(cf) / 'meta.json').read_text())
    if Path(cf_meta['collection']).resolve() != Path(collection).resolve():
      raise ValueError(f'{cf} holds the CF of {cf_meta["collection"]}, not of {collection}')
    pairs = [json.loads(line) for line in (Path(cf) / 'pairs.jsonl').open()]
  items.update(pair_items(pairs, items, end, kappa and kappa[0], decision_error_margin))
  complete = bool(run['done'] and run['env']['terminated'] and cursor[-1] + 1 == end)
  summary = _summary(items, source, meta, collection, corpus, cf, cf_meta, kappa, complete,
                     outcome_error_prob, decision_error_margin)
  return items, summary


def fact_items(facts, kinds, cursor, ffl, answers, threshold):
  """TEACHER_ADD_FACT of the flat funded legal decisions; `kinds` names the
  facts' kind codes (gym ts_v1_corpus.KINDS)."""
  filled = ffl & facts['filled'][cursor]
  kind = np.where(filled, facts['kind'][cursor], -1).astype(np.int8)
  held = np.where(filled, facts['held_minutes'][cursor], 0).astype(np.int32)
  realized = np.full(len(cursor), -1, np.int8)
  for e, name in enumerate(L.EVENTS):
    realized[filled & (kind == kinds.index(name)) & (held <= L.HORIZON)] = e
  realized[filled & (held > L.HORIZON)] = REALIZED.index('OPEN_AT_H')
  known = realized >= 0
  p = np.stack([answers[f'answer_{k}'] for k in ANSWER_P], -1)
  p_realized = np.where(known, p[np.arange(len(p)), np.maximum(realized, 0)], np.nan)
  return dict(
      fact_valid=filled, fact_kind=kind, fact_held=held,
      fact_exit_cursor=np.where(filled, facts['exit_cursor'][cursor], -1).astype(np.int32),
      fact_ready_at=np.where(filled, facts['observed_until'][cursor], -1).astype(np.int32),
      realized=realized, p_realized=p_realized.astype(np.float32),
      outcome_error=known & (np.where(known, p_realized, 1.) < threshold))


def executed_items(rows, env):
  """STUDENT_EXECUTED_NATIVE: the Native trade of each executed, filled ADD,
  the entry the env opened at the next cursor (a ts_v1 ADD fills there)."""
  n = len(rows)
  closed = {t['entry_cursor']: t for t in env['attributions']}
  still = {e['entry_cursor'] for e in env['entries'].values()}
  out = dict(executed_valid=np.zeros(n, bool), executed_open=np.zeros(n, bool),
             executed_reason=np.full(n, '', '<U24'), executed_held=np.zeros(n, np.int32),
             executed_net_pnl=np.full(n, np.nan), executed_ready_at=np.full(n, -1, np.int32))
  for i, r in enumerate(rows):
    if r['executed'] != 1 or r['reject_reason']:
      continue
    entry = r['cursor'] + 1
    if entry in closed:
      t = closed[entry]
      out['executed_valid'][i] = True
      out['executed_reason'][i] = t['reason']
      out['executed_held'][i] = t['held_market_bars']
      out['executed_net_pnl'][i] = float(t['net_trade_pnl'])
      out['executed_ready_at'][i] = t['exit_cursor']
    elif entry in still:
      out['executed_open'][i] = True
    else:
      raise ValueError(f'The filled ADD at cursor {r["cursor"]} has no Native trade from {entry}')
  return out


def strict_items(strict, cursor, ffl, end):
  """HINDSIGHT_ORACLE_TEACHING of the flat funded legal decisions."""
  valid = ffl & strict['valid'][cursor] & ~strict['tie'][cursor]
  return dict(strict_valid=valid,
              strict_action=np.where(valid, strict['action'][cursor], -1).astype(np.int8),
              strict_ready_at=np.where(valid, end, -1).astype(np.int32))


def pair_items(pairs, items, end, kappa, margin):
  """VERIFIED_CF_NATIVE of the fork rows; decision errors need kappa."""
  n = len(items['cursor'])
  out = dict(cf_fork=np.zeros(n, bool), cf_valid=np.zeros(n, bool),
             cf_status=np.full(n, '', '<U24'), cf_reason=np.full(n, '', '<U24'),
             cf_g_add=np.full(n, np.nan), cf_g_wait=np.full(n, np.nan),
             cf_delta_g=np.full(n, np.nan), cf_regret=np.full(n, np.nan),
             cf_ready_at=np.full(n, -1, np.int32), cf_return_ready_at=np.full(n, -1, np.int32),
             teachers_agree=np.zeros(n, bool), decision_error=np.zeros(n, bool))
  for p in pairs:
    i = p['cursor']
    if items['executed'][i] != p['reference_action']:
      raise ValueError(f'The CF pair at cursor {i} forks from action {p["reference_action"]}, '
                       f'the collection executed {items["executed"][i]}')
    out['cf_fork'][i] = True
    out['cf_status'][i], out['cf_reason'][i] = p['status'], p['reason']
    if p['status'] != 'VALID_COMPLETE' or p['delta_g'] is None:
      continue  # NOT_COMPARABLE and UNAVAILABLE_BUDGET abstain
    dg = p['delta_g']
    out['cf_valid'][i] = True
    out['cf_g_add'][i], out['cf_g_wait'][i], out['cf_delta_g'][i] = p['g_add'], p['g_wait'], dg
    # The other action's return minus the executed one's.
    out['cf_regret'][i] = dg if p['reference_action'] == 0 else -dg
    out['cf_ready_at'][i] = end if p['coupled_at'] is None else p['coupled_at']
    out['cf_return_ready_at'][i] = end
    out['teachers_agree'][i] = abs(dg) > TIE and bool(items['strict_valid'][i]) and (
        items['strict_action'][i] == int(dg > 0))
  if kappa is not None:
    out['decision_error'] = out['cf_valid'] & (np.where(
        out['cf_valid'], out['cf_regret'], 0.) > margin * kappa)
  return out


def _summary(items, source, meta, collection, corpus, cf, cf_meta, kappa, complete,
             outcome_error_prob, decision_error_margin):
  def count(name):
    valid = items[f'{name}_valid']
    return dict(source=SOURCES[name], items=int(valid.sum()),
                cursors=len(np.unique(items['cursor'][valid])))

  e = errors(items)
  by = {name: int((items['outcome_error'] & (items['realized'] == k)).sum())
        for k, name in enumerate(REALIZED)}
  return dict(
      contract=source['contract'], fold=source['fold'], policy=meta['policy'],
      collection=str(Path(collection).resolve()), corpus=str(Path(corpus).resolve()),
      cf=cf and str(Path(cf).resolve()), cf_policy=cf_meta and cf_meta['policy'],
      complete=complete, end=int(source['flat_end']), decisions=len(items['cursor']),
      flat_funded_legal=int(items['flat_funded_legal'].sum()),
      teaching_order=OFFLINE_TRAIN, time_ordered=TIME_ORDERED_NOT_SUPPORTED,
      release='OFFLINE_TRAIN: every valid feedback of TRAIN material; ready_at kept, gates nothing',
      sources={name: count(name) for name in SOURCES},
      abstained=dict(
          fact_censored_before_h=int((items['fact_valid'] & (items['realized'] < 0)).sum()),
          fact_not_filled=int((items['flat_funded_legal'] & ~items['fact_valid']).sum()),
          executed_open_at_end=int(items['executed_open'].sum()),
          cf_not_comparable=int((items['cf_status'] == 'NOT_COMPARABLE').sum()),
          cf_unavailable_budget=int((items['cf_status'] == 'UNAVAILABLE_BUDGET').sum())),
      cf_coverage=(round(float(items['cf_valid'].sum()) / int(items['cf_fork'].sum()), 4)
                   if items['cf_fork'].any() else None),
      realized={name: int((items['realized'] == k).sum()) for k, name in enumerate(REALIZED)},
      teachers_agree=int(items['teachers_agree'].sum()),
      thresholds=dict(outcome_error_prob=outcome_error_prob,
                      decision_error_margin=decision_error_margin,
                      kappa_q=kappa and kappa[0], kappa_source=kappa and kappa[1],
                      decision_errors='TAGGED' if kappa else 'DECISION_ERRORS_NOT_TAGGED'),
      errors=dict(outcome=len(e['outcome']), decision=len(e['decision']), demo=len(e['demo']),
                  outcome_by_realized=by, decision_by_executed=dict(
                      wait=int((items['decision_error'] & (items['executed'] == 0)).sum()),
                      add=int((items['decision_error'] & (items['executed'] == 1)).sum()))),
      created=time.strftime('%Y-%m-%d %H:%M:%S'))


def write(out, items, summary):
  """items.npz, then summary.json last: a folder with a summary holds
  whole items."""
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  tmp = out / 'items.tmp.npz'
  np.savez(tmp, **items)
  tmp.rename(out / ITEMS)
  tmp = out / f'{SUMMARY}.tmp'
  tmp.write_text(json.dumps(summary, indent=1))
  tmp.rename(out / SUMMARY)


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--collection', required=True, help='formal pi_k collection of one contract')
  parser.add_argument('--corpus', required=True, help='TRAIN corpus of the same contract')
  parser.add_argument('--cf', default=None, help='CF folder of the same collection')
  parser.add_argument('--kappa-cf', action='append', default=None,
                      help='CF folders of the round whose pooled Delta G std is kappa_Q '
                           '(default: --cf)')
  parser.add_argument('--mix', default='uniform_contract', help='contract weights of kappa_Q')
  parser.add_argument('--kappa-floor', type=float, default=None,
                      help='replaces a zero or smaller std of real pairs (value anchor rule)')
  parser.add_argument('--out', required=True, help='feedback folder of the contract')
  parser.add_argument('--order', default=OFFLINE_TRAIN, choices=ORDERS,
                      help=f'teaching order; {TIME_ORDERED} is refused '
                           f'({TIME_ORDERED_NOT_SUPPORTED})')
  parser.add_argument('--outcome-error-prob', type=float, default=DEFAULTS['outcome_error_prob'])
  parser.add_argument('--decision-error-margin', type=float,
                      default=DEFAULTS['decision_error_margin'])
  args = parser.parse_args(argv)
  check_order(args.order)
  kappa, folders = None, args.kappa_cf or ([args.cf] if args.cf else [])
  if folders:
    from .value import cf_kappa
    try:
      kappa = cf_kappa(folders, args.mix, args.kappa_floor)
    except ValueError as e:
      print(f'DECISION_ERRORS_NOT_TAGGED: {e}')
  items, summary = build(args.collection, args.corpus, args.cf, kappa,
                         args.outcome_error_prob, args.decision_error_margin)
  write(args.out, items, summary)
  print(json.dumps({k: summary[k] for k in ('contract', 'policy', 'sources', 'errors')}, indent=1))


if __name__ == '__main__':
  main()
