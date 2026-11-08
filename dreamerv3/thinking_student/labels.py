"""Teaching labels from the TS corpus facts (A3, A6 L2-L3, B1.4).

One Outcome label describes the Native trade "ADD now, then hold" from a flat
account. Its first ending is one of the three Native ways a trade closes --
TP, SL (the env's: fixed distances, TP5 / SL30 by default, or under the V2
strategy the trade's own structural SL and 2R TP) or another forced
exit (OTHER_EXIT: main roll, contract expiry, margin call) -- in a
market-minute bin, or the trade is still open after the
last edge H (OPEN_AT_H: "not ended within H", never "will not reach TP").
A trade the data boundary cut before it ended (RIGHT_CENSORED) only tells us
that it had not ended by that age. The MFE targets are the largest
executable favourable move within 15 / 60 / 120 market minutes of the fill,
whatever the exit, within the official trading day (gym ts_v1_grades).
Demonstration labels are the graded teacher's: ADD where it grades the trade
GOOD or better, WAIT where it grades it POOR, nothing for FAIR or no grade
(L_policy_demo); the V2 positive teacher writes its ADD / WAIT rows into the
same arrays (gym ts_v2_positive.graded_teacher). Labels are TRAIN teaching
data only and never enter the policy input.
"""
import numpy as np

# Upper edges (market minutes) of the first-ending bins; about one official
# day is 345 valid minutes, the last edge H is roughly eight days.
OUTCOME_BINS = (5, 10, 15, 30, 60, 120, 240, 345, 690, 1380, 2760)
NBINS = len(OUTCOME_BINS)
HORIZON = OUTCOME_BINS[-1]
# Event e ending in bin k -> e * NBINS + k; still open after H -> OPEN_AT_H.
EVENTS = ('TP', 'SL', 'OTHER_EXIT')
OPEN_AT_H = len(EVENTS) * NBINS
OUTCOME_CLASSES = OPEN_AT_H + 1
RISK_FLAGS = ('early10_5', 'early10_15')
MAE_HORIZONS = ('mae_15', 'mae_60', 'mae_full')
MAE_COMPLETE = {'mae_15': 'mae_15_complete', 'mae_60': 'mae_60_complete',
                'mae_full': 'path_complete'}
# Ticks per unit of the MAE / MFE heads (student ts.outcome_points): the V1
# SL distance 30 for the corpora of fixed TP / SL distances; under the V2
# structural stop its max_risk_points (10), so 1.0 is one planned R
# (outcome_points; user 2026-11-03). Ticks of the corpus's product (gym
# envs.entry_product, user 2026-11-08): one student setting for every
# product; the labels divide the MAE / MFE facts (points) by
# outcome_scale, the unit in points.
MAE_SCALE = 30.0
MFE_HORIZONS = ('mfe_15', 'mfe_60', 'mfe_120')
MFE_SCALE = MAE_SCALE

LABEL_SPACE = {
    'outcome_valid': (),
    'outcome_allowed': (OUTCOME_CLASSES,),
    'outcome_risk': (len(RISK_FLAGS),),
    'outcome_risk_valid': (len(RISK_FLAGS),),
    'outcome_mae': (len(MAE_HORIZONS),),
    'outcome_mae_valid': (len(MAE_HORIZONS),),
    'outcome_mfe': (len(MFE_HORIZONS),),
    'outcome_mfe_valid': (len(MFE_HORIZONS),),
    'outcome_quality': (),
    'outcome_quality_valid': (),
}
# demo_weight is 0 on rows without a demonstration and otherwise carries the
# class weight of the curriculum.
DEMO_SPACE = {'demo_action': (), 'demo_weight': ()}
# L4 (S3): complete Native returns of one frozen policy pi_k -- the executed
# action's (q_real) and, at a fork, the other action's branch (q_cf).
Q_SPACE = {'q_real': (), 'q_real_valid': (), 'q_cf': (), 'q_cf_valid': ()}


def time_bin(minutes):
  """1..5 -> 0, 6..10 -> 1, ..., beyond the last edge -> NBINS."""
  return int(np.searchsorted(OUTCOME_BINS, minutes, side='left'))


def allowed_classes(kind, minutes):
  """Classes consistent with what the native trade actually showed. A trade
  ends after at least one held minute; a censor age of 0 carries no
  information."""
  if type(minutes) is not int or minutes < int(kind in EVENTS):
    raise ValueError(f'{kind}: held market minutes {minutes!r} is not possible')
  allowed = np.zeros(OUTCOME_CLASSES, bool)
  if kind in EVENTS:
    b = time_bin(minutes)
    allowed[OPEN_AT_H if b == NBINS else EVENTS.index(kind) * NBINS + b] = True
  elif kind == 'RIGHT_CENSORED':
    # Bins ending at or before the censor age are known to hold no ending.
    m = int(np.searchsorted(OUTCOME_BINS, minutes, side='right'))
    for e in range(len(EVENTS)):
      allowed[e * NBINS + m:(e + 1) * NBINS] = True
    allowed[OPEN_AT_H] = True
  else:
    raise ValueError(f'Not a native trade outcome: {kind}')
  return allowed


def _zeros(space, batch_shape=()):
  return {k: np.zeros((*batch_shape, *shape), np.float32)
          for k, shape in space.items()}


def empty_labels(batch_shape=()):
  """Every teaching label (Outcome, demonstration, Q), all without information."""
  return _zeros({**LABEL_SPACE, **DEMO_SPACE, **Q_SPACE}, batch_shape)


def outcome_points(source):
  """Ticks per unit of the MAE / MFE labels of a corpus (its source.json; the
  student's ts.outcome_points): the max_risk_points of a V2 structural
  stop, else MAE_SCALE."""
  entry = ((source['inputs'].get('strategy') or {}).get('entry') or {})
  return float(entry['max_risk_points']) if entry.get('stop_rule') else MAE_SCALE


def outcome_scale(source):
  """Points per unit of the MAE / MFE labels of a corpus: outcome_points in
  points of its product (source.json tick_points, gym ts_v1_corpus)."""
  return outcome_points(source) * source['tick_points']


def outcome_labels(facts, kinds, graded, points=MAE_SCALE):
  """Outcome labels of every cursor of a corpus facts table whose `kind`
  codes index `kinds` (gym ts_v1_corpus.KINDS), with the MFE targets of its
  graded table; MAE / MFE in units of `points` (outcome_scale). A cursor
  has a label exactly when its ADD filled; unknown parts stay zero with
  valid=0."""
  n = len(facts['filled'])
  out = _zeros(LABEL_SPACE, (n,))
  for t in np.flatnonzero(facts['filled']):
    out['outcome_valid'][t] = 1.
    out['outcome_allowed'][t] = allowed_classes(
        kinds[facts['kind'][t]], int(facts['held_minutes'][t]))
    for i, name in enumerate(RISK_FLAGS):
      if facts[name + '_valid'][t]:
        out['outcome_risk'][t, i] = float(facts[name][t])
        out['outcome_risk_valid'][t, i] = 1.
    for i, name in enumerate(MAE_HORIZONS):
      if facts[MAE_COMPLETE[name]][t]:
        out['outcome_mae'][t, i] = float(facts[name][t]) / points
        out['outcome_mae_valid'][t, i] = 1.
    for i, name in enumerate(MFE_HORIZONS):
      if graded[name + '_complete'][t]:
        out['outcome_mfe'][t, i] = float(graded[name][t]) / points
        out['outcome_mfe_valid'][t, i] = 1.
    # The env reward of the whole trade (V1: the Quality TP reward; V2: its
    # net PnL / reward_scale_cny); unknown until it closed. The name stays.
    if facts['path_complete'][t] and np.isfinite(facts['reward'][t]):
      out['outcome_quality'][t] = float(facts['reward'][t])
      out['outcome_quality_valid'][t] = 1.
  return out


def demo_labels(graded):
  """(action, valid) per cursor of a graded table: ADD for GOOD or better,
  WAIT for POOR; FAIR and an ungraded cursor demonstrate nothing."""
  valid = graded['valid'].astype(bool)
  return np.where(valid, graded['action'], 0).astype(np.float32), valid


def class_weights(action, valid, balance=1.0):
  """(w_wait, w_add) with mean 1 over the demonstrations; balance=1 gives
  both classes equal total weight, balance=0 keeps natural frequencies."""
  n = np.array([(valid & (action == 0)).sum(), (valid & (action == 1)).sum()], np.float64)
  if not n.all():
    raise ValueError(f'Both demonstrated actions are needed, got WAIT/ADD counts {n.tolist()}')
  p = n / n.sum()
  w = p ** -balance
  return tuple(float(x) for x in w / (p * w).sum())
