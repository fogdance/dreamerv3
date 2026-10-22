"""Outcome teaching labels from native teacher trade facts (A3, B1.4).

One label describes the counterfactual "ADD now, then hold" native trade from
the decision state. TP and SL are competing first terminations on a
market-minute grid. A trade that is still open at the data boundary, or that
a forced non-TP/SL exit closed, only tells us that neither TP nor SL happened
before that age (right-censoring); it is never relabelled as a failure.
Labels are TRAIN teaching data only and never enter the policy input.
"""
import numpy as np

# Upper edges (market minutes) of the first-termination bins; about one
# official day is 345 valid minutes, the last edge is roughly eight days.
OUTCOME_BINS = (5, 10, 15, 30, 60, 120, 240, 345, 690, 1380, 2760)
NBINS = len(OUTCOME_BINS)
# TP in bin k -> k, SL in bin k -> NBINS + k, still open after the last edge.
SURVIVE = 2 * NBINS
OUTCOME_CLASSES = 2 * NBINS + 1
RISK_FLAGS = ('early10_5', 'early10_15')
MAE_HORIZONS = ('mae_15', 'mae_60', 'mae_full')
MAE_SCALE = 30.0  # points, the native SL distance
TERMINATIONS = ('TP5', 'SL30')
CENSORINGS = ('OTHER_EXIT', 'RIGHT_CENSORED')
# The teacher L1 fact fields `outcome_label` reads; corpus caches keep only these.
L1_FIELDS = (
    'immediate_outcome_verified', 'exit_type', 'held_market_minutes',
    'early10_5', 'early10_5_valid', 'early10_15', 'early10_15_valid',
    'mae_15', 'mae_15_complete', 'mae_60', 'mae_window_complete', 'mae_full',
    'path_complete', 'configured_reward')

LABEL_SPACE = {
    'outcome_valid': (),
    'outcome_allowed': (OUTCOME_CLASSES,),
    'outcome_risk': (len(RISK_FLAGS),),
    'outcome_risk_valid': (len(RISK_FLAGS),),
    'outcome_mae': (len(MAE_HORIZONS),),
    'outcome_mae_valid': (len(MAE_HORIZONS),),
    'outcome_quality': (),
    'outcome_quality_valid': (),
}


def time_bin(minutes):
  """1..5 -> 0, 6..10 -> 1, ..., beyond the last edge -> NBINS."""
  return int(np.searchsorted(OUTCOME_BINS, minutes, side='left'))


def allowed_classes(kind, minutes):
  """Classes consistent with what the native trace actually observed."""
  if type(minutes) is not int or minutes < 1:
    raise ValueError(f'Held market minutes must be a positive int, got {minutes!r}')
  allowed = np.zeros(OUTCOME_CLASSES, bool)
  if kind in TERMINATIONS:
    b = time_bin(minutes)
    allowed[SURVIVE if b == NBINS else b + (NBINS if kind == 'SL30' else 0)] = True
  elif kind in CENSORINGS:
    # Bins ending at or before the censor age are known to hold no TP/SL.
    m = int(np.searchsorted(OUTCOME_BINS, minutes, side='right'))
    allowed[m:NBINS] = True
    allowed[NBINS + m:2 * NBINS] = True
    allowed[SURVIVE] = True
  else:
    raise ValueError(f'Not a native trade outcome: {kind}')
  return allowed


def empty_labels(batch_shape=()):
  return {k: np.zeros((*batch_shape, *shape), np.float32)
          for k, shape in LABEL_SPACE.items()}


def outcome_label(l1):
  """Project one teacher L1 fact record; invalid parts stay zero with valid=0."""
  out = empty_labels()
  kind = l1.get('exit_type') if l1 else None
  if not l1 or not l1.get('immediate_outcome_verified') or kind not in (
      *TERMINATIONS, *CENSORINGS):
    return out
  minutes = l1['held_market_minutes']
  out['outcome_valid'][()] = 1.
  out['outcome_allowed'][:] = allowed_classes(kind, minutes)
  for i, name in enumerate(RISK_FLAGS):
    if l1[name + '_valid']:
      out['outcome_risk'][i] = float(bool(l1[name]))
      out['outcome_risk_valid'][i] = 1.
  complete = dict(mae_15=l1['mae_15_complete'], mae_60=l1['mae_window_complete'][1],
                  mae_full=l1['path_complete'])
  for i, name in enumerate(MAE_HORIZONS):
    if complete[name]:
      out['outcome_mae'][i] = float(l1[name]) / MAE_SCALE
      out['outcome_mae_valid'][i] = 1.
  # Official Quality TP reward of the whole trade; unknown until it closed.
  if l1['path_complete'] and l1.get('configured_reward') is not None:
    out['outcome_quality'][()] = float(l1['configured_reward'])
    out['outcome_quality_valid'][()] = 1.
  return out


def label_arrays(facts):
  """Stack `outcome_label` over a sequence of L1 fact records (None allowed)."""
  out = empty_labels((len(facts),))
  for i, l1 in enumerate(facts):
    if l1 and l1.get('immediate_outcome_verified'):
      for k, v in outcome_label(l1).items():
        out[k][i] = v
  return out
