"""Inherit earlier weights into the Thinking Student (C2 4.1).

`inherit_f1`: only parameters whose meaning survives the new architecture are
copied: the market encoder (with the account/mask input rows removed), the
market RSSM (its action slot now always sees zeros) and the market decoder
head. `inherit_student`: an earlier Thinking Student checkpoint whose
parameter tree has since changed, optionally through a named layout
migration. Everything not copied keeps its fresh initialization and is
listed as such; nothing is reset silently.
"""
import pickle
from pathlib import Path

import numpy as np

from . import labels as L

# F1 minute-encoder input rows in DictConcat (sorted key) order.
F1_MINUTE_ROWS = {'action_mask': (0, 2), 'market_seq_e_shared': (2, 1082),
                  'position_state': (1082, 1091)}
F1_MINUTE_KERNEL = 'enc/minute/mlp0/kernel'
PREFIX_MAP = (('enc/', 'market_enc/'), ('dyn/', 'market_dyn/'),
              ('dec/mlp/', 'market_dec/mlp/'),
              ('dec/vec/market_seq_e_shared/', 'market_dec/vec/market_seq_e_shared/'))


def load_params(path):
  path = Path(path)
  if path.is_dir():
    path = path / 'agent.pkl'
  with path.open('rb') as f:
    return pickle.load(f)['params']


def plan(source, target):
  """Return {target_key: array} plus reused/sliced/initialized key lists."""
  values, reused, sliced = {}, [], []
  for key, value in source.items():
    for old, new in PREFIX_MAP:
      if not key.startswith(old):
        continue
      dest = new + key[len(old):]
      if dest not in target:
        break
      value = np.asarray(value)
      if key == F1_MINUTE_KERNEL:
        lo, hi = F1_MINUTE_ROWS['market_seq_e_shared']
        if value.shape[0] != F1_MINUTE_ROWS['position_state'][1]:
          raise ValueError(f'Unexpected F1 minute kernel rows {value.shape}')
        value = value[lo:hi]
        sliced.append(dest)
      else:
        reused.append(dest)
      if value.shape != tuple(target[dest].shape) or value.dtype != target[dest].dtype:
        raise ValueError(f'{key} -> {dest}: {value.shape}/{value.dtype} '
                         f'!= {target[dest].shape}/{target[dest].dtype}')
      values[dest] = value
      break
  initialized = sorted(k for k in target if k not in values and not k.startswith('opt'))
  return values, sorted(reused), sorted(sliced), initialized


def checkpoint_file(path):
  """A run folder (its latest checkpoint), a checkpoint folder, or a file."""
  path = Path(path)
  pointer = path / 'ckpt' / 'latest'
  if pointer.exists():
    path = path / 'ckpt' / pointer.read_text().strip()
  return path / 'agent.pkl' if path.is_dir() else path


# Kernels whose input rows are [fusion state | Outcome evidence].
EVIDENCE_KERNELS = ('value/mlp/linear0/kernel', 'decision/reason_in/kernel',
                    'decision/gate/kernel')
EVIDENCE = L.OUTCOME_CLASSES + len(L.RISK_FLAGS) + len(L.MAE_HORIZONS) + 1
ACCOUNT_KERNEL = 'account_enc/mlp/linear0/kernel'
ACCOUNT_TAIL = 3  # input rows after account_native: the valid flag and the 2-way mask


def r2r3(key, old, new):
  """S2 student -> Review R2/R3 layout (2026-10-23); None for other keys.

  Account input [position_state 9 | account_native | valid | mask 2]: the
  new account fields were appended to account_native and get zero rows, so
  the encoder computes what it did until they are learned. Outcome events:
  S2 TP 11 | SL 11 | SURVIVE -> TP 11 | SL 11 | OTHER_EXIT 11 | OPEN_AT_H;
  the OTHER_EXIT bins keep their fresh head columns and SURVIVE ("no TP/SL
  within H") becomes OPEN_AT_H. Evidence S2 probs 23 | risk 2 | MAE 3 |
  quality | latent -> probs 34 | risk 2 | MAE 3 | quality: zero rows for the
  OTHER_EXIT probabilities, the unsupervised latent rows are dropped.
  """
  n = 2 * L.NBINS  # TP and SL bins, the same in both layouts
  if key == ACCOUNT_KERNEL:
    if old.shape[1:] != new.shape[1:] or old.shape[0] >= new.shape[0]:
      raise ValueError(f'{key}: {old.shape} is not a narrower account input of {new.shape}')
    out = np.zeros_like(new)
    out[:old.shape[0] - ACCOUNT_TAIL] = old[:-ACCOUNT_TAIL]
    out[-ACCOUNT_TAIL:] = old[-ACCOUNT_TAIL:]
    return out
  if key in ('outcome/event/kernel', 'outcome/event/bias'):
    if old.shape[:-1] != new.shape[:-1] or old.shape[-1] != n + 1:
      raise ValueError(f'{key}: {old.shape} is not the S2 event head of {new.shape}')
    out = np.array(new)
    out[..., :n] = old[..., :n]
    out[..., L.OPEN_AT_H] = old[..., n]
    return out
  if key in EVIDENCE_KERNELS:
    units = new.shape[0] - EVIDENCE
    tail = EVIDENCE - L.OUTCOME_CLASSES  # risk, MAE and quality rows
    if old.shape[1:] != new.shape[1:] or old.shape[0] < units + n + 1 + tail:
      raise ValueError(f'{key}: {old.shape} is not an S2 evidence input of {new.shape}')
    out = np.zeros_like(new)
    out[:units + n] = old[:units + n]
    out[units + L.OPEN_AT_H] = old[units + n]
    out[units + L.OUTCOME_CLASSES:] = old[units + n + 1:units + n + 1 + tail]
    return out
  return None


MIGRATIONS = {'r2r3': r2r3}


def inherit_student(agent, source_path, migrate=None, log=print):
  """Continue from a Thinking Student checkpoint with another parameter tree.

  Keys with the same shape and dtype are copied together with the update
  counters; `migrate` names a layout change (MIGRATIONS) that places the
  old values of its reshaped keys exactly. Other reshaped keys (e.g. a
  widened head and its optimizer moments) and new keys (e.g. a new stage
  optimizer) keep their fresh initialization; keys the model no longer has
  are dropped.
  """
  file = checkpoint_file(source_path)
  with file.open('rb') as f:
    data = pickle.load(f)
  source, target = data['params'], agent.params
  remap = MIGRATIONS[migrate] if migrate else lambda *_: None
  values, reused, remapped, reshaped = {}, [], [], []
  for key in sorted(source.keys() & target.keys()):
    old, new = np.asarray(source[key]), np.asarray(target[key])
    change = f'{key}: {old.shape} -> {new.shape}'
    if old.shape == new.shape and old.dtype == new.dtype:
      values[key] = old
      reused.append(key)
    elif (value := remap(key, old, new)) is not None:
      values[key] = value
      remapped.append(change)
    else:
      reshaped.append(change)
  lists = dict(source=str(file), migrate=migrate, reused=reused, remapped=remapped,
               reshaped=reshaped, initialized=sorted(target.keys() - source.keys()),
               dropped=sorted(source.keys() - target.keys()))
  current = agent.save()
  current['params'].update(values)
  current['counters'] = data['counters']
  agent.load(current)
  log(f'Inherit {file}: reused={len(reused)} remapped={len(remapped)} '
      f'reshaped={len(reshaped)} initialized={len(lists["initialized"])} '
      f'dropped={len(lists["dropped"])}')
  for name in ('remapped', 'reshaped', 'initialized', 'dropped'):
    for key in lists[name]:
      log(f'  {name}: {key}')
  return lists


def inherit_f1(agent, source_path, log=print):
  """Copy inheritable F1 weights into `agent` in place; return the lists."""
  source = load_params(source_path)
  target = {k: v for k, v in agent.params.items()}
  values, reused, sliced, initialized = plan(source, target)
  if not values:
    raise ValueError('No F1 parameter matched the Thinking Student market brain')
  data = agent.save()
  data['params'].update(values)
  agent.load(data)
  log(f'F1 inherit from {source_path}: reused={len(reused)} sliced={len(sliced)} '
      f'initialized={len(initialized)}')
  for name, keys in (('sliced', sliced), ('initialized', initialized)):
    for k in keys:
      log(f'  {name}: {k}')
  return dict(reused=reused, sliced=sliced, initialized=initialized)
