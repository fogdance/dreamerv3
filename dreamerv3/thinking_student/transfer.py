"""Inherit F1 market weights into the Thinking Student (C2 4.1).

Only parameters whose meaning survives the new architecture are copied:
the market encoder (with the account/mask input rows removed), the market
RSSM (its action slot now always sees zeros) and the market decoder head.
Everything else keeps its fresh initialization and is listed as such.
"""
import pickle
from pathlib import Path

import numpy as np

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
