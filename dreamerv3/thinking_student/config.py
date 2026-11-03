"""Thinking Student V1 configuration on top of the canonical Dreamer presets.

The student settings are configs/thinking_student/student.yaml (C2 4.2
implementation settings, recorded with every policy snapshot; not part of the
frozen A1-A6 contract).
"""
import copy
from pathlib import Path

import elements
import numpy as np
import yaml

from ..configuration import read_config_yaml

# The student settings live with the other Thinking Student configs.
STUDENT = Path(__file__).parents[2] / 'configs' / 'thinking_student' / 'student.yaml'
_SETTINGS = yaml.safe_load(STUDENT.read_text())
TS_DEFAULTS, SMALL = _SETTINGS['ts'], _SETTINGS['small']


def make_config(preset='size12m', seed=0, batch_size=8, batch_length=64,
                device='cpu', updates=None):
  """preset: a Dreamer size preset, or 'small' for CPU engineering checks."""
  definitions = read_config_yaml(Path(__file__).parents[1] / 'configs.yaml')
  defaults = copy.deepcopy(definitions['defaults'])
  defaults['agent']['ts'] = copy.deepcopy(TS_DEFAULTS)
  cfg = elements.Config(defaults)
  if preset == 'small':
    cfg = cfg.update(SMALL)
  else:
    cfg = cfg.update(definitions[preset])
  cfg = cfg.update({
      'seed': seed, 'batch_size': batch_size, 'batch_length': batch_length,
      'report_length': batch_length, 'replay_context': 0,
      'jax.platform': 'cpu' if device == 'cpu' else 'cuda',
      'jax.compute_dtype': 'float32' if device == 'cpu' else 'bfloat16',
      'jax.precompile': False, 'jax.profiler': False, 'jax.prealloc': False,
      'jax.sync_train_metrics': True,
  })
  cfg = cfg.update(updates or {})
  return dict(cfg)


def observation_space(market=None):
  """The student's inputs; `market` (ts.market) sets the 15m context length."""
  from gym_trading_env.research.fg_s3_e_observation import E_FIELDS_V2
  from gym_trading_env.research.ts_v1_account import ACCOUNT_NATIVE_SHAPE
  keys = ('market_seq_e_shared', 'market_context_15m', 'market_macd_5m',
          'position_state', 'action_mask', 'reward', 'is_first', 'is_last',
          'is_terminal')
  spaces = {k: elements.Space(E_FIELDS_V2[k][1], E_FIELDS_V2[k][0]) for k in keys}
  depth = dict(market or TS_DEFAULTS['market'])['context_15m']
  spaces['market_context_15m'] = elements.Space(np.float32, (depth, 7))
  spaces['account_native'] = elements.Space(np.float32, ACCOUNT_NATIVE_SHAPE)
  return spaces


def action_space():
  return {'action': elements.Space(np.int32, (), 0, 2)}
