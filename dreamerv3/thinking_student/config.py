"""Thinking Student V1 configuration on top of the canonical Dreamer presets.

Numbers here are implementation settings (C2 4.2), recorded with every policy
snapshot; they are not part of the frozen A1-A6 contract.
"""
import copy
from pathlib import Path

import elements
import numpy as np

from ..configuration import read_config_yaml

ACCOUNT_NATIVE_DIM = 10

TS_DEFAULTS = {
    'account': {'units': 1024, 'layers': 2},
    'fusion': {'units': 1024},
    'outcome': {'units': 1024, 'layers': 2, 'evidence': 32},
    'value': {'units': 1024, 'layers': 2},
    # beta_q = 0: the value anchor contributes nothing to pi0 (A6 L3); S3
    # raises it once Q has policy-conditioned supervision.
    'decision': {'units': 1024, 'layers': 2, 'alpha_min': 0.1, 'alpha_max': 1.0,
                 'beta_q': 0.0, 'kappa_q': 1.0, 'clip_q': 3.0, 'clip_d': 5.0},
    # Training stage (A6) whose optimizer `train` steps. L2 adapts SharedFusion
    # only when declared; burn-in rows after a market reset carry no L_outcome
    # (outcome_burnin) or L_policy_demo (demo_burnin).
    'stage': 'L1', 'l2_fusion': True, 'outcome_burnin': 16, 'demo_burnin': 16,
    # Behavior policy mu of mode='explore': a uniform legal action with
    # probability explore_eps, else the formal pi_k action.
    'explore_eps': 0.05,
    'loss_scales': {'dyn': 1.0, 'rep': 0.1, 'rec': 1.0, 'account': 1.0,
                    'outcome_event': 1.0, 'outcome_risk': 1.0,
                    'outcome_mae': 1.0, 'outcome_quality': 1.0,
                    'policy_demo': 1.0},
}

SMALL = {
    'agent.enc.simple.units': 16, 'agent.dec.simple.layers': 1,
    'agent.dec.simple.units': 16,
    'agent.dyn.rssm.deter': 32, 'agent.dyn.rssm.hidden': 16,
    'agent.dyn.rssm.stoch': 2, 'agent.dyn.rssm.classes': 4,
    'agent.dyn.rssm.blocks': 4,
    'agent.ts.account.units': 16, 'agent.ts.fusion.units': 16,
    'agent.ts.outcome.units': 16, 'agent.ts.outcome.evidence': 4,
    'agent.ts.value.units': 16, 'agent.ts.decision.units': 16,
    'agent.ts.outcome_burnin': 2, 'agent.ts.demo_burnin': 2,
}


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


def observation_space():
  from gym_trading_env.research.fg_s3_e_observation import E_FIELDS_V2
  keys = ('market_seq_e_shared', 'market_context_15m', 'market_macd_5m',
          'position_state', 'action_mask', 'reward', 'is_first', 'is_last',
          'is_terminal')
  spaces = {k: elements.Space(E_FIELDS_V2[k][1], E_FIELDS_V2[k][0]) for k in keys}
  spaces['account_native'] = elements.Space(np.float32, (ACCOUNT_NATIVE_DIM,))
  return spaces


def action_space():
  return {'action': elements.Space(np.int32, (), 0, 2)}
