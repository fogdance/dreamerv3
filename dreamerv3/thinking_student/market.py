"""A2 Market Brain: causal market encoder plus an exogenous market RSSM.

The market world never receives the student's action or account as an input.
The original Dreamer RSSM core is kept with its exact parameter shapes so F1
weights can be inherited; its action slot is fed a constant zero vector.
"""
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn
from .. import rssm

MARKET_KEYS = ('market_seq_e_shared', 'market_context_15m', 'market_macd_5m')
MARKET_DECODER_KEYS = ('market_seq_e_shared',)
# Width of the original WAIT/ADD action slot in the F1 RSSM core.
NULL_ACTION_DIM = 2


class MarketEncoder(nj.Module):
  """1m minute MLP + completed 15m buckets + partial 5m MACD, market only.
  With macd False (V2) the MACD input has no branch and no parameter."""

  units: int = 1024
  layers: int = 3
  norm: str = 'rms'
  act: str = 'silu'
  symlog: bool = True
  context_units: int = 128
  macd_units: int = 16
  macd: bool = True

  def __init__(self, obs_space, **kw):
    self.depth = obs_space['market_context_15m'].shape[0]
    expected = {'market_seq_e_shared': (60, 18), 'market_context_15m': (self.depth, 7),
                'market_macd_5m': (4,)}
    if {k: obs_space[k].shape for k in expected} != expected:
      raise ValueError('Market Brain requires the E market observation shapes')
    self.obs_space = {k: obs_space[k] for k in MARKET_KEYS}
    self.minute = rssm.Encoder(
        {'market_seq_e_shared': obs_space['market_seq_e_shared']},
        units=self.units, layers=self.layers, norm=self.norm, act=self.act,
        symlog=self.symlog, **kw, name='minute')
    self.kw = kw

  def __call__(self, obs, reset, training, single=False):
    bshape = reset.shape
    _, _, z1m = self.minute(
        {}, {'market_seq_e_shared': obs['market_seq_e_shared']}, reset,
        training, single)
    context = obs['market_context_15m']
    if context.shape[-2:] != (self.depth, 7):
      raise ValueError(f'15m context {context.shape[-2:]} is not the student\'s ({self.depth}, 7)')
    mask = context[..., 6]
    values = jnp.where(mask[..., None] == 1, context[..., :6], 0.)
    clean = jnp.concatenate([values, mask[..., None]], -1)
    x = nn.cast(clean.reshape((-1, self.depth * 7)))
    z15m = jax.nn.gelu(self.sub(
        'context15', nn.Linear, self.context_units, **self.kw)(x))
    z15m = jnp.where((mask == 1).any(-1).reshape((-1, 1)), z15m, 0.)
    parts = [z1m.reshape((-1, self.units)), z15m]
    if self.macd:
      macd = obs['market_macd_5m']
      macd_valid = macd[..., 3:4] == 1
      clean_macd = jnp.where(macd_valid, macd[..., :3], 0.)
      zm = jax.nn.gelu(self.sub('macd5', nn.Linear, self.macd_units, **self.kw)(
          nn.cast(clean_macd.reshape((-1, 3)))))
      parts.append(jnp.where(macd_valid.reshape((-1, 1)), zm, 0.))
    x = jnp.concatenate(parts, -1)
    tokens = self.sub('fusion', nn.Linear, self.units, **self.kw)(x)
    return tokens.reshape((*bshape, self.units))


class MarketRSSM:
  """RSSM whose transition is driven by market time only (A2/R01).

  Wraps the original core (same parameter names and shapes) and always feeds
  its action slot a zero vector, so no caller can route a student action in.
  """

  def __init__(self, name, **kw):
    space = {'null': elements.Space(np.float32, (NULL_ACTION_DIM,))}
    self.rssm = rssm.RSSM(space, **kw, name=name)

  def initial(self, bsize):
    return self.rssm.initial(bsize)

  def _null(self, bshape):
    return {'null': jnp.zeros((*bshape, NULL_ACTION_DIM), nn.COMPUTE_DTYPE)}

  def observe(self, carry, tokens, reset, training, single=False):
    return self.rssm.observe(
        carry, tokens, self._null(reset.shape), reset, training, single)

  def loss(self, carry, tokens, reset, training):
    return self.rssm.loss(
        carry, tokens, self._null(reset.shape), reset, training)

  def imagine(self, carry, length, training):
    B = carry['deter'].shape[0]
    carry, feat, _ = self.rssm.imagine(
        carry, self._null((B, length)), length, training)
    return carry, feat


def feat2tensor(feat):
  stoch = feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1))
  return jnp.concatenate([nn.cast(feat['deter']), nn.cast(stoch)], -1)
