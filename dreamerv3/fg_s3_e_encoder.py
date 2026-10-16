"""S3 E: one shared-minute MLP plus a fixed 15m context MLP and fusion."""
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn
from . import rssm


class S3EMultiscaleEncoder(nj.Module):
    units: int = 1024
    layers: int = 3
    norm: str = 'rms'
    act: str = 'silu'
    symlog: bool = True
    context_units: int = 128
    macd_units: int = 16

    def __init__(self, obs_space, **kw):
        expected = {'market_seq_e_shared': (60, 18), 'market_context_15m': (24, 7),
                    'position_state': (9,), 'action_mask': (2,), 'market_macd_5m': (4,)}
        if set(obs_space) != set(expected) or any(
                s.shape != expected[k] or s.dtype != np.float32 for k, s in obs_space.items()):
            raise ValueError('E Encoder ObsSpace shape/dtype/keys differ')
        if self.context_units != 128 or self.layers != 3 or self.units < 1:
            raise ValueError('E V2 requires three 1m layers and fixed context128')
        self.obs_space = obs_space
        self.minute = rssm.Encoder({k: s for k, s in obs_space.items()
            if k not in ('market_context_15m', 'market_macd_5m')}, units=self.units, layers=self.layers,
            norm=self.norm, act=self.act, symlog=self.symlog, **kw, name='minute')
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def __call__(self, carry, obs, reset, training, single=False):
        allowed = {*self.obs_space, 'market_seq', 'reward', 'is_first', 'is_last',
                   'is_terminal', 'market_advance'}
        if set(obs) - allowed:
            raise ValueError('E Encoder rejects unknown/future observation fields')
        bdims = 1 if single else 2
        if reset.ndim != bdims:
            raise ValueError('E Encoder batch/time rank differs')
        good = jnp.bool_(True)
        for k, s in self.obs_space.items():
            x = obs[k]
            if x.shape != (*reset.shape, *s.shape) or x.dtype != jnp.float32:
                raise ValueError('E Encoder input shape/dtype differs: '+k)
            good &= jnp.isfinite(x).all()
        macd = obs['market_macd_5m']
        good &= ((macd[..., 3] == 0) | (macd[..., 3] == 1)).all()
        context = obs['market_context_15m']
        mask = context[..., 6]
        good &= ((mask == 0) | (mask == 1)).all()

        def bad(_):
            def fail():
                raise ValueError('E Encoder nonfinite input or nonbinary valid mask')
            jax.debug.callback(fail)
            return jnp.int32(0)
        jax.lax.cond(good, lambda _: jnp.int32(0), bad, None)

        minute_obs = {k: obs[k] for k in self.obs_space if k not in ('market_context_15m', 'market_macd_5m')}
        _, _, z1m = self.minute({}, minute_obs, reset, training, single)
        values = jnp.where(mask[..., None] == 1, context[..., :6], 0.)
        clean = jnp.concatenate([values, mask[..., None]], -1)
        x = nn.cast(clean.reshape((-1, 168)))
        z15m = jax.nn.gelu(self.sub('context15', nn.Linear, 128, **self.kw)(x))
        z15m = jnp.where((mask == 1).any(-1).reshape((-1, 1)), z15m, 0.)
        macd_valid = macd[..., 3:4] == 1
        clean_macd = jnp.where(macd_valid, macd[..., :3], 0.)
        zm = jax.nn.gelu(self.sub('macd5', nn.Linear, self.macd_units, **self.kw)(
            nn.cast(clean_macd.reshape((-1, 3)))))
        zm = jnp.where(macd_valid.reshape((-1, 1)), zm, 0.)
        x = jnp.concatenate([z1m.reshape((-1, self.units)), z15m, zm], -1)
        tokens = self.sub('fusion', nn.Linear, self.units, **self.kw)(x)
        return carry, {}, tokens.reshape((*reset.shape, self.units))
