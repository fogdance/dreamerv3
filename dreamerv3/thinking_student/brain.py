"""A2-A5 student brain after the Market Brain.

AccountEncoder -> SharedFusion -> Outcome / Value -> Intuition + Reasoning ->
Integrator -> one Final Actor. Stop-gradient boundaries follow A5 and the
ABC review F-01: Reasoning and the gate read SG(Outcome) and SG(Q), and the
explicit value anchor reads SG(Q_ADD - Q_WAIT).
"""
import jax
import jax.numpy as jnp
import ninjax as nj

import embodied.jax
import embodied.jax.nets as nn

sg = jax.lax.stop_gradient
f32 = jnp.float32

# Market-minute bin edges for the first TP/SL termination; anything later is
# the survival class (still open at the last edge, or right-censored).
OUTCOME_BINS = (5, 15, 30, 60, 120, 240, 480)
OUTCOME_CLASSES = 2 * len(OUTCOME_BINS) + 1
RISK_FLAGS = ('early10_5', 'early10_15')
MAE_HORIZONS = ('mae_15', 'mae_60', 'mae_full')
WAIT, ADD = 0, 1


def add_legal(position_state, action_mask):
  """Outcome is defined only for a legal ADD from a flat account (A3/C1-07)."""
  return (position_state[..., 1] == 0) & (action_mask[..., ADD] > 0.5)


class AccountEncoder(nj.Module):

  units: int = 256
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'

  def __call__(self, position_state, account_native, action_mask, valid):
    native = account_native * valid[..., None]
    x = jnp.concatenate([
        nn.symlog(position_state), nn.symlog(native), valid[..., None],
        action_mask], -1)
    return self.sub('mlp', nn.MLP, self.layers, self.units,
                    act=self.act, norm=self.norm)(x)


class SharedFusion(nj.Module):

  units: int = 256
  act: str = 'silu'
  norm: str = 'rms'

  def __call__(self, market, account):
    x = jnp.concatenate([nn.cast(market), nn.cast(account)], -1)
    return self.sub('mlp', nn.MLP, 1, self.units, act=self.act, norm=self.norm)(x)


class OutcomeBrain(nj.Module):
  """Competing-risk first TP/SL time distribution + path risk + quality."""

  units: int = 256
  layers: int = 2
  evidence: int = 32
  act: str = 'silu'
  norm: str = 'rms'

  def __call__(self, state, legal):
    h = self.sub('mlp', nn.MLP, self.layers, self.units,
                 act=self.act, norm=self.norm)(state)
    event_logits = f32(self.sub('event', nn.Linear, OUTCOME_CLASSES)(h))
    risk_logits = f32(self.sub('risk', nn.Linear, len(RISK_FLAGS))(h))
    mae = f32(self.sub('mae', nn.Linear, len(MAE_HORIZONS))(h))
    quality = f32(self.sub('quality', nn.Linear, 1)(h))[..., 0]
    latent = jnp.tanh(f32(self.sub('latent', nn.Linear, self.evidence)(h)))
    probs = jax.nn.softmax(event_logits, -1)
    nbins = len(OUTCOME_BINS)
    stats = jnp.concatenate([
        probs, jax.nn.sigmoid(risk_logits), mae, quality[..., None]], -1)
    valid = f32(legal)
    evidence = jnp.concatenate([stats, latent], -1) * valid[..., None]
    return dict(
        event_logits=event_logits, risk_logits=risk_logits, mae=mae,
        quality=quality, valid=valid, evidence=evidence,
        p_tp=probs[..., :nbins].sum(-1) * valid,
        p_sl=probs[..., nbins:2 * nbins].sum(-1) * valid,
        p_open=probs[..., -1] * valid)


class ValueBrain(nj.Module):
  """Policy-conditioned Q_WAIT / Q_ADD with a neutral start (Q_ADD == Q_WAIT)."""

  units: int = 256
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'

  def __call__(self, state, evidence):
    x = jnp.concatenate([nn.cast(state), nn.cast(sg(evidence))], -1)
    h = self.sub('mlp', nn.MLP, self.layers, self.units,
                 act=self.act, norm=self.norm)(x)
    # Zero output kernel and a shared bias give identical action values until
    # Q supervision arrives (A6 L3 bootstrap); both heads then learn freely.
    q = f32(self.sub('q', nn.Linear, 2, bias=False, winit='zeros')(h))
    shared = f32(self.sub('qbias', nn.Linear, 1, winit='zeros')(
        jnp.ones_like(h[..., :1])))
    return q + shared


class DecisionBrain(nj.Module):
  """Intuition + structured Reasoning + bounded gate + one Final Actor head."""

  units: int = 256
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'
  alpha_min: float = 0.1
  alpha_max: float = 1.0
  beta_q: float = 1.0
  kappa_q: float = 1.0
  clip_q: float = 3.0
  clip_d: float = 5.0

  def __call__(self, state, evidence, q, legal):
    mlp = dict(act=self.act, norm=self.norm)
    e, qv = sg(evidence), sg(q)
    dq = jnp.where(legal, qv[..., ADD] - qv[..., WAIT], 0.)
    d = self.sub('intuition', nn.MLP, self.layers, self.units, **mlp)(state)
    rin = jnp.concatenate(
        [nn.cast(state), nn.cast(e), nn.cast(qv), nn.cast(dq[..., None])], -1)
    r = self.sub('reason', nn.MLP, self.layers, self.units, **mlp)(rin)
    reason_pref = f32(self.sub('reason_pref', nn.Linear, 1)(r))[..., 0]
    gin = jnp.concatenate([nn.cast(state), nn.cast(e), nn.cast(qv)], -1)
    g = f32(self.sub('gate', nn.Linear, 1)(gin))[..., 0]
    alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(g)
    wr = self.sub('reason_proj', nn.Linear, self.units, bias=False)(r)
    h = self.sub('integrator', nn.Norm, self.norm)(
        d + nn.cast(alpha[..., None]) * wr)
    correction = self.clip_d * jnp.tanh(
        f32(self.sub('final', nn.Linear, 1)(h))[..., 0])
    anchor = self.beta_q * jnp.clip(dq / self.kappa_q, -self.clip_q, self.clip_q)
    z = anchor + correction
    logits = jnp.stack([jnp.zeros_like(z), z], -1)
    return dict(logits=logits, alpha=alpha, anchor=anchor,
                correction=correction, reason_pref=reason_pref, delta_q=dq)


def masked_policy(logits, action_mask):
  return embodied.jax.outs.MaskedCategorical(
      logits, action_mask > 0.5, unimix=0.0, allow_fallback=False)
