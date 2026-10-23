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
import optax

from .labels import (MAE_HORIZONS, NBINS, OPEN_AT_H, OUTCOME_BINS,
                     OUTCOME_CLASSES, RISK_FLAGS)

sg = jax.lax.stop_gradient
f32 = jnp.float32
WAIT, ADD = 0, 1
QUICK_TP_BINS = sum(edge <= 15 for edge in OUTCOME_BINS)


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
  """First ending of the trade "ADD now, then hold" -- TP, SL or another
  forced exit in a market-minute bin, or still open at H -- plus path risk
  and quality. Every answer is conditional on the horizon H; evidence holds
  only supervised answers (34 classes, 2 risks, 3 MAEs, quality)."""

  units: int = 256
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'

  def __call__(self, state, legal):
    h = self.sub('mlp', nn.MLP, self.layers, self.units,
                 act=self.act, norm=self.norm)(state)
    event_logits = f32(self.sub('event', nn.Linear, OUTCOME_CLASSES)(h))
    risk_logits = f32(self.sub('risk', nn.Linear, len(RISK_FLAGS))(h))
    mae = f32(self.sub('mae', nn.Linear, len(MAE_HORIZONS))(h))
    quality = f32(self.sub('quality', nn.Linear, 1)(h))[..., 0]
    probs = jax.nn.softmax(event_logits, -1)
    valid = f32(legal)
    evidence = jnp.concatenate([
        probs, jax.nn.sigmoid(risk_logits), mae, quality[..., None]], -1)
    evidence = evidence * valid[..., None]
    event = lambda e: probs[..., e * NBINS:(e + 1) * NBINS].sum(-1) * valid
    return dict(
        event_logits=event_logits, risk_logits=risk_logits, mae=mae,
        quality=quality, valid=valid, evidence=evidence,
        p_tp_h=event(0), p_sl_h=event(1), p_other_h=event(2),
        p_open_h=probs[..., OPEN_AT_H] * valid,
        p_tp_15=probs[..., :QUICK_TP_BINS].sum(-1) * valid)


def masked_mean(x, mask):
  return (x * mask).sum() / jnp.maximum(mask.sum(), 1.)


def outcome_loss(out, data, weight):
  """L_outcome on teacher labels; only legal-ADD states with valid labels count.

  Event term: discrete-time competing-risk likelihood. The label lists every
  class consistent with the native trace (one class for a trade that ended or
  was still open at H, the later bins of every ending plus OPEN_AT_H for a
  censored trade), so the NLL is -log sum(p[allowed]) and a trade with no
  informative age contributes 0.
  """
  valid = data['outcome_valid'] * weight * out['valid']
  logp = jax.nn.log_softmax(out['event_logits'], -1)
  allowed = (data['outcome_allowed'] > 0.5) | (valid[..., None] <= 0)
  nll = -jax.nn.logsumexp(jnp.where(allowed, logp, -jnp.inf), -1)
  risk_mask = data['outcome_risk_valid'] * valid[..., None]
  risk = optax.sigmoid_binary_cross_entropy(
      out['risk_logits'], data['outcome_risk'])
  mae_mask = data['outcome_mae_valid'] * valid[..., None]
  quality_mask = data['outcome_quality_valid'] * valid
  losses = {
      'outcome_event': masked_mean(nll, valid),
      'outcome_risk': masked_mean(risk, risk_mask),
      'outcome_mae': masked_mean(jnp.square(out['mae'] - data['outcome_mae']), mae_mask),
      'outcome_quality': masked_mean(
          jnp.square(out['quality'] - data['outcome_quality']), quality_mask),
  }
  labelled = data['outcome_valid'] * weight
  # Rows whose first ending within H was observed as a TP or an SL.
  observed = valid * (data['outcome_allowed'].sum(-1) == 1)
  tp = observed * data['outcome_allowed'][..., :NBINS].sum(-1)
  sl = observed * data['outcome_allowed'][..., NBINS:2 * NBINS].sum(-1)
  metrics = {
      'outcome/labels': valid.sum(),
      'outcome/label_not_legal': (labelled * (1. - out['valid'])).sum(),
      'outcome/label_tp': masked_mean(tp, valid),
      'outcome/label_sl': masked_mean(sl, valid),
      'outcome/p_tp_h': masked_mean(out['p_tp_h'], valid),
      'outcome/p_tp_15': masked_mean(out['p_tp_15'], valid),
      'outcome/p_sl_h': masked_mean(out['p_sl_h'], valid),
      'outcome/p_other_h': masked_mean(out['p_other_h'], valid),
      'outcome/p_open_h': masked_mean(out['p_open_h'], valid),
      'outcome/p_tp_on_tp': masked_mean(out['p_tp_h'], tp),
      'outcome/p_tp_on_sl': masked_mean(out['p_tp_h'], sl),
      # Denominators of the masked means, so evaluations can pool batches.
      'outcome/n_risk': risk_mask.sum(), 'outcome/n_mae': mae_mask.sum(),
      'outcome/n_quality': quality_mask.sum(), 'outcome/n_tp': tp.sum(),
      'outcome/n_sl': sl.sum(),
  }
  return losses, metrics


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
  """Intuition + structured Reasoning + bounded gate + one Final Actor head.

  Q reaches Reasoning and the gate only as stopped decision context through
  zero-initialized projections (A5, S2->S3): neither an untrained nor a newly
  trained Q can move the demonstrated behavior prior. Q changes the action
  through the explicit anchor beta_Q * clip(dQ / kappa_Q); Reasoning learns
  to use the Q context only from later independent demonstrations (S4).
  """

  units: int = 256
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'
  alpha_min: float = 0.1
  alpha_max: float = 1.0
  beta_q: float = 0.0
  kappa_q: float = 1.0
  clip_q: float = 3.0
  clip_d: float = 5.0

  def __call__(self, state, evidence, q, legal):
    mlp = dict(act=self.act, norm=self.norm)
    e, qv = sg(evidence), sg(q)
    dq = jnp.where(legal, qv[..., ADD] - qv[..., WAIT], 0.)
    q_ctx = nn.cast(jnp.concatenate([qv, dq[..., None]], -1))
    x = jnp.concatenate([nn.cast(state), nn.cast(e)], -1)
    d = self.sub('intuition', nn.MLP, self.layers, self.units, **mlp)(state)
    pre = self.sub('reason_in', nn.Linear, self.units)(x) + self.sub(
        'reason_q', nn.Linear, self.units, bias=False, winit='zeros')(q_ctx)
    r = nn.act(self.act)(self.sub('reason_norm', nn.Norm, self.norm)(pre))
    if self.layers > 1:
      r = self.sub('reason', nn.MLP, self.layers - 1, self.units, **mlp)(r)
    reason_pref = f32(self.sub('reason_pref', nn.Linear, 1)(r))[..., 0]
    g = f32(self.sub('gate', nn.Linear, 1)(x) + self.sub(
        'gate_q', nn.Linear, 1, bias=False, winit='zeros')(q_ctx))[..., 0]
    alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(g)
    wr = self.sub('reason_proj', nn.Linear, self.units, bias=False)(r)
    h = self.sub('integrator', nn.Norm, self.norm)(
        d + nn.cast(alpha[..., None]) * wr)
    correction = self.clip_d * jnp.tanh(
        f32(self.sub('final', nn.Linear, 1)(h))[..., 0])
    anchor = self.beta_q * jnp.clip(dq / self.kappa_q, -self.clip_q, self.clip_q)
    z = anchor + correction
    logits = jnp.stack([jnp.zeros_like(z), z], -1)
    return dict(logits=logits, z=z, alpha=alpha, anchor=anchor,
                correction=correction, reason_pref=reason_pref, delta_q=dq)


def masked_policy(logits, action_mask):
  """MaskedSoftmax([0, z]): probabilities for teaching and diagnostics."""
  return embodied.jax.outs.MaskedCategorical(
      logits, action_mask > 0.5, unimix=0.0, allow_fallback=False)


def formal_action(z, action_mask):
  """DETERMINISTIC_MASKED_ARGMAX, the formal pi_k (A1): ADD exactly when the
  Native mask allows it and z > 0; z == 0 and every other case is WAIT."""
  return jnp.where((action_mask[..., ADD] > 0.5) & (z > 0), ADD, WAIT).astype(jnp.int32)
