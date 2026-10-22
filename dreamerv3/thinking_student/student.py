"""Thinking Student V1: one complete student, one policy entry (A1).

`StudentModel.policy` is the only decision path: Market Brain -> Account ->
Fusion -> Outcome -> Value -> Intuition/Reasoning/Integrator -> Final Actor ->
Native mask. Training, autonomous collection and Teacher-free exams all call
it; `mode='eval'` selects greedy execution of the same distribution.
"""
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

import embodied.jax
import embodied.jax.nets as nn

from .. import rssm
from . import brain
from . import market
from .config import action_space, make_config, observation_space

sg = jax.lax.stop_gradient
f32 = jnp.float32

# Parameter groups by module prefix; A6 stages authorize groups, never ad hoc keys.
GROUPS = {
    'market': ('market_enc', 'market_dyn', 'market_dec'),
    'account': ('account_enc', 'account_aux'),
    'fusion': ('fusion',),
    'outcome': ('outcome',),
    'value': ('value',),
    'decision': ('decision',),
}
# S0 implements L1 only; L2-L6 optimizers arrive with their supervision (S1-S4).
STAGE_GROUPS = {'L1': ('market', 'account')}


class AccountAux(nj.Module):
  """L_account: the account embedding must keep the Native account facts."""

  def __call__(self, account):
    pos = f32(self.sub('position', nn.Linear, 9)(account))
    can_add = f32(self.sub('can_add', nn.Linear, 1)(account))[..., 0]
    return pos, can_add


class StudentModel(embodied.jax.Agent):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    if set(act_space) != {'action'} or act_space['action'].classes != 2:
      raise ValueError('Thinking Student acts with exactly WAIT/ADD')
    ts = config.ts
    enc = config.enc.simple
    self.market_enc = market.MarketEncoder(
        obs_space, units=enc.units, layers=3, norm=enc.norm, act=enc.act,
        symlog=enc.symlog, winit=enc.winit, name='market_enc')
    self.market_dyn = market.MarketRSSM('market_dyn', **config.dyn.rssm)
    dec_space = {k: obs_space[k] for k in market.MARKET_DECODER_KEYS}
    self.market_dec = rssm.Decoder(dec_space, **config.dec.simple, name='market_dec')
    self.account_enc = brain.AccountEncoder(**ts.account, name='account_enc')
    self.account_aux = AccountAux(name='account_aux')
    self.fusion = brain.SharedFusion(**ts.fusion, name='fusion')
    self.outcome = brain.OutcomeBrain(**ts.outcome, name='outcome')
    self.value = brain.ValueBrain(**ts.value, name='value')
    self.decision = brain.DecisionBrain(**ts.decision, name='decision')
    modules = dict(
        market_enc=self.market_enc, market_dyn=self.market_dyn.rssm,
        market_dec=self.market_dec, account_enc=self.account_enc,
        account_aux=self.account_aux, fusion=self.fusion, outcome=self.outcome,
        value=self.value, decision=self.decision)
    self.group_modules = {
        g: [modules[n] for n in names] for g, names in GROUPS.items()}
    l1 = [m for g in STAGE_GROUPS['L1'] for m in self.group_modules[g]]
    self.opt_l1 = embodied.jax.Optimizer(
        l1, _make_opt(**config.opt), summary_depth=1, name='opt_l1')
    self.scales = dict(ts.loss_scales)

  @property
  def policy_keys(self):
    names = [n for g in ('market', 'account', 'fusion', 'outcome', 'value',
                         'decision') for n in GROUPS[g]]
    names.remove('market_dec')
    names.remove('account_aux')
    return '^(' + '|'.join(names) + ')/'

  @property
  def ext_space(self):
    # 0 for corpus records written before account_native existed.
    return {'account_valid': elements.Space(np.float32, ())}

  def init_policy(self, batch_size):
    return {'market': self.market_dyn.initial(batch_size)}

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  # ---- the single decision path ----

  def observe_market(self, carry, obs, training, single):
    reset = obs['is_first']
    tokens = self.market_enc(obs, reset, training, single)
    mcarry, _, feat = self.market_dyn.observe(
        carry['market'], tokens, reset, training, single)
    return {'market': mcarry}, feat

  def think(self, market_feat, obs, account_valid):
    account = self.account_enc(
        obs['position_state'], obs['account_native'], obs['action_mask'],
        account_valid)
    state = self.fusion(market.feat2tensor(market_feat), account)
    legal = brain.add_legal(obs['position_state'], obs['action_mask'])
    outcome = self.outcome(state, legal)
    q = self.value(state, outcome['evidence'])
    decision = self.decision(state, outcome['evidence'], q, legal)
    return dict(account=account, state=state, legal=legal, outcome=outcome,
                q=q, decision=decision)

  def policy(self, carry, obs, mode='train'):
    carry, feat = self.observe_market(carry, obs, training=False, single=True)
    valid = jnp.ones(obs['is_first'].shape, f32)
    t = self.think(feat, obs, valid)
    dist = brain.masked_policy(t['decision']['logits'], obs['action_mask'])
    greedy = mode == 'eval'
    action = dist.pred() if greedy else dist.sample(nj.seed())
    action = action.astype(jnp.int32)
    out = answer_sheet(t, dist)
    out['greedy'] = jnp.full(action.shape, greedy)
    out['finite'] = {k: jnp.isfinite(v).all(range(1, v.ndim))
                     for k, v in dict(feat=market.feat2tensor(feat),
                                      logits=t['decision']['logits']).items()}
    return carry, {'action': action}, out

  # ---- S0 learning: L_market + L_account (A6 L1) ----

  def train(self, carry, data):
    metrics, (carry, mets) = self.opt_l1(
        self.l1_loss, carry, data, training=True, has_aux=True)
    metrics.update(mets)
    return carry, {}, metrics

  def l1_loss(self, carry, data, training):
    reset = data['is_first']
    tokens = self.market_enc(data, reset, training)
    mcarry, _, losses, feat, _, mets = self.market_dyn.loss(
        carry['market'], tokens, reset, training)
    _, _, recons = self.market_dec({}, feat, reset, training)
    for key, recon in recons.items():
      losses[key] = recon.loss(sg(data[key]))
    valid = data['account_valid']
    account = self.account_enc(
        data['position_state'], data['account_native'], data['action_mask'],
        valid)
    pos, can_add = self.account_aux(account)
    losses['account_position'] = jnp.square(
        pos - data['position_state']).mean(-1)
    losses['account_can_add'] = optax.sigmoid_binary_cross_entropy(
        can_add, f32(data['action_mask'][..., 1] > 0.5))
    scales = {'dyn': self.scales['dyn'], 'rep': self.scales['rep'],
              'account_position': self.scales['account'],
              'account_can_add': self.scales['account']}
    scales.update({k: self.scales['rec'] for k in recons})
    loss = sum(scales[k] * f32(v).mean() for k, v in losses.items())
    # Forward the rest of the brain on stopped inputs: diagnostics only, so
    # L1 never moves decision parameters (and their params exist for policy).
    t = self.think(sg(feat), sg(data), sg(valid))
    # Native WAIT is always legal; zero dummy batches at init carry no mask.
    mask = data['action_mask'].at[..., brain.WAIT].set(1.)
    dist = brain.masked_policy(t['decision']['logits'], mask)
    mets = {**mets, **{f'loss/{k}': f32(v).mean() for k, v in losses.items()}}
    mets['brain/p_add'] = dist.prob(jnp.ones(reset.shape, jnp.int32)).mean()
    mets['brain/delta_q_abs'] = jnp.abs(t['decision']['delta_q']).mean()
    mets['brain/outcome_valid'] = t['outcome']['valid'].mean()
    return loss, ({'market': mcarry}, mets)

  def report(self, carry, data):
    _, (carry, mets) = self.l1_loss(carry, data, training=False)
    return carry, mets


def answer_sheet(t, dist):
  """B1.2 student answer recorded before the action is executed."""
  o, q, d = t['outcome'], t['q'], t['decision']
  return {
      'answer/p_add': dist.prob(jnp.ones(d['alpha'].shape, jnp.int32)),
      'answer/q_wait': q[..., brain.WAIT],
      'answer/q_add': q[..., brain.ADD],
      'answer/delta_q': d['delta_q'],
      'answer/alpha': d['alpha'],
      'answer/anchor': d['anchor'],
      'answer/correction': d['correction'],
      'answer/reason_pref': d['reason_pref'],
      'answer/outcome_valid': o['valid'],
      'answer/p_tp': o['p_tp'],
      'answer/p_sl': o['p_sl'],
      'answer/p_open': o['p_open'],
      'answer/quality': o['quality'] * o['valid'],
  }


def _make_opt(lr=4e-5, agc=0.3, eps=1e-20, beta1=0.9, beta2=0.999,
              momentum=True, nesterov=False, wd=0.0, wdregex=r'/kernel$',
              schedule='const', warmup=1000, anneal=0):
  del momentum, wd, wdregex, schedule, anneal
  sched = optax.constant_schedule(lr)
  if warmup:
    ramp = optax.linear_schedule(0.0, lr, warmup)
    sched = optax.join_schedules([ramp, sched], [warmup])
  return optax.chain(
      embodied.jax.opt.clip_by_agc(agc),
      embodied.jax.opt.scale_by_rms(beta2, eps),
      embodied.jax.opt.scale_by_momentum(beta1, nesterov),
      optax.scale_by_learning_rate(sched))


def make_agent(logdir, preset='size12m', device='cpu', **kwargs):
  cfg = elements.Config(make_config(preset, device=device, **kwargs))
  agent_cfg = elements.Config(
      **dict(cfg.agent), logdir=str(logdir), seed=cfg.seed, jax=dict(cfg.jax, transfer_guard=False),
      batch_size=cfg.batch_size, batch_length=cfg.batch_length,
      replay_context=cfg.replay_context, report_length=cfg.report_length,
      replica=0, replicas=1)
  return StudentModel(observation_space(), action_space(), agent_cfg)
