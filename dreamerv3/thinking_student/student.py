"""Thinking Student V1: one complete student, one policy entry (A1).

`StudentModel.policy` is the only decision path: Market Brain -> Account ->
Fusion -> Outcome -> Value -> Intuition/Reasoning/Integrator -> Final Actor ->
Native mask. Training, autonomous collection, counterfactual continuation and
Teacher-free exams all call it. mode='policy' executes the formal pi_k
(DETERMINISTIC_MASKED_ARGMAX); mode='explore' executes the separate behavior
policy mu (epsilon mixture around the formal action) and records mu(a|s).
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
from . import labels
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


def stage_groups(ts):
  """A6 2.2: the groups each stage optimizer may update. L_outcome adapts
  SharedFusion only when the config declares it; L_policy_demo trains only
  the decision group (Intuition/Reasoning/Gate/Integrator/Final) and
  L_reason only its Reasoning path; the Q losses of L4 train only the
  Value Brain."""
  return {'L1': ('market', 'account'),
          'L2': ('outcome', 'fusion') if ts.l2_fusion else ('outcome',),
          'L3': ('decision',),
          'L4': ('value',)}


def warm(reset, burnin):
  """1 once the market state has observed `burnin` steps since its last reset."""
  idx = jnp.arange(reset.shape[1])
  last = jax.lax.cummax(jnp.where(reset, idx, -burnin), axis=1)
  return f32(idx - last >= burnin)


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
        symlog=enc.symlog, winit=enc.winit, macd=ts.market.macd, name='market_enc')
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
    self.stage_groups = stage_groups(ts)
    if ts.stage not in self.stage_groups:
      raise ValueError(f'Unknown training stage {ts.stage}')
    # Each stage keeps its own optimizer state over its own groups (A6 2.1).
    self.opts = {
        stage: embodied.jax.Optimizer(
            [m for g in groups for m in self.group_modules[g]],
            _make_opt(**config.opt), summary_depth=1, name=f'opt_{stage.lower()}')
        for stage, groups in self.stage_groups.items()}
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
    # account_valid: 0 where a data source has no exact account_native.
    # outcome_*, demo_*: TRAIN teacher labels for L_outcome / L_policy_demo;
    # q_*: Q^{pi_k} targets for L4. Never a policy input.
    spaces = {'account_valid': elements.Space(np.float32, ())}
    spaces.update({k: elements.Space(np.float32, shape) for k, shape in {
        **labels.LABEL_SPACE, **labels.DEMO_SPACE, **labels.Q_SPACE}.items()})
    return spaces

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

  def think(self, market_feat, obs, account_valid, stop_state=False):
    account = self.account_enc(
        obs['position_state'], obs['account_native'], obs['action_mask'],
        account_valid)
    state = self.fusion(market.feat2tensor(market_feat), account)
    if stop_state:
      state = sg(state)
    legal = brain.add_legal(obs['position_state'], obs['action_mask'])
    outcome = self.outcome(state, legal)
    q = self.value(state, outcome['evidence'])
    decision = self.decision(state, outcome['evidence'], q, legal)
    return dict(account=account, state=state, legal=legal, outcome=outcome,
                q=q, decision=decision)

  def policy(self, carry, obs, mode='policy'):
    if mode not in ('policy', 'explore'):
      raise ValueError(f"Policy mode must be 'policy' or 'explore', got {mode!r}")
    carry, feat = self.observe_market(carry, obs, training=False, single=True)
    valid = jnp.ones(obs['is_first'].shape, f32)
    t = self.think(feat, obs, valid)
    mask = obs['action_mask']
    dist = brain.masked_policy(t['decision']['logits'], mask)
    proposed = brain.formal_action(t['decision']['z'], mask)
    action, explored, mu = proposed, jnp.zeros(proposed.shape, bool), jnp.ones(proposed.shape, f32)
    if mode == 'explore':
      # Behavior policy mu: with probability eps a uniform legal action.
      eps = self.config.ts.explore_eps
      k1, k2 = jax.random.split(nj.seed())
      explored = jax.random.uniform(k1, proposed.shape) < eps
      legal_add = mask[..., brain.ADD] > 0.5
      coin = jax.random.bernoulli(k2, 0.5, proposed.shape) & legal_add
      action = jnp.where(explored, coin.astype(jnp.int32), proposed)
      mu = (1 - eps) * f32(action == proposed) + eps / (1 + f32(legal_add))
    out = answer_sheet(t, dist, self.config.ts.outcome_points)
    out.update(proposed=proposed, mu_prob=mu, explored=explored)
    out['finite'] = {k: jnp.isfinite(v).all(range(1, v.ndim))
                     for k, v in dict(feat=market.feat2tensor(feat),
                                      logits=t['decision']['logits']).items()}
    return carry, {'action': action}, out

  # ---- learning: one optimizer step of the configured A6 stage ----

  def train(self, carry, data):
    stage = self.config.ts.stage
    if nj.creating():
      # Create every stage optimizer at init so all checkpoints share one
      # parameter tree and a run can switch stages without reshaping it.
      for other in self.opts:
        if other != stage:
          self._update(other, carry, data)
    carry, metrics = self._update(stage, carry, data)
    return carry, {}, metrics

  def _update(self, stage, carry, data):
    lossfn = getattr(self, f'{stage.lower()}_loss')
    metrics, (carry, mets) = self.opts[stage](
        lossfn, carry, data, training=True, has_aux=True)
    metrics.update(mets)
    return carry, metrics

  def report(self, carry, data):
    lossfn = getattr(self, f'{self.config.ts.stage.lower()}_loss')
    _, (carry, mets) = lossfn(carry, data, training=False)
    return carry, mets

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

  def l3_loss(self, carry, data, training):
    """The behavior prior (A6 L3, refreshed in L5) on the graded teacher's
    demonstrations. L_policy_demo trains the decision group on the teaching
    view [0, correction]: the value anchor is neutral there, so the residual
    learns the demonstrations and never the Q signal (A6 2.3). L_reason
    trains the Reasoning path to answer "the graded teacher demonstrates
    ADD" (reason_pref, never part of z). Upstream outputs enter stopped; once Q
    is supervised, its stopped context reaches Reasoning and the gate, and
    only these independent demonstrations teach them to use it."""
    reset = data['is_first']
    tokens = self.market_enc(data, reset, training)
    mcarry, _, feat = self.market_dyn.observe(
        carry['market'], tokens, reset, training)
    t = self.think(sg(feat), data, data['account_valid'], stop_state=True)
    d = t['decision']
    # Native WAIT is always legal; zero dummy batches at init carry no mask.
    mask = data['action_mask'].at[..., brain.WAIT].set(1.)
    prior = d['correction']
    dist = brain.masked_policy(jnp.stack([jnp.zeros_like(prior), prior], -1), mask)
    demo = (data['demo_weight'] > 0) & t['legal']
    action = jnp.where(demo, data['demo_action'], 0.).astype(jnp.int32)
    warmed = warm(reset, self.config.ts.demo_burnin)
    weight = warmed * data['demo_weight'] * f32(demo)
    n = jnp.maximum(f32(weight > 0).sum(), 1.)
    losses = {
        'policy_demo': (weight * -dist.logp(action)).sum() / n,
        'reason': (weight * optax.sigmoid_binary_cross_entropy(
            d['reason_pref'], f32(action))).sum() / n,
    }
    loss = sum(self.scales[k] * v for k, v in losses.items())
    rate = lambda x, w: (w * x).sum() / jnp.maximum(w.sum(), 1.)
    p_add = dist.prob(jnp.ones(reset.shape, jnp.int32))
    p_reason = jax.nn.sigmoid(d['reason_pref'])
    adds = f32(weight > 0) * data['demo_action']
    waits = f32(weight > 0) * (1 - data['demo_action'])
    decide = f32(t['legal'] & (warmed > 0))
    take = f32(brain.formal_action(prior, mask) == brain.ADD)
    formal = f32(brain.formal_action(d['z'], mask) == brain.ADD)
    reason = f32(d['reason_pref'] > 0)
    mets = {
        'loss/policy_demo': losses['policy_demo'], 'loss/reason': losses['reason'],
        'policy/n_demo': f32(weight > 0).sum(),
        'policy/n_add': adds.sum(), 'policy/n_wait': waits.sum(),
        'policy/n_legal': decide.sum(),
        # The prior (teaching view) and, with the anchor, the formal pi_k.
        'policy/p_add_on_add': rate(p_add, adds),
        'policy/p_add_on_wait': rate(p_add, waits),
        'policy/acc_add': rate(take, adds),
        'policy/acc_wait': rate(1 - take, waits),
        'policy/add_rate': rate(take, decide),
        'policy/formal_add_rate': rate(formal, decide),
        'reason/p_add_on_add': rate(p_reason, adds),
        'reason/p_add_on_wait': rate(p_reason, waits),
        'reason/acc_add': rate(reason, adds),
        'reason/acc_wait': rate(1 - reason, waits),
        'brain/alpha': d['alpha'].mean(),
        'brain/correction_abs': jnp.abs(d['correction']).mean(),
        'brain/delta_q_abs': jnp.abs(d['delta_q']).mean(),
    }
    return loss, ({'market': mcarry}, mets)

  def l2_loss(self, carry, data, training):
    """L_outcome (A6 L2) on Native ADD facts. Market and Account are frozen:
    their outputs enter Fusion as stopped inputs, as in the policy forward."""
    reset = data['is_first']
    tokens = self.market_enc(data, reset, training)
    mcarry, _, feat = self.market_dyn.observe(
        carry['market'], tokens, reset, training)
    account = self.account_enc(
        data['position_state'], data['account_native'], data['action_mask'],
        data['account_valid'])
    state = self.fusion(sg(market.feat2tensor(feat)), sg(account))
    legal = brain.add_legal(data['position_state'], data['action_mask'])
    out = self.outcome(state, legal)
    weight = warm(reset, self.config.ts.outcome_burnin)
    losses, mets = brain.outcome_loss(out, data, weight)
    loss = sum(self.scales[k] * v for k, v in losses.items())
    mets.update({f'loss/{k}': v for k, v in losses.items()})
    return loss, ({'market': mcarry}, mets)

  def l4_loss(self, carry, data, training):
    """L_q_real, L_q_cf and L_q_pair (A6 L4) on the targets of one frozen
    pi_k. Every upstream output enters stopped, so only the Value Brain
    learns Q^{pi_k}; the targets are complete Native returns, so there is
    no bootstrap and no target copy. No Actor loss: Q reaches the decision
    only through the A5 anchor once beta_Q is raised in a new snapshot."""
    reset = data['is_first']
    tokens = self.market_enc(data, reset, training)
    mcarry, _, feat = self.market_dyn.observe(
        carry['market'], tokens, reset, training)
    t = self.think(sg(feat), data, data['account_valid'], stop_state=True)
    q, legal = t['q'], f32(t['legal'])
    warmed = warm(reset, self.config.ts.q_burnin)
    action = data['action'].astype(jnp.int32)
    own = jnp.take_along_axis(q, action[..., None], -1)[..., 0]
    other = jnp.take_along_axis(q, (1 - action)[..., None], -1)[..., 0]
    real = warmed * data['q_real_valid']
    cf = warmed * data['q_cf_valid'] * legal
    dq = q[..., brain.ADD] - q[..., brain.WAIT]
    dg = jnp.where(action == brain.ADD, data['q_real'] - data['q_cf'],
                   data['q_cf'] - data['q_real'])
    losses = {
        'q_real': brain.masked_mean(jnp.square(own - data['q_real']), real),
        'q_cf': brain.masked_mean(jnp.square(other - data['q_cf']), cf),
        'q_pair': brain.masked_mean(jnp.square(dq - dg), cf),
    }
    loss = sum(self.scales[k] * v for k, v in losses.items())
    signed = cf * f32(dg != 0)
    mets = {f'loss/{k}': v for k, v in losses.items()}
    mets.update({
        'q/n_real': real.sum(), 'q/n_cf': cf.sum(), 'q/n_sign': signed.sum(),
        'q/target_real': brain.masked_mean(data['q_real'], real),
        'q/pred_real': brain.masked_mean(own, real),
        'q/delta_g': brain.masked_mean(dg, cf),
        'q/delta_q_on_cf': brain.masked_mean(dq, cf),
        'q/sign_agree': brain.masked_mean(f32(jnp.sign(dq) == jnp.sign(dg)), signed),
        'brain/delta_q_abs': brain.masked_mean(jnp.abs(dq), legal * warmed),
    })
    return loss, ({'market': mcarry}, mets)


def answer_sheet(t, dist, points):
  """B1.2 student answer recorded before the action is executed; `points`
  per unit of the MFE head (ts.outcome_points)."""
  o, q, d = t['outcome'], t['q'], t['decision']
  return {
      'answer/p_add': dist.prob(jnp.ones(d['alpha'].shape, jnp.int32)),
      'answer/z': d['z'],
      'answer/q_wait': q[..., brain.WAIT],
      'answer/q_add': q[..., brain.ADD],
      'answer/delta_q': d['delta_q'],
      'answer/alpha': d['alpha'],
      'answer/anchor': d['anchor'],
      'answer/correction': d['correction'],
      'answer/reason_pref': d['reason_pref'],
      'answer/outcome_valid': o['valid'],
      'answer/p_tp_h': o['p_tp_h'],
      'answer/p_tp_15': o['p_tp_15'],
      'answer/p_sl_h': o['p_sl_h'],
      'answer/p_other_h': o['p_other_h'],
      'answer/p_open_h': o['p_open_h'],
      'answer/quality': o['quality'] * o['valid'],
      # MFE answers in points, like the graded table's targets.
      **{f'answer/{name}': o['mfe'][..., i] * points * o['valid']
         for i, name in enumerate(labels.MFE_HORIZONS)},
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
  return StudentModel(observation_space(cfg.agent.ts.market), action_space(), agent_cfg)
