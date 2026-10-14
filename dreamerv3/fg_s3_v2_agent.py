"""DAgger WM+BC and Gate-controlled original RL on the original networks."""
from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
import elements
import embodied

from .agent import Agent
from .fg_s3c_agent import S3CAgent
from .fg_s3_tutor_loss import agent_extensions, eligible_teacher, pair_mask, tutor_objective


@contextmanager
def masked_joint_optimizer(optimizer, actor_active, critic_active, extra_active=None):
    """Keep the original optimizer schema, schedule and a single update call.

    Zero gradients alone do not freeze inherited momentum or weight decay.
    Freeze both the final update and all parameter-specific optimizer leaves.
    Shared optimizer clocks advance with WM updates, as in the source optimizer.
    This temporary transformation exists only during tracing and is restored.
    """
    original = optimizer.opt

    def update(grads, state, params=None):
        rules = {'pol/': actor_active, 'val/': critic_active, **(extra_active or {})}
        active = {k: next((value for prefix, value in rules.items() if k.startswith(prefix)),
                         jnp.bool_(True)) for k in params}
        grads = {k: jnp.where(active[k], v, jnp.zeros_like(v)) for k, v in grads.items()}
        updates, candidate = original.update(grads, state, params)
        updates = {k: jnp.where(active[k], v, jnp.zeros_like(v)) for k, v in updates.items()}

        def restore(path, new, old):
            keys = [p.key for p in path if isinstance(p, jax.tree_util.DictKey) and p.key in active]
            return jnp.where(active[keys[-1]], new, old) if keys else new

        return updates, jax.tree_util.tree_map_with_path(restore, candidate, state)

    optimizer.opt = optax.GradientTransformation(original.init, update)
    try:
        yield
    finally:
        optimizer.opt = original


class S3V2Agent(S3CAgent):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        self.mode = config.s3_v2_mode
        if self.mode not in ('a2', 'c2', 'tutor', 'b2'):
            raise ValueError('S3-V2 supports A2, C2, Tutor or B2')
        if self.mode == 'b2':
            from .fg_s3_b2_outcome import b2_extensions, DecisionAdapter, MixSchedule
            self.b2 = b2_extensions({'mode': 'b2', **{k: config[k] for k in
                ('b2_head_units', 'b2_event_lambda', 'b2_early_lambda', 'b2_coefficient_max',
                 'b2_delta_max', 'b2_mix_warmup', 'b2_mix_ramp', 'b2_min_event_updates')}})
            head = dict(layers=1, units=config.b2_head_units, act=config.policy.act,
                        norm=config.policy.norm, winit=config.policy.winit, outscale=1.)
            self.b2_event = embodied.jax.MLPHead(elements.Space(np.int32, (), 0, 6),
                'categorical', **head, name='b2_event')
            self.b2_early = embodied.jax.MLPHead(elements.Space(bool, (), 0, 2),
                'binary', **head, name='b2_early')
            self.b2_adapter = DecisionAdapter(config.b2_coefficient_max, config.b2_delta_max,
                                             name='b2_adapter')
            self.b2_schedule = MixSchedule(config.b2_mix_warmup, config.b2_mix_ramp,
                config.b2_min_event_updates, name='b2_schedule')
            # The original optimizer holds this list by reference. Extend it
            # before outer Agent._init_params(), never inside a train call.
            self.modules.extend([self.b2_event, self.b2_early, self.b2_adapter])
        if self.mode == 'tutor':
            self.tutor = agent_extensions({'mode': 'tutor', **{
                k: config[k] for k in ('tutor_alpha', 'tutor_pair_lambda', 'tutor_pair_tau',
                                      'tutor_hard_kappa', 'tutor_hard_wmax')}})
        if config.replay_context != 1 or config.replay_context_mode != 'recompute_prefix':
            raise ValueError('S3-V2 requires one causal recomputed prefix')
        if not np.isfinite(config.bc_lambda) or config.bc_lambda < 0:
            raise ValueError('S3-V2 BC lambda must be finite and nonnegative')

    def training_loss_fn(self, data, effective_stepid):
        # Construct precisely the original complete NN/optimizer schema. This
        # initialization is overwritten by SourceAdmission before interaction.
        if nj.creating():
            return self.b2_init_loss if self.mode == 'b2' else self.loss
        k = self.config.replay_context
        y, valid = data['teacher_action'][:, k:], data['teacher_valid'][:, k:]
        if y.shape != effective_stepid.shape[:2] or valid.shape != y.shape:
            raise ValueError('S3-V2 teacher effective shape differs')
        def bad(_):
            def fail():
                raise ValueError('S3_V2_TEACHER_REPLAY_CONTEXT_SHIFT')
            jax.debug.callback(fail)
            return jnp.int32(0)
        jax.lax.cond(jnp.all(data['stepid'][:, k:] == effective_stepid),
                     lambda _: jnp.int32(0), bad, None)
        extra = {}
        if self.mode == 'b2':
            from gym_trading_env.research.fg_s3_b2_outcome_catalog import FIELDS
            labels = {name: data[name][:, k:] for name in FIELDS}
            if any(x.shape != y.shape for x in labels.values()):
                raise ValueError('B2 outcome effective shape differs from Replay')
            extra['outcomes'] = labels
        return partial(self.v2_loss, teacher_y=y, teacher_valid=valid, **extra)

    @property
    def ext_space(self):
        space = super().ext_space
        if self.mode == 'b2':
            space.update(outcome_event=elements.Space(np.int32, (), 0, 6),
                outcome_event_valid=elements.Space(bool), outcome_early10=elements.Space(np.int32, (), 0, 2),
                outcome_early_valid=elements.Space(bool))
        return space

    @property
    def policy_keys(self):
        if self.mode == 'b2':
            return '^(enc|dyn|dec|pol|b2_event|b2_early|b2_adapter|b2_schedule)/'
        return super().policy_keys

    def b2_predict(self, repfeat, bdims):
        inp = self.feat2tensor(repfeat)
        return self.b2_event(inp, bdims), self.b2_early(inp, bdims)

    def policy_distribution(self, repfeat, mask, bdims):
        if self.mode != 'b2':
            return super().policy_distribution(repfeat, mask, bdims)
        policy = self.pol(self.feat2tensor(repfeat), bdims)
        event, early = self.b2_predict(repfeat, bdims)
        probs = jax.nn.softmax(event.logits, -1)
        early_prob = jax.nn.sigmoid(early.logit)
        delta, (parts, early_part) = self.b2_adapter(probs, early_prob)
        mix = self.b2_schedule.read()
        raw = policy[self.action_key].logits
        final = raw.at[..., 1].add(mix * delta)
        unclipped = parts.sum(-1) + early_part
        # Raw Actor categorical has no unimix. Mask/unimix is applied once to
        # each separate distribution, never reapplied to transformed logits.
        dist = self._masked_policy({self.action_key: embodied.jax.outs.Categorical(final)}, mask, False)
        base = self._masked_policy(policy, mask, False)[self.action_key]
        diagnostics = {'b2/p_add_base': base.prob(jnp.ones(raw.shape[:-1], jnp.int32)),
            'b2/p_add_final': dist[self.action_key].prob(jnp.ones(raw.shape[:-1], jnp.int32)),
            'b2/delta_logit': mix * delta,
            'b2/raw_delta_logit': delta,  # Legacy name: historically the CLIPPED adapter value.
            'b2/adapter_unclipped_delta_logit': unclipped,
            'b2/adapter_clipped_delta_logit': delta,
            'b2/clip_saturation': jnp.abs(unclipped) >= self.config.b2_delta_max,
            'b2/base_logit_add_minus_wait': raw[..., 1] - raw[..., 0],
            'b2/final_logit_add_minus_wait': final[..., 1] - final[..., 0],
            'b2/mix_value': jnp.broadcast_to(mix, raw.shape[:-1]),
            'b2/outcome_probs': probs, 'b2/early_prob': early_prob,
            'b2/adapter_parts': parts, 'b2/adapter_early': early_part,
            'b2/argmax_changed': base.pred() != dist[self.action_key].pred()}
        return dist, diagnostics

    def b2_init_loss(self, carry, obs, prevact, training):
        base, aux = self.loss(carry, obs, prevact, training)
        feat = aux[2]['repfeat']
        event, early = self.b2_predict(feat, 2)
        delta, _ = self.b2_adapter(jax.nn.softmax(event.logits, -1), jax.nn.sigmoid(early.logit))
        base += jnp.float32(0) * (event.logits.mean() + early.logit.mean() + delta.mean()
                                 + self.b2_schedule.read())
        return base, aux

    def b2_bc_loss(self, repfeat, mask, y, valid):
        valid = eligible_teacher(mask, y, valid)
        safe_mask = jnp.where(valid[..., None], mask > .5, jnp.array([True, False]))
        feat = jax.tree.map(jax.lax.stop_gradient, repfeat)
        policy, diagnostics = self.policy_distribution(feat, safe_mask, bdims=2)
        dist = policy[self.action_key]
        nll = -dist.logp(jnp.where(valid, y, 0))
        from .fg_s3_b2_outcome import masked_mean
        bc = masked_mean(nll, valid)
        predicted, actual = dist.pred() == 1, y == 1
        extra = {'bc/nll': bc, 'bc/valid_count': valid.sum(),
            'bc/add_count': (valid & actual).sum(), 'bc/wait_count': (valid & ~actual).sum(),
            'bc/tp': (valid & predicted & actual).sum(), 'bc/fn': (valid & ~predicted & actual).sum(),
            'bc/fp': (valid & predicted & ~actual).sum(), 'bc/tn': (valid & ~predicted & ~actual).sum(),
            'bc/p_add_mean': masked_mean(diagnostics['b2/p_add_final'], valid),
            'bc/accuracy': masked_mean((predicted == actual).astype(jnp.float32), valid)}
        for key in ('p_add_base', 'p_add_final', 'delta_logit', 'mix_value', 'argmax_changed',
                    'adapter_unclipped_delta_logit', 'adapter_clipped_delta_logit', 'clip_saturation',
                    'base_logit_add_minus_wait', 'final_logit_add_minus_wait'):
            extra['b2/'+key+'_mean'] = masked_mean(diagnostics['b2/'+key], valid)
        return bc, extra

    def wm_only(self, carry, obs, prevact, training):
        (ec, dc, xc, ee, de, xe, tokens, feat, _, _, _, losses, metrics) = self.world_model(
            carry, obs, prevact, training)
        loss = sum(v.mean() * self.scales[k] for k, v in losses.items())
        metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
        return loss, ((ec, dc, xc), (ee, de, xe), dict(repfeat=feat, losses=losses), metrics)

    def _components(self, carry, obs, prevact, training, *, rl):
        fn = self.loss if rl else self.wm_only
        loss, (carry, entries, outs, metrics) = fn(carry, obs, prevact, training)
        wm_keys = [k for k in self.scales if k not in ('policy', 'value', 'repval')]
        wm = sum(outs['losses'][k].mean() * self.scales[k] for k in wm_keys)
        common = {k: v for k, v in metrics.items() if
                  k.startswith(('avail/post_', 'avail/prior_')) or
                  k in {f'loss/{x}' for x in wm_keys}}
        common.update({'v2/wm_loss': wm, 'v2/rl_loss': loss - wm,
            'avail/img_fallback_rate': metrics.get('avail/img_fallback_rate', jnp.float32(0)),
            'avail/img_action_invalid_rate': metrics.get('avail/img_action_invalid_rate', jnp.float32(0))})
        return loss, (carry, entries, outs['repfeat'], common)

    def v2_loss(self, carry, obs, prevact, training, *, teacher_y, teacher_valid, outcomes=None):
        ready = self.avail_actor_gate.read() > 0
        if self.mode == 'c2':
            base, (carry, entries, feat, metrics) = nj.cond(ready,
                partial(self._components, training=training, rl=True),
                partial(self._components, training=training, rl=False),
                carry, obs, prevact)
        else:
            base, (carry, entries, feat, metrics) = self._components(
                carry, obs, prevact, training, rl=False)
        # Lambda zero uses the same WM/RL route without BC forward/RNG calls.
        if self.mode == 'tutor':
            bc, extra = self.tutor_loss(feat, obs, teacher_y, teacher_valid)
        elif self.mode == 'b2' and self.config.bc_lambda:
            bc, extra = self.b2_bc_loss(feat, obs['action_mask'], teacher_y, teacher_valid)
        elif self.config.bc_lambda:
            bc, extra = self.bc_loss(feat, obs['action_mask'], teacher_y, teacher_valid)
        else:
            bc, extra = jnp.float32(0), {'bc/valid_count': teacher_valid.sum()}
        total = base + jnp.float32(self.config.bc_lambda) * bc
        if self.mode == 'b2':
            from .fg_s3_b2_outcome import outcome_objective
            event, early = self.b2_predict(feat, 2)
            el, al, outcome_metrics = outcome_objective(event, early, outcomes)
            total += self.config.b2_event_lambda * el + self.config.b2_early_lambda * al
            metrics.update(outcome_metrics)
            metrics['b2/mix_value'] = self.b2_schedule.read()
            metrics.update(self.b2_schedule.diagnostics())
            _ = self.b2_adapter.values
        if self.mode == 'tutor':
            total += jnp.float32(self.tutor['tutor_pair_lambda']) * extra['tutor/pair_loss']
        # Ninjax requires every joint optimizer parameter to be accessed by
        # its loss. Read weights only, with an exact zero derivative; never
        # evaluate the Critic or run its normalizers/slow target for A2.
        # Module.values registers its qualified keys as accessed in Ninjax.
        # Unused leaves then receive exact zero gradients from jax.grad.
        _ = self.pol.values, self.val.values
        active = (self.config.bc_lambda > 0) & (extra['bc/valid_count'] > 0)
        metrics.update(extra)
        metrics.update({'bc/nll': extra.get('bc/nll', bc), 'bc/active': jnp.float32(active),
            'bc/lambda': jnp.float32(self.config.bc_lambda), 'bc/total_loss': total,
            'avail/ready': jnp.float32(ready),
            'v2/rl_active': jnp.float32(ready & (self.mode == 'c2'))})
        return total, (carry, entries, {}, metrics)

    def tutor_loss(self, repfeat, obs, y, valid):
        valid = eligible_teacher(obs['action_mask'], y, valid)
        evaluated = bool(self.config.bc_lambda or self.tutor['tutor_pair_lambda'])
        if evaluated:
            safe_mask = jnp.where(valid[..., None], obs['action_mask'] > .5,
                                 jnp.array([True, False]))
            feat = jax.lax.stop_gradient(self.feat2tensor(repfeat))
            dist = self._masked_policy(self.pol(feat, bdims=2), safe_mask, False)[self.action_key]
            nll, logits = -dist.logp(jnp.where(valid, y, 0)), dist.logits
        else:
            nll, logits = jnp.zeros_like(y, dtype=jnp.float32), jnp.zeros((*y.shape, 2))
        bc, metrics = tutor_objective(nll, logits, y, valid, obs['is_first'],
            alpha=self.tutor['tutor_alpha'], tau=self.tutor['tutor_pair_tau'],
            hard_kappa=self.tutor['tutor_hard_kappa'], hard_wmax=self.tutor['tutor_hard_wmax'],
            evaluated=evaluated)
        metrics['tutor/pair_active'] = jnp.float32(
            (self.tutor['tutor_pair_lambda'] > 0) & (metrics['tutor/pair_count'] > 0))
        metrics['tutor/actor_active'] = jnp.float32(
            ((self.config.bc_lambda > 0) & jnp.any(valid)) | (metrics['tutor/pair_active'] > 0))
        return bc, metrics

    def train(self, carry, data):
        if nj.creating():
            return Agent.train(self, carry, data)
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        rl = (self.mode == 'c2') & (self.avail_actor_gate.read() > 0)
        bc = (self.config.bc_lambda > 0) & jnp.any(data['teacher_valid'][:, 1:])
        if self.mode == 'tutor':
            y = data['teacher_action'][:, 1:]
            valid = eligible_teacher(obs['action_mask'], y, data['teacher_valid'][:, 1:])
            bc = ((self.config.bc_lambda > 0) & jnp.any(valid)) | (
                (self.tutor['tutor_pair_lambda'] > 0) & jnp.any(pair_mask(y, valid, obs['is_first'])))
        extra_active = {}
        if self.mode == 'b2':
            extra_active = {
                'b2_event/': (self.config.b2_event_lambda > 0) & jnp.any(data['outcome_event_valid'][:, 1:]),
                'b2_early/': (self.config.b2_early_lambda > 0) & jnp.any(data['outcome_early_valid'][:, 1:]),
                'b2_adapter/': bc & (self.b2_schedule.read() > 0)}
        rules = {'pol/': bc | rl, 'val/': rl, **extra_active}
        frozen_params = {k: v for k, v in nj.context().items() if k.startswith(tuple(rules))}
        with masked_joint_optimizer(self.opt, bc | rl, rl, extra_active):
            metrics, (carry, _, _, extra) = self.opt(
                self.training_loss_fn(data, stepid), carry, obs, prevact,
                training=True, has_aux=True)
        # optax.apply_updates still adds a zero update to inactive parameters;
        # that changed inherited -0.0 bits in the real CLOSED GPU check.
        # Select the original parameter itself to preserve exact frozen state.
        nj.context().update({k: jnp.where(next(value for prefix, value in rules.items()
            if k.startswith(prefix)), nj.context()[k], old) for k, old in frozen_params.items()})
        metrics.update(extra)
        if self.mode == 'b2':
            self.b2_schedule.advance(extra_active['b2_event/'])
            metrics['b2/mix_next'] = self.b2_schedule.read()
        if self.mode == 'c2':
            def update_slow():
                self.slowval.update()
                return jnp.int32(0)
            nj.cond(rl, update_slow, lambda: jnp.int32(0))
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, {}, metrics
