"""DAgger WM+BC and Gate-controlled original RL on the original networks."""
from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from .agent import Agent
from .fg_s3c_agent import S3CAgent


@contextmanager
def masked_joint_optimizer(optimizer, actor_active, critic_active):
    """Keep the original optimizer schema, schedule and a single update call.

    Zero gradients alone do not freeze inherited momentum or weight decay.
    Freeze both the final update and all parameter-specific optimizer leaves.
    Shared optimizer clocks advance with WM updates, as in the source optimizer.
    This temporary transformation exists only during tracing and is restored.
    """
    original = optimizer.opt

    def update(grads, state, params=None):
        active = {k: actor_active if k.startswith('pol/') else
                  critic_active if k.startswith('val/') else jnp.bool_(True)
                  for k in params}
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
        if self.mode not in ('a2', 'c2'):
            raise ValueError('S3-V2 supports A2 or C2')
        if config.replay_context != 1 or config.replay_context_mode != 'recompute_prefix':
            raise ValueError('S3-V2 requires one causal recomputed prefix')
        if not np.isfinite(config.bc_lambda) or config.bc_lambda < 0:
            raise ValueError('S3-V2 BC lambda must be finite and nonnegative')

    def training_loss_fn(self, data, effective_stepid):
        # Construct precisely the original complete NN/optimizer schema. This
        # initialization is overwritten by SourceAdmission before interaction.
        if nj.creating():
            return self.loss
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
        return partial(self.v2_loss, teacher_y=y, teacher_valid=valid)

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
            'v2/rl_actor_loss': metrics.get('loss/policy', jnp.float32(0)) * self.scales['policy'],
            'v2/rl_critic_loss': sum(metrics.get('loss/'+k, jnp.float32(0)) * self.scales[k]
                                      for k in ('value', 'repval') if k in self.scales),
            'avail/img_fallback_rate': metrics.get('avail/img_fallback_rate', jnp.float32(0)),
            'avail/img_action_invalid_rate': metrics.get('avail/img_action_invalid_rate', jnp.float32(0))})
        return loss, (carry, entries, outs['repfeat'], common)

    def v2_loss(self, carry, obs, prevact, training, *, teacher_y, teacher_valid):
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
        if self.config.bc_lambda:
            bc, extra = self.bc_loss(feat, obs['action_mask'], teacher_y, teacher_valid)
        else:
            bc, extra = jnp.float32(0), {'bc/valid_count': teacher_valid.sum()}
        total = base + jnp.float32(self.config.bc_lambda) * bc
        # Ninjax requires every joint optimizer parameter to be accessed by
        # its loss. Read weights only, with an exact zero derivative; never
        # evaluate the Critic or run its normalizers/slow target for A2.
        # Module.values registers its qualified keys as accessed in Ninjax.
        # Unused leaves then receive exact zero gradients from jax.grad.
        _ = self.pol.values, self.val.values
        active = (self.config.bc_lambda > 0) & (extra['bc/valid_count'] > 0)
        metrics.update(extra)
        metrics.update({'bc/nll': bc, 'bc/active': jnp.float32(active),
            'bc/lambda': jnp.float32(self.config.bc_lambda), 'bc/total_loss': total,
            'avail/ready': jnp.float32(ready),
            'v2/rl_active': jnp.float32(ready & (self.mode == 'c2'))})
        return total, (carry, entries, {}, metrics)

    def train(self, carry, data):
        if nj.creating():
            return Agent.train(self, carry, data)
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        rl = (self.mode == 'c2') & (self.avail_actor_gate.read() > 0)
        bc = (self.config.bc_lambda > 0) & jnp.any(data['teacher_valid'][:, 1:])
        frozen_params = {k: v for k, v in nj.context().items() if k.startswith(('pol/', 'val/'))}
        with masked_joint_optimizer(self.opt, bc | rl, rl):
            metrics, (carry, _, _, extra) = self.opt(
                self.training_loss_fn(data, stepid), carry, obs, prevact,
                training=True, has_aux=True)
        # optax.apply_updates still adds a zero update to inactive parameters;
        # that changed inherited -0.0 bits in the real CLOSED GPU check.
        # Select the original parameter itself to preserve exact frozen state.
        nj.context().update({k: jnp.where(bc | rl if k.startswith('pol/') else rl,
            nj.context()[k], old) for k, old in frozen_params.items()})
        metrics.update(extra)
        if self.mode == 'c2':
            def update_slow():
                self.slowval.update()
                return jnp.int32(0)
            nj.cond(rl, update_slow, lambda: jnp.int32(0))
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, {}, metrics
