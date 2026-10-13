"""Outcome supervision and bounded risk-directed policy correction for B2."""
import math

import jax
import jax.numpy as jnp
import ninjax as nj

EVENTS = ('TP_FAST', 'TP_MID', 'TP_LATE', 'SL', 'OTHER', 'OPEN_AT_H')
DEFAULTS = dict(b2_head_units=128, b2_event_lambda=.1, b2_early_lambda=.1,
    b2_coefficient_max=2., b2_delta_max=2., b2_mix_warmup=4,
    b2_mix_ramp=16, b2_min_event_updates=4)


def b2_extensions(settings):
    result = {'s3_v2_mode': 'b2'}
    for key, default in DEFAULTS.items():
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('B2 setting must be finite: '+key)
        if isinstance(default, int):
            if type(value) is not int or value < (0 if key == 'b2_mix_warmup' else 1):
                raise ValueError('B2 count setting is invalid: '+key)
        elif value < 0 or (key.endswith('_max') and value == 0):
            raise ValueError('B2 loss/bound setting is invalid: '+key)
        result[key] = value
    return result


class DecisionAdapter(nj.Module):
    """Six constrained magnitudes; OTHER is fixed zero, no free intercept."""
    def __init__(self, coefficient_max=2., delta_max=2.):
        self.maximum = coefficient_max
        self.delta_max = delta_max

    def __call__(self, event_prob, early_prob):
        def initialize():
            init = jnp.minimum(jnp.array([1., .5, .25, 1., 1., 1.]), self.maximum / 2)
            p = init / self.maximum
            return jnp.log(p) - jnp.log1p(-p)
        raw = self.value('raw_weights', initialize)
        w = jnp.float32(self.maximum) * jax.nn.sigmoid(raw)
        signed = jnp.stack([w[0], w[1], w[2], -w[3], jnp.float32(0), -w[4]])
        # BC cannot train the predictions into pretending that risk is smaller.
        event_prob, early_prob = jax.tree.map(jax.lax.stop_gradient, (event_prob, early_prob))
        parts = event_prob * signed
        early_part = -early_prob * w[5]
        delta = jnp.clip(parts.sum(-1) + early_part, -self.delta_max, self.delta_max)
        return delta, (parts, early_part)


class MixSchedule(nj.Module):
    def __init__(self, warmup, ramp, minimum):
        self.warmup, self.ramp, self.minimum = warmup, ramp, minimum
        self.updates = nj.Variable(jnp.array, 0, jnp.int32, name='updates')
        self.event_updates = nj.Variable(jnp.array, 0, jnp.int32, name='event_updates')

    def read(self):
        value = jnp.clip((self.updates.read() - self.warmup) / self.ramp, 0., 1.)
        return jnp.where(self.event_updates.read() >= self.minimum, value, jnp.float32(0))

    def advance(self, event_active):
        self.updates.write(self.updates.read() + 1)
        self.event_updates.write(self.event_updates.read() + event_active.astype(jnp.int32))


def masked_mean(values, valid):
    return jnp.where(valid, values, 0.).sum() / jnp.maximum(valid.sum(), 1)


def outcome_objective(event, early, labels):
    y, valid = labels['outcome_event'], labels['outcome_event_valid']
    e, ev = labels['outcome_early10'], labels['outcome_early_valid']
    if y.shape != valid.shape or y.shape != e.shape or e.shape != ev.shape:
        raise ValueError('B2 effective outcome tensors must align')
    bad = (valid & ((y < 0) | (y >= 6))) | (ev & ((e != 0) & (e != 1)))
    def fail(_):
        def reject():
            raise ValueError('B2_INVALID_CERTIFIED_OUTCOME_LABEL')
        jax.debug.callback(reject)
        return jnp.int32(0)
    jax.lax.cond(jnp.any(bad), fail, lambda _: jnp.int32(0), None)
    event_loss = masked_mean(event.loss(jnp.where(valid, y, 0)), valid)
    early_loss = masked_mean(early.loss(jnp.where(ev, e, 0)), ev)
    prob = jax.nn.softmax(event.logits, -1)
    early_prob = jax.nn.sigmoid(early.logit)
    target = jax.nn.one_hot(jnp.where(valid, y, 0), 6)
    metrics = {'b2/event_loss': event_loss, 'b2/early_loss': early_loss,
        'b2/event_valid_count': valid.sum(), 'b2/early_valid_count': ev.sum(),
        'b2/event_brier': masked_mean(jnp.square(prob-target).sum(-1), valid),
        'b2/early_brier': masked_mean(jnp.square(early_prob-e), ev),
        'b2/early_target_mean': masked_mean(e, ev),
        'b2/early_prob_mean': masked_mean(early_prob, ev)}
    for i, name in enumerate(EVENTS):
        member = valid & (y == i)
        metrics['b2/count_'+name] = member.sum()
        metrics['b2/recall_'+name] = masked_mean((event.pred() == i).astype(jnp.float32), member)
        metrics['b2/prob_'+name] = masked_mean(prob[..., i], valid)
    return event_loss, early_loss, metrics
