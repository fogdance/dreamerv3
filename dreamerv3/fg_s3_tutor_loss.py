"""Static Tutor objectives on one masked Actor distribution; no replay state."""
import math

import jax
import jax.numpy as jnp


DEFAULTS = dict(tutor_alpha=.8, tutor_pair_lambda=.1, tutor_pair_tau=1.,
                tutor_hard_kappa=0., tutor_hard_wmax=2.)


def agent_extensions(settings):
    """Shared frozen configuration for training, exact restore and evaluation."""
    mode = settings['mode']
    if mode not in ('a2', 'c2', 'tutor'):
        raise ValueError('Unknown S3-V2 mode')
    result = {'s3_v2_mode': mode}
    if mode != 'tutor':
        return result
    for key, default in DEFAULTS.items():
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Tutor setting must be a finite number: '+key)
        result[key] = float(value)
    if not 0 <= result['tutor_alpha'] <= 1 or result['tutor_pair_lambda'] < 0:
        raise ValueError('Tutor requires alpha in [0,1] and nonnegative pair lambda')
    if result['tutor_pair_tau'] <= 0 or result['tutor_hard_wmax'] < 1:
        raise ValueError('Tutor requires positive tau and hard weight maximum >= 1')
    if result['tutor_hard_kappa'] != 0:
        raise ValueError('Tutor V0 static curriculum requires hard_kappa=0; E2 is deferred')
    return result


def eligible_teacher(mask, y, valid):
    """Fail closed if a supposedly certified label violates its public contract."""
    valid = jnp.asarray(valid, bool)
    both = jnp.all(mask > .5, axis=-1)
    bad = valid & (~both | ((y != 0) & (y != 1)))
    def fail(_):
        def reject():
            raise ValueError('TUTOR_CERTIFIED_LABEL_REQUIRES_TWO_LEGAL_ACTIONS')
        jax.debug.callback(reject)
        return jnp.int32(0)
    jax.lax.cond(jnp.any(bad), fail, lambda _: jnp.int32(0), None)
    return valid & both


def pair_mask(y, valid, is_first):
    """P -> N pairs only inside the same batch member and reset segment."""
    if y.ndim != 2 or valid.shape != y.shape or is_first.shape != y.shape:
        raise ValueError('Tutor needs aligned effective [batch,time] tensors')
    episode = jnp.cumsum(is_first.astype(jnp.int32), axis=1)
    pos, neg = valid & (y == 1), valid & (y == 0)
    return (pos[:, :, None] & neg[:, None, :]
            & (episode[:, :, None] == episode[:, None, :]))


def tutor_objective(nll, logits, y, valid, is_first, *, alpha=.8, tau=1., evaluated=True):
    """Return BC mix and diagnostics; logits are final post-mask/unimix logits.

    Invalid positions are finite before any pair subtraction. All reductions
    use static shapes; empty groups/pairs have zero losses and finite gradients.
    `evaluated=False` supplies counts without evaluating the Actor at lambda=0.
    """
    if nll.shape != y.shape or logits.shape != (*y.shape, 2):
        raise ValueError('Tutor distribution must match the effective binary-action time axis')
    valid = jnp.asarray(valid, bool)
    pos, neg = valid & (y == 1), valid & (y == 0)
    count, np_, nn_ = valid.sum(), pos.sum(), neg.sum()
    def mean(value, mask):
        return jnp.where(mask, value, 0.).sum() / jnp.maximum(mask.sum(), 1)
    natural = mean(nll, valid)
    both = (np_ > 0) & (nn_ > 0)
    balanced = jnp.where(both, .5 * (mean(nll, pos) + mean(nll, neg)), natural)
    mix = alpha * natural + (1-alpha) * balanced
    # Even an external -inf sentinel cannot generate 0*inf or inf-inf here.
    safe_logits = jnp.where(valid[..., None], logits, 0.)
    z = safe_logits[..., 1] - safe_logits[..., 0]
    pairs = pair_mask(y, valid, is_first)
    n_pairs = pairs.sum()
    gap = z[:, :, None] - z[:, None, :]
    pair = mean(jax.nn.softplus(-gap / tau), pairs) if evaluated else jnp.float32(0)
    padd = jax.nn.softmax(safe_logits, -1)[..., 1] if evaluated else jnp.zeros_like(nll)
    predicted = padd > .5
    difficulty = jnp.where(y == 1, 1-padd, padd)
    metrics = {
        'bc/nll': natural, 'bc/valid_count': count.astype(jnp.float32),
        'bc/add_count': np_.astype(jnp.float32), 'bc/wait_count': nn_.astype(jnp.float32),
        'bc/tp': (pos & predicted).sum(), 'bc/fn': (pos & ~predicted).sum(),
        'bc/fp': (neg & predicted).sum(), 'bc/tn': (neg & ~predicted).sum(),
        'bc/accuracy': mean((predicted == (y == 1)).astype(jnp.float32), valid),
        'bc/always_wait_accuracy': mean((y == 0).astype(jnp.float32), valid),
        'bc/p_add_mean': mean(padd, valid),
        'teacher/valid_fraction': count / valid.size,
        'tutor/natural_ce': natural, 'tutor/balanced_ce': balanced, 'tutor/bc_mix_loss': mix,
        'tutor/eligible_pos': np_, 'tutor/eligible_neg': nn_,
        'tutor/pos_fraction_valid': np_ / jnp.maximum(count, 1),
        'tutor/pair_loss': pair, 'tutor/pair_count': n_pairs,
        'tutor/pairable_batch': (n_pairs > 0).astype(jnp.float32),
        'tutor/p_add_pos_mean': mean(padd, pos), 'tutor/p_add_neg_mean': mean(padd, neg),
        'tutor/fn_rate': mean((~predicted).astype(jnp.float32), pos),
        'tutor/fp_rate': mean(predicted.astype(jnp.float32), neg),
        'tutor/pair_accuracy': mean((gap > 0).astype(jnp.float32), pairs),
        'tutor/logit_gap_pos_neg': jnp.where(both, mean(z, pos)-mean(z, neg), 0.),
        'tutor/difficulty_mean': mean(difficulty, valid),
        'tutor/difficulty_max': jnp.where(valid, difficulty, 0.).max(),
        'tutor/difficulty_weight_mean': (count > 0).astype(jnp.float32),
        'tutor/difficulty_weight_max': (count > 0).astype(jnp.float32),
        'tutor/pos_defined': (np_ > 0).astype(jnp.float32),
        'tutor/neg_defined': (nn_ > 0).astype(jnp.float32),
        'tutor/pair_defined': (n_pairs > 0).astype(jnp.float32),
        'tutor/scores_evaluated': jnp.float32(evaluated),
    }
    return mix, metrics
