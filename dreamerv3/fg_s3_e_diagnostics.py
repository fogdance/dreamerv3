"""Read-only gradient instrumentation for bounded E acceptance runs."""
import ninjax as nj
import optax

from .fg_s3_v2_agent import _S3V2Model


class EAcceptanceModel(_S3V2Model):
    def train(self, carry, data):
        if nj.creating():
            return super().train(carry, data)
        original = self.opt.opt
        metrics = {}

        def update(grads, state, params=None):
            for name, prefix in (('minute', 'enc/minute/'),
                                 ('context15', 'enc/context15/'), ('fusion', 'enc/fusion/')):
                leaves = {k: v for k, v in grads.items() if k.startswith(prefix)}
                if not leaves:
                    raise ValueError('E acceptance is missing gradient branch: '+name)
                metrics['opt/e_'+name+'_grad_norm'] = optax.global_norm(leaves)
            return original.update(grads, state, params)

        # The existing masked joint optimizer wraps this observer while tracing.
        # Every gradient/state/update is passed through unchanged. No new leaves,
        # optimizer, loss, RNG call or production transfer exception is added.
        self.opt.opt = optax.GradientTransformation(original.init, update)
        try:
            carry, outputs, result = super().train(carry, data)
        finally:
            self.opt.opt = original
        return carry, outputs, {**result, **metrics}
