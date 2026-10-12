"""Weighted adapter over S3-A's ORIGINAL pol and existing pure NLL kernel."""
import math

import numpy as np

from dreamerv3.fg_s3a_actor import ActorBC
from embodied.jax import internal


def validate_weights(weights, rows):
    w = np.ones(rows,np.float64) if weights is None else np.asarray(weights,np.float64)
    if w.shape != (rows,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('S3-B weights must be finite, nonnegative, and row aligned')
    with np.errstate(over='ignore'):
        total = float(w.sum())
    if not math.isfinite(total):
        raise ValueError('S3-B raw weight denominator overflow')
    return w, total


class WeightedActorBC(ActorBC):
    """Independent instance/state; no modification of any S3-A module or API."""
    def __init__(self, agent, opt_config=None):
        super().__init__(agent,opt_config)
        self.bc_attempts = 0
        self.skipped_batches = 0

    @property
    def s3b_bc_opt(self):
        return self.s3a_bc_opt

    @s3b_bc_opt.setter
    def s3b_bc_opt(self,value):
        self.s3a_bc_opt = value

    def _batch(self,features,mask,labels,valid=None):
        # Original legality/feature/label/device checks remain authoritative.
        x,m,y,_ = super()._batch(features,mask,labels)
        w,total = validate_weights(valid,len(labels))
        if not total:
            raise ValueError('Zero-weight batch must be skipped before the gradient kernel')
        # Multiplication of all weights by one common positive constant leaves
        # the weighted NLL unchanged. This preserves very small raw denominators
        # without per-class or per-contract normalization.
        scaled = (w/w.max()).astype(np.float32)
        if ((w > 0) & (scaled == 0)).any() or not np.isfinite(scaled).all():
            raise FloatingPointError('Positive weight underflows at the original float32 Loss boundary')
        return x,m,y,internal.device_put(scaled,self.agent.train_sharded)

    def update(self,features,mask,labels,weights=None):
        w,total = validate_weights(weights,len(labels))
        if not total:
            super()._batch(features,mask,labels)
            self.bc_attempts += 1
            self.skipped_batches += 1
            self.freeze_audit(require_actor_change=self.bc_updates > 0)
            return dict(phase='S3B_ACTOR_ONLY',status='ZERO_WEIGHT_SKIPPED',
                        skipped=True,bc_updates=self.bc_updates,bc_attempts=self.bc_attempts,
                        skipped_batches=self.skipped_batches,effective_weight_sum=0.,
                        original_optimizer_updated=False)
        metrics = super().update(features,mask,labels,valid=w)
        self.bc_attempts += 1
        y = np.asarray(labels)
        scaled = w/w.max()
        ess = float(scaled.sum()**2/np.square(scaled).sum())
        metrics.update(phase='S3B_ACTOR_ONLY',skipped=False,bc_attempts=self.bc_attempts,
            skipped_batches=self.skipped_batches,effective_weight_sum=total,
            weight_max=float(w.max()),weight_min_positive=float(w[w > 0].min()),
            loss_denominator_rescale=float(w.max()),batch_ess=ess,
            sum_w_ADD=float(w[y == 1].sum()),sum_w_WAIT=float(w[y == 0].sum()),
            single_label_batch=bool(len(np.unique(y[w > 0])) == 1),original_optimizer_updated=False)
        return metrics

    def freeze_audit(self,require_actor_change=True):
        return dict(super().freeze_audit(require_actor_change),
                    optimizer_namespace='s3b_bc_opt',bc_attempts=self.bc_attempts,
                    skipped_batches=self.skipped_batches)
