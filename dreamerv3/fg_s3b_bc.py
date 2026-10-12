"""S3-B V2: original pol, frozen population mean, unbiased batch NLL."""
import math

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from dreamerv3.fg_s3a_actor import ActorBC
from embodied.jax import internal

LOSS_VERSION = 'global_train_mean_v2'

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
    def __init__(self, agent, opt_config=None, *, global_mean_weight):
        self._global_mean_weight = float(global_mean_weight)
        if not math.isfinite(self._global_mean_weight) or self._global_mean_weight <= 0:
            raise ValueError('S3-B requires a finite positive frozen global mean weight')
        super().__init__(agent,opt_config)
        self.bc_attempts = 0
        self.skipped_batches = 0

        def forward(features, mask, labels):
            features = jax.lax.stop_gradient(features)
            dist = agent.model._masked_policy(agent.model.pol(features,bdims=1),mask,
                                              allow_fallback=False)[agent.model.action_key]
            return dist.logp(labels),jax.nn.softmax(dist.logits)[...,1]
        pure = nj.pure(forward)

        def loss(params,features,mask,labels,normalized_weights):
            _,(logp,padd) = pure(params,features,mask,labels,seed=0,create=False,modify=False)
            # B is fixed by tensor shape. No batch sum or batch maximum enters
            # this denominator. Sum/B also preserves the original all-ones NLL.
            return (-logp*normalized_weights).sum()/jnp.float32(labels.shape[0]),padd

        gradient = jax.value_and_grad(loss,has_aux=True)
        def update(params,state,features,mask,labels,normalized_weights):
            (value,padd),grads = gradient(params,features,mask,labels,normalized_weights)
            changes,next_state = self.tx.update(grads,state,params)
            next_params = optax.apply_updates(params,changes)
            finite = jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(
                (next_params,next_state,grads))]).all() & jnp.isfinite(value)
            metrics = dict(loss=value,grad_norm=optax.global_norm(grads),
                update_norm=optax.global_norm(changes),param_delta=optax.global_norm(
                    jax.tree.map(lambda a,b:a-b,next_params,params)),finite=finite)
            return next_params,next_state,metrics,padd
        actor_sharding = {k:agent.train_params_sharding[k] for k in self.actor_keys}
        sh = agent.train_sharded
        self._loss_grad = jax.jit(gradient,in_shardings=(actor_sharding,sh,sh,sh,sh))
        self._update = jax.jit(update,in_shardings=(actor_sharding,None,sh,sh,sh,sh))

    @property
    def global_mean_weight(self):
        return self._global_mean_weight

    @property
    def s3b_bc_opt(self):
        return self.s3a_bc_opt

    @s3b_bc_opt.setter
    def s3b_bc_opt(self,value):
        self.s3a_bc_opt = value

    def _batch(self,features,mask,labels,valid=None):
        # Original legality/feature/label/device checks remain authoritative.
        x,m,y,_ = super()._batch(features,mask,labels)
        if valid is None:
            return super()._batch(features,mask,labels)
        w,total = validate_weights(valid,len(labels))
        if not total:
            raise ValueError('Zero-weight batch must be skipped before the gradient kernel')
        with np.errstate(over='ignore',under='ignore'):
            scaled = (w/self.global_mean_weight).astype(np.float32)
        if ((w > 0) & (scaled == 0)).any() or not np.isfinite(scaled).all():
            raise FloatingPointError('Frozen-mean normalized weight exceeds the finite positive float32 boundary')
        return x,m,y,internal.device_put(scaled,self.agent.train_sharded)

    def update(self,features,mask,labels,weights=None):
        w,total = validate_weights(weights,len(labels))
        if not total:
            super()._batch(features,mask,labels)
            self.bc_attempts += 1
            self.skipped_batches += 1
            self.freeze_audit(require_actor_change=self.bc_updates > 0)
            return dict(phase='S3B_ACTOR_ONLY',status='ZERO_WEIGHT_SKIPPED',loss_version=LOSS_VERSION,
                        global_mean_weight=self.global_mean_weight,
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
            loss_version=LOSS_VERSION,global_mean_weight=self.global_mean_weight,
            batch_rows=len(w),batch_ess=ess,
            sum_w_ADD=float(w[y == 1].sum()),sum_w_WAIT=float(w[y == 0].sum()),
            single_label_batch=bool(len(np.unique(y[w > 0])) == 1),original_optimizer_updated=False)
        return metrics

    def freeze_audit(self,require_actor_change=True):
        return dict(super().freeze_audit(require_actor_change),
                    optimizer_namespace='s3b_bc_opt',bc_attempts=self.bc_attempts,
                    skipped_batches=self.skipped_batches,loss_version=LOSS_VERSION,
                    global_mean_weight=self.global_mean_weight)
