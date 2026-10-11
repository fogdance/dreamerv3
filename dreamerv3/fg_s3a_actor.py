"""Composition over the ORIGINAL pol; independent Optax state, no RL updates."""
from __future__ import annotations

import hashlib
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from embodied.jax import internal

DEFAULT_OPT = dict(lr=4e-5, agc=0.3, eps=1e-20, beta1=0.9, beta2=0.999,
                   momentum=True, nesterov=False, wd=0., schedule='const', warmup=0, anneal=0)


def parameter_digests(params):
    result = {}
    for key, value in sorted(params.items()):
        a = np.asarray(jax.device_get(value))
        result[key] = hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
    return result


def classification_metrics(labels, p_add):
    y, p = np.asarray(labels,dtype=np.int32), np.asarray(p_add,dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid classification metric inputs')
    pred=p>.5; positive=y==1
    tp=int((pred & positive).sum()); fp=int((pred & ~positive).sum())
    fn=int((~pred & positive).sum()); tn=int((~pred & ~positive).sum())
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else None
    # Average precision with tied probabilities evaluated at shared thresholds.
    # An all-constant score has AP exactly equal to the positive class prior.
    order=np.argsort(-p,kind='stable'); yy=positive[order]
    ends=np.r_[np.flatnonzero(np.diff(p[order]))+1,len(y)]
    cumulative=np.cumsum(yy)[ends-1]
    ap=float(np.sum(np.diff(np.r_[0,cumulative])*cumulative/ends)/positive.sum()) if positive.any() else None
    return dict(teacher_ADD=int(positive.sum()),teacher_WAIT=int((~positive).sum()),
        tp=tp,fp=fp,fn=fn,tn=tn,precision=precision,recall=recall,
        pr_auc=ap,pr_auc_definition='average_precision_tied_thresholds',
        balanced_accuracy=(recall+tn/(tn+fp))/2 if positive.any() and (~positive).any() else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        accuracy=(tp+tn)/len(y),predicted_ADD_fraction=float(pred.mean()),
        p_ADD_mean=float(p.mean()),p_ADD_min=float(p.min()),p_ADD_max=float(p.max()),
        p_ADD_teacher_ADD=float(p[positive].mean()) if positive.any() else None,
        p_ADD_teacher_WAIT=float(p[~positive].mean()) if (~positive).any() else None,
        brier=float(np.mean((p-y)**2)),
        always_wait_accuracy=float((~positive).mean()),always_wait_pr_auc=float(positive.mean()),
        status='ALL_WAIT_COLLAPSE' if not pred.any() else 'CLASSIFICATION_ONLY')


class ActorBC:
    """Only pol/* is a differentiable argument; source state is never donated."""

    def __init__(self, agent, opt_config=None):
        self.agent=agent
        if agent.train_mesh.size != 1 or agent.policy_mesh.size != 1:
            raise ValueError('S3-A v1 supports one CPU or one GPU; distributed meshes are unsupported')
        self.opt_config=dict(DEFAULT_OPT,**(opt_config or {}))
        if set(self.opt_config)!=set(DEFAULT_OPT) or self.opt_config['wd'] or self.opt_config['warmup'] or self.opt_config['schedule']!='const':
            raise ValueError('S3-A baseline requires unweighted BC, constant LR, no decay/warmup')
        if self.opt_config['lr']<=0 or self.opt_config['agc']<0:
            raise ValueError('Invalid BC optimizer configuration')
        self.actor_keys=sorted(k for k in agent.params if k.startswith('pol/'))
        if not self.actor_keys:
            raise ValueError('Original pol parameter namespace is missing')
        self.source_digests=parameter_digests(agent.params)
        self.source_shapes={k:(v.shape,str(v.dtype)) for k,v in agent.params.items()}
        self.source_counters=dict(updates=int(agent.n_updates),batches=int(agent.n_batches))
        self.frozen_refs={k:v for k,v in agent.params.items() if k not in self.actor_keys}
        self.tx=agent.model._make_opt(**self.opt_config)
        self.s3a_bc_opt=self.tx.init(self.actor_params)
        self.bc_updates=0

        def forward(features, mask, labels):
            features=jax.lax.stop_gradient(features)
            dist=agent.model._masked_policy(agent.model.pol(features,bdims=1),mask,
                                            allow_fallback=False)[agent.model.action_key]
            return dist.logp(labels), jax.nn.softmax(dist.logits)[...,1]
        pure=nj.pure(forward)

        def loss(params,features,mask,labels,valid):
            _,(logp,padd)=pure(params,features,mask,labels,seed=0,create=False,modify=False)
            return (-logp*valid).sum()/valid.sum(),padd

        def update(params,state,features,mask,labels,valid):
            (value,padd),grads=jax.value_and_grad(loss,has_aux=True)(params,features,mask,labels,valid)
            changes,next_state=self.tx.update(grads,state,params)
            next_params=optax.apply_updates(params,changes)
            finite=jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves((next_params,next_state,grads))]).all() & jnp.isfinite(value)
            metrics=dict(loss=value,grad_norm=optax.global_norm(grads),
                update_norm=optax.global_norm(changes),param_delta=optax.global_norm(
                    jax.tree.map(lambda a,b:a-b,next_params,params)),finite=finite)
            return next_params,next_state,metrics,padd

        actor_sharding={k:agent.train_params_sharding[k] for k in self.actor_keys}
        sh=agent.train_sharded
        self._update=jax.jit(update,in_shardings=(actor_sharding,None,sh,sh,sh,sh))
        self._predict=jax.jit(lambda params,x,m,y:loss(params,x,m,y,jnp.ones(y.shape,jnp.float32)),
                             in_shardings=(actor_sharding,sh,sh,sh))

    @property
    def actor_params(self):
        return {k:self.agent.params[k] for k in self.actor_keys}

    def _batch(self,features,mask,labels,valid=None):
        x=np.asarray(features,np.float32);m=np.asarray(mask,np.float32);y=np.asarray(labels,np.int32)
        v=np.ones(y.shape,np.float32) if valid is None else np.asarray(valid,np.float32)
        if (x.ndim!=2 or m.shape!=(len(x),2) or y.shape!=(len(x),) or v.shape!=y.shape
                or not np.isfinite(x).all() or not np.isin(y,[0,1]).all()
                or not np.isin(m,[0,1]).all() or not m.all()
                or not np.isin(v,[0,1]).all() or not v.sum()):
            raise ValueError('BC requires finite legal empty ADD/WAIT rows and nonempty validity')
        return tuple(internal.device_put(a,self.agent.train_sharded) for a in (x,m,y,v))

    def probabilities(self,features,mask):
        x,m,y,_=self._batch(features,mask,np.zeros(len(features),np.int32))
        _,p=self._predict(self.actor_params,x,m,y)
        return np.asarray(jax.device_get(p))

    def update(self,features,mask,labels,valid=None):
        x,m,y,v=self._batch(features,mask,labels,valid)
        params,state,metrics,p=self._update(self.actor_params,self.s3a_bc_opt,x,m,y,v)
        metrics={k:np.asarray(jax.device_get(a)).item() for k,a in metrics.items()}
        if not metrics['finite']:
            raise FloatingPointError('BC update is nonfinite; weights/state were not published')
        # The original param dictionary keeps exactly the same keys and shapes.
        with self.agent.train_lock:
            self.agent.params.update(params)
        self.s3a_bc_opt=state;self.bc_updates+=1
        if any(self.agent.params[k] is not a for k,a in self.frozen_refs.items()):
            raise RuntimeError('BC replaced a non-Actor state entry')
        self.agent.sync_policy_from_train()
        self.assert_policy_sync()
        selected=np.asarray(valid,dtype=bool) if valid is not None else np.ones(len(labels),bool)
        return dict(metrics,bc_updates=self.bc_updates,phase='S3A_ACTOR_ONLY',
                    **{'pol/'+k:metrics[k] for k in ('grad_norm','update_norm','param_delta')},
                    **classification_metrics(np.asarray(labels)[selected],np.asarray(jax.device_get(p))[selected]))

    def assert_policy_sync(self):
        if self.agent.pending_sync is not None:
            raise RuntimeError('BC policy sync is still pending')
        for k in self.agent.policy_keys:
            np.testing.assert_array_equal(jax.device_get(self.agent.policy_params[k]),jax.device_get(self.agent.params[k]))

    def freeze_audit(self,require_actor_change=True):
        current=parameter_digests(self.agent.params)
        if set(current)!=set(self.source_digests) or any((v.shape,str(v.dtype))!=self.source_shapes[k] for k,v in self.agent.params.items()):
            raise RuntimeError('Source parameter key/shape/dtype contract changed')
        changed=[k for k in current if current[k]!=self.source_digests[k]]
        if any(not k.startswith('pol/') for k in changed):
            raise RuntimeError('A frozen parameter, joint optimizer, Norm or Gate changed')
        if require_actor_change and not changed:
            raise RuntimeError('Actor has not changed')
        if dict(updates=int(self.agent.n_updates),batches=int(self.agent.n_batches))!=self.source_counters:
            raise RuntimeError('Original joint training counters changed during BC')
        self.assert_policy_sync()
        return dict(status='PASS',total_keys=len(current),frozen_keys=len(self.frozen_refs),
                    changed_keys=changed,source_digests=self.source_digests,current_digests=current,
                    bc_updates=self.bc_updates,original_updates=int(self.agent.n_updates),
                    original_batches=int(self.agent.n_batches),policy_sync=True)
