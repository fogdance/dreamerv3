"""Concurrent original RL + equal P4 BC; original modules and optimizer only."""
from functools import partial

import elements
import jax
import jax.numpy as jnp
import numpy as np

from .agent import Agent


def _raise_shift():
    raise ValueError('S3C_TEACHER_REPLAY_CONTEXT_SHIFT')


class S3CAgent(Agent):
    @property
    def ext_space(self):
        return {**super().ext_space, 'teacher_action':elements.Space(np.int32,()),
                'teacher_valid':elements.Space(np.bool_,())}

    def training_loss_fn(self, data, effective_stepid):
        weight=float(self.config.bc_lambda)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('BC lambda must be finite and nonnegative')
        if weight == 0:
            return self.loss  # identical base RNG, forward, loss and update path.
        k=int(self.config.replay_context)
        y,v=data['teacher_action'][:,k:],data['teacher_valid'][:,k:]
        if y.shape != effective_stepid.shape[:2] or v.shape != y.shape:
            raise ValueError('S3C teacher effective shape differs')
        def bad(_):
            jax.debug.callback(_raise_shift)
            return jnp.int32(0)
        jax.lax.cond(jnp.all(data['stepid'][:,k:]==effective_stepid),
                     lambda _:jnp.int32(0),bad,None)
        return partial(self.s3c_loss,teacher_y=y,teacher_valid=v)

    def bc_loss(self, repfeat, mask, y, valid):
        valid=jnp.asarray(valid,bool)
        safe_mask=jnp.where(valid[...,None],mask>.5,
                            jnp.broadcast_to(jnp.array([True,False]),mask.shape))
        safe_y=jnp.where(valid,y,0)
        feat=jax.lax.stop_gradient(self.feat2tensor(repfeat))
        dist=self._masked_policy(self.pol(feat,bdims=2),safe_mask,False)[self.action_key]
        nll=-dist.logp(safe_y)
        vf=valid.astype(jnp.float32);count=vf.sum();denom=jnp.maximum(count,1.)
        loss=jnp.where(count>0,(nll*vf).sum()/denom,0.)
        padd=jax.nn.softmax(dist.logits,-1)[...,1]
        predicted=padd>.5;actual=y==1
        metrics={'bc/nll':loss,'bc/valid_count':count,
            'bc/add_count':(vf*actual).sum(),'bc/wait_count':(vf*~actual).sum(),
            'bc/tp':(vf*(predicted&actual)).sum(),'bc/fp':(vf*(predicted&~actual)).sum(),
            'bc/fn':(vf*(~predicted&actual)).sum(),'bc/tn':(vf*(~predicted&~actual)).sum(),
            'bc/accuracy':(vf*(predicted==actual)).sum()/denom,
            'bc/always_wait_accuracy':(vf*~actual).sum()/denom,
            'bc/p_add_mean':(vf*padd).sum()/denom}
        return loss,metrics

    def s3c_loss(self,carry,obs,prevact,training,*,teacher_y,teacher_valid):
        base,(carry,entries,outs,metrics)=super().loss(carry,obs,prevact,training)
        bc,extra=self.bc_loss(outs['repfeat'],obs['action_mask'],teacher_y,teacher_valid)
        gate=self.avail_actor_gate.read().astype(jnp.float32)
        weight=jnp.float32(self.config.bc_lambda)
        total=base+gate*weight*bc
        metrics={**metrics,**extra,'bc/base_loss':base,'bc/total_loss':total,
            'bc/lambda':weight,'bc/gate_ready':gate,
            'bc/active':gate*(weight>0)*(extra['bc/valid_count']>0)}
        return total,(carry,entries,outs,metrics)
