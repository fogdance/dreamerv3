"""Original Agent policy with safe probability diagnostics for masked ADD."""
import jax
from .agent import Agent


class S3AAgent(Agent):
    def policy(self,carry,obs,mode='train'):
        original=self.config
        # Original diagnostics call prob(ADD), which rejects illegal events.
        # The policy, Encoder, RSSM and reset/action carry remain the original.
        self.config=original.update(update_diagnostics=False)
        try:
            carry,action,out=super().policy(carry,obs,mode)
        finally:
            self.config=original
        dist=self._masked_policy(self.pol(self.feat2tensor(carry[1]),1),
                                 obs['action_mask'],False)[self.action_key]
        out['p_add']=jax.nn.softmax(dist.logits)[...,1]
        return carry,action,out
