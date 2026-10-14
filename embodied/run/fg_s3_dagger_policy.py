"""Host-side TRAIN mixing; only executed actions enter the Actor's carry."""
import copy

import numpy as np


def replace_prevact(carry, action):
    """Patch Dreamer's real tuple-of-trees with leaf lists from Agent._split."""
    if not isinstance(carry, tuple) or len(carry) != 4 or set(carry[-1]) != {'action'}:
        raise ValueError('Unsupported Dreamer policy carry contract')
    old = carry[-1]['action']
    if isinstance(old, list):
        if len(old) != len(action):
            raise ValueError('Policy carry batch differs from executed action')
        value = [np.asarray(action[i], dtype=old[i].dtype) for i in range(len(old))]
    else:
        value = np.asarray(action, dtype=old.dtype)
        if value.shape != old.shape:
            raise ValueError('Policy previous-action shape changed')
    return (*carry[:3], {'action': value})


class ScriptedActor:
    def __init__(self, seed, probability):
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Scripted ADD probability must be in [0, 1]')
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self.probability = probability

    def __call__(self, carry, obs):
        action = ((self.rng.random(len(obs['is_first'])) < self.probability)
                  & obs['action_mask'][:, 1].astype(bool)
                  & ~obs['is_last']).astype(np.int32)
        return {'prev_action': action.copy()}, {'action': action}, {}

    def save(self):
        return copy.deepcopy(self.rng.bit_generator.state)

    def load(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state)


class MixedExecutionPolicy:
    """Driver-compatible policy with a separate, protected decision mailbox.

    context() and query() are TRAIN host callbacks. Their results never enter
    actor obs, actor carry, or the returned policy outs / native Replay fields.
    """
    def __init__(self, actor, query, context, *, beta, seed, dreamer_carry=False,
                 observation_schema=None):
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('Oracle mixing probability must be in [0, 1]')
        self.actor, self.query, self.context = actor, query, context
        self.beta, self.dreamer_carry = beta, dreamer_carry
        from gym_trading_env.research.fg_s3_e_observation import fields_for
        self.observation_schema = observation_schema
        self.observation_keys = set(fields_for(observation_schema)) - {'action'}
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.decision = None

    def __call__(self, carry, obs):
        if any(k.startswith(('teacher', 'oracle', 'q_')) for k in obs):
            raise ValueError('Teacher fields leaked into policy observation')
        if len(obs['is_first']) != 1:
            raise ValueError('D0 supports one actual Env')
        if set(obs) != self.observation_keys:
            raise ValueError('Unknown fields in the original policy observation contract')
        if self.observation_schema is not None:
            from gym_trading_env.research.fg_s3_e_observation import validate_e_observation
            validate_e_observation(obs, batched=True)
        previous = None
        if carry is not None:
            previous = int(np.asarray(carry[-1]['action'] if self.dreamer_carry
                                      else carry['prev_action']).reshape(-1)[0])
        next_carry, proposed, outs = self.actor(carry, obs)
        if any(k.startswith(('teacher', 'oracle', 'q_')) for k in outs):
            raise ValueError('Teacher fields leaked into Actor output')
        p = int(proposed['action'][0])
        if p not in (0, 1) or obs['action_mask'][0, 0] != 1:
            raise ValueError('Invalid Actor proposal or WAIT mask')
        ctx = self.context(obs)
        answer = self.query(ctx, obs['action_mask'][0])
        active = not bool(obs['is_last'][0])
        draw = float(self.rng.random()) if active else None
        use_oracle = bool(active and answer['valid'] and answer['certified'] and draw < self.beta)
        chosen = answer['action'] if use_oracle else p
        executed = int(active and chosen == 1 and bool(obs['action_mask'][0, 1]))
        action = np.array([executed], np.int32)
        if self.dreamer_carry:
            next_carry = replace_prevact(next_carry, action)
        else:
            next_carry = {'prev_action': action.copy()}
        self.decision = dict(context=ctx, oracle=answer, actor_proposal=p,
                             actor_prev_action=previous, executed_action=executed,
                             oracle_selected=use_oracle, beta=self.beta, beta_draw=draw)
        return next_carry, {'action': action}, outs

    def save(self):
        if not hasattr(self.actor, 'save'):
            raise ValueError('Agent exact state must be separately registered for model resume')
        return dict(beta=self.beta, rng=copy.deepcopy(self.rng.bit_generator.state),
                    actor=self.actor.save(), decision=copy.deepcopy(self.decision))

    def load(self, state):
        if state['beta'] != self.beta:
            raise ValueError('D0 mixing policy identity changed')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.actor.load(state['actor'])
        self.decision = copy.deepcopy(state['decision'])
