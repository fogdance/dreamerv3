"""The offline context contract, including causality and stop-gradient."""
from types import SimpleNamespace

import elements
import jax
import jax.numpy as jnp
import numpy as np

from dreamerv3.agent import Agent


class Encoder:
  entry_space = {}

  def __call__(self, carry, obs, reset, **kwargs):
    value = obs['market_seq']
    return {'h': value[:, -1]}, {}, value


class Dynamics:
  entry_space = {'deter': None, 'stoch': None}

  def observe(self, carry, tokens, action, reset, **kwargs):
    np.testing.assert_array_equal(action['action'], 0)
    return {'deter': tokens[:, -1]}, {}, {}


def model():
  initial = lambda b: ({'h': jnp.zeros(b)}, {'deter': jnp.zeros(b)}, {}, {'action': jnp.zeros(b, jnp.int32)})
  return SimpleNamespace(config=elements.Config(dict(replay_context=1, replay_context_mode='recompute_prefix')),
      obs_space={'market_seq': None, 'is_first': None, 'reward': None}, act_space={'action': None},
      enc=Encoder(), dyn=Dynamics(), dec=SimpleNamespace(entry_space={}), init_train=initial)


def batch():
  return dict(market_seq=jnp.arange(65, dtype=jnp.float32)[None],
      is_first=jnp.array([[True] + [False] * 64]), reward=jnp.arange(65, dtype=jnp.float32)[None],
      action=jnp.arange(65, dtype=jnp.int32)[None] % 2,
      stepid=jnp.arange(65, dtype=jnp.uint8)[None, :, None], consec=jnp.zeros((1,65),jnp.int32))


def test_reconstruction_contract_alignment_causality_and_stopped_gradient():
  agent = model()
  assert set(Agent.ext_space.fget(agent)) == {'consec', 'stepid'}
  data = batch()
  carry, obs, prevact, stepid = Agent._apply_replay_context(agent, agent.init_train(1), data)
  np.testing.assert_array_equal(obs['reward'], data['reward'][:, 1:])
  np.testing.assert_array_equal(prevact['action'], data['action'][:, :-1])
  np.testing.assert_array_equal(stepid, data['stepid'][:, 1:])
  assert obs['market_seq'].shape == (1,64)
  changed = dict(data, market_seq=data['market_seq'].at[:, 1:].set(999))
  other = Agent._apply_replay_context(agent, agent.init_train(1), changed)[0]
  np.testing.assert_array_equal(carry[1]['deter'], other[1]['deter'])
  def context_value(value):
    incoming = dict(data, market_seq=value)
    return Agent._apply_replay_context(agent, agent.init_train(1), incoming)[0][1]['deter'].sum()
  np.testing.assert_array_equal(jax.grad(context_value)(data['market_seq']), 0)


def test_legacy_input_contract_still_requires_latent_cache():
  agent = model()
  agent.config = agent.config.update(replay_context_mode='stored')
  assert {'dyn/deter', 'dyn/stoch'} <= set(Agent.ext_space.fget(agent))
