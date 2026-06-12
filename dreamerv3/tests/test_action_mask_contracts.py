from types import SimpleNamespace

import elements
import embodied
import embodied.jax
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3.agent import Agent, hard_nonempty_mask
from dreamerv3.rssm import RSSM
from embodied.jax.outs import MaskedCategorical


def _f32(value):
  return np.asarray(value, dtype=np.float32)


class _MaskSequenceEnv:

  def __init__(self, episode_length=5):
    self.episode_length = episode_length
    self.index = 0
    self.done = True

  @property
  def obs_space(self):
    return {
        'state': elements.Space(np.int32),
        'action_mask': elements.Space(np.float32, (3,), 0.0, 1.0),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, 3),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if bool(action['reset']) or self.done:
      self.index = 0
      self.done = False
      is_first = True
    else:
      self.index += 1
      is_first = False
    self.done = self.index == self.episode_length - 1
    mask = np.eye(3, dtype=np.float32)[self.index % 3]
    return {
        'state': np.int32(self.index),
        'action_mask': mask,
        'reward': np.float32(0.0),
        'is_first': np.bool_(is_first),
        'is_last': np.bool_(self.done),
        'is_terminal': np.bool_(False),
    }

  def close(self):
    pass


def _masked_policy(carry, obs, **kwargs):
  del kwargs
  action = np.argmax(obs['action_mask'], axis=-1).astype(np.int32)
  return carry, {'action': action}, {}


def test_driver_and_replay_preserve_mask_action_alignment_across_resets():
  replay = embodied.Replay(length=6, capacity=100, chunksize=8, seed=0)
  transitions = []
  driver = embodied.Driver([_MaskSequenceEnv], parallel=False)
  driver.on_step(lambda tran, worker: transitions.append(tran.copy()))
  driver.on_step(replay.add)
  try:
    driver(_masked_policy, steps=20)
  finally:
    driver.close()

  assert any(bool(tran['is_first']) for tran in transitions)
  assert any(bool(tran['is_last']) for tran in transitions)
  for tran in transitions:
    expected = int(tran['state']) % 3
    assert int(np.argmax(tran['action_mask'])) == expected
    if not bool(tran['is_last']):
      assert int(tran['action']) == expected

  batch = replay.sample(2)
  expected = batch['state'] % 3
  assert np.array_equal(np.argmax(batch['action_mask'], axis=-1), expected)
  executable = ~batch['is_last']
  assert np.array_equal(batch['action'][executable], expected[executable])


def test_apply_replay_context_shifts_actions_but_not_masks():
  agent = object.__new__(Agent)
  agent.obs_space = {
      'action_mask': object(),
      'is_first': object(),
      'is_last': object(),
      'is_terminal': object(),
      'reward': object(),
  }
  agent.act_space = {'action': object()}
  agent.config = SimpleNamespace(replay_context=0)
  data = {
      'action_mask': jnp.asarray([[
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
      ]]),
      'action': jnp.asarray([[0, 1, 2]], jnp.int32),
      'is_first': jnp.asarray([[True, False, False]]),
      'is_last': jnp.asarray([[False, False, True]]),
      'is_terminal': jnp.asarray([[False, False, False]]),
      'reward': jnp.zeros((1, 3), jnp.float32),
      'stepid': jnp.zeros((1, 3, 20), jnp.uint8),
  }
  carry = ({}, {}, {}, {'action': jnp.asarray([2], jnp.int32)})

  _, obs, prevact, _ = agent._apply_replay_context(carry, data)

  assert np.array_equal(np.asarray(obs['action_mask']), np.asarray(data['action_mask']))
  assert np.array_equal(np.asarray(prevact['action']), [[2, 0, 1]])


def _make_rssm(name='dyn'):
  act_space = {'action': elements.Space(np.int32, (), 0, 3)}
  return RSSM(
      act_space,
      deter=8,
      hidden=8,
      stoch=2,
      classes=3,
      blocks=2,
      imglayers=1,
      obslayers=1,
      dynlayers=1,
      name=name,
  )


def _rssm_loss_fn(dyn):
  return lambda carry, tokens, actions, reset: dyn.loss(
      carry, tokens, actions, reset, False)


def test_prior_feature_does_not_read_current_observation_and_resets_history():
  dyn = _make_rssm()
  loss_fn = _rssm_loss_fn(dyn)
  carry = dyn.initial(1)
  tokens = jnp.arange(20, dtype=jnp.float32).reshape(1, 5, 4) / 10
  actions = {'action': jnp.asarray([[0, 1, 2, 0, 1]], jnp.int32)}
  reset = jnp.asarray([[True, False, False, False, False]])
  params = nj.init(loss_fn)({}, carry, tokens, actions, reset, seed=0)

  changed_current = tokens.at[:, 2].set(1000.0)
  _, base = nj.pure(loss_fn)(params, carry, tokens, actions, reset, seed=1)
  _, changed = nj.pure(loss_fn)(
      params, carry, changed_current, actions, reset, seed=1)
  base_post, base_prior = base[3], base[4]
  changed_post, changed_prior = changed[3], changed[4]
  assert np.allclose(_f32(base_prior['deter'][:, 2]), _f32(changed_prior['deter'][:, 2]))
  assert np.allclose(_f32(base_prior['logit'][:, 2]), _f32(changed_prior['logit'][:, 2]))
  assert not np.allclose(_f32(base_post['logit'][:, 2]), _f32(changed_post['logit'][:, 2]))

  reset = jnp.asarray([[True, False, False, True, False]])
  changed_history = tokens.at[:, :3].set(-1000.0)
  changed_actions = {'action': actions['action'].at[:, :3].set(2)}
  params = nj.init(loss_fn)({}, carry, tokens, actions, reset, seed=2)
  _, base = nj.pure(loss_fn)(params, carry, tokens, actions, reset, seed=3)
  _, changed = nj.pure(loss_fn)(
      params, carry, changed_history, changed_actions, reset, seed=3)
  assert np.allclose(_f32(base[4]['deter'][:, 3]), _f32(changed[4]['deter'][:, 3]))
  assert np.allclose(_f32(base[4]['logit'][:, 3]), _f32(changed[4]['logit'][:, 3]))


def test_rssm_imagination_actions_use_current_not_next_latent():
  dyn = _make_rssm()
  start = dyn.initial(3)
  start['deter'] = jnp.asarray([
      [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  ])
  start['stoch'] = jnp.repeat(
      jax.nn.one_hot(jnp.arange(3), 3)[:, None, :], 2, axis=1)

  def policy(feat):
    return {'action': jnp.argmax(feat['stoch'][..., 0, :], -1).astype(jnp.int32)}

  imagine_fn = lambda carry: dyn.imagine(carry, policy, 4, False)
  params = nj.init(imagine_fn)({}, start, seed=0)
  _, (_, imgfeat, imgact) = nj.pure(imagine_fn)(params, start, seed=1)
  current = {
      key: jnp.concatenate([value[:, None], imgfeat[key][:, :-1]], 1)
      for key, value in start.items()
  }
  expected = policy(current)['action']
  assert np.array_equal(np.asarray(imgact['action']), np.asarray(expected))

  next_actions = policy(imgfeat)['action']
  assert np.any(np.asarray(expected) != np.asarray(next_actions))


def test_actor_hard_mask_has_zero_availability_gradient():
  avail_logits = jnp.asarray([2.0, 1.0, -2.0])
  policy_logits = jnp.asarray([0.2, -0.3, 4.0])

  def actor_loss(avail, policy):
    mask, _ = hard_nonempty_mask(jax.nn.sigmoid(avail), 0.5)
    dist = MaskedCategorical(policy, mask, unimix=0.1, allow_fallback=True)
    return -dist.logp(jnp.asarray(0)) - 0.01 * dist.entropy()

  avail_grad, policy_grad = jax.grad(actor_loss, (0, 1))(
      avail_logits, policy_logits)
  assert np.array_equal(np.asarray(avail_grad), np.zeros(3, np.float32))
  assert np.linalg.norm(np.asarray(policy_grad)) > 0
  assert float(policy_grad[2]) == 0.0


class _PriorAvailabilityHarness(nj.Module):

  def __init__(self):
    self.dyn = _make_rssm(name='dyn')
    space = elements.Space(bool, (3,), 0, 2)
    self.avail = embodied.jax.MLPHead(
        space, output='binary', layers=1, units=8, name='avail')

  def __call__(self, tokens, actions, reset, target):
    carry = self.dyn.initial(tokens.shape[0])
    _, _, _, _, priorfeat, _ = self.dyn.loss(
        carry, tokens, actions, reset, False)
    inp = jnp.concatenate([
        priorfeat['deter'],
        priorfeat['stoch'].reshape((*priorfeat['stoch'].shape[:-2], -1)),
    ], -1)
    return self.avail(inp, 2).loss(target).mean()


def test_prior_availability_loss_updates_head_and_dynamics():
  model = _PriorAvailabilityHarness(name='model')
  tokens = jnp.arange(40, dtype=jnp.float32).reshape(2, 5, 4) / 10
  actions = {'action': jnp.asarray([
      [0, 1, 2, 0, 1],
      [1, 2, 0, 1, 2],
  ], jnp.int32)}
  reset = jnp.asarray([
      [True, False, False, False, False],
      [True, False, False, False, False],
  ])
  target = jax.nn.one_hot(
      (jnp.arange(5)[None] + jnp.arange(2)[:, None]) % 3,
      3,
      dtype=bool,
  )
  params = nj.init(model)({}, tokens, actions, reset, target, seed=0)

  def grad_loss(*args):
    return nj.grad(model, params.keys())(*args)

  _, (_, _, grads) = nj.pure(grad_loss)(
      params, tokens, actions, reset, target, seed=1)
  avail_norm = sum(
      float(jnp.abs(value).sum())
      for key, value in grads.items()
      if '/avail/' in key
  )
  dyn_norm = sum(
      float(jnp.abs(value).sum())
      for key, value in grads.items()
      if '/dyn/' in key
  )
  assert avail_norm > 0
  assert dyn_norm > 0
