from functools import partial as bind

import elements
import embodied
import numpy as np

import utils


class TrainLoopAgent(utils.TestAgent):

  def stream(self, stream):
    return stream

  def train(self, carry, data):
    expected = sorted(
        (set(self.obs_space) | {k for k in self.act_space if k != 'reset'})
        | {'consec', 'stepid'})
    assert sorted(data.keys()) == expected, (sorted(data.keys()), expected)
    batch_size, length = data['count'].shape
    carry, = carry
    assert carry.shape == (batch_size,)
    assert not any(k.startswith('log/') for k in data.keys())
    self._stats['replay_steps'] += batch_size * length
    for t in range(length):
      current = data['count'][:, t]
      reset = data['is_first'][:, t]
      target = (1 - reset) * (carry + 1) + reset * current
      assert (current == target).all()
      carry = current

    outs = {}
    metrics = {}
    return (carry,), outs, metrics


class TestTrain:

  def test_run_loop(self, tmpdir):
    args = self._make_args(tmpdir)
    agent = self._make_agent()
    embodied.run.train(
        lambda: agent, bind(self._make_replay, args),
        self._make_env, bind(self._make_stream, args),
        self._make_logger, args)
    stats = agent.stats()
    print('Stats:', stats)
    replay_steps = args.steps * args.train_ratio
    assert stats['lifetime'] >= 1  # Otherwise decrease log and ckpt interval.
    assert np.allclose(stats['env_steps'], args.steps, 100, 0.1)
    assert np.allclose(stats['replay_steps'], replay_steps, 100, 0.1)
    assert stats['reports'] >= 1
    assert stats['saves'] >= 2
    assert stats['loads'] == 0
    args = args.update(steps=2 * args.steps)
    embodied.run.train(
        lambda: agent, bind(self._make_replay, args),
        self._make_env, bind(self._make_stream, args),
        self._make_logger, args)
    stats = agent.stats()
    assert stats['loads'] == 1
    assert np.allclose(stats['env_steps'], args.steps, 100, 0.1)

  def _make_agent(self):
    env = self._make_env(0)
    agent = TrainLoopAgent(env.obs_space, env.act_space)
    env.close()
    return agent

  def _make_env(self, index):
    from embodied.envs import dummy
    return dummy.Dummy('disc', size=(64, 64), length=100)

  def _make_replay(self, args):
    kwargs = {'length': args.batch_length, 'capacity': 1e4}
    return embodied.replay.Replay(**kwargs)

  def _make_stream(self, args, replay, mode):
    fn = bind(replay.sample, args.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=args.batch_length if mode == 'train' else args.report_length,
        consec=1,
        prefix=args.replay_context,
        strict=(mode == 'train'),
        contiguous=True)
    return stream

  def _make_logger(self):
    return elements.Logger(elements.Counter(), [
        elements.logger.TerminalOutput(),
    ])

  def _make_args(self, logdir):
    return elements.Config(
        steps=1000,
        train_ratio=32.0,
        log_every=0.1,
        report_every=0.2,
        save_every=0.2,
        report_batches=1,
        from_checkpoint='',
        usage=dict(psutil=True, nvsmi=False, gputil=False, malloc=False, gc=False),
        debug=False,
        logdir=str(logdir),
        envs=4,
        batch_size=8,
        batch_length=16,
        replay_context=0,
        report_length=8,
        consec_report=1,
        checkpoint_retention=dict(
            enabled=False, steps=[0], directory='ckpt_retained'),
        action_mask_warmup=dict(
            enabled=False,
            min_train_samples=25000,
            consecutive_reports=5,
            regression_reports=5,
            stop_on_regression=True,
            post_accuracy=0.995,
            post_exact=0.990,
            prior_accuracy=0.990,
            prior_exact=0.980,
            prior_fpr=0.005,
            img_fallback=0.0001,
        ),
        action_mask_actor_initial=False,
    )
