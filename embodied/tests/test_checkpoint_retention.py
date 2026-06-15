import json
import pathlib
from functools import partial as bind
import time

import elements
import embodied
import numpy as np

from embodied.envs import dummy
from embodied.run.checkpoint_retention import StepCheckpointRetention


class FakeCheckpoint:

  def __init__(self, latest):
    self._latest = pathlib.Path(latest)

  def latest(self):
    return self._latest


def test_step_checkpoint_retention_copies_first_save_after_target(tmp_path):
  source = tmp_path / 'ckpt' / 'source'
  source.mkdir(parents=True)
  (source / 'done').write_text('')
  (source / 'step.pkl').write_bytes(b'test')

  retention = StepCheckpointRetention(tmp_path, {
      'enabled': True,
      'steps': [500, 700],
      'directory': 'ckpt_retained',
  })
  checkpoint = FakeCheckpoint(source)

  retention.maybe_retain(checkpoint, 499)
  assert not (tmp_path / 'ckpt_retained').exists()

  retention.maybe_retain(checkpoint, 510)
  retained = tmp_path / 'ckpt_retained' / 'step_000000000500'
  assert retained.is_dir()
  assert (retained / 'done').exists()
  assert not (tmp_path / 'ckpt_retained' / 'step_000000000700').exists()

  manifest = json.loads(
      (tmp_path / 'ckpt_retained' / 'retention_manifest.json').read_text())
  assert manifest['targets']['500']['retained_at_step'] == 510


def test_step_checkpoint_retention_accepts_comma_separated_steps(tmp_path):
  source = tmp_path / 'ckpt' / 'source'
  source.mkdir(parents=True)
  (source / 'done').write_text('')

  retention = StepCheckpointRetention(tmp_path, {
      'enabled': True,
      'steps': '500, 700',
  })
  retention.maybe_retain(FakeCheckpoint(source), 800)

  assert (tmp_path / 'ckpt_retained' / 'step_000000000500').is_dir()
  assert (tmp_path / 'ckpt_retained' / 'step_000000000700').is_dir()


def test_step_checkpoint_retention_writes_seed_metadata(tmp_path):
  source = tmp_path / 'ckpt' / 'source'
  source.mkdir(parents=True)
  (source / 'done').write_text('')
  metadata = {
      'experiment_seed': 101,
      'dreamer_seed': 101,
      'train_env_seed': 101101,
      'replay_seed': 101202,
      'eval_env_seed': 0,
      'matched_random_seed': 20260615,
  }

  retention = StepCheckpointRetention(tmp_path, {
      'enabled': True,
      'steps': [10],
      'directory': 'ckpt_retained',
  }, metadata=metadata)
  retention.maybe_retain(FakeCheckpoint(source), 12)

  manifest = json.loads(
      (tmp_path / 'ckpt_retained' / 'retention_manifest.json').read_text())
  assert manifest['metadata'] == metadata
  assert manifest['targets']['10']['retained_at_step'] == 12


def test_step_checkpoint_retention_seed_metadata_is_repeatable(tmp_path):
  metadata = {
      'experiment_seed': 101,
      'dreamer_seed': 101,
      'train_env_seed': 101101,
      'replay_seed': 101202,
      'eval_env_seed': 0,
      'matched_random_seed': 20260615,
  }
  manifests = []
  for name in ('run_a', 'run_b'):
    root = tmp_path / name
    source = root / 'ckpt' / 'source'
    source.mkdir(parents=True)
    (source / 'done').write_text('')
    retention = StepCheckpointRetention(root, {
        'enabled': True,
        'steps': [10],
        'directory': 'ckpt_retained',
    }, metadata=metadata)
    retention.maybe_retain(FakeCheckpoint(source), 12)
    manifest = json.loads(
        (root / 'ckpt_retained' / 'retention_manifest.json').read_text())
    manifests.append(manifest['metadata'])

  assert manifests == [metadata, metadata]


class MinimalAgent:

  def __init__(self, obs_space, act_space):
    self.obs_space = obs_space
    self.act_space = act_space
    self._stats = {
        'env_steps': 0, 'replay_steps': 0, 'reports': 0,
        'saves': 0, 'loads': 0, 'created': time.time()}

  def stream(self, stream):
    return stream

  def stats(self):
    stats = self._stats.copy()
    stats['lifetime'] = time.time() - stats.pop('created')
    return stats

  def init_policy(self, batch_size):
    return (np.zeros(batch_size, np.float32),)

  def init_train(self, batch_size):
    return (np.zeros(batch_size, np.float32),)

  def init_report(self, batch_size):
    return ()

  def policy(self, carry, obs, mode='train'):
    del mode
    batch_size = len(obs['is_first'])
    self._stats['env_steps'] += batch_size
    actions = {
        key: np.stack([space.sample() for _ in range(batch_size)])
        for key, space in self.act_space.items() if key != 'reset'}
    return carry, actions, {}

  def train(self, carry, data):
    batch_size, length = data['is_first'].shape
    self._stats['replay_steps'] += batch_size * length
    return carry, {}, {'dummy_loss': np.float32(0.0)}

  def report(self, carry, data):
    del data
    self._stats['reports'] += 1
    return carry, {'scalar': np.float32(0.0)}

  def save(self):
    self._stats['saves'] += 1
    return self._stats.copy()

  def load(self, data):
    self._stats = data
    self._stats['loads'] += 1


def test_train_loop_retains_step_targets_and_final_due_target(tmp_path):
  env = dummy.Dummy('disc', size=(32, 32), length=50)
  agent = MinimalAgent(env.obs_space, env.act_space)
  env.close()

  warmup_config = dict(
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
  )
  args = elements.Config(
      steps=1000,
      train_ratio=32.0,
      log_every=0.1,
      report_every=0.2,
      save_every=0.2,
      report_batches=1,
      from_checkpoint='',
      usage=dict(
          psutil=False, nvsmi=False, gputil=False, malloc=False, gc=False),
      debug=True,
      logdir=str(tmp_path),
      envs=2,
      batch_size=4,
      batch_length=16,
      replay_context=0,
      report_length=8,
      consec_report=1,
      checkpoint_retention=dict(
          enabled=True, steps=[200, 500, 900], directory='ckpt_retained'),
      seed_protocol=dict(
          experiment_seed=101,
          dreamer_seed=101,
          train_env_seed=101101,
          replay_seed=101202,
          eval_env_seed=0,
          matched_random_seed=20260615),
      action_mask_warmup=warmup_config,
      action_mask_actor_initial=False,
  )

  def make_env(index):
    del index
    return dummy.Dummy('disc', size=(32, 32), length=50)

  def make_replay(args):
    return embodied.replay.Replay(length=args.batch_length, capacity=1e4)

  def make_stream(replay, mode):
    stream = embodied.streams.Stateless(bind(
        replay.sample, args.batch_size, mode))
    return embodied.streams.Consec(
        stream,
        length=args.batch_length if mode == 'train' else args.report_length,
        consec=1,
        prefix=args.replay_context,
        strict=(mode == 'train'),
        contiguous=True)

  def make_logger():
    return elements.Logger(elements.Counter(), [
        elements.logger.TerminalOutput()])

  embodied.run.train(
      lambda: agent,
      bind(make_replay, args),
      make_env,
      make_stream,
      make_logger,
      args)

  retained = sorted(
      path.name for path in (tmp_path / 'ckpt_retained').iterdir()
      if path.is_dir())
  assert retained == [
      'step_000000000200',
      'step_000000000500',
      'step_000000000900',
  ]
  manifest = json.loads(
      (tmp_path / 'ckpt_retained' / 'retention_manifest.json').read_text())
  assert manifest['metadata']['dreamer_seed'] == 101
  assert manifest['metadata']['replay_seed'] == 101202
  assert sorted(manifest['targets']) == ['200', '500', '900']
  for target, record in manifest['targets'].items():
    assert pathlib.Path(record['path']).is_dir()
    assert int(record['retained_at_step']) >= int(target)
