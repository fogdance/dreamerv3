import elements
import pytest

from dreamerv3 import main as dreamer_main


def _base_config(**overrides):
  values = dict(
      seed=0,
      script='train',
      experiment_seed=101,
      dreamer=dict(seed=101),
      env=dict(train_seed=0, eval_seed=0),
      replay=dict(seed=0),
      audit=dict(
          matched_random_seed=20260615,
          entry_eval_version='entry_eval_test',
          entry_eval_config='/data/logdir/trading_contracts/test/configs/entry_eval/test.yaml',
          split_manifest_hash='abc123',
          execution_timing='signal_on_close_plus_spread'),
      seed_protocol=dict(
          experiment_seed=0,
          dreamer_seed=0,
          train_env_seed=101,
          replay_seed=202,
          eval_env_seed=0,
          matched_random_seed=20260615,
          config_seed=0,
          config_seed_sync='',
          dreamer_seed_source='dreamer.seed',
          train_env_seed_source='experiment_seed*1000+101',
          replay_seed_source='experiment_seed*1000+202',
          eval_env_seed_source='env.eval_seed',
          matched_random_seed_source='audit.matched_random_seed',
          entry_eval_version='',
          entry_eval_config='',
          split_manifest_hash='',
          execution_timing='',
          train_env_seed_usage='',
          replay_seed_usage='',
          eval_env_seed_usage='',
          eval_policy_mode='',
          matched_random_seed_usage='',
          gymnasium_constructor_seed_usage='',
          eval_determinism=''),
  )
  values.update(overrides)
  return elements.Config(values)


def test_seed_protocol_derives_env_and_replay_seeds():
  config = dreamer_main._resolve_seed_protocol(_base_config())

  assert config.seed == 101
  assert config.seed_protocol.config_seed == 101
  assert config.seed_protocol.experiment_seed == 101
  assert config.seed_protocol.dreamer_seed == 101
  assert config.seed_protocol.train_env_seed == 101101
  assert config.seed_protocol.replay_seed == 101202
  assert config.env.train_seed == 101101
  assert config.replay.seed == 101202
  assert config.seed_protocol.eval_env_seed == 0
  assert config.seed_protocol.matched_random_seed == 20260615
  assert config.seed_protocol.entry_eval_version == 'entry_eval_test'
  assert config.seed_protocol.entry_eval_config == '/data/logdir/trading_contracts/test/configs/entry_eval/test.yaml'
  assert config.seed_protocol.execution_timing == 'signal_on_close_plus_spread'
  assert config.seed_protocol.train_env_seed_usage == 'first_reset_only_per_env'
  assert config.seed_protocol.eval_env_seed_usage == 'not_used_per_day_forced_start'
  assert config.seed_protocol.eval_policy_mode == 'pred'
  assert config.seed_protocol.matched_random_seed_usage == 'fixed_across_dreamer_seeds'


def test_legacy_seed_sets_dreamer_seed_when_dreamer_seed_default():
  config = dreamer_main._resolve_seed_protocol(_base_config(
      seed=303,
      dreamer=dict(seed=0)))

  assert config.seed == 303
  assert config.dreamer.seed == 303
  assert config.seed_protocol.dreamer_seed == 303
  assert config.seed_protocol.dreamer_seed_source == 'seed'


def test_seed_and_dreamer_seed_mismatch_fails_fast():
  with pytest.raises(ValueError, match='must match'):
    dreamer_main._resolve_seed_protocol(_base_config(
        seed=101,
        dreamer=dict(seed=202)))


def test_seed_protocol_honors_explicit_env_and_replay_seeds():
  config = dreamer_main._resolve_seed_protocol(_base_config(
      env=dict(train_seed=7, eval_seed=8),
      replay=dict(seed=9)))

  assert config.seed_protocol.train_env_seed == 7
  assert config.seed_protocol.eval_env_seed == 8
  assert config.seed_protocol.replay_seed == 9


def test_seed_protocol_requires_explicit_dreamer_seed():
  with pytest.raises(ValueError, match='dreamer.seed'):
    dreamer_main._resolve_seed_protocol(_base_config(dreamer=dict(seed=None)))


def test_env_reset_seed_is_per_worker_train_seed():
  config = dreamer_main._resolve_seed_protocol(_base_config())
  config = config.update(script='train')

  assert dreamer_main._env_reset_seed(config, 0) == 101101
  assert dreamer_main._env_reset_seed(config, 3) == 101104


def test_env_reset_seed_uses_eval_seed_for_eval_scripts():
  config = dreamer_main._resolve_seed_protocol(_base_config(
      env=dict(train_seed=101101, eval_seed=5)))
  config = config.update(script='eval_only')

  assert dreamer_main._env_reset_seed(config, 0) == 5
  assert dreamer_main._env_reset_seed(config, 2) == 7


def test_make_replay_passes_replay_seed(monkeypatch, tmp_path):
  captured = {}

  class FakeReplay:

    def __init__(self, **kwargs):
      captured.update(kwargs)

  monkeypatch.setattr(dreamer_main.embodied.replay, 'Replay', FakeReplay)
  config = elements.Config(dict(
      batch_length=32,
      batch_size=8,
      report_length=16,
      consec_train=1,
      consec_report=1,
      replay_context=0,
      logdir=str(tmp_path),
      replicas=1,
      replica=0,
      jax=dict(compute_dtype='float32'),
      replay=dict(
          size=100000,
          online=True,
          chunksize=1024,
          fracs=dict(uniform=1.0, priority=0.0, recency=0.0),
          prio=dict(
              exponent=0.8,
              maxfrac=0.5,
              initial=float('inf'),
              zero_on_sample=True),
          recexp=1.0),
      seed_protocol=dict(replay_seed=101202),
  ))

  dreamer_main.make_replay(config, 'replay')

  assert captured['seed'] == 101202


def test_gymnasium_use_seed_is_rejected_for_trading_env():
  config = elements.Config(dict(
      task='gymnasium_CustomTradingEnv-v0',
      seed=101,
      script='train',
      env=dict(
          train_seed=None,
          eval_seed=0,
          gymnasium=dict(config_path='tests/test.yaml', use_seed=True)),
      seed_protocol=dict(train_env_seed=101101, eval_env_seed=0),
  ))

  with pytest.raises(ValueError, match='env.gymnasium.use_seed=True'):
    dreamer_main.make_env(config, 0)
