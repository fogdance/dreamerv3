from types import SimpleNamespace
import importlib

import elements
import pytest
from embodied.envs import dummy

from embodied.run.action_mask_guard import require_formal_actor
from embodied.run.action_mask_warmup import ActionMaskWarmup
from embodied.run.train import _apply_action_mask_warmup_gate


def _warmup_config(**overrides):
  values = dict(
      enabled=True,
      min_train_samples=100,
      consecutive_reports=3,
      regression_reports=2,
      stop_on_regression=True,
      post_accuracy=0.995,
      post_exact=0.990,
      prior_accuracy=0.990,
      prior_exact=0.980,
      prior_fpr=0.005,
      img_fallback=0.0001,
  )
  values.update(overrides)
  return SimpleNamespace(**values)


class _GateAgent:

  def __init__(self, enabled):
    self.enabled = bool(enabled)
    self.set_calls = []

  def get_avail_actor_enabled(self):
    return self.enabled

  def set_avail_actor_enabled(self, enabled):
    self.enabled = bool(enabled)
    self.set_calls.append(self.enabled)


def test_train_warmup_preserves_loaded_formal_actor_gate():
  warmup = ActionMaskWarmup(_warmup_config(), actor_enabled=False)
  agent = _GateAgent(True)

  _apply_action_mask_warmup_gate(agent, warmup, step=123)

  assert warmup.actor_enabled
  assert warmup.switched_step == 123
  assert agent.enabled
  assert agent.set_calls == [True]


def test_train_warmup_keeps_warmup_checkpoint_disabled():
  warmup = ActionMaskWarmup(_warmup_config(), actor_enabled=False)
  agent = _GateAgent(False)

  _apply_action_mask_warmup_gate(agent, warmup, step=123)

  assert not warmup.actor_enabled
  assert warmup.switched_step == -1
  assert not agent.enabled
  assert agent.set_calls == [False]


def test_require_formal_actor_rejects_missing_gate_interface():
  with pytest.raises(TypeError, match="get_avail_actor_enabled"):
    require_formal_actor(object(), "test")


def test_require_formal_actor_rejects_warmup_checkpoint():
  with pytest.raises(RuntimeError, match="avail_actor_gate/value=0"):
    require_formal_actor(_GateAgent(False), "test")


def test_require_formal_actor_accepts_formal_checkpoint():
  require_formal_actor(_GateAgent(True), "test")


class _FakeCheckpoint:

  def __init__(self, *args, **kwargs):
    pass

  def load(self, *args, **kwargs):
    pass


def _make_logger():
  return elements.Logger(elements.Counter(), [
      elements.logger.TerminalOutput()])


def _make_env(_index):
  return dummy.Dummy("disc", size=(32, 32), length=5)


def _eval_args(path):
  return elements.Config(
      from_checkpoint="fake",
      logdir=str(path),
      usage=dict(psutil=False, nvsmi=False, gputil=False, malloc=False, gc=False),
      envs=1,
      debug=True,
      log_every=999,
      steps=0,
  )


def test_eval_only_rejects_warmup_checkpoint(monkeypatch, tmp_path):
  module = importlib.import_module("embodied.run.eval_only")
  monkeypatch.setattr(module.elements, "Checkpoint", _FakeCheckpoint)

  with pytest.raises(RuntimeError, match="avail_actor_gate/value=0"):
    module.eval_only(
        lambda: _GateAgent(False),
        _make_env,
        _make_logger,
        _eval_args(tmp_path / "eval"))


def test_live_trading_rejects_warmup_checkpoint(monkeypatch, tmp_path):
  module = importlib.import_module("embodied.run.live_trading")
  monkeypatch.setattr(module.elements, "Checkpoint", _FakeCheckpoint)

  args = _eval_args(tmp_path / "live").update(envs=1)
  with pytest.raises(RuntimeError, match="avail_actor_gate/value=0"):
    module.live_trading(
        lambda: _GateAgent(False),
        _make_env,
        _make_logger,
        args)
