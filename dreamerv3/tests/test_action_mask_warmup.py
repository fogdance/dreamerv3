from types import SimpleNamespace

from embodied.run.action_mask_warmup import ActionMaskWarmup


def _config(**overrides):
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


def _metrics(**overrides):
  values = {
      "avail/post_accuracy": 0.999,
      "avail/post_exact_accuracy": 0.999,
      "avail/prior_accuracy": 0.999,
      "avail/prior_exact_accuracy": 0.999,
      "avail/prior_fpr": 0.001,
      "avail/img_fallback_rate": 0.0,
  }
  values.update(overrides)
  return values


def test_warmup_requires_consecutive_passing_reports():
  state = ActionMaskWarmup(_config())
  assert not state.ready(99)
  assert state.ready(100)
  assert not state.update(_metrics(), 10)
  assert not state.update(_metrics(**{"avail/prior_fpr": 0.1}), 20)
  assert not state.update(_metrics(), 30)
  assert not state.update(_metrics(), 40)
  assert state.update(_metrics(), 50)
  assert state.actor_enabled
  assert state.switched_step == 50


def test_warmup_state_round_trips_checkpoint():
  state = ActionMaskWarmup(_config())
  state.update(_metrics(), 10)
  restored = ActionMaskWarmup(_config())
  restored.load(state.save())
  assert restored.save() == state.save()
  assert not restored.update(_metrics(), 20)
  assert restored.update(_metrics(), 30)


def test_formal_training_stops_after_consecutive_regressions():
  state = ActionMaskWarmup(_config())
  state.load({
      "actor_enabled": True,
      "reports": 3,
      "consecutive_passes": 3,
      "consecutive_regressions": 0,
      "switched_step": 30,
      "last_passed": True,
  })
  bad = _metrics(**{"avail/prior_fpr": 0.1})
  assert not state.update(bad, 10)
  try:
    state.update(bad, 20)
  except RuntimeError as exc:
    assert "regressed" in str(exc)
  else:
    raise AssertionError("Availability regression did not stop training")


def test_disabled_warmup_preserves_explicit_actor_mode():
  state = ActionMaskWarmup(_config(enabled=False), actor_enabled=False)
  assert not state.actor_enabled
  assert not state.update({}, 10)


def test_auto_warmup_rejects_actor_bypass():
  try:
    ActionMaskWarmup(_config(), actor_enabled=True)
  except ValueError as exc:
    assert "avail_actor_enabled=False" in str(exc)
  else:
    raise AssertionError("Auto warm-up accepted an already enabled actor")
