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


def _fg_add_metrics(positives=32, post_tp=32, prior_tp=32):
  return _metrics(**{
      'avail/post_add_positives': positives,
      'avail/prior_add_positives': positives,
      'avail/post_add_recall': post_tp / positives if positives else float('nan'),
      'avail/prior_add_recall': prior_tp / positives if positives else float('nan'),
  })


def _fg_add_config(**overrides):
  return _config(add_recall_enabled=True, min_add_positives=16,
      post_add_recall=0.8, prior_add_recall=0.8, **overrides)


def test_fg_gate_rejects_always_disallow_add_even_with_perfect_global_metrics():
  state = ActionMaskWarmup(_fg_add_config())
  # The degenerate classifier predicts ADD illegal on all 175 legal states.
  degenerate = _fg_add_metrics(positives=175, post_tp=0, prior_tp=0)
  for step in (10, 20, 30, 40, 50):
    assert not state.update(degenerate, step)
  assert not state.actor_enabled


def test_fg_gate_rejects_no_positive_support_even_with_fabricated_recall():
  state = ActionMaskWarmup(_fg_add_config())
  data = _fg_add_metrics(positives=0)
  data['avail/post_add_recall'] = 1.0
  data['avail/prior_add_recall'] = 1.0
  assert not state.update(data, 10)
  assert not state.actor_enabled


def test_fg_gate_requires_prior_and_posterior_add_recall():
  state = ActionMaskWarmup(_fg_add_config())
  assert not state.update(_fg_add_metrics(prior_tp=24), 10)
  assert not state.update(_fg_add_metrics(post_tp=24), 20)
  for step in (30, 40):
    assert not state.update(_fg_add_metrics(post_tp=29, prior_tp=29), step)
  assert state.update(_fg_add_metrics(post_tp=29, prior_tp=29), 50)
  assert state.actor_enabled
  assert state.switched_step == 50


def test_fg_gate_rejects_too_few_positive_samples():
  state = ActionMaskWarmup(_fg_add_config())
  for step in (10, 20, 30):
    assert not state.update(_fg_add_metrics(positives=4, post_tp=4, prior_tp=4), step)
  assert not state.actor_enabled


def test_fg_enabled_gate_does_not_treat_insufficient_support_as_regression():
  state = ActionMaskWarmup(_fg_add_config())
  for step in (10, 20, 30):
    state.update(_fg_add_metrics(), step)
  assert state.actor_enabled
  for step, support in ((40, 0), (50, 4), (60, 0), (70, 4), (80, 0)):
    assert not state.update(_fg_add_metrics(positives=support,
        post_tp=support, prior_tp=support), step)
  assert state.actor_enabled
  assert state.consecutive_regressions == 0
  assert state.metrics()['add_support_sufficient'] == 0.0


def test_fg_missing_support_still_stops_on_observed_global_regression():
  state = ActionMaskWarmup(_fg_add_config())
  for step in (10, 20, 30):
    state.update(_fg_add_metrics(), step)
  bad = _fg_add_metrics(positives=0)
  bad['avail/prior_fpr'] = 0.1
  assert not state.update(bad, 40)
  try:
    state.update(bad, 50)
  except RuntimeError as exc:
    assert 'regressed' in str(exc)
  else:
    raise AssertionError('Global regression was hidden by missing ADD support')
