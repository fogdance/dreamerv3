import math


METRICS = {
    "post_accuracy": ("avail/post_accuracy", "min"),
    "post_exact": ("avail/post_exact_accuracy", "min"),
    "prior_accuracy": ("avail/prior_accuracy", "min"),
    "prior_exact": ("avail/prior_exact_accuracy", "min"),
    "prior_fpr": ("avail/prior_fpr", "max"),
    "img_fallback": ("avail/img_fallback_rate", "max"),
}


class ActionMaskWarmup:

  def __init__(self, config, actor_enabled=False):
    self.enabled = bool(config.enabled)
    self.min_train_samples = int(config.min_train_samples)
    self.required_passes = int(config.consecutive_reports)
    self.regression_limit = int(config.regression_reports)
    self.stop_on_regression = bool(config.stop_on_regression)
    self.thresholds = {
        name: float(getattr(config, name)) for name in METRICS}
    if self.min_train_samples < 1:
      raise ValueError("min_train_samples must be at least 1")
    if self.required_passes < 1:
      raise ValueError("consecutive_reports must be at least 1")
    if self.regression_limit < 1:
      raise ValueError("regression_reports must be at least 1")
    if self.enabled and actor_enabled:
      raise ValueError(
          "Auto action-mask warm-up requires agent.avail_actor_enabled=False")
    self.actor_enabled = bool(actor_enabled)
    self.reports = 0
    self.consecutive_passes = 0
    self.consecutive_regressions = 0
    self.switched_step = -1
    self.last_passed = False

  def ready(self, train_samples):
    return self.enabled and int(train_samples) >= self.min_train_samples

  def update(self, metrics, step):
    if not self.enabled:
      return False
    passed = self._passed(metrics)
    self.reports += 1
    self.last_passed = passed
    if not self.actor_enabled:
      self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
      if self.consecutive_passes >= self.required_passes:
        self.actor_enabled = True
        self.switched_step = int(step)
        self.consecutive_regressions = 0
        return True
      return False
    self.consecutive_regressions = 0 if passed else self.consecutive_regressions + 1
    if self.stop_on_regression and self.consecutive_regressions >= self.regression_limit:
      raise RuntimeError(
          "Availability regressed after masked actor activation for "
          f"{self.consecutive_regressions} consecutive reports")
    return False

  def metrics(self):
    return {
        "enabled": float(self.enabled),
        "actor_enabled": float(self.actor_enabled),
        "reports": self.reports,
        "consecutive_passes": self.consecutive_passes,
        "consecutive_regressions": self.consecutive_regressions,
        "last_passed": float(self.last_passed),
        "switched_step": self.switched_step,
    }

  def save(self):
    return {
        "actor_enabled": self.actor_enabled,
        "reports": self.reports,
        "consecutive_passes": self.consecutive_passes,
        "consecutive_regressions": self.consecutive_regressions,
        "switched_step": self.switched_step,
        "last_passed": self.last_passed,
    }

  def load(self, data):
    self.actor_enabled = bool(data["actor_enabled"])
    self.reports = int(data["reports"])
    self.consecutive_passes = int(data["consecutive_passes"])
    self.consecutive_regressions = int(data["consecutive_regressions"])
    self.switched_step = int(data["switched_step"])
    self.last_passed = bool(data["last_passed"])

  def _passed(self, metrics):
    for name, (key, mode) in METRICS.items():
      if key not in metrics:
        return False
      value = float(metrics[key])
      if not math.isfinite(value):
        return False
      threshold = self.thresholds[name]
      if mode == "min" and value < threshold:
        return False
      if mode == "max" and value > threshold:
        return False
    return True
