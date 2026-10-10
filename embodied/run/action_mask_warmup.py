import math


# Availability for action index 1 is the ADD_3 action in the FG task.
ADD_RECALL_METRICS = {
    'post_add_recall': ('avail/post_add_recall', 'min'),
    'prior_add_recall': ('avail/prior_add_recall', 'min'),
}

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
    self.add_recall_enabled = bool(getattr(config, 'add_recall_enabled', False))
    self.min_add_positives = int(getattr(config, 'min_add_positives', 1))
    active_metrics = {**METRICS, **(ADD_RECALL_METRICS if self.add_recall_enabled else {})}
    self.thresholds = {
        name: float(getattr(config, name)) for name in active_metrics}
    if self.add_recall_enabled and self.min_add_positives < 1:
      raise ValueError('min_add_positives must be at least 1')
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
    self.last_add_support_sufficient = not self.add_recall_enabled

  def ready(self, train_samples):
    return self.enabled and int(train_samples) >= self.min_train_samples

  def update(self, metrics, step):
    if not self.enabled:
      return False
    passed = self._passed(metrics)
    self.reports += 1
    self.last_passed = passed
    self.last_add_support_sufficient = self._has_add_support(metrics)
    if not self.actor_enabled:
      self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
      if self.consecutive_passes >= self.required_passes:
        self.actor_enabled = True
        self.switched_step = int(step)
        self.consecutive_regressions = 0
        return True
      return False
    if (not self.last_add_support_sufficient and
        self._metrics_passed(metrics, METRICS)):
      # No labelled ADD opportunities: recall cannot be assessed. This
      # interrupts a regression streak but does not certify a passing report.
      self.consecutive_regressions = 0
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
        "add_support_sufficient": float(self.last_add_support_sufficient),
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
        "last_add_support_sufficient": self.last_add_support_sufficient,
    }

  def load(self, data):
    self.actor_enabled = bool(data["actor_enabled"])
    self.reports = int(data["reports"])
    self.consecutive_passes = int(data["consecutive_passes"])
    self.consecutive_regressions = int(data["consecutive_regressions"])
    self.switched_step = int(data["switched_step"])
    self.last_passed = bool(data["last_passed"])
    self.last_add_support_sufficient = bool(data.get(
        "last_add_support_sufficient", not self.add_recall_enabled))

  def _has_add_support(self, metrics):
    if self.add_recall_enabled:
      # A batch of only WAIT-only states is not evidence of ADD recall.
      for prefix in ('post', 'prior'):
        support = float(metrics.get(f'avail/{prefix}_add_positives', 0))
        if not math.isfinite(support) or support < self.min_add_positives:
          return False
    return True

  def _passed(self, metrics):
    if not self._has_add_support(metrics):
      return False
    active_metrics = {**METRICS, **(ADD_RECALL_METRICS if self.add_recall_enabled else {})}
    return self._metrics_passed(metrics, active_metrics)

  def _metrics_passed(self, metrics, active_metrics):
    for name, (key, mode) in active_metrics.items():
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
