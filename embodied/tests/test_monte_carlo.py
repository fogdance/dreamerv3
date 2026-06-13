from embodied.run.monte_carlo import _require_formal_actor


class _Agent:

  def __init__(self, enabled):
    self.enabled = enabled

  def get_avail_actor_enabled(self):
    return self.enabled


def test_monte_carlo_requires_actor_gate_interface():
  try:
    _require_formal_actor(object())
  except TypeError as exc:
    assert "get_avail_actor_enabled" in str(exc)
  else:
    raise AssertionError("Monte Carlo accepted an agent without actor gate")


def test_monte_carlo_rejects_warmup_checkpoint():
  try:
    _require_formal_actor(_Agent(False))
  except RuntimeError as exc:
    assert "avail_actor_gate/value=0" in str(exc)
  else:
    raise AssertionError("Monte Carlo accepted a warm-up checkpoint")


def test_monte_carlo_accepts_formal_checkpoint():
  _require_formal_actor(_Agent(True))
