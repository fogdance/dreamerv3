import json
import pickle
from pathlib import Path

import elements
import ruamel.yaml as yaml

from dreamerv3.action_mask_asset import (
    ActionMaskAssetError,
    require_masked_actor_asset,
    validate_asset,
)


def _config(directory, enabled=True, required=True):
  return elements.Config({
      "task": "gymnasium_CustomTradingEnv-v0",
      "env": {"gymnasium": {"config_path": "data/trading_stage1.yaml"}},
      "agent": {
          "avail_actor_enabled": enabled,
          "dyn": {"typ": "rssm", "rssm": {"deter": 8}},
          "enc": {"typ": "simple", "simple": {"units": 8}},
          "dec": {"typ": "simple", "simple": {"units": 8}},
          "rewhead": {"units": 8},
          "conhead": {"units": 8},
          "availhead": {"units": 8},
          "policy": {"units": 8},
          "value": {"units": 8},
          "policy_dist_disc": "categorical",
          "policy_dist_cont": "bounded_normal",
          "avail_threshold": 0.5,
      },
      "action_mask_asset": {
          "required": required,
          "directory": str(directory),
          "checkpoint": "checkpoint",
          "window": 2,
      },
      "run": {"from_checkpoint": ""},
  })


def _make_asset(tmp_path, prior_fpr=0.001):
  directory = Path(tmp_path) / "asset"
  checkpoint = directory / "ckpt" / "checkpoint"
  checkpoint.mkdir(parents=True)
  (directory / "ckpt" / "latest").write_text("checkpoint")
  (checkpoint / "agent.pkl").write_bytes(b"agent")
  (checkpoint / "step.pkl").write_bytes(pickle.dumps(100))
  (checkpoint / "done").touch()
  config = _config(directory)
  yaml.YAML(typ="safe").dump(dict(config), directory / "config.yaml")
  reports = []
  for step in (80, 90, 110):
    reports.append({
        "step": step,
        "train/avail/post_accuracy": 0.999,
        "train/avail/post_exact_accuracy": 0.999,
        "train/avail/post_fpr": 0.01,
        "train/avail/post_fnr": 0.01,
        "train/avail/prior_accuracy": 0.999,
        "train/avail/prior_exact_accuracy": 0.999,
        "train/avail/prior_fpr": prior_fpr,
        "train/avail/prior_fnr": 0.01,
        "train/avail/img_fallback_rate": 0.0,
    })
  (directory / "metrics.jsonl").write_text(
      "".join(json.dumps(report) + "\n" for report in reports))
  return directory, checkpoint


def test_valid_asset_uses_only_reports_before_checkpoint(tmp_path):
  directory, checkpoint = _make_asset(tmp_path)
  result = validate_asset(directory, _config(directory), window=2)
  assert result.checkpoint == checkpoint.resolve()
  assert result.last_report_step == 90


def test_missing_asset_blocks_masked_actor(tmp_path):
  try:
    require_masked_actor_asset(_config(tmp_path / "missing"))
  except ActionMaskAssetError as exc:
    assert "directory not found" in str(exc)
  else:
    raise AssertionError("Missing availability asset was accepted")


def test_failed_prior_gate_blocks_asset(tmp_path):
  directory, _ = _make_asset(tmp_path, prior_fpr=0.1)
  try:
    validate_asset(directory, _config(directory), window=2)
  except ActionMaskAssetError as exc:
    assert "prior_fpr" in str(exc)
  else:
    raise AssertionError("Failed availability metrics were accepted")


def test_incompatible_model_blocks_asset(tmp_path):
  directory, _ = _make_asset(tmp_path)
  config = _config(directory).update({"agent.dyn.rssm.deter": 16})
  try:
    validate_asset(directory, config, window=2)
  except ActionMaskAssetError as exc:
    assert "agent.dyn.rssm.deter" in str(exc)
  else:
    raise AssertionError("Incompatible availability asset was accepted")


def test_incompatible_threshold_blocks_asset(tmp_path):
  directory, _ = _make_asset(tmp_path)
  config = _config(directory).update({"agent.avail_threshold": 0.9})
  try:
    validate_asset(directory, config, window=2)
  except ActionMaskAssetError as exc:
    assert "agent.avail_threshold" in str(exc)
  else:
    raise AssertionError("Incompatible availability threshold was accepted")


def test_actor_cannot_enable_without_required_asset(tmp_path):
  config = _config(tmp_path, required=False)
  try:
    require_masked_actor_asset(config)
  except ActionMaskAssetError as exc:
    assert "required=true" in str(exc)
  else:
    raise AssertionError("Masked actor started without a required asset")


def test_actor_requires_pinned_checkpoint(tmp_path):
  directory, _ = _make_asset(tmp_path)
  config = _config(directory).update({"action_mask_asset.checkpoint": ""})
  try:
    require_masked_actor_asset(config)
  except ActionMaskAssetError as exc:
    assert "pinned" in str(exc)
  else:
    raise AssertionError("Masked actor started without a pinned checkpoint")


def test_invalid_pinned_checkpoint_is_rejected(tmp_path):
  directory, _ = _make_asset(tmp_path)
  try:
    validate_asset(directory, checkpoint="../outside", window=2)
  except ActionMaskAssetError as exc:
    assert "Invalid checkpoint name" in str(exc)
  else:
    raise AssertionError("Invalid pinned checkpoint was accepted")
