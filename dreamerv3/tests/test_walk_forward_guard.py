import hashlib
import json
from pathlib import Path

import elements
import pytest
import ruamel.yaml as yaml

from dreamerv3.walk_forward_guard import (
    WalkForwardGuardError,
    validate_walk_forward_split,
)


def _write_csv(path: Path, first_day: str, last_day: str):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      "Date,Open,High,Low,Close,Volume,OpenInterest\n"
      f"{first_day} 09:01:00,1,1,1,1,1,1\n"
      f"{last_day} 15:00:00,1,1,1,1,1,1\n")


def _write_yaml(path: Path, data_path: str):
  path.parent.mkdir(parents=True, exist_ok=True)
  data = {
      "trading": {
          "data_path": data_path,
          "data_interval": "1m",
      },
      "training": {
          "randomize_start": True,
          "start_clock": "future_night",
      },
  }
  y = yaml.YAML()
  with path.open("w") as f:
    y.dump(data, f)


def _write_entry_eval(root: Path, version: str, *, timing="signal_on_close_plus_spread"):
  out = root / version
  out.mkdir(parents=True, exist_ok=True)
  split = [{
      "name": "fold_test",
      "train_days": [20240101, 20240102],
      "validation_days": [20240103],
      "test_days": [20240104],
  }]
  (out / "split_manifest.json").write_text(json.dumps(split))
  entry_eval_config = out / "entry_eval.yaml"
  y = yaml.YAML()
  with entry_eval_config.open("w") as f:
    y.dump({
        "version": version,
        "entry_evaluator": {
            "execution_timing": timing,
        },
        "walk_forward": {
            "split_manifest_path": str(out / "split_manifest.json"),
        },
    }, f)
  digest = hashlib.sha256((out / "split_manifest.json").read_bytes()).hexdigest()
  return out, entry_eval_config, digest


def _config(tmp_path: Path, *, entry_eval_config: Path, split_hash: str, allow_non_holdout=False):
  return elements.Config(dict(
      script="train",
      audit=dict(
          entry_eval_version="entry_eval_test",
          entry_eval_config=str(entry_eval_config),
          split_manifest_hash=split_hash,
          execution_timing="signal_on_close_plus_spread",
      ),
      env=dict(
          gymnasium=dict(
              config_path=str(tmp_path / "env.yaml"),
          ),
      ),
      walk_forward_guard=dict(
          enabled=True,
          allow_non_holdout_training=allow_non_holdout,
          data_root=str(tmp_path / "data"),
      ),
  ))


def test_walk_forward_guard_passes_clean_train_only_split(tmp_path):
  _, entry_eval_config, split_hash = _write_entry_eval(tmp_path / "artifacts" / "entry_eval", "entry_eval_test")
  _write_yaml(tmp_path / "env.yaml", "TRAIN")
  _write_csv(tmp_path / "data" / "TRAIN_1m.csv", "2024-01-01", "2024-01-02")

  result = validate_walk_forward_split(
      _config(tmp_path, entry_eval_config=entry_eval_config, split_hash=split_hash),
      dreamer_root=tmp_path)

  assert result["status"] == "pass"
  assert result["role_map_source"] == "split_manifest.json:last_fold"
  assert result["entry_eval_config"] == str(entry_eval_config)
  assert result["training_data"]["min_day"] == 20240101
  assert result["training_data"]["max_day"] == 20240102
  assert result["roles"]["validation"]["min_day"] == 20240103
  assert result["roles"]["test"]["min_day"] == 20240104


def test_walk_forward_guard_rejects_stale_split_hash(tmp_path):
  _, entry_eval_config, _ = _write_entry_eval(tmp_path / "artifacts" / "entry_eval", "entry_eval_test")
  _write_yaml(tmp_path / "env.yaml", "TRAIN")
  _write_csv(tmp_path / "data" / "TRAIN_1m.csv", "2024-01-01", "2024-01-02")

  with pytest.raises(WalkForwardGuardError, match="split_manifest_hash mismatch"):
    validate_walk_forward_split(
        _config(tmp_path, entry_eval_config=entry_eval_config, split_hash="stale"),
        dreamer_root=tmp_path)


def test_walk_forward_guard_rejects_training_data_overlap(tmp_path):
  _, entry_eval_config, split_hash = _write_entry_eval(tmp_path / "artifacts" / "entry_eval", "entry_eval_test")
  _write_yaml(tmp_path / "env.yaml", "TRAIN")
  _write_csv(tmp_path / "data" / "TRAIN_1m.csv", "2024-01-01", "2024-01-04")

  with pytest.raises(WalkForwardGuardError, match="overlaps validation"):
    validate_walk_forward_split(
        _config(tmp_path, entry_eval_config=entry_eval_config, split_hash=split_hash),
        dreamer_root=tmp_path)


def test_walk_forward_guard_rejects_training_data_before_split_train(tmp_path):
  _, entry_eval_config, split_hash = _write_entry_eval(tmp_path / "artifacts" / "entry_eval", "entry_eval_test")
  _write_yaml(tmp_path / "env.yaml", "TRAIN")
  _write_csv(tmp_path / "data" / "TRAIN_1m.csv", "2023-12-29", "2024-01-02")

  with pytest.raises(WalkForwardGuardError, match="starts before split train"):
    validate_walk_forward_split(
        _config(tmp_path, entry_eval_config=entry_eval_config, split_hash=split_hash),
        dreamer_root=tmp_path)


def test_walk_forward_guard_allows_explicit_non_holdout_debug(tmp_path):
  _, entry_eval_config, split_hash = _write_entry_eval(tmp_path / "artifacts" / "entry_eval", "entry_eval_test")
  _write_yaml(tmp_path / "env.yaml", "TRAIN")
  _write_csv(tmp_path / "data" / "TRAIN_1m.csv", "2024-01-01", "2024-01-04")

  result = validate_walk_forward_split(
      _config(
          tmp_path,
          entry_eval_config=entry_eval_config,
          split_hash=split_hash,
          allow_non_holdout=True),
      dreamer_root=tmp_path)

  assert result["status"] == "non_holdout"
  assert result["validation_overlap_count"] == 1
  assert result["test_overlap_count"] == 1


def test_walk_forward_guard_skips_when_audit_metadata_is_empty(tmp_path):
  config = elements.Config(dict(
      script="train",
      audit=dict(
          entry_eval_version="",
          entry_eval_config="",
          split_manifest_hash="",
          execution_timing=""),
      walk_forward_guard=dict(enabled=True),
  ))

  result = validate_walk_forward_split(config, dreamer_root=tmp_path)

  assert result["status"] == "skipped"


def _default_config():
  path = Path(__file__).resolve().parents[1] / "configs.yaml"
  configs = yaml.YAML(typ="safe").load(path.read_text())
  return elements.Config(configs["defaults"])


def test_declared_guard_result_schema_accepts_skipped_result():
  config = _default_config().update(script="live_trading")
  guard_result = {
      "status": "skipped",
      "reason": "guard only enforces script=train",
  }

  updated = config.update(walk_forward_guard_result=guard_result)

  assert updated.walk_forward_guard_result.status == "skipped"
  assert updated.walk_forward_guard_result.reason == "guard only enforces script=train"
  assert updated.script == "live_trading"


def test_declared_guard_result_schema_accepts_pass_result(tmp_path):
  _, entry_eval_config, split_hash = _write_entry_eval(
      tmp_path / "artifacts" / "entry_eval", "entry_eval_test")
  _write_yaml(tmp_path / "env.yaml", "TRAIN")
  _write_csv(tmp_path / "data" / "TRAIN_1m.csv", "2024-01-01", "2024-01-02")
  guard_result = validate_walk_forward_split(
      _config(tmp_path, entry_eval_config=entry_eval_config, split_hash=split_hash),
      dreamer_root=tmp_path)

  updated = _default_config().update(walk_forward_guard_result=guard_result)

  assert updated.walk_forward_guard_result.status == "pass"
  assert updated.walk_forward_guard_result.training_data.min_day == 20240101
  assert updated.walk_forward_guard_result.roles.validation.days == 1
  assert updated.walk_forward_guard_result.validation_overlap_count == 0
  assert updated.walk_forward_guard_result.validation_overlap_days == (0,)
