import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import ruamel.yaml as yaml


BLOCKING_METRICS = {
    "post_accuracy": ("train/avail/post_accuracy", "min", 0.995),
    "post_exact": ("train/avail/post_exact_accuracy", "min", 0.990),
    "prior_accuracy": ("train/avail/prior_accuracy", "min", 0.990),
    "prior_exact": ("train/avail/prior_exact_accuracy", "min", 0.980),
    "prior_fpr": ("train/avail/prior_fpr", "max", 0.005),
    "img_fallback": ("train/avail/img_fallback_rate", "max", 0.0001),
}

DIAGNOSTIC_METRICS = {
    "post_fpr": "train/avail/post_fpr",
    "post_fnr": "train/avail/post_fnr",
    "prior_fnr": "train/avail/prior_fnr",
}

COMPAT_PREFIXES = (
    "agent.dyn.",
    "agent.enc.",
    "agent.dec.",
    "agent.rewhead.",
    "agent.conhead.",
    "agent.availhead.",
    "agent.policy.",
    "agent.value.",
)

COMPAT_KEYS = (
    "task",
    "env.gymnasium.config_path",
    "agent.policy_dist_disc",
    "agent.policy_dist_cont",
    "agent.avail_threshold",
)


class ActionMaskAssetError(RuntimeError):
  pass


@dataclass(frozen=True)
class AssetValidation:
  directory: Path
  checkpoint: Path
  checkpoint_step: int
  first_report_step: int
  last_report_step: int
  blocking_metrics: dict
  diagnostic_metrics: dict


def _flatten(mapping, prefix=""):
  result = {}
  for key, value in mapping.items():
    name = f"{prefix}.{key}" if prefix else str(key)
    if isinstance(value, dict):
      result.update(_flatten(value, name))
    else:
      result[name] = value
  return result


def _flat_config(config):
  flat = dict(config.flat) if hasattr(config, "flat") else _flatten(config)
  return {
      key: tuple(value) if isinstance(value, list) else value
      for key, value in flat.items()
  }


def _load_yaml(path):
  try:
    return yaml.YAML(typ="safe").load(path.read_text())
  except Exception as exc:
    raise ActionMaskAssetError(f"Cannot read asset config {path}: {exc}") from exc


def _resolve_checkpoint(directory, pinned=""):
  ckpt_root = directory / "ckpt"
  if pinned:
    name = pinned
  else:
    latest = ckpt_root / "latest"
    if not latest.is_file():
      raise ActionMaskAssetError(f"Missing checkpoint pointer: {latest}")
    name = latest.read_text().strip()
  checkpoint = (ckpt_root / name).resolve()
  if checkpoint.parent != ckpt_root.resolve():
    raise ActionMaskAssetError(f"Invalid checkpoint name: {name!r}")
  required = ("agent.pkl", "step.pkl", "done")
  missing = [name for name in required if not (checkpoint / name).is_file()]
  if missing:
    raise ActionMaskAssetError(
        f"Incomplete checkpoint {checkpoint}; missing: {', '.join(missing)}")
  try:
    step = int(pickle.loads((checkpoint / "step.pkl").read_bytes()))
  except Exception as exc:
    raise ActionMaskAssetError(
        f"Cannot read checkpoint step from {checkpoint}: {exc}") from exc
  return checkpoint, step


def _check_compatibility(source_config, target_config):
  source = _flat_config(source_config)
  target = _flat_config(target_config)
  keys = set(COMPAT_KEYS)
  keys.update(
      key for key in set(source) | set(target)
      if key.startswith(COMPAT_PREFIXES))
  mismatches = []
  for key in sorted(keys):
    source_value = source.get(key, "<missing>")
    target_value = target.get(key, "<missing>")
    if source_value != target_value:
      mismatches.append(f"{key}: asset={source_value!r}, run={target_value!r}")
  if mismatches:
    details = "\n  ".join(mismatches[:20])
    suffix = "\n  ..." if len(mismatches) > 20 else ""
    raise ActionMaskAssetError(
        "Availability asset is incompatible with the requested run:\n  "
        f"{details}{suffix}")


def _load_reports(path, checkpoint_step, window):
  required = {
      spec[0] for spec in BLOCKING_METRICS.values()
  } | set(DIAGNOSTIC_METRICS.values())
  reports = []
  try:
    stream = path.open(encoding="utf-8")
  except OSError as exc:
    raise ActionMaskAssetError(f"Cannot read asset metrics {path}: {exc}") from exc
  with stream:
    for lineno, line in enumerate(stream, 1):
      try:
        report = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ActionMaskAssetError(
            f"Invalid JSON at {path}:{lineno}: {exc}") from exc
      if report.get("step", checkpoint_step + 1) <= checkpoint_step:
        if required <= report.keys():
          reports.append(report)
  if len(reports) < window:
    raise ActionMaskAssetError(
        f"Need {window} complete reports at or before checkpoint step "
        f"{checkpoint_step}, found {len(reports)}")
  return reports[-window:]


def _check_metrics(reports):
  summaries = {}
  failures = []
  for name, (key, mode, threshold) in BLOCKING_METRICS.items():
    values = [float(report[key]) for report in reports]
    if not all(math.isfinite(value) for value in values):
      failures.append(name)
      summaries[name] = {
          "observed": float("nan"), "mode": mode, "threshold": threshold}
      continue
    observed = min(values) if mode == "min" else max(values)
    summaries[name] = {
        "observed": observed, "mode": mode, "threshold": threshold}
    passed = observed >= threshold if mode == "min" else observed <= threshold
    if not passed:
      failures.append(name)
  diagnostics = {}
  for name, key in DIAGNOSTIC_METRICS.items():
    values = [float(report[key]) for report in reports]
    diagnostics[name] = {"min": min(values), "max": max(values)}
  if failures:
    raise ActionMaskAssetError(
        f"Availability gate failed: {', '.join(failures)}")
  return summaries, diagnostics


def validate_asset(directory, target_config=None, window=5, checkpoint=""):
  if window < 1:
    raise ActionMaskAssetError("Asset metric window must be at least 1")
  directory = Path(directory).expanduser().resolve()
  if not directory.is_dir():
    raise ActionMaskAssetError(f"Availability asset directory not found: {directory}")
  checkpoint, checkpoint_step = _resolve_checkpoint(directory, checkpoint)
  source_config = _load_yaml(directory / "config.yaml")
  if target_config is not None:
    _check_compatibility(source_config, target_config)
  reports = _load_reports(directory / "metrics.jsonl", checkpoint_step, window)
  blocking, diagnostics = _check_metrics(reports)
  return AssetValidation(
      directory=directory,
      checkpoint=checkpoint,
      checkpoint_step=checkpoint_step,
      first_report_step=int(reports[0]["step"]),
      last_report_step=int(reports[-1]["step"]),
      blocking_metrics=blocking,
      diagnostic_metrics=diagnostics,
  )


def require_masked_actor_asset(config):
  actor_enabled = bool(config.agent.avail_actor_enabled)
  asset_required = bool(config.action_mask_asset.required)
  if not actor_enabled and not asset_required:
    return config
  if actor_enabled and not asset_required:
    raise ActionMaskAssetError(
        "Masked actor training requires action_mask_asset.required=true")
  directory = str(config.action_mask_asset.directory).strip()
  if not directory:
    raise ActionMaskAssetError(
        "Masked actor training requires action_mask_asset.directory")
  checkpoint = str(config.action_mask_asset.checkpoint).strip()
  if actor_enabled and not checkpoint:
    raise ActionMaskAssetError(
        "Masked actor training requires a pinned action_mask_asset.checkpoint")
  result = validate_asset(
      directory, target_config=config, window=int(config.action_mask_asset.window),
      checkpoint=checkpoint)
  requested = str(config.run.from_checkpoint).strip()
  if requested and Path(requested).expanduser().resolve() != result.checkpoint:
    raise ActionMaskAssetError(
        "run.from_checkpoint must be empty or match the validated availability "
        f"asset checkpoint: {result.checkpoint}")
  print(
      "Availability asset READY:",
      f"{result.checkpoint} (step {result.checkpoint_step}, reports "
      f"{result.first_report_step}..{result.last_report_step})")
  return config.update({"run.from_checkpoint": str(result.checkpoint)})
