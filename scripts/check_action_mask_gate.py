#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreamerv3.action_mask_asset import ActionMaskAssetError, validate_asset


def main():
  parser = argparse.ArgumentParser(
      description="Validate a checkpoint-bound availability warm-up asset.")
  parser.add_argument("asset", type=Path, help="Warm-up asset directory")
  parser.add_argument("--window", type=int, default=5)
  parser.add_argument("--checkpoint", default="", help="Pinned checkpoint name")
  args = parser.parse_args()
  try:
    result = validate_asset(
        args.asset, window=args.window, checkpoint=args.checkpoint)
  except ActionMaskAssetError as exc:
    print(f"BLOCKED: {exc}", file=sys.stderr)
    return 1
  print(
      f"READY: {result.checkpoint} step={result.checkpoint_step} "
      f"reports={result.first_report_step}..{result.last_report_step}")
  for name, summary in result.blocking_metrics.items():
    relation = ">=" if summary["mode"] == "min" else "<="
    print(
        f"PASS {name}: worst={summary['observed']:.8g} "
        f"required {relation} {summary['threshold']:.8g}")
  for name, summary in result.diagnostic_metrics.items():
    print(
        f"INFO {name}: min={summary['min']:.8g} max={summary['max']:.8g}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
