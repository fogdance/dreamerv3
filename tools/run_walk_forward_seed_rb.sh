#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <seed_name> <experiment_seed> [logdir] [pilot1m|formal3m]" >&2
  exit 2
fi

SEED_NAME="$1"
EXPERIMENT_SEED="$2"
LOGDIR="${3:-/data/logdir/action-mask-rb8y-wf1-ms-${SEED_NAME}-20260617}"
PROFILE="${4:-pilot1m}"

case "$PROFILE" in
  pilot1m)
    RUN_STEPS="1e6"
    RETENTION_STEPS=(5e5 7e5 9e5)
    ;;
  formal3m)
    RUN_STEPS="3e6"
    RETENTION_STEPS=(1.5e6 2.1e6 2.7e6)
    ;;
  *)
    echo "Unknown profile: $PROFILE (expected pilot1m or formal3m)" >&2
    exit 2
    ;;
esac

TRAIN_ENV_SEED=$((EXPERIMENT_SEED * 1000 + 101))
REPLAY_SEED=$((EXPERIMENT_SEED * 1000 + 202))
MATCHED_RANDOM_SEED=20260615

CONTRACT_ROOT="${RB_CONTRACT_ROOT:-/data/logdir/trading_contracts/rb8y_fixed_holdout_v1}"
SPLIT_NAME="rb8y_fixed_holdout_v1"
ENTRY_EVAL_VERSION="$SPLIT_NAME"
ENTRY_EVAL_CONFIG="$CONTRACT_ROOT/configs/entry_eval/${SPLIT_NAME}_signal_close.yaml"
SPLIT_MANIFEST_HASH="${RB_SPLIT_MANIFEST_HASH:-$(cat "$CONTRACT_ROOT/artifacts/walk_forward_splits/$SPLIT_NAME/split_manifest.sha256")}"
EXECUTION_TIMING="signal_on_close_plus_spread"
ENV_CONFIG="$CONTRACT_ROOT/configs/env/${SPLIT_NAME}_train.yaml"
PYTHON="/home/v/miniconda3/envs/dreamerv3/bin/python"

cd "$(dirname "$0")/.."

exec "$PYTHON" dreamerv3/main.py \
  --configs action_mask_formal \
  --logdir "$LOGDIR" \
  --run.steps "$RUN_STEPS" \
  --run.checkpoint_retention.steps "${RETENTION_STEPS[@]}" \
  --env.gymnasium.config_path "$ENV_CONFIG" \
  --experiment_seed "$EXPERIMENT_SEED" \
  --dreamer.seed "$EXPERIMENT_SEED" \
  --env.train_seed "$TRAIN_ENV_SEED" \
  --env.eval_seed 0 \
  --replay.seed "$REPLAY_SEED" \
  --audit.matched_random_seed "$MATCHED_RANDOM_SEED" \
  --audit.entry_eval_version "$ENTRY_EVAL_VERSION" \
  --audit.entry_eval_config "$ENTRY_EVAL_CONFIG" \
  --audit.split_manifest_hash "$SPLIT_MANIFEST_HASH" \
  --audit.execution_timing "$EXECUTION_TIMING"

# -----------------------------------------------------------------------------
# Reproducibility notes for the RB 8Y walk-forward experiment.
#
# This script only starts training. Validation checkpoint selection and the
# final test/equity audits are shown below as comments so the whole experiment
# can be reproduced without searching chat history.
#
# Contract:
#   product: RB
#   contract_root: /data/logdir/trading_contracts/rb8y_fixed_holdout_v1
#   train env data: 2017-12-25 .. 2024-12-31
#   full eval data: 2017-12-25 .. 2025-12-02
#   observation: 60x18
#   execution_timing: signal_on_close_plus_spread
#   entry_eval_config:
#     $CONTRACT_ROOT/configs/entry_eval/rb8y_fixed_holdout_v1_signal_close.yaml
#   split_manifest_hash:
#     read from $CONTRACT_ROOT/artifacts/walk_forward_splits/rb8y_fixed_holdout_v1/split_manifest.sha256
#
# Usage:
#
#   cd /home/v/Documents/work/dreamerv3
#   ./tools/run_walk_forward_seed_rb.sh <seed_name> <experiment_seed> <logdir> <pilot1m|formal3m>
#
# Stage 1: RB 8Y pilot 1M.
# Retention: 500k / 700k / 900k / latest.
#
#   ./tools/run_walk_forward_seed_rb.sh seed_a 101 /data/logdir/action-mask-rb8y-pilot1m-seed-a-20260617 pilot1m
#   ./tools/run_walk_forward_seed_rb.sh seed_b 202 /data/logdir/action-mask-rb8y-pilot1m-seed-b-20260617 pilot1m
#   ./tools/run_walk_forward_seed_rb.sh seed_c 303 /data/logdir/action-mask-rb8y-pilot1m-seed-c-20260617 pilot1m
#
# Stage 2: RB 8Y long-run debug 3M.
# Retention: 1.5M / 2.1M / 2.7M / latest.
#
#   ./tools/run_walk_forward_seed_rb.sh seed_a 101 /data/logdir/action-mask-rb8y-formal3m-seed-a-20260617 formal3m
#   ./tools/run_walk_forward_seed_rb.sh seed_b 202 /data/logdir/action-mask-rb8y-formal3m-seed-b-20260617 formal3m
#   ./tools/run_walk_forward_seed_rb.sh seed_c 303 /data/logdir/action-mask-rb8y-formal3m-seed-c-20260617 formal3m
#
# Build or reuse the RB entry-eval dataset before checkpoint audits.
# Keep outputs outside source repositories:
#
#   cd /home/v/Documents/work/gym-trading-env
#   CONTRACT_ROOT=/data/logdir/trading_contracts/rb8y_fixed_holdout_v1
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/run_entry_capability.py \
#     --config "$CONTRACT_ROOT/configs/entry_eval/rb8y_fixed_holdout_v1_signal_close.yaml" \
#     --output "$CONTRACT_ROOT/artifacts/entry_eval/rb8y_fixed_holdout_v1"
#
# Validation checkpoint selection protocol:
#   Run deterministic per-day replay for validation only on each retained
#   checkpoint. Select the checkpoint using validation only. Do not inspect test
#   until one checkpoint has been selected for that seed.
#
# Example for one pilot seed/checkpoint:
#
#   cd /home/v/Documents/work/gym-trading-env
#   SEED_NAME=seed_a
#   LOGDIR=/data/logdir/action-mask-rb8y-pilot1m-seed-a-20260617
#   CONTRACT_ROOT=/data/logdir/trading_contracts/rb8y_fixed_holdout_v1
#   CKPT_STEP=000000900000
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_checkpoint_audit.py \
#     --dreamer-root /home/v/Documents/work/dreamerv3 \
#     --run-logdir "$LOGDIR" \
#     --checkpoint "$LOGDIR/ckpt_retained/step_${CKPT_STEP}" \
#     --entry-eval-dir "$CONTRACT_ROOT/artifacts/entry_eval/rb8y_fixed_holdout_v1" \
#     --env-config-path "$CONTRACT_ROOT/configs/env/rb8y_fixed_holdout_v1_full.yaml" \
#     --output-dir "/data/logdir/audits/action-mask-rb8y-pilot1m-20260617/${SEED_NAME}_${CKPT_STEP}_validation_signal_close" \
#     --collect \
#     --collect-mode per_day \
#     --roles validation \
#     --matched-random-seed 20260615 \
#     --jax-platform cpu
#
# Repeat validation for the retained checkpoint steps:
#
#   pilot1m:  CKPT_STEP=000000500000 / 000000700000 / 000000900000
#   formal3m: CKPT_STEP=000001500000 / 000002100000 / 000002700000
#
# Also evaluate latest if needed:
#
#   --checkpoint "$LOGDIR/ckpt/latest"
#
# Test protocol after validation selection:
#   Run test exactly once for the validation-selected checkpoint.
#
# Example for selected pilot 900k:
#
#   cd /home/v/Documents/work/gym-trading-env
#   SEED_NAME=seed_a
#   LOGDIR=/data/logdir/action-mask-rb8y-pilot1m-seed-a-20260617
#   CONTRACT_ROOT=/data/logdir/trading_contracts/rb8y_fixed_holdout_v1
#   SELECTED_CKPT="$LOGDIR/ckpt_retained/step_000000900000"
#   OUT="/data/logdir/audits/action-mask-rb8y-pilot1m-20260617/${SEED_NAME}_selected_test_signal_close"
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_checkpoint_audit.py \
#     --dreamer-root /home/v/Documents/work/dreamerv3 \
#     --run-logdir "$LOGDIR" \
#     --checkpoint "$SELECTED_CKPT" \
#     --entry-eval-dir "$CONTRACT_ROOT/artifacts/entry_eval/rb8y_fixed_holdout_v1" \
#     --env-config-path "$CONTRACT_ROOT/configs/env/rb8y_fixed_holdout_v1_full.yaml" \
#     --output-dir "$OUT" \
#     --collect \
#     --collect-mode per_day \
#     --roles test \
#     --matched-random-seed 20260615 \
#     --jax-platform cpu
#
# Equity and floating drawdown audit for the selected test result:
#
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_equity_curve_audit.py \
#     --attribution-dir "$OUT" \
#     --entry-eval-config "$CONTRACT_ROOT/configs/entry_eval/rb8y_fixed_holdout_v1_signal_close.yaml" \
#     --output-dir "${OUT}_equity_curve"
#
# Multi-seed aggregate report after all selected test audits are complete:
#
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/report_dreamer_multiseed_repeatability.py \
#     --experiment-name action-mask-rb8y-pilot1m-20260617 \
#     --output-dir /data/logdir/audits/action-mask-rb8y-pilot1m-20260617 \
#     --seed-report seed_a=/data/logdir/audits/action-mask-rb8y-pilot1m-20260617/seed_a_selected_test_signal_close/summary.json \
#     --seed-report seed_b=/data/logdir/audits/action-mask-rb8y-pilot1m-20260617/seed_b_selected_test_signal_close/summary.json \
#     --seed-report seed_c=/data/logdir/audits/action-mask-rb8y-pilot1m-20260617/seed_c_selected_test_signal_close/summary.json
# -----------------------------------------------------------------------------
