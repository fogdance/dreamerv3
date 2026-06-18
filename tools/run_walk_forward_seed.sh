#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <seed_name> <experiment_seed> [logdir]" >&2
  exit 2
fi

SEED_NAME="$1"
EXPERIMENT_SEED="$2"
LOGDIR="${3:-/data/logdir/action-mask-wf1-ms-${SEED_NAME}-20260615}"

TRAIN_ENV_SEED=$((EXPERIMENT_SEED * 1000 + 101))
REPLAY_SEED=$((EXPERIMENT_SEED * 1000 + 202))
MATCHED_RANDOM_SEED=20260615

ENTRY_EVAL_VERSION="entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1"
SPLIT_MANIFEST_HASH="eb0f7073d35afd4b76f73eddcbf11cf24539287f502ec7e83726d48343ff598e"
EXECUTION_TIMING="signal_on_close_plus_spread"
ENV_CONFIG="/home/v/Documents/work/gym-trading-env/configs/env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml"
PYTHON="/home/v/miniconda3/envs/dreamerv3/bin/python"

cd "$(dirname "$0")/.."

exec "$PYTHON" dreamerv3/main.py \
  --configs action_mask_formal \
  --logdir "$LOGDIR" \
  --env.gymnasium.config_path "$ENV_CONFIG" \
  --experiment_seed "$EXPERIMENT_SEED" \
  --dreamer.seed "$EXPERIMENT_SEED" \
  --env.train_seed "$TRAIN_ENV_SEED" \
  --env.eval_seed 0 \
  --replay.seed "$REPLAY_SEED" \
  --audit.matched_random_seed "$MATCHED_RANDOM_SEED" \
  --audit.entry_eval_version "$ENTRY_EVAL_VERSION" \
  --audit.split_manifest_hash "$SPLIT_MANIFEST_HASH" \
  --audit.execution_timing "$EXECUTION_TIMING"

# -----------------------------------------------------------------------------
# Reproducibility notes for the JM walk-forward experiment.
#
# This script only starts training. The validation checkpoint selection and the
# final test/equity audits are intentionally shown below as comments so the
# whole experiment can be reproduced without searching chat history.
#
# Contract:
#   product: JM
#   train: 2024-06-03 .. 2025-07-31
#   validation: 2025-08-01 .. 2025-08-29
#   test: 2025-09-01 .. 2025-12-02
#   observation: 60x18
#   execution_timing: signal_on_close_plus_spread
#   entry_eval_version: entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1
#   split_manifest_hash: eb0f7073d35afd4b76f73eddcbf11cf24539287f502ec7e83726d48343ff598e
#
# Training commands:
#
#   cd /home/v/Documents/work/dreamerv3
#
#   ./tools/run_walk_forward_seed.sh seed_a 101 /data/logdir/action-mask-wf1-ms-seed-a-20260615
#   ./tools/run_walk_forward_seed.sh seed_b 202 /data/logdir/action-mask-wf1-ms-seed-b-20260615
#   ./tools/run_walk_forward_seed.sh seed_c 303 /data/logdir/action-mask-wf1-ms-seed-c-20260615
#
# Optional rerun if a seed needs to be repeated after a failed run:
#
#   ./tools/run_walk_forward_seed.sh seed_c2 303 /data/logdir/action-mask-wf1-ms-seed-c2-20260618
#
# Expected retained checkpoints, if checkpoint retention is enabled in configs:
#
#   $LOGDIR/ckpt_retained/step_000000500000
#   $LOGDIR/ckpt_retained/step_000000700000
#   $LOGDIR/ckpt_retained/step_000000900000
#   $LOGDIR/ckpt/latest
#
# Build or reuse the entry-eval dataset before checkpoint audits:
#
#   cd /home/v/Documents/work/gym-trading-env
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/run_entry_capability.py \
#     --config configs/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1.yaml \
#     --output artifacts/entry_eval/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1
#
# Validation checkpoint selection protocol:
#   Run deterministic per-day replay for validation only on each retained
#   checkpoint. Select the checkpoint using validation only. Do not inspect test
#   until one checkpoint has been selected for that seed.
#
# Example for one seed/checkpoint:
#
#   cd /home/v/Documents/work/gym-trading-env
#   SEED_NAME=seed_a
#   LOGDIR=/data/logdir/action-mask-wf1-ms-seed-a-20260615
#   CKPT_STEP=000000900000
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_checkpoint_audit.py \
#     --dreamer-root /home/v/Documents/work/dreamerv3 \
#     --run-logdir "$LOGDIR" \
#     --checkpoint "$LOGDIR/ckpt_retained/step_${CKPT_STEP}" \
#     --entry-eval-dir artifacts/entry_eval/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1 \
#     --env-config-path configs/env_trading_stage1_jm_walk_forward_full_20240603_20251202.yaml \
#     --output-dir "artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/${SEED_NAME}_${CKPT_STEP}_validation_signal_close" \
#     --collect \
#     --collect-mode per_day \
#     --roles validation \
#     --matched-random-seed 20260615 \
#     --jax-platform cpu
#
# Repeat the validation audit for:
#
#   CKPT_STEP=000000500000
#   CKPT_STEP=000000700000
#   CKPT_STEP=000000900000
#
# Also evaluate latest if needed:
#
#   --checkpoint "$LOGDIR/ckpt/latest"
#
# Test protocol after validation selection:
#   Run test exactly once for the validation-selected checkpoint.
#
# Example for selected 900k:
#
#   cd /home/v/Documents/work/gym-trading-env
#   SEED_NAME=seed_a
#   LOGDIR=/data/logdir/action-mask-wf1-ms-seed-a-20260615
#   SELECTED_CKPT="$LOGDIR/ckpt_retained/step_000000900000"
#   OUT="artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/${SEED_NAME}_selected_test_signal_close"
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/dreamer_checkpoint_audit.py \
#     --dreamer-root /home/v/Documents/work/dreamerv3 \
#     --run-logdir "$LOGDIR" \
#     --checkpoint "$SELECTED_CKPT" \
#     --entry-eval-dir artifacts/entry_eval/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1 \
#     --env-config-path configs/env_trading_stage1_jm_walk_forward_full_20240603_20251202.yaml \
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
#     --entry-eval-config configs/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1.yaml \
#     --output-dir "${OUT}_equity_curve"
#
# Multi-seed aggregate report after all selected test audits are complete:
#
#   /home/v/miniconda3/envs/dreamerv3/bin/python tools/report_dreamer_multiseed_repeatability.py \
#     --experiment-name action-mask-wf1-multiseed-20260615 \
#     --output-dir artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615 \
#     --seed-report seed_a=artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_a_selected_test_signal_close/summary.json \
#     --seed-report seed_b=artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_b_selected_test_signal_close/summary.json \
#     --seed-report seed_c=artifacts/dreamer_walk_forward_multiseed/action-mask-wf1-multiseed-20260615/seed_c_selected_test_signal_close/summary.json
# -----------------------------------------------------------------------------

