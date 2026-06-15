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
