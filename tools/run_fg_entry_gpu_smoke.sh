#!/usr/bin/env bash
# Bounded GPU-only wiring validation; preserves FG business rules and 50M architecture.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dreamer_root=$(dirname -- "$script_dir")
env_root=${FG_ENV_ROOT:-$(dirname -- "$dreamer_root")/gym-trading-env}
if [[ ! -d "$env_root/src/gym_trading_env" ]]; then
  echo "Missing environment repository: $env_root" >&2
  exit 2
fi
export JAX_PLATFORMS=cuda
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$dreamer_root:$env_root/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${FG_PYTHON:-/home/v/miniconda3/envs/dreamerv3/bin/python}
dataset=${1:?Usage: bash tools/run_fg_entry_gpu_smoke.sh DATASET NEW_SMOKE_LOGDIR}
smoke_logdir=${2:?Provide a new smoke log directory}
export JAX_COMPILATION_CACHE_DIR="${FG_JAX_CACHE:-$(dirname -- "$smoke_logdir")/jax_compile_cache}"
if [[ -e "$smoke_logdir" ]]; then
  echo 'Refusing to reuse an existing smoke directory' >&2
  exit 2
fi
"$python_bin" -c 'import jax; devices=jax.devices(); assert devices and all(d.platform == "gpu" for d in devices), devices; print(devices)'
cd -- "$dreamer_root"
args=(--configs defaults fg_entry_v1 fg_entry_dataset_v1
  --fg_entry.manifest "$dataset/manifest.json"
  --env.gymnasium.config_path "$dataset/train_dataset.yaml"
  --logdir "$smoke_logdir" --experiment_seed 0 --dreamer.seed 0
  --jax.platform cuda --jax.prealloc False --logger.outputs jsonl
  --batch_size 1 --batch_length 4 --report_length 4 --replay_context 0
  --run.train_ratio 1 --run.save_every 1 --run.log_every 1 --run.report_every 1)
"$python_bin" -u -m dreamerv3.main "${args[@]}" --run.steps 20
"$python_bin" - "$smoke_logdir" <<'PY'
import pickle,sys
import json
from pathlib import Path
root=Path(sys.argv[1]);latest=root/'ckpt'/(root/'ckpt/latest').read_text().strip()
assert pickle.loads((latest/'step.pkl').read_bytes()) == 20, 'Missing step-20 checkpoint'
assert (latest/'driver.pkl').is_file(), 'Missing financial Driver checkpoint'
state=pickle.loads((latest/'driver.pkl').read_bytes())['envs'][0]['env']['account']
(root/'smoke_before_resume.json').write_text(json.dumps(state,indent=2))
PY
# Same logdir restores agent, replay, Driver and the complete financial account.
"$python_bin" -u -m dreamerv3.main "${args[@]}" --run.steps 30
"$python_bin" tools/fg_entry_gpu_smoke_audit.py "$smoke_logdir"
