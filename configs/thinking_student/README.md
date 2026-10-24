# Thinking Student V1 configs

Every Thinking Student configuration lives in this directory; the code
assumes no other path. Absolute paths below point to data and run folders
on `/data`, never to another worktree.

| File | What it sets | Read by |
| --- | --- | --- |
| `student.yaml` | student settings (heads, burn-ins, ε of μ, loss scales) and the CPU test preset `small` | `thinking_student/config.py` |
| `formal_v1.yaml` | formal training on every TRAIN contract: paths, contracts, budgets, rounds, CF, value anchor, exams (zero-carry windows; L1, L2 and the first pi_0000) | `formal_train --config` |
| `formal_v1_carry.yaml` | the formal training after the state-check: L1/L2 inherited from `formal_v1` ckpt 22000; L3–L5 windows start from the MarketCarry stores (`student.window_start: carry`) | `formal_train --config` |
| `c4_smoke_2c.yaml` | the same flow on FG005 + FG405 with small budgets | `formal_train --config` |
| `markets/pool.json` | market pool: 18 TRAIN rows (corpora) and 1 VALIDATION row (FG409, DEV); TEST is sealed and not listed | gym `ts_v1_corpus --pool`, `exam --pool`, `paths.markets` |
| `markets/envs/`, `markets/contracts/`, `markets/strategy_quality_tp_v1.yaml` | Native env of each contract, its contract terms, the Single3 strategy master with the Quality TP reward | through `pool.json` |

- **Network size:** a Dreamer size preset is named in `student.preset`
  (formal: `size50m`); its definition stays in `dreamerv3/configs.yaml`.
- **Market data:** the 1-minute CSV of each contract is data, not
  configuration. `pool.json` and the env files give its absolute path.
- **Built artifacts, not configs:**
  - A corpus copies its env, contract and strategy into
    `<corpus>/inputs/` when it is built, and trains and collects from that
    copy.
  - Each formal run writes its resolved config to `<run_root>/config.yaml`.

## Commands (from `gym-trading-env/`)

```bash
PY="env PYTHONPATH=.:src:../dreamerv3 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/v/miniconda3/envs/dreamerv3/bin/python -u"
CFG=../dreamerv3/configs/thinking_student

# teaching material of one TRAIN contract (CPU)
$PY -m gym_trading_env.research.ts_v1_corpus --contract CZCE.FG809 \
    --pool $CFG/markets/pool.json --out /data/logdir/fg_single3/thinking_student/corpus --workers 24

# formal training (GPU): pretrain, rounds, exams, progress
$PY -m dreamerv3.thinking_student.formal_train pretrain --config $CFG/formal_v1.yaml --until L2  # L1, L2 only
$PY -m dreamerv3.thinking_student.formal_train pretrain --config $CFG/formal_v1_carry.yaml  # stores, L3, pi_0000
$PY -m dreamerv3.thinking_student.formal_train iterate  --config $CFG/formal_v1_carry.yaml --until cf
$PY -m dreamerv3.thinking_student.formal_train iterate  --config $CFG/formal_v1_carry.yaml --rounds 1
$PY -m dreamerv3.thinking_student.formal_train evaluate --config $CFG/formal_v1_carry.yaml --policy pi_0001
$PY -m dreamerv3.thinking_student.formal_train status   --config $CFG/formal_v1_carry.yaml

# market carry stores by hand; state-check of windows that start from them
$PY -m dreamerv3.thinking_student.run market-carry --policy RUN/policies/pi_0000 --corpus CORPUS --out RUN/market_carry
$PY -m dreamerv3.thinking_student.run state-check --policy RUN/policies/pi_0000 \
    --collection RUN/experience/pi_0000/collect/CZCE.FG005 --corpus CORPUS_FG005 \
    --window-start carry --carry-root RUN/market_carry --seed-offset 1000003 --windows 1000
```
