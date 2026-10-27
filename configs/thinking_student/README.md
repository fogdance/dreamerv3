# Thinking Student V1 configs

Every Thinking Student configuration lives in this directory; the code
assumes no other path. Absolute paths below point to data and run folders
on `/data`, never to another worktree.

| File | What it sets | Read by |
| --- | --- | --- |
| `student.yaml` | student settings (heads, burn-ins, ε of μ, loss scales) and the CPU test preset `small` | `thinking_student/config.py` |
| `formal_v1.yaml` | formal training on every TRAIN contract: paths, contracts, budgets, rounds, CF, value anchor, exams (zero-carry windows; L1, L2 and the first pi_0000) | `formal_train --config` |
| `formal_v1_carry.yaml` | the formal training after the state-check: L1/L2 inherited from `formal_v1` ckpt 22000; L3–L5 windows start from the MarketCarry stores (`student.window_start: carry`); rounds 7–9 add L6 (`iteration.l6`, OFFLINE_TRAIN) | `formal_train --config` |
| `formal_v1_l1_l2_10x_continue.yaml` | the formal run with ten times the steps of `formal_v1_carry` (L1 100k, L2 120k, L3 120k, L4 5000, L5 1000, L6 L2 review 2000) and ten times its WAIT forks; trained in a worktree of its own, the development run stays on `formal_v1_carry` | `formal_train --config` |
| `formal_v1_carry_pi2_cf_explore.yaml` | an exploration run that goes on from `pi_0002` of the development run with the FG201 CF supplement: `formal_v1_carry` with its own `run_root`, `iteration.cf_coverage: exploratory` and L6 from round 3; its header lists the links that prepare the run | `formal_train iterate --config` |
| `c4_smoke_2c.yaml` | the same flow on FG005 + FG405 with small budgets | `formal_train --config` |
| `markets/pool.json` | market pool: 18 TRAIN rows (corpora) and 1 VALIDATION row (FG409, DEV); TEST is sealed and not listed | gym `ts_v1_corpus --pool`, `exam --pool`, `paths.markets` |
| `markets/envs/`, `markets/contracts/`, `markets/strategy_quality_tp_v1.yaml` | Native env of each contract, its contract terms, the Single3 strategy master with the Quality TP reward | through `pool.json` |

- **Network size:** a Dreamer size preset is named in `student.preset`
  (formal: `size50m`); its definition stays in `dreamerv3/configs.yaml`.
- **CF budget:** every legal fork first gets a Native probe of
  `iteration.cf_probe_steps` (64). `iteration.cf_budget` is the
  continuation quota of each contract: a stratified sample of the probes
  that did not end goes on from its saved state up to `cf_max_steps`; the
  others are `UNAVAILABLE_BUDGET` without a label. A finished fork's
  snapshot in `<cf>/work/` is deleted once the CF is complete; only the
  unfinished probes keep theirs, so a larger quota can resume them.
- **L4 coverage:** with `iteration.cf_min_coverage` (0.8), a round whose
  valid pairs cover too few forks of a contract skips L4 and keeps the
  value brain and anchor of its pi_k (`cf_coverage: representative`, the
  default). An exploration config sets `cf_coverage: exploratory`: L4 then
  learns from the valid pairs it has. The round records
  `S3_Q_TARGETS_EXPLORATORY` and lists the low contracts in `L4_coverage`.
- **L6 remediation:** `l6.remedial_fraction: 0` (E-7 option A, user
  2026-10-28): the L2 review and L5 take natural windows only. The teaching
  items, the natural L2 review and the L4 decision-error weight
  (`remedial_boost`) stay.
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
$PY -m dreamerv3.thinking_student.formal_train iterate  --config $CFG/formal_v1_carry.yaml --rounds 5  # rounds 2-6
$PY -m dreamerv3.thinking_student.formal_train iterate  --config $CFG/formal_v1_carry.yaml --rounds 3  # rounds 7-9, L6
$PY -m dreamerv3.thinking_student.formal_train iterate  --config $CFG/formal_v1_carry.yaml --rounds 1
$PY -m dreamerv3.thinking_student.formal_train evaluate --config $CFG/formal_v1_carry.yaml --policy pi_0001
$PY -m dreamerv3.thinking_student.formal_train status   --config $CFG/formal_v1_carry.yaml

# the 10x run, from the gym worktree next to its own dreamerv3 worktree:
# L1 went on by hand (run.py train-market --updates 90000); then
$PY -m dreamerv3.thinking_student.formal_train pretrain --config $CFG/formal_v1_l1_l2_10x_continue.yaml

# the pi_2 -> pi_3 exploration run: make the links of its header once, then
# iterate only (never pretrain); round 3 reuses the round-3 experience
$PY -m dreamerv3.thinking_student.formal_train iterate  --config $CFG/formal_v1_carry_pi2_cf_explore.yaml --rounds 1

# S5 teaching items of one contract by hand (CPU, seconds; iterate writes them with L6 on)
$PY -m dreamerv3.thinking_student.feedback --collection RUN/experience/pi_0006/collect/CZCE.FG001 \
    --corpus CORPUS_FG001 --cf RUN/experience/pi_0006/cf/CZCE.FG001 \
    --kappa-cf RUN/experience/pi_0006/cf/CZCE.FG001 --kappa-cf ... --kappa-floor 0.05 \
    --out RUN/experience/pi_0006/feedback/CZCE.FG001

# market carry stores by hand; state-check of windows that start from them
$PY -m dreamerv3.thinking_student.run market-carry --policy RUN/policies/pi_0000 --corpus CORPUS --out RUN/market_carry
$PY -m dreamerv3.thinking_student.run state-check --policy RUN/policies/pi_0000 \
    --collection RUN/experience/pi_0000/collect/CZCE.FG005 --corpus CORPUS_FG005 \
    --window-start carry --carry-root RUN/market_carry --seed-offset 1000003 --windows 1000
```
