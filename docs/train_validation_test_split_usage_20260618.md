# DreamerV3 Train / Validation / Test Split Usage

Date: 2026-06-18

Status: code-based audit, no training behavior changed

## Final Answer

Short answer:

DreamerV3 has generic **train + eval** execution modes, but the current code
does not provide a native, first-class **train + validation + test** dataset
contract for trading. For the current trading experiments, we are not using
DreamerV3's built-in `train_eval`/`parallel` eval streams as validation/test.
Instead, validation and test are implemented outside the training loop by the
checkpoint audit pipeline.

Current DreamerV3 walk-forward trading runs use the three datasets as follows:

| Stage | How It Is Used | Official Walk-Forward Meaning |
| --- | --- | --- |
| Training | `dreamerv3/main.py --script train` builds one training env from `--env.gymnasium.config_path`; all replay and model updates come from this env only. | The env config must point to a train-only market file. |
| Validation | Not part of the main training loop. Validation is a post-training deterministic checkpoint audit using `tools/dreamer_checkpoint_audit.py --roles validation`. | Used only to choose among retained checkpoints. |
| Test | Not part of the main training loop. Test is a post-selection deterministic checkpoint audit using `--roles test`. | Run once only after checkpoint selection. |

The important implementation fact is that DreamerV3 itself does not know a
native `train/validation/test` market split during `script=train`. The split is
enforced by:

1. the env YAML used during training;
2. the entry-eval `split_manifest.json`;
3. the checkpoint audit command choosing `--roles validation` or `--roles test`.

For JM, the current split is clean and chronological. For RB, the current
training env and current entry-eval split manifest are not yet aligned for
formal validation/test use; details are in the RB section below.

## Does DreamerV3 Natively Support Three Datasets?

Code-based answer:

| Capability | Exists In Code? | Current Trading Use? | Notes |
| --- | --- | --- | --- |
| One training env and one training replay | Yes | Yes | `script=train` uses this path. |
| Generic train + eval envs | Yes | No, not as official walk-forward validation/test | `script=train_eval` and `script=parallel` have eval drivers/replays. |
| Separate validation and test env configs | No first-class config contract found | No | There is no built-in `env.validation.config_path` and `env.test.config_path` flow. |
| Validation checkpoint selection | Not native to training loop | Yes, via audit scripts | We audit retained checkpoints with `--roles validation`. |
| Test run-once after selection | Not native to training loop | Yes, procedural via audit scripts | We audit selected checkpoint with `--roles test`. |

So the precise statement is:

```text
DreamerV3 supports generic evaluation, but not the exact trading
train/validation/test protocol we need. The current project implements the
three-way split procedurally with train-only env configs plus post-training
checkpoint audits.
```

The current code could be extended to make validation/test first-class, but that
is not how it works today.

## Program-Level Guard

The startup path now includes a metadata-only walk-forward guard:

```text
dreamerv3/walk_forward_guard.py
```

`dreamerv3/main.py` calls this guard after config parsing and seed resolution,
but before creating the env, agent, replay, or training loop.

The guard checks:

1. `audit.entry_eval_version` is configured for formal walk-forward runs.
2. `audit.split_manifest_hash` is configured and matches the actual
   `split_manifest.json` SHA256.
3. `audit.execution_timing` matches the entry-eval manifest when the manifest is
   available.
4. The training env config exists.
5. The training market CSV can be resolved from `trading.data_path` and
   `trading.data_interval`.
6. The training data date range does not extend beyond the split train role.
7. The training data date range does not overlap validation or test trading
   days.

Guard result is saved into the run config as:

```text
walk_forward_guard_result
```

This makes the split source auditable in `config.yaml`.

### Formal vs Debug

Default behavior:

```text
walk_forward_guard.enabled: true
walk_forward_guard.allow_non_holdout_training: false
```

If the guard detects that training data covers validation/test dates, formal
training fails fast.

For explicit pilot/debug runs only, the run can set:

```text
--walk_forward_guard.allow_non_holdout_training true
```

In that case the run is allowed to start, but the guard result status is:

```text
non_holdout
```

Such a run must not be reported as formal validation/test or holdout evidence.

## Code Paths

### DreamerV3 Main Entry

Relevant file:

`/home/v/Documents/work/dreamerv3/dreamerv3/main.py`

Key behavior:

- Lines 167-176 load `dreamerv3/configs.yaml`, parse command-line overrides,
  and resolve the seed protocol.
- Lines 217-224 dispatch `script=train` to `embodied.run.train(...)`.
- Lines 234-242 dispatch `script=train_eval` to a generic train/eval run, but
  this is not the official trading walk-forward validation/test protocol.
- Lines 357-385 build replay. The replay seed is `seed_protocol.replay_seed`.
- Lines 388-432 build envs. For `gymnasium`, `reset_seed` is passed into the
  wrapper from `env.train_seed` or `env.eval_seed`.

### Training Loop

Relevant file:

`/home/v/Documents/work/dreamerv3/embodied/run/train.py`

Key behavior:

- Lines 102-107 create only training envs and add every transition to the same
  replay buffer.
- Lines 119-135 train the agent from that replay buffer.
- Lines 137-151 save normal and retained checkpoints.
- Lines 157-162 run `agent.policy(..., mode='train')`.
- Lines 164-170 run report batches from the same training replay. This is model
  reporting, not walk-forward validation.

There is no validation/test replay in `script=train`.

### Generic `train_eval` Is Not The Official Split

Relevant file:

`/home/v/Documents/work/dreamerv3/embodied/run/train_eval.py`

`train_eval` creates a train driver and an eval driver, but `main.py` passes
`bind(make_env, config)` for both. Unless a separate env config is explicitly
injected, this generic eval env uses the same config source. It should not be
treated as the official trading validation/test split.

`train_eval` has two datasets at most:

- train env/replay;
- eval env/replay.

It does not distinguish validation from test, does not select retained
checkpoints by validation, and does not enforce test run-once semantics.

### Generic `parallel` Eval Is Also Not The Official Split

Relevant file:

`/home/v/Documents/work/dreamerv3/embodied/run/parallel.py`

`parallel` can separate train and eval traffic:

- `parallel_actor` dispatches eval workers with `mode='eval'` and train workers
  with `mode='train'`;
- `parallel_replay` writes eval transitions into `replay_eval`;
- `parallel_learner` can report from the eval replay.

This is still a generic train/eval mechanism, not a three-way
train/validation/test protocol. In `main.py`, `script=parallel` also receives
`bind(make_env, config)` for both train and eval env factories. Without explicit
additional wiring, train and eval use the same env config source.

### `eval_only` Is One Evaluation Dataset

Relevant file:

`/home/v/Documents/work/dreamerv3/embodied/run/eval_only.py`

`eval_only` loads a checkpoint and runs one env config with
`agent.policy(..., mode='eval')`. It can be pointed at a validation or test env
manually, but it does not know which role it is evaluating and does not enforce
validation-only checkpoint selection or test run-once rules.

### Checkpoint Audit

Relevant file:

`/home/v/Documents/work/gym-trading-env/tools/dreamer_checkpoint_audit.py`

Key behavior:

- Lines 264-277 map trading days to roles using the last entry-eval fold in
  `split_manifest.json`.
- Lines 242-261 select candidate start rows by role and `start_clock`.
- Lines 905-919 in `collect_mode=per_day` force one deterministic start row per
  selected trading day.
- Lines 965-967 run the loaded checkpoint policy with `mode="eval"`.

This is the official checkpoint validation/test mechanism used by the current
walk-forward workflow.

### Entry-Eval Split Generation

Relevant file:

`/home/v/Documents/work/gym-trading-env/src/gym_trading_env/research/entry_analysis.py`

Generic fold behavior:

- Lines 848-861 split sorted trading days into seven blocks and create three
  walk-forward folds.
- Lines 864-900 build purged fold masks for supervised entry baselines.

Important distinction:

- Ridge/XGBoost entry capability uses all generated folds.
- Dreamer checkpoint audit uses only the last fold as the current overall
  `train/validation/test` role map.

JM currently uses a custom one-fold split manifest. RB currently uses the
generic three-fold split manifest.

## Current Training Command Contract

The main training scripts are:

| Product | Script | Training Env Config |
| --- | --- | --- |
| JM | `/home/v/Documents/work/dreamerv3/tools/run_walk_forward_seed.sh` | `/home/v/Documents/work/gym-trading-env/configs/env_trading_stage1_jm_walk_forward_train_20240603_20250731.yaml` |
| RB | `/home/v/Documents/work/dreamerv3/tools/run_walk_forward_seed_rb.sh` | `/home/v/Documents/work/gym-trading-env/configs/env_trading_stage1_rb8y_walk_forward_train_20171225_20241231.yaml` |

Both scripts call:

```bash
python dreamerv3/main.py --configs action_mask_formal --script train ...
```

`script` is inherited from `defaults.script: train`.

Both scripts pass:

```text
--experiment_seed
--dreamer.seed
--env.train_seed
--env.eval_seed 0
--replay.seed
--audit.matched_random_seed
--audit.entry_eval_version
--audit.split_manifest_hash
--audit.execution_timing signal_on_close_plus_spread
```

The audit metadata is saved in `config.yaml`, but validation/test are not run
inside the training process.

## JM Current Split

Entry-eval artifact:

`/home/v/Documents/work/gym-trading-env/artifacts/entry_eval/entry_eval_jm_walk_forward_20240603_20251202_signal_close_v1/split_manifest.json`

Current hash:

```text
eb0f7073d35afd4b76f73eddcbf11cf24539287f502ec7e83726d48343ff598e
```

Current split:

| Role | Date Range | Trading Days |
| --- | --- | ---: |
| train | 2024-06-03 .. 2025-07-31 | 284 |
| validation | 2025-08-01 .. 2025-08-29 | 21 |
| test | 2025-09-01 .. 2025-12-02 | 61 |

Training env:

```text
data_path: JM_WF1_TRAIN_20240603_20250731
randomize_start: true
start_clock: future_night
```

Full audit env:

```text
data_path: 18M_DCE_JM2601
randomize_start: true
start_clock: future_night
```

Conclusion:

JM is currently aligned:

- training uses train-only data;
- validation/test are absent from the training env;
- validation/test are replayed only by forced per-day audit over the full env;
- validation is used for checkpoint selection;
- test should be run once after validation selection.

The existing detailed split report is:

`/home/v/Documents/work/gym-trading-env/docs/walk_forward_data_split_visibility_report.md`

## RB Current Split

Entry-eval artifact:

`/home/v/Documents/work/gym-trading-env/artifacts/entry_eval/entry_eval_rb8y_walk_forward_20171225_20251202_signal_close_v1/split_manifest.json`

Actual current hash:

```text
b1f2f9a048c09c7663ccadcf3c162cf7f55f2818445a108975696f37b6593845
```

The RB training script currently defaults to this same hash:

```text
b1f2f9a048c09c7663ccadcf3c162cf7f55f2818445a108975696f37b6593845
```

The RB script metadata has been updated to the current generated artifact hash.
The remaining problem is split/env overlap, not hash mismatch.

Current RB entry-eval folds:

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| fold_1 | 2017-12-25 .. 2020-04-01 | 2020-04-02 .. 2021-05-21 | 2021-05-24 .. 2022-07-08 |
| fold_2 | 2017-12-25 .. 2021-05-21 | 2021-05-24 .. 2022-07-08 | 2022-07-11 .. 2023-08-23 |
| fold_3 | 2017-12-25 .. 2022-07-08 | 2022-07-11 .. 2023-08-23 | 2023-08-24 .. 2024-10-17 |

Dreamer audit uses the last fold, so current RB role map is:

| Role | Date Range | Trading Days |
| --- | --- | ---: |
| train | 2017-12-25 .. 2022-07-08 | 1101 |
| validation | 2022-07-11 .. 2023-08-23 | 275 |
| test | 2023-08-24 .. 2024-10-17 | 275 |

But the current RB training env is:

```text
data_path: RB_8Y_WF_TRAIN_20171225_20241231
```

This training env covers validation/test dates from the current last fold. As a
result, current RB artifacts are useful for pilot training and debugging, but
should not be treated as a clean formal validation/test protocol until one of
these is fixed:

1. regenerate a fixed RB split manifest whose train role matches
   `2017-12-25 .. 2024-12-31`, with validation/test after that range; or
2. restrict RB training env data to the last-fold train range
   `2017-12-25 .. 2022-07-08`; or
3. explicitly define a different RB official split and update both env configs
   and split manifest hash together.

With the guard enabled, the default RB script fails as formal holdout training.
To run the current RB setup as explicit pilot/debug only:

```bash
ALLOW_NON_HOLDOUT=true ./tools/run_walk_forward_seed_rb.sh seed_a 101 /data/logdir/action-mask-rb8y-pilot1m-seed-a-20260617 pilot1m
```

## Visibility Rules

### During Training

Dreamer can see only what the training env loads from its `config_path`.

For `script=train`:

- env transitions go into one training replay;
- batch sampling comes from that replay;
- model initialization, exploration, and training are controlled by
  `dreamer.seed` / resolved `config.seed`;
- random training starts are controlled by `env.train_seed` on first reset and
  the env RNG after that;
- validation/test roles are not sampled unless the training env config itself
  contains those dates.

Therefore the train-only env config is the hard boundary. If a full env config
is passed to training, validation/test leakage occurs.

### During Validation

Validation is an offline deterministic checkpoint audit:

```bash
tools/dreamer_checkpoint_audit.py \
  --collect \
  --collect-mode per_day \
  --roles validation \
  --checkpoint <retained_checkpoint>
```

It loads:

- the saved Dreamer checkpoint;
- the run `config.yaml`;
- the full evaluation env config;
- the entry-eval candidate table and `split_manifest.json`.

It overrides start-row selection so only validation days are replayed. It uses
`agent.policy(..., mode="eval")`.

The validation result is allowed to choose the checkpoint step, for example
500k vs 700k vs 900k vs latest.

### During Test

Test is the same audit mechanism with:

```bash
--roles test
```

The test run should happen only once after validation selection. Test must not
be used to choose checkpoints, tune seeds, tune reward, or tune env settings.

## Relationship To Entry-Eval Candidates

Entry-eval artifacts are offline research artifacts. They are not fed into
Dreamer training.

They provide:

- candidate IDs;
- split roles;
- fixed-exit counterfactual outcomes;
- oracle fixed-exit reference;
- matched-random same-count baselines;
- attribution joins for Dreamer trades.

Dreamer training does not read `candidates.parquet`, `outcomes.parquet`, or
`split_manifest.json`. The audit scripts read them after training.

## Current Official Interpretation

### JM

Current JM walk-forward results can be interpreted under a valid
train/validation/test protocol if commands follow the documented sequence:

1. train with train-only env config;
2. evaluate retained checkpoints on validation;
3. select checkpoint using validation only;
4. evaluate selected checkpoint once on test.

### RB

Current RB run commands are operationally valid for training, but the current
validation/test role map is not clean relative to the current training env date
range. RB needs split/config alignment before any result is called a formal
holdout or formal validation/test result.

## Must Not Be Confused

- `run.eval_envs` in `action_mask_formal` is not the official walk-forward
  validation/test protocol for `script=train`.
- `agent.report(...)` during training is not validation; it reports on the
  training replay.
- `train_eval` and `parallel` are generic Dreamer modes; they are only official
  validation/test if their env config and split roles are explicitly wired for
  that purpose.
- `eval_env_seed` is recorded for audit metadata. In per-day checkpoint audit,
  selected validation/test starts are forced by split role, not randomly sampled.

## Recommended Hard Rules

For future Dreamer trading experiments:

1. Always use a train-only env config for training.
2. Pin and record `entry_eval_version` and `split_manifest_hash`.
3. Reject if the current split manifest hash differs from the run config
   metadata.
4. For validation/test, always use `collect-mode per_day`.
5. Validation may choose the checkpoint; test may not.
6. Never use the full eval env config for training unless the experiment is
   explicitly not a holdout experiment.
7. For RB, fix the split/env mismatch before formal reporting.
