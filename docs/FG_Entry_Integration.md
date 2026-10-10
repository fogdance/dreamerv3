# FG entry integration

Independent branch: `feat/fg-entry-v1.1`. Companion environment registers `FixedDirectionEntryEnv-v0`; expose its `src` directory through PYTHONPATH. Historical training profiles remain unchanged; new default switches are disabled.

## Formal training admission

`fg_entry_v1` uses two actions (WAIT=0, ADD_3=1), mandatory frozen FG train manifest, account/Driver checkpointing and an always available WAIT. Uncertain imagined availability falls back only to WAIT. Availability warm-up remains required; a warm-up checkpoint is rejected by formal evaluation.

```sh
python -m dreamerv3.main --configs defaults fg_entry_v1 fg_entry_dataset_v1 \
  --fg_entry.manifest DATASET/manifest.json \
  --env.gymnasium.config_path DATASET/train_dataset.yaml \
  --logdir NEW_LOGDIR --experiment_seed 0 --dreamer.seed 0
```

Use separate directories for seeds 0, 1, 2. Only records with user-confirmed complete data coverage and completed main-roll/expiry lifecycle are admitted; an artificial cutoff cannot discard remaining positions. No formal real-data training has been run yet.

## Comparable evaluation

```sh
python tools/fg_entry_evaluate.py --model-config RUN/config.yaml \
  --checkpoint RUN/ckpt/COMPLETE_CHECKPOINT_DIRECTORY \
  --env-config DATASET/REAL_CONTRACT_test.yaml --output NEW_COMPARISON --seed 0
```

Same financial environment and conservative fill protocol for Always ADD, Random ADD, 20-bar/5-point retracement ADD and Dreamer. Also outputs matched-frequency Random. Reports include TP count/wait/reward, unfinished positions, maximum floating loss and account net PNL, plus fills, per-entry attribution, reward events and account equity. Timing ability is not asserted until out-of-sample multiple-seed evidence is reviewed.

## Verification evidence

13 targeted tests passed (26.06 seconds). Tests cover two-action WAIT fallback while preserving legacy argmax fallback, mandatory FG manifest, serial/parallel Driver restoration of account and pending action/carry, and actual Gym OrderEnforcing restoration of a losing three-lot position. The wrapper uses reset(options=resume) to restore both Gym wrapper flags and the same financial account.

`tools/fg_entry_smoke.py` creates synthetic fixtures, performs two actual world-model/actor updates through replay/stream, then executes masked policy inference. This is strictly wiring verification, not a market backtest or a trained timing model. Result: 49 finite scalar metrics and legal WAIT action, recorded at `/tmp/fg-dreamer-smoke-w5sts2_g/result.json` during this session.

Account checkpointing is supported by the ordinary train script. The FG profile is not admitted for distributed parallel train scripts until those scripts implement matching account checkpoint semantics.

## Unified business configuration

The companion repository maintains directions, costs, cutoff, excluded FG805, expiry protocol and reward contract only in `configs/fg_entry_v1/fg.yaml`. Dataset construction emits frozen per-contract snapshots and the train selector. Editing the master invalidates frozen manifest admission until rebuilding. dataset_v3 has 24 main segments and 12 complete train lifecycles; real-data Gym reset and admission were verified.
