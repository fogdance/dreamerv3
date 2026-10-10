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

## Current authorization boundary

The user explicitly prohibits starting formal training. Do not run the training command until authorized again. The earlier launch attempt exited with code 1 during JAX backend initialization (AssertionError), before the training loop; its log is `/tmp/fg-entry-v1/train-fg-seed0.log`. No market-data training result is claimed.

## GPU smoke authorization after reboot

The user authorizes only a few GPU smoke steps and explicitly prohibits CPU training and formal training. `tools/run_fg_entry_gpu_smoke.sh DATASET NEW_LOGDIR` preserves the 50M architecture, WAIT warm-up and business rules, uses batch 1 / sequence 4 / ratio 1, runs to step 20, then reloads the same financial checkpoint and runs to step 30. JAX_PLATFORMS=cuda prevents CPU fallback. The script is bounded regardless of the million-step profile default.

Reboot cleared `/tmp/fg-entry-v1`. Commits were recovered without changing original working trees to `/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/worktrees/{env,dreamerv3}`; regenerated data and smoke logs remain in the parent persistent artifact directory. Earlier launch logs and generated results in /tmp were lost at reboot; previous test observations remain recorded in committed documents.

GPU smoke completed: 30 environment steps / 7 updates, finite losses and account cursor 19→29 after checkpoint restoration. See `docs/FG_Entry_GPU_Smoke_Acceptance.md` for observed bugs, fixes and limits. Full formal-batch memory and timing skill remain unverified.

## Final FG observation schema

The latest user revision removes expiry_remaining. Inputs are now `market_seq` (60,18), `position_state` (9,), and `action_mask` (2,). Position fields and identity semantics are frozen in companion `docs/FG_Entry_Observation_V1.md`; the existing encoder consumes the new dict without architectural changes. Previous smoke checkpoints use the old schema and are not new-schema training evidence. Six directly affected Gym/Driver/FG regressions passed with GPU available; no training was performed for this revision. Strict position-independent actor scores are no longer promised; their differences are a future dedicated comparison.


## 2026-10-10 原仓库开发入口

当前开发目录是 `/home/v/Documents/work/dreamerv3`，分支 `feat/fg-entry-v1.1`。环境代码位于 `/home/v/Documents/work/gym-trading-env`，同名分支。旧 artifacts 工作树已移除；本文较早章节的工作树路径仅为历史记录。迁移当时冻结数据集为 `/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/dataset_v1`。

`tools/run_fg_entry_gpu_smoke.sh` 默认引用同级 `gym-trading-env`，也支持 `FG_ENV_ROOT`。迁移后 6 项 FG 接口和账户恢复测试通过，脚本语法检查通过。未启动训练；旧 GPU smoke 检查点保留用于历史审计，不代表当前九维 Obs 已完成训练验证。


## 当前冻结数据集：按换主力边界切分

使用 `/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/dataset_v2_main_split/manifest.json` 与同目录 `train_dataset.yaml`。训练截止 2022-12-09，验证截止 2024-08-19，测试截止 2025-12-31；切分统一维护在 env 仓库 `configs/fg_entry_v1/fg.yaml` 的 `data.split`，禁止在合约主力区段中途切分。旧 dataset_v1 仅作历史审计，配置哈希过时，不用于当前训练或恢复。此次不启动训练。


## 当前数据入口：345 分钟导出及因果分钟索引

使用 `/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/dataset_v3_345_audited` 下的 `manifest.json` 和 `train_dataset.yaml`。切分日期不变。导出截断 23:01～23:30，保留早期交易日；行情索引按预先声明的 345 分钟时钟计算并固定除以 344。此前数据集仅作历史审计；统一配置和行情时钟版本变化后不恢复旧金融检查点。未启动训练，GPU 训练 smoke 尚未重跑。核对 env 文档 `docs/FG_Entry_Minute_Clock_Fix.md`。
