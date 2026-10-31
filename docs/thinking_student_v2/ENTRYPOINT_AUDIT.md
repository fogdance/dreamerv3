# V2 正向教材：入口审查

日期：2026-10-31
依据：`Thinking_Student_V2_Positive_Entry_Corpus_Claude_Spec_V0_3.md` 第 2.1 节第 1 步、第 5 节第 1 步。
本文只记录现有实现和数据的事实，以及 V2 复用哪些部分。

## 一、约束

- 只遵守工作区根目录的 `AGENTS.md`（Thinking Student V1 专线），两仓自己的 `AGENTS.md` 不适用。
- 保留 Native 金融规则、合法动作 Mask、因果 Observation 和 TRAIN/DEV/TEST 隔离。
- 本轮只用 TRAIN；DEV、PAPER 和 TEST 都不用于挑选正例。
- V1 的运行结果只读。
- 本轮不训练学生、不启动 1x，也不接入 L3/L5。
- 生产代码只改一处：用户 2026-10-31 要求 TP5/SL30 可配置（见第六节）。默认值不变。

## 二、数据

| 项目 | 事实 |
|---|---|
| 市场池 | `dreamerv3/configs/thinking_student/markets/pool.json`：18 个 TRAIN，加 1 个 VALIDATION（FG409，DEV）；TEST 封存，不在池中 |
| 1m 行情 | 掘金导出的 CSV：`Date` 是 bar 结束时间，`bob` 是 bar 开始时间，另有 `trading_date`、`Volume`、`OpenInterest` |
| 交易时段 | 21:00–23:00、09:00–10:15、10:30–11:30、13:30–15:00，每天 345 个市场分钟（`strategy_quality_tp_v1.yaml: market_clock`） |
| 逐笔数据 | 没有。触发只能在 1m 收盘后确认，按下一根 bar 的开盘价成交 |

## 三、方向来源

- 方向写在 `strategy_quality_tp_v1.yaml: directions`（第 15 行起）：每个合约一个固定区间，`date_basis: trading_date`。
- 由 `envs/entry_env.py:341 direction_at` 读取。
- 这些区间由用户配置，没有任何记录说明它们在当时就已知。
- 结论：V2 的每一行都标为 `direction_source = DIRECTION_ASSUMED`，并记录配置路径。**不能据此宣称已验证独立交易收益。**

TRAIN 中方向为 SHORT 的合约共 9 个：FG809、FG901、FG905、FG005、FG201、FG205、FG209、FG301、FG405。
其余 9 个 TRAIN 合约是 LONG，本轮不做镜像。

## 四、15 分钟 K 线

- 来源：`gym-trading-env/e_data_v1/core.py:248 MarketData._buckets`。
  - 按交易时段切 15 分钟，每个 bucket 必须有 15 根有效的 1m bar、日历连续、属于同一个交易日，才算 `complete`。
  - 不完整的 bucket 不会被补造，会在 `events` 中记为 `BUCKET_REJECTED`。
  - 每个 bucket 带 `available_at`：CSV 行情就是 bucket 结束时间。
- 获取：Native env 带 E profile 构造时（`ts_v1_corpus.native_env(config, market=True)`），`env._e_market` 就是这份 `MarketData`（`entry_env.py:251 MarketData.from_env`），与 Native 共用有效分钟和日历。
- V2 只用 `complete_buckets`，并只在 `available_at ≤` 决策时刻时可见。

## 五、Native 成交与费用

**入场**（`envs/entry_env.py:607 _native_step`）：
- 游标 t 的决策在 bar t 收盘后做出。
- ADD 在 bar t+1 的 `bob`、以开盘价成交。
  - 做空时成交价为 Open − spread，向下取整到 tick（`_price`，第 292 行）。
  - 要求该 bar Volume ≥ 3、方向和截止时间不变、时钟有效、资金足够（`_open_entry`，第 474 行）。
- 同一根 bar 的最高价不会被当作成交价。

**出场**（`envs/entry_execution.py:108 single3_exit_decision`，由 `_exit_request` 第 569 行调用）：
- SL：不利报价触及止损价就成交，做空时是 High + 1 ≥ SL。
  - 如果开盘就跳空越过止损，按开盘报价成交。
  - 同一根 bar 同时触及 TP 和 SL 时，SL 优先，并标记为 `ambiguous_tp_sl`。
- TP：有利报价必须越过 TP 价至少 1 tick，做空时是 Low + 1 ≤ TP − 1；按 TP 价成交。
- 入场那根 bar 的 TP 只用收盘价判断。

**费用**（`strategy_quality_tp_v1.yaml: costs`，第 127 行）：

| 项目 | 数值 |
|---|---|
| 开仓 / 平仓手续费 | 每手 3 / 3 元 |
| 每手乘数 | 20 |
| 点差 | 1 |
| 滑点 | 0 |
| 下单量 | 固定 3 手、单仓位 |

即每点 60 元，一笔交易手续费 18 元（例：TP5 = +282 元，SL30 = −1,818 元）。

## 六、TP / SL 距离

- **改之前：**
  - env 本身已可配置 `entry.take_profit_points / stop_loss_points`（gym e58c27e），默认 TP5/SL30。
  - 但下列教材路径写死了 5/30：
    - `research/fg_single3_oracle.py`：`CandidateIndex.candidate` 的 SL/TP 价、C0–C3 打分器、资金证明中的 TP 净收益下界。
    - `research/ts_v1_grades.py`：检查 TP 是否位于测得成交价 +5 和 1 个 tick 处。
    - `research/ts_v1_corpus.py`：种类名 `KINDS = ('TP5','SL30',…)` 和 `native_check`。
    - `research/fg_s3_quality_teacher_v2.py`、`fg_s3_quality_catalog_v2.py`：种类名。
    - Dreamer `thinking_student/labels.py`：`EVENTS`。
    - Dreamer `thinking_student/run.py`：`FACT_KINDS`，以及 `check_collection` 的 5/30 默认值。
- **本轮（用户 2026-10-31）：**
  - 上述路径都改为读取 env 的距离。
  - 种类改名为 `TP` / `SL`。整数代码不变，已有教材仍可读取。
  - `ts_v1_corpus --take-profit-points / --stop-loss-points` 把距离写入 `<corpus>/inputs/env.yaml`，并在 `source.json inputs.rules` 中记录。
  - 学生持仓输入的尺度（/5、/30）和 Outcome MAE/MFE 头的尺度（30）保持不变。它们是模型的输入尺度，不是规则；2026-10-30 的 PAPER 考试也要求这一点。
  - Quality TP 奖励中的 early-adverse 10 点是奖励常数，不是 TP/SL 距离，保持不变。
  - `SINGLE3_FILL_MODEL` 这个身份名称不变，距离已经通过 env 配置进入 `config_hash`。
- **V2 的用法：**
  - V2 的每笔交易都在真实 Native env 中执行：`stop_loss_points = R`、`take_profit_points = 2R`，R 的取值为 5…10。
  - 先由 `CandidateIndex` 给出索引结果，再用 `reference_candidate` 逐分钟复核。
  - 生产 Env 的成交和费用规则不变。

## 七、已有教材与读取

- `research/ts_v1_corpus.py` 按游标保存 `corpus/<contract>/{inputs,market,facts,strict,graded,behaviors}`，游标就是 env 的行号。
- Dreamer 的 `thinking_student/corpus.py` 用 `C.load_arrays` 读取。
- V2 的每行都带 `contract` 和 `cursor`（即 env 行号），可以按游标与已有的 `corpus/<contract>/market` 关联。不改 V1 教材和 V1 Reader。
- 资金：`fg_single3_oracle.funds_certificate` 证明，在固定 TRAIN 条款下，空仓账户的资金和保证金都不构成约束。所以每个候选都可以从空仓账户单独计算。

## 八、复用与新增

| 用途 | 复用 |
|---|---|
| env 和输入 | `ts_v1_corpus.train_row`、`write_inputs(rules)`、`native_env` |
| 15m bucket | `env._e_market.complete_buckets` |
| 交易事实 | `fg_single3_oracle.CandidateIndex`（已参数化）、`reference_candidate`、`empty_at` |
| 市场分钟 | `env.market_clock_counter.held` |
| 并行 | `ts_v1_corpus._pool`（spawn，每个 worker 1 个原生线程） |

新增：
- Gym `research/ts_v2_positive.py`：事件、候选、正例、审计和 CLI。
- Gym `research/ts_v2_positive_plot.py`：matplotlib 图。
- 配置：`dreamerv3/configs/thinking_student/v2_positive_entry.yaml`。
- 测试：`gym-trading-env/tests/entry_v1/test_ts_v2_positive.py`。
- 文档：本目录下的 `DESIGN.md` 和 `REPORT.md`。
- 输出目录：`/data/logdir/fg_single3/thinking_student/V2_POSITIVE_ENTRY_20261031/`。
