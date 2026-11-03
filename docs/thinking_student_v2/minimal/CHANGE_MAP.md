# V2 最小改造：改动清单（CHANGE_MAP）

依据：`thinkstudent/Thinking_Student_V2_Minimal_Implementation_Plan_V0_5.md`（用户 2026-10-31）。

| 项目 | 内容 |
|---|---|
| 规则 | 只遵守工作区根目录的 `AGENTS.md`（Thinking Student V1 专线）：只提交不推送；GPU 训练前先问；TEST 保持封存；PAPER 和实盘不进训练 |
| 分支 | `feat/ts-v2-minimal-20261031`（gym 与 dreamer 两个仓库），从开发主线 HEAD 拉出：gym `0e1ef3b`，dreamer `58a623b` |
| Worktree | `/data/research/fg-s3-thinking-student-v2-20261031`。V1 的 worktree、配置、语料和 run 都不动 |
| 教材 | `/data/logdir/fg_single3/thinking_student/V2_1_POSITIVE_ENTRY_20261031`（规则 V2，用户在 K 线软件里看过），只读，不重新生成 |
| 用户决定（2026-10-31） | 计划风险上限 R ≤ 10 |

每个新键缺省时都保持原有行为：V1 的 `config_hash`、reward hash 和产出不变。

**简写：**
- **G** = `gym-trading-env/src/gym_trading_env`
- **E** = `gym-trading-env/e_data_v1`
- **D** = `dreamerv3/dreamerv3/thinking_student`
- **C** = `dreamerv3/configs/thinking_student`

## 1. 教材 → 现有 L3/L5 教师入口

**真实能力：**
- Dreamer 从 `corpus/<c>/graded/` 读示范，这个路径是写死的（`D/corpus.py` 的 `TSCorpus.__init__` 和 `corpus_sources`，`D/feedback.py` 的 `_items`）。
- 读取的数组：`valid`、`action`、`grade`、`ready_at`、`mfe_*`。
- `valid=False` 的分钟 `demo_weight` 为 0，不会被默认成 WAIT。
- 所以只需要把教材映射成同样的数组，L3/L5 的 loss 不改。

| 位置 | 改动 |
|---|---|
| `G/research/ts_v1_corpus.py:produce` + CLI | 新参数 `teacher=<V2_1 目录>`：`graded/` 改由下一行的映射写出，不再调用 `ts_v1_grades.grade()` |
| `G/research/ts_v2_positive.py:graded_teacher`（新函数） | 映射规则见下表 |
| `G/research/ts_v1_grades.py:measure` | 校验 TP 时用每笔交易自己的 TP（`env.planned_exit`），不再用固定距离 |
| `D/run.py`（L3 报告） | 教师名称从 summary 读取，不再写死 `GRADED_ENTRY_TEACHER` |
| `D/corpus.py:corpus_sources` | 各语料的 `inputs.rules` 也必须相同，V1 和 V2 语料不能混用 |

**映射规则：**

| 教材行 | 映射到 |
|---|---|
| `ADD`、`ACTIVATION_ADD` | ADD（`grade` 2 / 3，`ready_at` = 出场 cursor）。条件：R ≤ 10；env 的止损线就是教材的线；新语料在同一 cursor 的 Native 事实复现了这笔交易（TP、同一个出场、同样的净盈亏），否则报错。不满足前两条的分别记为 `RISK_ABOVE_MAX`、`STOP_LINE_DIFFERS`，写进 summary |
| `WAIT_FAR`、`WAIT_APPROACH`、`WAIT_NEAR_NO_VOLUME` | WAIT，只取还保留正例的事件；`grade` 为 −1（V2 的 WAIT 不是“POOR 交易”），L5 反馈汇总另按 action 计数（`demo_actions`） |
| `ACTIVATION` 和其它所有分钟 | `valid=False` |

## 2. 策略：SHORT_ONLY / 3 手 / 单仓位，结构止损 + 2R

**真实能力：**
- 现在只支持固定距离：`entry.take_profit_points` / `stop_loss_points`，即成交价 ± N 点。
- **不支持** `SL = 已确认前高 + 5`，也不支持 `TP = 成交 − 2R`。
- 增量：在现有的 TP/SL 计算点加一个参数化分支。不加 ADD 合法性 Gate，不加状态机，不另建 Env。

**线的选法（实测）：** 教材在一次触碰开始时定下这条线，整个触碰期间不变。这是事件状态，env 不能照搬。两种无状态的选法在 5,164 个 R ≤ 10 正例上的结果：

| 选法 | 与教材的线一致 |
|---|---|
| 活着的线中 ≥ 成交价的最低一条 | 4,159（80.5%） |
| 再要求价格已经离开过（1m 收盘 ≤ H − 10，与教材 `departure_points` 相同） | **5,163**（仅 1 个不一致） |

- 采用第二种。
- 不一致的 1 个在 FG405：触碰在 1594 上开始之后一分钟，下方又确认了一条 1593。
- 该正例记为 `STOP_LINE_DIFFERS`，不进教材。

| 位置 | 改动 |
|---|---|
| `G/envs/entry_env.py` | 新增 `entry` 键：`stop_rule: prior_15m_high`、`stop_above_points`、`max_risk_points`、`target_r`、`prior_high{pivot_left, pivot_right, lookback_buckets, departure_points}`。不需要 `min_risk_points`：H* ≥ 成交价，所以 R ≥ 5 |
| `G/envs/entry_env.py:planned_exit`（新方法） | 决策 cursor s 时活着的线（已确认、未破、未过期，且价格已经离开过：此前有一根 1m 收盘 ≤ H − 10）中，取 ≥ 成交价的最低 H*，R = H* + 5 − 成交价，SL = 成交 + R，TP = 成交 − 2R。`stop_source` 取值：STRUCTURAL（R ≤ 10）；CAPPED（R > 10，取 R = 10）；NO_LINE（没有线，取 R = 10）。LONG 报错。无状态：不记录触碰或事件 |
| `G/envs/entry_env.py:_compute_stop_loss_price / _compute_take_profit_price` | 设置了 `stop_rule` 时调用 `planned_exit`；ENTRY 成交记录 `stop_loss_price / take_profit_price / risk_points / stop_source` |
| `G/envs/entry_env.py:settle_exit` | TP 守卫改用这笔持仓自己的 TP 价格 |
| `G/research/ts_v2_positive.py:prior_highs` | 纯 pivot 部分拆成 `pivot_levels()`，“已离开”拆成 `departed_at()`，env 和教材共用；教材的输出不变 |
| `G/research/fg_single3_oracle.py:CandidateIndex.candidate` | SL / TP 阈值读成交后持仓上的价格；NET_PNL 下各候选的 reward 都是出场净 PnL / 尺度（见第 3 节） |
| `D/run.py:check_collection` | 按每笔交易自己的 SL / TP 核对 |
| `G/envs/entry_env.py:fill_model` | 设置了 `stop_rule` 时记为 `minute_structural_risk_first_v1`（同一根 K 线内仍先判止损）；V1 仍是 `minute_tp5_sl30_risk_first_v1`，hash 不变 |
| `G/research/ts_v1_corpus.py`（`source.json`、CLI） | `inputs.rules` 记录实际的出场规则（V2：`stop_rule`，不再写未使用的 TP5/SL30）。CLI 拒绝 `--strategy` 与 `--take-profit/stop-loss-points` 同时给，`--teacher` 必须配 `--strategy`，且不能与 `--grades` 同时给 |

## 3. Reward：净 PnL / 固定尺度；Q 用 R2

**真实能力：** 训练用的 Reward 是 R0：Quality TP、early10 惩罚、SL 固定 −3，没有净 PnL 选项；但每个出场事件已经带有含双边手续费的 `net_trade_pnl`。

| 位置 | 改动 |
|---|---|
| `G/rewards/single3_reward.py` | 新 schema `FG_SINGLE3_REWARD_NET_PNL_V1` + `reward_scale_cny`：每步 reward = Σ 出场的 `net_trade_pnl` / 尺度。原来的 scorer 继续负责出场校验和状态 |
| 尺度 | **618 元**：一次完整 10 点止损（10 × 3 手 × 20 + 18 元费用），−1 就是一次最大计划亏损，与 R2 的 1818 对 SL30 的含义相同 |
| `G/research/ts_v1_corpus.py:write_inputs` | `strategy=` 参数：叠加 V2 策略文件（entry 键 + reward），不再强制 QUALITY_TP |
| `G/research/fg_single3_oracle.py:CandidateIndex.candidate` → `ts_v1_corpus.candidate_facts` | NET_PNL 下 `reward = net_pnl / 尺度`（出场那一步），也就是 L2 的 `outcome_quality`；`native_check` 逐分钟核对它 |
| `D/q_return.py`（从 R2 分支 ffb9d28 带入）+ `D/formal_train.py` 接线 | `iteration.q_return: net_pnl_h2760`（H = 2760 内不折扣）。尺度从语料 env 读（缺省 1818）。NET_PNL 下原来的 SL −2.9 检查改为逐笔核对 reward == 净 PnL / 尺度 |

**一致性：** Replay 和 obs 里的 `reward`、L2 `outcome_quality`、CF 回报、Q 标签都来自同一个 env reward，不再遗留 R0。

**Outcome 标签（Review 后，用户 2026-11-03）：**
- Outcome 的 TP / SL / 其它出场类别本来就读 Native 交易，不改。
- MAE / MFE 头的单位原来写死为 30 点（SL30）。现在由语料决定（`D/labels.py:outcome_points`）：V2 用 `max_risk_points` 10，1.0 = 一个 R；V1 仍是 30。
  - 学生的 `ts.outcome_points` 必须与之相等，训练时检查，不一致就报错。
  - `formal_train` 的 `student.outcome_points` 经 `run.py --outcome-points` 传给每个新学生，快照里有记录。MFE 答案换算回点数时用它。
- early10 风险标志不变：5 / 15 分钟内不利 10 点仍是一个真实发生的事实。R ≤ 10 时它几乎等于“这段时间内止损”。
- `outcome_quality` 键名不变。内容是整笔交易的 env reward（V1：Quality TP；V2：净 PnL / 618），说明文字已改。

## 4. Observation：删 MACD、15m 24 → 48

**真实能力：**
- 学生的输入是：
  - `market_seq_e_shared` (60, 18)，本身不含 MACD
  - `market_context_15m` **(24, 7)**
  - `market_macd_5m` (4,)
  - 账户输入
- MACD 是 `MarketEncoder` 里单独的一个分支。
- 24 写死在 `E/core.py:snapshot`、`G/envs/entry_env.py`（obs space 和形状检查）、`D/market.py`（168 = 24 × 7）。

| 位置 | 改动 |
|---|---|
| `E/core.py:MarketData` | `context_buckets`（缺省 24）用于截取、左侧补零和 meta；仍然只用完整的 bucket，且 `available_at ≤ cutoff` |
| `G/envs/entry_env.py` | `entry.context_15m_buckets`（V2：48）传给 MarketData，也用于 obs space 和形状检查 |
| `C/student.yaml`、`D/config.py:observation_space`、`D/market.py:MarketEncoder` | `ts.market: {macd, context_15m}`。V2 用 `{false, 48}`：没有 `macd5` 分支，fusion 只接 1m + 15m。env 仍然输出 `market_macd_5m`，供旧项目使用，但没有任何 V2 参数读取它 |
| `D/run.py`、`D/formal_train.py` | `student.market` 传给每个新建的学生（`--market-macd / --context-15m`）；快照会记录它，`load_policy` 从快照恢复 |
| `D/transfer.py:plan` | `context15/kernel` 从 168 行加宽到 336 行，较早的 24 个 bucket 用零行（与现有 MACD 加宽方式相同） |

**维度：**

| 输入 | 现在 | V2 |
|---|---|---|
| 15m 上下文 | 24 × 7 = 168 | 48 × 7 = 336 |
| MACD | 4 | 0 |
| fusion 输入 | units + 128 + 16 | units + 128 |

**新 run：** 15m 长度变了，所以语料的 `market/` 要按 V2 env 重新生成，并开新的 run。

## 配置（`C/`）

| 文件 | 内容 |
|---|---|
| `markets/strategy_v2_short_structural.yaml` | 策略叠加：结构止损、R ≤ 10、2R、48 个 bucket、净 PnL / 618 |
| `formal_v2_minimal.yaml` | 即 V2_CONFIG：1x 预算。用户 2026-11-03 在训练 worktree `fg-s3-thinking-student-v2-1x-20261103` 上开始运行（`FORMAL_V2_MINIMAL_1X_20261103`） |
| `trial_v2_minimal.yaml` | 短试训 |

两份训练配置都设 `save_every: 0`，即每个阶段只保存最终 ckpt（size50m 约 388 MB 一个；用户 2026-11-03）。

**语料与合约：**
- 新语料根目录：`/data/logdir/fg_single3/thinking_student/corpus_v2_minimal_20261031`
- TRAIN 是 9 个 SHORT 合约：FG809、FG901、FG905、FG005、FG201、FG205、FG209、FG301、FG405
- DEV 是 FG409（SHORT）
- LONG 合约不强制改成 SHORT

## 提交

| 仓库 | 提交 |
|---|---|
| gym | `35e5d90` 四项改动；`dcf89b7` Review 后的修复 |
| dreamer | `8a46bcf` 学生市场开关、R2 净 PnL Q、V2 配置；`3abffab` Review 后的修复；`6f05725` 1x 的 run 目录；`db0239a` TODO、9 个语料 |

验证见 `TEST_REPORT.md`。
