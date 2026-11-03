# V2 最小改造：验证报告（TEST_REPORT）

依据：V0.5 计划第 3 步。改动见 `CHANGE_MAP.md`。

| 项目 | 内容 |
|---|---|
| 代码 | gym `dcf89b7`、dreamer `3abffab`（测试时）。之后 dreamer 只有 `6f05725`（1x 的 run 目录）和 `db0239a`（TODO） |
| 语料 | `/data/logdir/fg_single3/thinking_student/corpus_v2_minimal_20261031`（9 个 SHORT TRAIN 合约，2026-11-03 20:20） |
| 核对脚本与输出 | `/data/logdir/fg_single3/thinking_student/V2_MINIMAL_TEST_20261103/checks/`、`out/`。只用 env 和语料，不用 GPU |
| 未做 | GPU 链路 smoke（用户 2026-11-03：不用做）。正式 1x `FORMAL_V2_MINIMAL_1X_20261103` 本身跑这条链，见第 7 节 |

**结论：**
- 教师只在保留的正例和它们触碰的 WAIT 上有监督，其它分钟都没有。
- 正例经 V2 env 成交、止损、止盈、出场、净盈亏，与教材逐笔一致。
- Reward = 净 PnL / 618，与权益变化逐分相符。
- V1 不变。

## 1. 教师 mask

**FG205 触碰 `@960`（H 1850；教材 11 个正例，全是 EXCELLENT）：**

| cursor | teach | R（教材） | valid | 示范 | grade | ready_at | 该分钟自己的语料交易 |
|---|---|---|---|---|---|---|---|
| 956 | WAIT_FAR | | True | WAIT | − | 1042 | SL −618 |
| 957–959 | WAIT_APPROACH | | True | WAIT | − | 1042 | TP +1182 |
| 960 | ACTIVATION_ADD | 12 | False | − | − | −1 | TP +1182 |
| 961–964、967 | ADD | 11 / 12 | False | − | − | −1 | TP +1182 |
| 965、966、970–972 | ADD | 10 | True | ADD | EXCELLENT | 1042 | TP +1182（出场 1042） |

- 2 个没有 timeline 行的分钟（968、969）`valid=False`。
- 触碰 `@980` 的 2 个正例 R 为 11 / 12，全部去掉。它的 WAIT 行（976–979）因此也 `valid=False`（`WAIT_OF_DROPPED_TOUCH`）。

**9 个合约**（`graded/summary.json`，与数组重新计数一致）：

| 合约 | 教材正例 | ADD（GOOD / EXCELLENT） | WAIT | R > 10 | 线不一致 | 去掉的触碰的 WAIT | ACTIVATION | 保留触碰 / 教材触碰 |
|---|---|---|---|---|---|---|---|---|
| FG809 | 434 | 244（221 / 23） | 95 | 190 | 0 | 18 | 10 | 13 / 15 |
| FG901 | 1,829 | 1,537（1,160 / 377） | 147 | 292 | 0 | 57 | 24 | 30 / 38 |
| FG905 | 723 | 637（530 / 107） | 73 | 86 | 0 | 14 | 13 | 13 / 15 |
| FG005 | 685 | 436（357 / 79） | 81 | 249 | 0 | 9 | 10 | 19 / 24 |
| FG201 | 211 | 139（46 / 93） | 82 | 72 | 0 | 33 | 24 | 50 / 68 |
| FG205 | 310 | 194（48 / 146） | 72 | 116 | 0 | 56 | 28 | 48 / 74 |
| FG209 | 910 | 529（203 / 326） | 203 | 381 | 0 | 88 | 40 | 81 / 103 |
| FG301 | 1,571 | 1,195（805 / 390） | 276 | 376 | 0 | 38 | 26 | 57 / 63 |
| FG405 | 574 | 252（114 / 138） | 154 | 321 | 1 | 58 | 18 | 48 / 71 |
| **合计** | **7,247** | **5,163（3,484 / 1,679）** | **1,183** | **2,083** | **1** | **371** | **193** | **359 / 471** |

- 示范以外的分钟 `valid` 全为 False（每个合约 0 个）。
- 语料测试 `test_the_corpus_teacher_is_the_material_and_nothing_else`（FG809）逐个 cursor 核对：ADD = R ≤ 10 的正例，WAIT = 保留触碰的 WAIT 行，其它分钟都无示范。
- L3 读到的：重启前在 FG205 上核对过（核对脚本随 /tmp 丢失，没有重跑）。
  - `TSCorpus` 的 `demo_valid` 正好是 graded 的 `valid`。
  - L3 的标签账户 P0 取到 WAIT 72 / ADD 194，类别权重 (1.847, 0.686)。
  - 1x 的 L3 报告会写出 9 个合约合并后的计数，SMOKE_REPORT 里核对。

**WAIT 示范在该分钟自己的语料交易：**

| teach | WAIT 行 | 自己的交易是 TP | 比例 | TP 的止损来源 |
|---|---|---|---|---|
| WAIT_FAR | 219 | 28 | 12.8% | CAPPED 28 |
| WAIT_APPROACH | 950 | 369 | 38.8% | CAPPED 366、STRUCTURAL 3 |
| WAIT_NEAR_NO_VOLUME | 14 | 6 | 42.9% | CAPPED 2、STRUCTURAL 4 |
| 合计 | 1,183 | 403 | 34.1% | CAPPED 396、STRUCTURAL 7 |

- 这些 WAIT 是教材按设计在触碰之前的分钟：价格离线还远，或者没有量。
- 当时成交，止损线在 10 点以外，所以几乎都是 CAPPED（R = 10）。
- 其中 34% 的这类交易仍然到了 TP。L3 会在这些分钟教 WAIT，这是教材的选择，不是映射的错误。

## 2. 真实交易（FG205，V2 语料 env，`minute_structural_risk_first_v1`）

**账户 A：** 在保留的正例上 ADD，上一笔出场后才做下一笔，共 37 笔。成交、SL、TP、出场 cursor、出场原因、净盈亏与教材逐笔相同（各 37/37）。

| 决策 | 成交 | 线 H* | SL | TP | R | 来源 | 出场 | 手续费 | 净盈亏 | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| 941 | 1846 | 1850 | 1855 | 1828 | 9 | STRUCTURAL | 1035 TP | 18 | 1,062 | 1.7184 |
| 1504 | 1812 | 1817 | 1822 | 1792 | 10 | STRUCTURAL | 1551 TP | 18 | 1,182 | 1.9126 |
| 1809 | 1829 | 1830 | 1835 | 1817 | 6 | STRUCTURAL | 1846 TP | 18 | 702 | 1.1359 |

**账户 B：** 能 ADD 且空仓时以 2% 概率 ADD（seed 0），共 284 笔。

| 来源 / 出场 | CAPPED | NO_LINE | STRUCTURAL |
|---|---|---|---|
| STOP_LOSS | 178 | 18 | 10 |
| TAKE_PROFIT | 72 | 5 | 1 |

| 决策 | 成交 | 线 H* | SL | TP | R | 来源 | 出场 | 净盈亏 |
|---|---|---|---|---|---|---|---|---|
| 3 | 1831 | 无 | 1841 | 1811 | 10 | NO_LINE | 5 SL | −618 |
| 202 | 1812 | 1858 | 1822 | 1792 | 10 | CAPPED | 204 SL | −618 |
| 270 | 1788 | 1858 | 1798 | 1768 | 10 | CAPPED | 298 TP | 1,182 |
| 1795 | 1826 | 1830 | 1835 | 1808 | 9 | STRUCTURAL | 1861 SL | −558 |
| 22144 | 1923 | 1927 | 1932 | 1905 | 9 | STRUCTURAL | 22187 TP | 1,062 |

- 每笔交易持有的 SL / TP 都与 `planned_exit` 无状态重算的结果相同。
- 9 笔在成交那根 K 线内就出场了，它们没有经过 `env.entries`，只做了重算。

## 3. 止损线与教材

`planned_exit` 在全部 7,247 个教材正例上的结果：

| 正例 | 结果 |
|---|---|
| R ≤ 10：5,164 | 全是 STRUCTURAL。5,163 个与教材同一条线、同样的 SL / TP |
| R 11–12：2,083 | 2,077 个 CAPPED 在教材的线上（R 取 10）。另有 2 个 CAPPED、4 个 STRUCTURAL 用了另一条线 |

- 唯一不一致的在 FG405 cursor 21330。成交 1589，教材的线是 1594，env 取了 1593：触碰在 1594 上开始后，下方又确认了一条 1593。
- 这个正例记为 `STOP_LINE_DIFFERS`，不进教材。

## 4. Reward 与 Q

- **env reward：** 第 2 节两个账户都是从开始跑到结束。
  - 每次出场那一步的 reward = 净盈亏 / 618。
  - 出场以外没有非零 reward。
  - Σ reward × 618 = Σ 净盈亏 = 权益变化：A 为 36,294.00，B 为 −34,572.00。
- **语料：** facts 的 `reward`（L2 `outcome_quality`）在 NET_PNL 下就是净盈亏 / 618。建语料时 `native_check` 在每个合约 32 个 cursor 上逐分钟核对它。
- **Q（`net_pnl_h2760`）：** `q_return.relabel` 在每一轮、每个合约都做以下核对，任何一项不符就停：
  - 参考路径复现 CF 存的回报；
  - 每笔出场的 reward 是它的净盈亏 / 618，落在 exit_cursor − 1；
  - 各笔交易之和等于账户净盈亏；
  - 每个耦合的 pair，分支 PnL − 参考 PnL = 存下的权益差。
- **单元测试：** `test_the_net_pnl_return_is_in_the_reward_unit` 核对 G(t) 的单位：一次 −618 的 SL 是 −1，+1182 的 TP 是 1182 / 618。
- **手算 G(t)：** 待 1x 第 1 轮 CF，在 SMOKE_REPORT 里给一个 pair 的手算结果。

## 5. Observation 与因果性

gym `tests/entry_v1/test_ts_v2_minimal.py`：
- 48 个 15m bucket 中，较新的 24 个与 V1 的上下文相同，较早的 24 个是真实的完整 bucket。
- 因果性：把 t + 1 之后的价格改动 +40、成交量改为 ×3，t 时刻的所有 obs（包括 48 × 7）和 `planned_exit` 都不变。

dreamer `dreamerv3/tests/test_ts_v2_minimal.py`：
- V2 学生没有 `macd5` 参数，`context15/kernel` 有 336 行。
- 改动 `market_macd_5m`，编码器输出完全不变；改动较早的 24 个 bucket，输出会变。
- 24 个 bucket 的输入直接报错，不会被 reshape。
- V1 学生不变：MACD 起作用，168 行。
- F1 继承：`context15/kernel` 的旧行放到较新的 24 个 bucket，较早的为零行，fusion 原样复用。
- 1x 的 `foundation/inherit.json` 与此一致：`widened` 只有 `market_enc/context15/kernel`。

## 6. V1 回归

重启前做过。这次重跑没有跑完就停了：代码没变，正式训练已经开始。

| 检查 | 结果 |
|---|---|
| FG809 V1 语料：base 代码（gym `0e1ef3b`）和 V2 代码分别生成 | 121 个文件，只有 `source.json` 不同：多了 `inputs.strategy: null` 和 repos SHA，其余是耗时 |
| V1 冻结策略 `FORMAL_V1_GRADED_20261028/policies/pi_0009` 在 FG809 语料 env 上跑 400 分钟（两套代码） | 数组和逐行记录都相同 |
| V1 的 `config_hash`、`observation_identity`、`reward_hash` | 与 base 代码相同（`test_the_v1_rules_and_identities_are_unchanged`） |

## 7. 链路

- GPU 链路 smoke（L1 → L5、collect、CF、考试，small 预设）没有做：用户 2026-11-03 决定不做。
- 1x 从 2026-11-03 20:37 开始在 GPU 上跑同一条链，size50m：F1 继承、L1 → L3、pi_0000 与考试，然后 6 轮 collect / CF / q_return / L4 / L5。
- 各阶段自带的检查在 1x 里同样执行。

## 8. 测试与 lint

| 范围 | 结果 |
|---|---|
| gym（V2 + 已有 TS / env / e_data 测试） | 112 项通过，含建好语料后的语料教师测试 |
| dreamer（V2 + S0–S5、C4） | 69 项通过。全套跑了 68 项；MAE/MFE 单位测试把比较改成 float32 后单独重跑通过 |
| ruff `--select F,E9`：dreamer 改动的 13 个文件 | 全部通过 |
| ruff `--select F,E9`：gym 改动的 9 个文件 | 4 条，都在 `fg_single3_oracle.py`（3 个未使用的 import、1 个未使用的变量），base `0e1ef3b` 里原样存在，不是这次引入的 |
