# V2.1：SHORT + LONG 改动清单（CHANGE_MAP）

依据：V2 最小改造 `../minimal/CHANGE_MAP.md`（V0.5），以及用户 2026-11-04 的决定。

| 项目 | 内容 |
|---|---|
| 起因 | V2 1x（`FORMAL_V2_MINIMAL_1X_20261103`）只训练了 SHORT。用户 2026-11-04：“这次只做 SHORT 是不合适的”“那就是 spec 描述错误”。V0.5 写的 `SHORT_ONLY` 作废 |
| 目标 | 重新做 V2.1 1x，SHORT 和 LONG 都包括，用新的 worktree 训练 |
| 用户决定（2026-11-04） | ① LONG 教材是 V2_1 规则的镜像，用户先看统计和 K 线图，通过后再建语料、训练；② 与 V1 1x 完全一致：18 个 TRAIN（9 SHORT + 9 LONG），DEV 只用 FG409，9 轮，L6 从第 7 轮开始。原因：“这样我们和早期的 1x 就可以做对比” |
| 规则 | 只遵守工作区根目录的 `AGENTS.md`：只提交不推送；GPU 训练前先问；TEST 保持封存；PAPER 和实盘不进训练，也不用来选策略或调参 |
| 分支 | `feat/ts-v2-1-20261104`（两个仓库），从 V2 分支 HEAD 拉出：gym `dcf89b7`，dreamer `d9bc7dd` |
| Worktree | 代码 `/data/research/fg-s3-thinking-student-v2-20261031`；训练另开新 worktree（第 6 节） |
| 不动的东西 | V2_1 SHORT 教材 `V2_1_POSITIVE_ENTRY_20261031` 不重新生成；V1 / V2 的配置、语料、run 都不改 |

**简写：** **G** = `gym-trading-env/src/gym_trading_env`，**D** = `dreamerv3/dreamerv3/thinking_student`，**C** = `dreamerv3/configs/thinking_student`。

## 1. LONG 的定义：空头空间里的镜像

把 LONG 市场的每个价格取负，就得到一个“空头空间”的市场，在它上面原样运行 SHORT 的扫描、pivot、离开、触碰逻辑，最后把价格换回真实值。

| 量 | 空头空间 |
|---|---|
| K 线 | high′ = −Low，low′ = −High，open′ = −Open，close′ = −Close |
| 报价 | sell_close′ = −（LONG 开仓买价，收盘时），sell_open′ = −（开仓买价，开盘时），buy_high′ = −（LONG 平仓卖价，在 Low） |
| 15m bucket | High′ = −Low |
| 取整 | floor(−x) = −ceil(x)，镜像后的报价与真实报价一一对应 |

换回真实价格后，LONG 的规则：

| 规则 | SHORT | LONG |
|---|---|---|
| 线 | 前 15m pivot 高点 H | 前 15m pivot 低点 L |
| 入场区 | [H − 7, H] | [L, L + 7] |
| 止损 | H + 5 | L − 5 |
| 止盈 | 成交 − 2R | 成交 + 2R |
| 离开 | 1m 收盘 ≤ H − 10 | 1m 收盘 ≥ L + 10 |
| R | SL − 成交 | 成交 − SL |

## 2. 改动

| 位置 | 改动 |
|---|---|
| `G/research/ts_v2_positive.py` | `SIDES`、`sign(side)`、`side_buckets`；`short_rows` 改为 `side_rows(pool, side, contracts)`；`load_config` 接受 SHORT 或 LONG；`market_arrays` 按方向给出空头空间的数组（附 `side`、`sign`）；`prior_highs` 用 `side_buckets` |
| 同上：记录与文字 | `_real`：把 `H`、`obs_*` 价格、`label_fill_expected` 换回真实价格；`_words`、`explain`、`wait_note`、`grade_reason` 按方向写 H / L、区间和止损。SHORT 的文字逐字节不变 |
| 同上：`produce`、`_trade_job`、`graded_teacher` | `label_tp = 成交 − s × 2R`；验证、`_room`、`planned_exit` 都带方向；source.json 记 `direction` |
| 同上：`combine`（新） | 把几份教材合成一份：每张表 = 表头 + 各份的数据行，逐字节、按给定顺序。要求版本相同、除 direction 外规则相同、合约不重叠、表头相同；source.json 记每份的目录、方向、合约和 sha256。CLI：`--combine A B --out X` |
| `G/research/ts_v2_positive_plot.py` | 按方向画线、区间、SL、▲/▼ 和 TP；标题写方向 |
| `G/envs/entry_env.py` | `STOP_RULES = (prior_15m_high, prior_15m_extreme)`。`_stop_lines(side)` 按方向缓存。`planned_exit` 在空头空间计算，s = ±1：SL = 成交 + s·R，TP = 成交 − s·2R。`prior_15m_high` 仍只允许 SHORT |
| 同上：`settle_exit` | TP 距离取绝对值（修正：LONG 时原来为负） |
| `G/research/ts_v1_corpus.py` | `load_strategy` 接受 `direction: BOTH`，且 BOTH 必须配 `prior_15m_extreme`；`write_inputs` 在 BOTH 下不检查方向；`produce` 把合约方向传给 `graded_teacher` |
| `D/run.py:check_collection` | TP / SL 距离取绝对值（修正：LONG 时原来为负，违规永远不会被标出） |
| `D/paper.py` | 有策略文件时，报告写策略名，不再写 V1 缺省的 “5 / 30”（V2 1x paper 报告的标签错误） |
| `C/v2_positive_entry_long.yaml`（新） | 与 `v2_positive_entry.yaml` 相同，只有 `direction: LONG` 不同 |
| `C/markets/strategy_v2_1_structural.yaml`（新） | `direction: BOTH`，`stop_rule: prior_15m_extreme`，其余与 V2 SHORT 策略相同（R ≤ 10，2R，48 个 15m bucket，NET_PNL / 618） |
| `C/formal_v2_1.yaml`（新） | `formal_v2_minimal.yaml` 改为：语料 `corpus_v2_1_20261104`，run `FORMAL_V2_1_1X_20261104`，策略 V2.1，18 个合约，`max_rounds: 9`，`l6.from_round: 7`。与 `formal_v1_graded.yaml` 的区别只有 V2 的键和 `save_every: 0` |

Dreamer 的训练代码本来就支持两个方向（V1 有 LONG），这次除 `check_collection` 外没有方向相关的改动。`labels.outcome_points` 对任何 `stop_rule` 都取 `max_risk_points`，所以 V2.1 语料的 MAE / MFE 单位也是 10 点。

## 3. 找到的 SHORT-only 假设和缺陷

| 位置 | 问题 | 处理 |
|---|---|---|
| `entry_env.settle_exit` | TP 距离 = 成交 − TP，LONG 时为负 | 取绝对值 |
| `run.check_collection` | TP / SL 距离同上，LONG 的违规不会被标出 | 取绝对值；新测试 |
| `ts_v2_positive_plot` | 坐标轴循环变量 `side` 覆盖了方向，标题写成 “direction bottom” | 循环变量改名 `spine` |
| `paper.markdown` | V2 策略的 paper 报告写成 “5 / 30” | 写策略名；V2 1x 报告已用同样的数字重新生成（第 7 节） |

## 4. 核对（`/data/logdir/fg_single3/thinking_student/V2_1_DEV_20261104`）

| 检查 | 结果 |
|---|---|
| SHORT FG205 用新代码重新生成，对比 V2_1 教材中的 FG205 行 | `positive_events`（74 行）、`positive_entries`（310 行）逐字节相同。`positive_timeline`（466 行）数值完全相同，列顺序不同：pandas 按第一行定列序，9 个合约一起跑时第一行是 WAIT，只跑 FG205 时第一行是 ACTIVATION_ADD。旧代码同样如此，不是这次的改动 |
| LONG FG401 教材（90 个事件，692 个正例），`checks/long_sanity.py` | 692/692 全部通过：SL = L − 5 < 成交 < TP = 成交 + 2R；R = 成交 − SL；TAKE_PROFIT 出场且出场价 ≥ TP；净盈亏 = 60 ×（出场 − 成交）− 18；报价在 [L, L + 7]；Native 验证通过 |
| `env.planned_exit(…, 'LONG')`，V2.1 env，FG401 全部正例 | R ≤ 10 的 445 个：STRUCTURAL，SL / TP 与教材相同；R 11–12 的 247 个：在同一条线上 CAPPED 到 10。`prior_15m_high` 的 env 拒绝 LONG |
| K 线图 `long_FG401/figures/01_FG401_20231024_1054.png` | L 1608，区间 [1608, 1615] 在线上方，SL 1603 在线下方，▲ 成交，TP 在上方 |

列顺序对合并的影响：`combine` 要求各份表头相同，不同就报错，不会悄悄拼错。

**9 个 LONG 合约的教材**（`V2_1_POSITIVE_ENTRY_LONG_20261104`，gym `d68d6a4` / dreamer `eb2d47c`，代码干净，400 秒）：

| 方向 | 合约 | 交易日 | 有正例的事件 | 正例 | EXCELLENT | R ≤ 10（保留） | 每 100 交易日保留 |
|---|---|---|---|---|---|---|---|
| SHORT（V2_1 原样） | 9 | 736 | 471 | 7,247 | 2,377 | 5,164 | 702 |
| LONG | 9 | 719 | 589 | 7,571 | 3,133 | 5,040 | 701 |

- 7,571 个 LONG 正例的算术检查全部通过（`checks/long_all/`）。
- `planned_exit` LONG 与教材的线一致 7,418 个；不一致 153 个（R ≤ 10 的 93 个、R 11–12 的 60 个），都在 4 次触碰里（FG909@15061 占 149 个）。原因都是触碰开始后 7–24 分钟才确认的一条更近的低点线：教材整个触碰用开始时的线，env 是无状态的，取最近的活线。与 SHORT FG405 的 1 个相同，建语料时记为 `STOP_LINE_DIFFERS` 丢弃（`checks/line_diff_*.txt`、`line_diff_timing.txt`）。预计 LONG 保留 4,947 个 ADD 示范，SHORT 5,163 个。
- 三张表的表头与 SHORT 教材相同；在 scratch 里试合并成功。

## 5. 测试

| 文件 | 新增 / 修改 |
|---|---|
| gym `tests/entry_v1/test_ts_v2_positive.py` | `side_rows` 两个方向；配置只接受 SHORT / LONG；FG401 LONG 市场是镜像（K 线、报价、15m pivot 低点、换回真实价格后的线 / SL / 区间 / R / 计划 TP）；`combine` 逐字节、拒绝重复合约、规则不同、表头不同 |
| gym `tests/entry_v1/test_ts_v2_minimal.py` | LONG 止损 = 前 15m 低点 − 5 或封顶（STRUCTURAL / CAPPED / NO_LINE 都出现）；LONG 交易按计划出场、净盈亏、reward = 净盈亏 / 618；V2.1 策略对 SHORT 的止损与 V0.5 相同；BOTH 必须配 `prior_15m_extreme` |
| dreamer `dreamerv3/tests/test_ts_v2_minimal.py` | paper 报告写策略名；`check_collection` 对 LONG 的 TP 不足、SL 越过都标出违规 |

结果：gym 116 项通过（V2 + 已有 TS / env / e_data），dreamer 18 项通过（V2、c4 paper、s2 `check_collection`）；ruff `--select F,E9` 全部通过。日志：`V2_1_DEV_20261104/tests/`。

## 6. 后续步骤（用户先审 LONG 教材）

1. 9 个 LONG TRAIN 合约的教材 → `V2_1_POSITIVE_ENTRY_LONG_20261104`，统计和 K 线图交给用户审。**在这里停下，等用户通过。**
2. 合并：`ts_v2_positive --combine V2_1_POSITIVE_ENTRY_20261031 V2_1_POSITIVE_ENTRY_LONG_20261104 --out V2_1_POSITIVE_ENTRY_BOTH_20261104`。
3. 18 个语料 → `corpus_v2_1_20261104`（命令在 `formal_v2_1.yaml` 头部），然后跑语料教师测试。
4. 新的训练 worktree 和 run 脚本 `FORMAL_V2_1_1X_20261104`（`--rounds 9`），命令交给用户在 tmux d3 启动。

## 7. V2 1x paper trading（2026-11-04，FG501–FG605，全部 SHORT）

`/data/logdir/fg_single3/thinking_student/PAPER_V2_MINIMAL_1X_20261104`。报告的出场标签已改正（原来误写 “5 / 30”，数字不变，原文件保存在 `logs/`）；逐笔统计见 `checks/trade_stats.txt`。

## 8. 提交

| 仓库 | 提交 |
|---|---|
| gym | `d68d6a4` |
| dreamer | 本提交（`git log -1 -- docs/thinking_student_v2/v2_1/CHANGE_MAP.md`） |
