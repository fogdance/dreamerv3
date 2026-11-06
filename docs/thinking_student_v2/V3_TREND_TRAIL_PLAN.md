# V3 长线趋势：入场不变，按 15m 波谷移动止损（新开发分支，改完直接 paper）

> 状态：用户 2026-11-06 批准的实施计划（TODO G-30）；用户要求"这个计划建议输出到文档"。分支 `feat/ts-v3-trend-trail-20261106`（两仓），worktree `/data/research/fg-s3-thinking-student-v3-trend-20261106`；证据在 `/data/logdir/fg_single3/thinking_student/V3_TREND_DEV_20261106/`。实施中的改动和结果在完成后补到文末。

## Context
用户 2026-11-06 提出一个新的长线策略，不是短线：
- **入场方式不变：** 沿用现在的规则闸门。15m 前高附近放量 ≥ 2 倍，报价在 [H−5, H] 内才能 ADD。
- **出场：** 没有固定止盈，靠 15m 下跌 N 字的波谷动态推进止损。下表以做空为例，做多是镜像，用波峰。

| 开仓后的状态 | 止损 |
|---|---|
| 0 个波谷 | 初始止损：paper 用触线 H + 5（R ≤ 10），训练时是成交价 + 5 |
| 价格离成交价 8 点 | 保本（成交价） |
| 1 个波谷 | 保本 |
| 2 个波谷 | 成交价 − 5（锁定 5 点盈利） |
| ≥ 3 个波谷 | 倒数第二个波谷的价格（跟着移动） |

- **用户已定：** 保本触发用 8 点；≥ 3 个波谷时用倒数第二个波谷；改完直接做 paper。
- **初始止损：** 训练沿用 V2.4，成交价 ±5；paper 用 H+5，和 V2.4 那次 paper 一样（用户 2026-11-06）。
- **其他要求：** 趋势识别代码从 `~/Documents/work/review_future/python` 拷过来；新建开发分支，不在 `integration/ts-next` 上改。

## 训练和 paper 分开配置（用户 2026-11-06："v2.4 是训练用成本价+5,paper使用h+5…我们要支持v2.4这种训练和paper有差异的方式"）
- **现状：** 这种方式靠 run.sh 给 exam 传 `--strategy` 来实现，配置里没有记录：
  - 训练用 formal 配置里的 `strategy`（V2.4：固定 5 点，TP 10）；
  - paper 用 run.sh 传进去的另一份（V2.4 + H+5）。
- **改为正式支持：**
  - formal 配置新增可选的 `paper.strategy`，不写时默认等于训练用的 strategy；
  - `exam` 的 paper 模式在没有 `--strategy` 时读取 run 配置里的 `paper.strategy`；
  - paper 报告和 review 导出同时写明训练策略和 paper 策略（version 和 sha256）。
- **V3 的组合：**
  - 训练：沿用 V2.4，固定 5 点、TP 10，直接用 V2.4 1x 已有的策略，不重训；
  - paper：`strategy_v3_trend_trail.yaml`。
  - paper 的初始止损用 H+5：触线线价 + 5，R ≤ 10，RULE / CAPPED / GAP，用户 2026-11-06 选定。
  - 保本的 8 点和锁定的 5 点都从成交价算。

## 分支
- 两仓都从 `integration/ts-next` 新建 `feat/ts-v3-trend-trail-20261106`（gym 86e551d，dreamer 6f7141b）。
- 新 worktree：`/data/research/fg-s3-thinking-student-v3-trend-20261106`。
- 证据和脚本放在 `/data/logdir/fg_single3/thinking_student/V3_TREND_DEV_20261106/`。

## 步骤

### 1. 拷贝趋势识别代码（gym）
- `market_structure/` 中的 `core.py`、`types.py`、`__init__.py`、`__main__.py`、`py.typed` 原样拷到 `src/gym_trading_env/market_structure/`。
- `tests/test_market_structure.py` 和 `fixtures/typescript-reference.json` 拷到 `tests/market_structure/`。
- 源目录在 review_future 仓里没有纳入 git，所以在 `__init__.py` 的注释里记下来源路径和各文件的 sha256，不改代码。
- 这些代码只用标准库，Python 3.11 可以直接用。

### 2. 波谷和波峰（因果，gym 新模块 `research/swing_trail.py`）
- 输入是 `env._e_market.complete_buckets` 里的完整 15m 桶（带 OHLC 和 `available_at`），转成小写的 OHLC。
- 对整个合约调用一次 `analyze_market_structure(left=2, right=2)`，默认参数（plateau last，ambiguous skip）。
- 再按 `confirm_index` 回放 `swing_log` 里的 added / replaced，得到每个桶收盘时已知的 H/L 拐点链。这一步避免对每个前缀都重算一遍。
- 测试保证回放结果和 `analyze(buckets[:b+1])['swings']` 在所有 b 上完全相同。如果不相同，就改为按前缀逐桶计算并缓存。
- **什么算开仓后的波谷：** 拐点所在的桶在成交那根 1m 所属的桶之后；只有在桶的 `available_at` ≤ 当前 1m 收盘时才算已知；按已知拐点链里的 L（做多时是 H）计数。
- 纯函数 `trail_stop(side, fill, stop, best, troughs, cfg)` 返回 `(新止损, stage)`。
  - 在空头空间计算，LONG 的价格取负。
  - stage 分 INITIAL / BREAKEVEN / LOCK / TROUGH。
  - 新止损只往有利方向移动，取 `min(当前, 候选)`。

### 3. env（`envs/entry_env.py`）
- **配置：** 新增可选的 `entry.exit_rule`：
  ```yaml
  kind: swing_trail
  timeframe: 15m
  pivot_left: 2
  pivot_right: 2
  breakeven_points: 8
  lock_points: 5
  ```
  - 只能和 rule_gate 一起使用。初始止损就是闸门原来的止损：有 `fixed_stop_points` 时是 FIXED，没有时是 H+5（RULE / CAPPED / GAP）。两种都支持，这次 paper 用 H+5。
  - 2026-11-06 补充（用户："V2.2 真实成本，使用V3的长线策略"；"不需要阀门"）：不再要求 rule_gate。不加闸门时初始止损是结构止损 H*+5 / L*−5（STRUCTURAL / CAPPED / NO_LINE，R ≤ 10），配置 `strategy_v2_2_structural_v3_trend.yaml`。
  - 不配 `exit_rule` 的 env 行为完全不变。
- **planned_exit：** 有 `exit_rule` 时 `take_profit_price = None`，plan 里记录 `exit_rule`。
- **出场判断：** `single3_exit_decision` 在 `tp_price is None` 时不再判断止盈。止损的成交规则不变：触价成交，跳空按开盘价，pending 照旧。
- **每根 1m 收盘后：** 在 `_native_step` 的出场循环之后，对仍持有的仓位：
  - 用这根 bar 的有利极值更新 best。做空用 Low，做多用 High；开仓那根只用 Close，和现有同 bar 规则一致。
  - 读取当前已知的波谷数，调用 `trail_stop`，把新值写进 `pos.stop_loss_price` 和 item 的 `stop_stage` / `best_price` / `stop_history`。新止损从下一分钟起生效。
  - 如果新止损已经在价格的另一侧（波谷确认时价格已经反弹回去），下一分钟按跳空规则以开盘价成交。
- **出场记录：** 原因仍记为 `STOP_LOSS`，这样奖励（net PnL）、exam 和 review 都不用改含义。`exit_metadata` 和 attribution 里加 `stop_stage` 和最终止损价。主力换月平仓（MAIN_ROLL）不变。
- **观测：** 持仓槽原来放"到 TP 的距离"，在 `exit_rule` 下改为"到当前止损的距离"（同一个缩放）。持仓时 ADD 本来就不合法（max_entries = 1），所以已训练策略在空仓时的决策不受影响。
- **缓存：** 拐点链和 `_gate` 一样在 `_load_market` 时清空，所以 `continue_data` 和 live 会重算。

### 4. 配置（dreamer `configs/thinking_student/markets/`）
- 新增 `strategy_v3_trend_trail.yaml`，version `V3_RULE_GATE_SWING_TRAIL`。
- 内容 = `strategy_v2_4_h5_stop.yaml`（V5 规则位置，规则文件是 `v2_positive_entry_v5(_long)_h5_stop.yaml`，SL = H+5，R ≤ 10；真实成本；net PnL 奖励；`reward_scale_cny` 618）+ 上面的 `exit_rule`。
- 旧文件都不动。

### 4b. 训练和 paper 的策略分开（dreamer）
- `formal_train` 的配置接受可选的 `paper.strategy`（相对配置文件的路径），并记入 run 配置。
- `exam --paper-pool` 没给 `--strategy` 时，从策略所在 run 的 `config.yaml` 读取 `paper.strategy`（`policies/pi_XXXX` 的上两级目录）；两个都没有时，沿用原来的行为。
- 报告头和 exam.json 记录 `train_strategy` 和 `paper_strategy`，各带 version 和 sha256。
- **已有 run 不改配置文件：** V2.4 1x 的配置不动；paper 的 run.sh 照旧显式传 `--strategy`。这一条用于以后的 run。

### 5. exam / paper 指标（dreamer）
- `exam.metrics` 增加：
  - 净盈亏 > 0 的交易占比（win_rate_net）；
  - 按 `stop_stage` 的出场计数；
  - 平均持仓市场分钟；
  - 每笔平均净点数。
- `paper.py` 的报告列出这些指标。原来的 TP / SL 列照旧输出（TP 会是 0）。
- 学生、规则内随机时机和每个规则位置都 ADD 三个账户都用同一个 env，不用改。

### 6. 测试（只跑相关的）
- **gym 新 `tests/entry_v1/test_swing_trail.py`：**
  1. 在一个真实合约上，回放拐点链 = 前缀 analyze 的结果（所有桶）。
  2. `trail_stop` 的各阶段（SHORT / LONG）：
     - 初始止损取自 plan（H+5 和 FIXED 各测一次）；8 点保本；1 个波谷保本；2 个波谷锁 5 点；3 个用第 2 个波谷，4 个用第 3 个；
     - 止损从不放宽。
  3. 真实合约上的 always-ADD 账户：
     - 没有 TAKE_PROFIT；
     - 止损单调，`stop_stage` 和止损价一致；
     - 出场价等于止损价，或者是跳空开盘价；
     - 用到的波谷 `available_at` 都不晚于止损更新的那一分钟（因果）。
- **另外要跑的：**
  - 拷过来的 `tests/market_structure`；
  - 现有的 `test_rule_gate.py`（env 有改动）；
  - dreamer GPU `test_ts_v2_3.py`（exam）。
- **lint：** ruff F/E9。

### 7. 提交和 paper
- 两仓都在 `feat/ts-v3-trend-trail-20261106` 上提交。DESIGN.md 加一节 V3 长线趋势；TODO 加 G-30。
- paper 在后台跑，目录 `PAPER_V3_TREND_V2_4_POLICIES_20261106`。
  - 策略：V2.4 1x 的 pi_0005（best_dev）和 pi_0009（最后一轮），不重训。
  - 合约：FG501–FG605。
  - 用 `strategy_v3_trend_trail.yaml`。
  - run.sh 照 `PAPER_V2_4_1X_H5STOP_20261105/run.sh` 写。
  - review 登记为 FORMAL_V2_4_1X 的 r0003；如果 TP 是 None 导致导出出错，就修导出。
- **报告：**
  - 对比表：V3 长线、V2.4 固定 5 点（−12,168 / −11,041）、V2.4 + H+5（+6,558 / +8,352）。
  - 列出各自的随机基准、每个规则位置都 ADD、胜率、各阶段出场分布、平均持仓时间、回撤、盈利月份。

## 不在这一步
- 不重训，也不改材料和教师。没有固定止盈以后，正例怎么定义要另外讨论。
- 不跑 SA / RB，也不跑 live。
- PAPER 不用于调参：8 点、5 点这些参数都由用户定。

## 实施记录（2026-11-06）
- **趋势识别代码：** 从 review_future 拷过来，原样放在 gym `src/gym_trading_env/market_structure/`，测试在 `tests/market_structure/`；源文件的 sha256 记在 `__init__.py`。拷过来的 5 个测试全过。
- **回放验证：** 在 FG205 的 1548 个完整 15m 桶上，按 `confirm_index` 回放得到的拐点链和逐个前缀重新分析的结果完全一致（0 处不同），所以采用回放。
- **TP 的处理（与上面的计划有一处不同）：**
  - plan 的 `take_profit_price` 设为 None，不会发生止盈出场；
  - 规则原来的 TP（target_r · R）另存为 `reference_tp_price`，只给持仓观测用，也就是 env 观测的持仓槽和 StudentEnv 账户特征的第 3 维；
  - 原因：已训练策略的这两个特征是按 TP 距离学的，换成到止损的距离会改变特征的含义。
- **smoke（TRAIN，CPU，每个规则位置都 ADD）：**
  - FG205：151 笔，−13,603，平均持仓 56 个市场分钟；
  - FG401：146 笔，+15,500，平均持仓 94 个市场分钟；
  - 两个合约都没有 TAKE_PROFIT；
  - 各阶段出场数：

    | 合约 | INITIAL | BREAKEVEN | LOCK | TROUGH |
    |---|---|---|---|---|
    | FG205 | 84 | 51 | 6 | 9 |
    | FG401 | 74 | 49 | 5 | 18 |
- **paper（`PAPER_V3_TREND_V2_4_POLICIES_20261106`）：**

  | | pi_0005（best_dev） | pi_0009（最后一轮） |
  |---|---|---|
  | 净盈亏 | +24,768 | +24,629 |
  | 随机（均值 ± 标准差，赢的种子数） | +12,121 ± 6,667（5/5） | +11,318 ± 11,346（4/5） |
  | 每个规则位置都 ADD | +23,507 | +23,507 |
  | 最大回撤 | 16,604 | 19,279 |
  | 盈利月份 | 10/17 | 10/17 |
  | 净胜率 | 16.4% | 16.3% |
  | 出场（INITIAL / BREAKEVEN / LOCK / TROUGH） | 230 / 216 / 22 / 63 | 223 / 217 / 22 / 61 |
  | 平均持仓（市场分钟） | 125 | 124 |

  review 已登记 r0003，validate 通过。
- **训练出场的配置：** 用户 2026-11-06 先提出"教材和老师训练的时候还是按照短线的训练或者长线来训练，这个要做成配置"，随后取消（"这个不做"）。材料和老师照旧按短线（TP 2R）训练，长线只用在 paper（`paths.paper_strategy`）。
