# V2 正向教材：设计

日期：2026-10-31
依据：Spec V0.3，以及用户同日对 V1 教材的 Review（FG205 截图：沿横线的多次接近都是入场点）。
参数集中在 `dreamerv3/configs/thinking_student/v2_positive_entry.yaml`（版本 `V2_POSITIVE_ENTRY_V2`）、`v2_positive_entry_v3.yaml`（V3，V2.2 用）和 `v2_positive_entry_v4.yaml`（V4，V2.3 用）；LONG 是各自的 `_long` 镜像。
**以下规则在生成之前就已冻结。** 之后任何改动都算新版本教材，并在 REPORT 中说明；不根据结果回头调参。

## 零、版本记录

### V1（2026-10-31 上午）
- 输出：`/data/logdir/fg_single3/thinking_student/V2_POSITIVE_ENTRY_20261031/`
- 代码：gym c402f86。
- 规则：每条 level 只产生一个事件；事件在远离或 60 分钟后结束；每个事件都需要触及 H 的那根 bar 本身 RV ≥ 2；只取最近的 level；区间 [H−5, H]，R ≤ 10；回看 46 个 bucket。

### V2（2026-10-31，用户 Review 之后）
- 输出：`/data/logdir/fg_single3/thinking_student/V2_1_POSITIVE_ENTRY_20261031/`

**用户指出 V1 遗漏的点**（FG205）：
- 2021-12-14 13:46–13:58：H=1850 的第二次接近。
- 2021-12-28 10:06–10:10：H=1778 第一次接近，没有放量。
- 2022-01-04 10:33–11:17：H=1722 的第三次接近。
- 2021-12-29 21:01：开盘第一分钟冲到 1774。

**改动及原因：**

| # | V1 | V2 | 原因 |
|---|---|---|---|
| 1 | 一条 level 只用一次（consumed） | 一条线可以被多次触及；每次远离之后，再接近就是新的 touch（事件） | 用户："后续再接近的时候，这些也是可选的入场点"（12-14、01-04） |
| 2 | 事件在激活后 60 分钟结束 | 取消窗口；touch 在远离或破位时结束 | 线没有被突破就一直有效 |
| 3 | 每个事件都要在触及 bar 上 RV ≥ 2 | 放量属于这条线：从线自己的 15m bucket 起，第一根 RV ≥ 2 且 High ≥ H−5 的 bar 标记这条线；之后的 touch 不需要再放量 | 12-28 那次接近没有放量（RV 1.2–1.8），但 1778 这个高点形成时有放量（12-27 15:00，RV 2.3） |
| 4 | 取最近的 level | 取已武装的 level 中 H 最低的那条（价格上方的第一条线） | 线长期有效之后，最先碰到的是最低那条阻力 |
| 5 | 区间 [H−5, H]，R ≤ 10 | 区间 [H−7, H]，R ≤ 12 | 接近时 High 常停在 H−4..H−5，收盘后的市价报价和下一根开盘成交通常在 H−6..H−7（12-14 13:46–13:50，12-28 全部） |
| 6 | 回看 46 个 bucket（约 2 天） | 115 个 bucket（约一周），破位提前结束 | 用户画的线跨越两天以上 |

**仍然不能纳入：** 2021-12-29 21:01 那个点。
- 这是夜盘开盘第一分钟：开 1759，冲高 1774，收 1757。
- 在收盘时决策、下一根开盘市价成交（1757），卖不到线附近。
- 只有提前挂在线上的限价单才能成交，而 Native 不模拟限价单。

### V3（2026-11-04，V2.2）
- 用户："之前修改了一点逻辑，去掉了放量的条件……恢复原有的教材高位放量逻辑"。
- 配置：`v2_positive_entry_v3.yaml`（SHORT）、`v2_positive_entry_v3_long.yaml`（LONG），`volume.activation: touch`。
- 只改 V2 的第 3 条，其余 5 条不变：

| # | V2 | V3 |
|---|---|---|
| 3 | 放量属于这条线，之后的 touch 不需要再放量（97% 的线都有放量，实际上不筛选） | 回到 Spec V0.3 / V1：**每次 touch 都要由它自己的一根 bar 开始**，这根 bar 同时满足 High ≥ H − 5 和 RV ≥ 2；之前接近 H 的 bar 都是 `WAIT_NEAR_NO_VOLUME`；touch 开始之后的候选不要求 RV ≥ 2 |

- 线自己的放量时刻（`line_spike`）仍然记录，但不再决定 touch 何时开始。
- 预计代价：12-28 10:06–10:10 那次接近（RV 1.2–1.8）不再满足放量条件，V2 时用户点出的这个点会被排除（生成 V3 教材后核对）。
- LONG 镜像同理：接近 L（Low ≤ L + 5）且 RV ≥ 2 的那根 bar 开始 touch（低位放量）。

### V4（2026-11-05，V2.3：规则闸门 + 真实成本）
> V4 的配置、材料和语料已按用户要求删除（2026-11-05），止损由 V5 取代；env 的闸门现在必须带 `fixed_stop_points`。规则位置、WAIT_IN_RULE 和成本照旧，下面的止损（H+5，RULE / CAPPED / GAP）只作记录。
- **用户：**
  - "外部已经判断方向。下跌趋势只有价格反弹至15分钟前高附近，并且出现2倍以上放量，学生才允许做空，上涨趋势同理。放量后的多个有效位置也可以入场。"
  - "规则决定学生能在哪里交易；大脑决定在这些位置中什么时候交易……要求学生尽量按照这个规则来入场。如果没有按照这个规则，就要找出根本原因，并修复"
  - "真实的交易成本，无滑点，1手玻璃1个点盈利20元，开仓成本2.1元，平仓成本2.1元。"
- 计划：`V2_3_RULE_GATE_PLAN.md`。根因证据：`/data/logdir/fg_single3/thinking_student/V2_3_DEV_20261105/root_cause/`。V2.2 在 TRAIN 上 86.5% 的 ADD 不在规则位置，因为 env 没有闸门。
- 配置：`v2_positive_entry_v4.yaml`（SHORT）、`v2_positive_entry_v4_long.yaml`（LONG）。touch 的规则与 V3 相同，只改四处：

| # | V3 | V4 | 原因 |
|---|---|---|---|
| 1 | 区间 [H−7, H] | [H−5, H] | 闸门只放行计划 R ≤ 10 的规则位置，这样候选正好就是 env 允许 ADD 的分钟 |
| 2 | R ≤ 12 | R ≤ 10 | 与 env 的 `max_risk_points` 一致；V2.2 的教师因此丢掉了 3,831 个正例（RISK_ABOVE_MAX） |
| 3 | 第一个正例之前的规则位置不标 | `timeline.wait_in_rule: true`：有正例的 touch 中，第一个正例之前的每个候选分钟标 `WAIT_IN_RULE` | 用户选"正例之前的位置标 WAIT"；V2.2 的 WAIT 全部在 touch 之前，规则位置上一条都没有 |
| 4 | 成本取 pool env 的 master 策略 | `costs: markets/costs_fg_real_20261105.yaml`（CZCE_FG 成本预设） | 真实成本：开仓、平仓各 2.1 元/手，spread 0，无滑点。决策报价就是 Close，成交价就是下一根 Open。止盈仍要越过 1 跳（用户："保留穿价 1 跳"） |

- **规则位置就是 env 的闸门。**
  - 这一条取代 V0.5 方案 §1 和 §7 的"不把前高/2倍放量新增成 Native 硬性合法性条件""不把规则硬塞入 ADD action mask"，依据是用户 2026-11-05 的新要求（"学生才允许做空""规则决定学生能在哪里交易"）和批准的计划。学生仍然自己决定在规则位置中何时 ADD；观测不加规则状态，这一点仍按 V0.5。
  - 策略 `markets/strategy_v2_3_rule_gate.yaml` 的 `entry.rule_gate` 指向这两份规则。
  - env 用材料自己的单次因果扫描（`ts_v2_positive.rule_pass`，经 `rule_cursors`）算出每一分钟是不是规则位置；规则位置之外 ADD 不合法。
  - 边界：env 传入市场的最后一分钟，所以 live 时最新一分钟也可以是规则位置；材料传入的仍是 TRAIN 的 flat_end，结果与 V3 的扫描相同（V3 配置重新生成 FG205 / FG401，逐字节一致）。
- **止损用本次 touch 自己的线。**
  - SL = H + 5（LONG：L − 5），R = H + 5 − fill。
  - R > 10 记 `CAPPED`，R 取 10；成交价已经越过 SL（只会在跳空时出现）记 `GAP`，R 取 10。
  - 开仓 fill 记录 `rule_event`（touch 编号）和 `touch`（这条线的第几次 touch）。
- **教师（`graded_teacher`）在闸门下：**
  - ADD = 正例，必须合法；SL 与材料的 `obs_sl` 不同直接报错。
  - WAIT 只取 `WAIT_IN_RULE` 行。
  - touch 之前的行（WAIT_FAR、WAIT_APPROACH、WAIT_NEAR_NO_VOLUME、ACTIVATION）在闸门下本来就不能 ADD，不做标签，计入 `dropped.WAIT_OFF_RULE`。
- **exam：** 每个账户报告合规率（`rule_compliance`，闸门下小于 1 就报错）、成交价在区间内的比例和 stop_source 分布；基准有两个：随机账户（规则内随机时机）和 always-ADD 账户（每个规则位置都 ADD）。


### V5（2026-11-05，V2.4：止损固定 5 点）
- **用户：** "止损固定5点"；范围选"全部：SL = 成交价+5"。
- 配置：`v2_positive_entry_v5.yaml` / `v2_positive_entry_v5_long.yaml`，在 V4 基础上只加 `entry.fixed_stop_points: 5`；策略 `markets/strategy_v2_4_fixed_stop.yaml`（version `V2_4_RULE_GATE_FIXED5`）。
- **止损：** 每个闸门 ADD 都是 SHORT SL = 成交价 + 5、TP = 成交价 − 10，LONG SL = 成交价 − 5、TP = 成交价 + 10（R = 5，2R），跳空成交也一样。`stop_source` 记 `FIXED`，不再有 RULE / CAPPED / GAP。
- **线 H 的作用不变：** 规则位置仍由决策报价在 [H−5, H] 决定，`stop_above_points` 只剩"线被突破，touch 结束"的作用。所以 V5 的规则位置与 V4 逐分钟相同（`test_rule_gate.py` 检查）。
- **材料：** 计划 R 与 `label_R` 都是 5；`obs_sl` = 决策报价 ± 5（计划值），`label_sl` = 实际成交价 ± 5；成交跳出 [H−5, H] 的候选也照常交易，不再记 FILL_OUT_OF_ZONE（`label_fill_in_zone` 仍保留在审计里）。
- **教师：** 闸门 SL 与 `label_sl` 比对，不一致直接报错。


### V6（2026-11-05，V2.5：闸门 + 触线止损 H+5）
- **用户：** "V2.4 的设置下，把止损换回 H+5，从材料、语料开始重新训练一次 1x"。依据：V2.4 的策略只把止损换成 H+5 来跑 paper，结果 +6,558（固定 5 点是 −12,168），见 `PAPER_V2_4_1X_H5STOP_20261105`。
- 配置：`v2_positive_entry_v6(_long).yaml` 就是 V5 去掉 `fixed_stop_points`；策略 `markets/strategy_v2_5_h5_stop.yaml`（version `V2_5_RULE_GATE_H5`）。
- **止损：** 闸门 ADD 的止损用本次触线的线：SL = H + 5（LONG：L − 5），R = SL − fill，R ≤ 10，TP 2R。R > 10 记 `CAPPED`（R 取 10）；成交价越过 SL 记 `GAP`（R 取 10）；其余记 `RULE`。这就是 V4 的止损，env 里对应的代码已经恢复。
- 规则位置、`WAIT_IN_RULE`、真实成本都与 V5 相同。

## 一、范围

- 只用 9 个方向为 SHORT 的 TRAIN 合约：FG809、FG901、FG905、FG005、FG201、FG205、FG209、FG301、FG405。
- 方向来自用户配置的区间，每行都标为 `direction_source = DIRECTION_ASSUMED`。
- 不用 LONG 合约，也不做镜像。
- 不用 DEV、PAPER 和 TEST。

## 二、线（因果前高 H）

1. 只用完整的 15 分钟 bucket，即 `MarketData.complete_buckets`，按结束时间排序。
2. **Pivot 定义：** 第 j 个 bucket 是 pivot high，当且仅当：
   - 它的 High 严格高于前 2 个完整 bucket 的 High；
   - 并且不低于后 2 个完整 bucket 的 High。
3. **可见时刻：** 第 j+2 个 bucket 的 `available_at`，即 bucket 结束时间。从第一个 bar 结束时间 ≥ 该时刻的游标起，这条线才可见。
4. **线的状态**（以 1m bar 计）：
   - `broken`：pivot bucket 结束之后，某根 bar 的 High + 点差 ≥ H + 5。这时 Native 的空单止损已经会被触发，线失效。
   - `expired`：当前已可见的完整 bucket 数，减去 j，超过 115（约一周）。
   - `spike`（放量标记）：从 pivot bucket 自己的第一根 bar 起，第一根同时满足 RV ≥ 2、High ≥ H − 5、尚未破位的 bar。这个时刻之后，这条线就是"放过量的线"。
   - `armed`（可以开始 touch）：从 pivot bucket 结束后第一次 Close ≤ H − 10 起；每次 touch 因远离结束后，从下一根 bar 起再次 armed。
5. **Active 线：** 在游标 s，取已可见、未破位、未过期、已 armed 的线中 H 最低的一条（H 相同时取较新的一条）。
6. **不做的事：**
   - 不做箱体识别。
   - 不用"事后最高点"反推 H。
   - 不用未来价格决定 H 在何时出现。

## 三、量比 RV

- RV_s = Volume_s / 前 20 根有效 1m bar 的平均 Volume。
- 每个交易时段（21:00、09:00、10:30、13:30）的第一分钟含集合竞价量，所以既不能作为放量标记，也不计入基准。
- 前面不足 20 根基准 bar 时，RV 记为未知。

## 四、Touch（事件）生命周期

每次只处理一个 touch，按游标顺序扫描，只用游标 s 及以前的数据。

1. **开始：** 没有进行中的 touch，active 线 H 在 bar t 同时满足：
   - High_t ≥ H − 5；
   - High_t + 点差 < H + 5（没有破位）；
   - 放量：V2 是这条线的 spike ≤ t（spike 可以就是 bar t 本身，这时 t 的 RV ≥ 2）；V3 是 bar t 自己 RV_t ≥ 2。
   之后这条线在本 touch 结束前不再 armed，touch 编号加 1。
   如果 bar t 接近 H 但没有满足放量条件，就继续等待（`WAIT_NEAR_NO_VOLUME`）：V2 是这条线还没放过量且 RV_t < 2；V3 是 RV_t < 2。
2. **结束：** touch 开始之后，第一根满足下列任一条件的 bar s：
   - **失效：** High_s + 点差 ≥ H + 5。线破位，不再使用。
   - **远离：** Close_s ≤ H − 10。线从 s+1 起再次 armed，可以有下一次 touch。
   - **过期：** 线超过回看期。
   - **边界：** 过了 Native 的 ADD 截止时间或 TRAIN 边界。
   结束的那根 bar 不再产生候选。

## 五、候选

- touch 内每个决策游标 s（包括 t 本身）都检查：决策时的卖出报价 `quote_s = Close_s − 点差`（向下取整）是否在 [H − 7, H] 内。
  - 候选**不要求** RV ≥ 2。
- 候选可执行的条件：
  - Native 空仓账户的 ADD Mask 打开；
  - 在 bar s+1 实际成交，成交价 `fill = Open_{s+1} − 点差`，并且 fill 也在 [H − 7, H] 内。如果因跳空落到区外，记为 `FILL_OUT_OF_ZONE`。
- **风险和目标：**
  - R = H + 5 − fill，取值 5 ≤ R ≤ 12。
  - SL = H + 5，与这条线的 H 相同。
  - 主目标 TP = fill − 2R。
  - 并行记录 TP = fill − ceil(1.5R)，只用于分析，不用于筛选。
- 同一 touch 的多个候选，以及同一条线不同 touch 的候选，都是**可以互相替代的入场方案**，各自从空仓账户单独计算。任何输出都不把它们的收益相加。

## 六、交易结果（离线标签）

- 每个 R 值构造一个真实 Native env：`stop_loss_points = R`、`take_profit_points = 2R`；1.5R 另外构造一组。
  - 每个合约最多 16 个 env（R = 5..12）。
- 先用 `CandidateIndex.candidate(s)` 给出交易。
  - 正例要再用 `reference_candidate` 逐分钟复核，两者的成交、出场和出场游标必须一致。
  - 测得的 fill 必须等于第五节预先算出的 fill。
- 记录的字段：
  - 成交时间和价格、出场原因、价格和游标、持仓市场分钟数。
  - Native 净盈亏 `net_pnl`（含手续费）。
  - `net_R` = net_pnl / (R × 3 手 × 20)。
  - MAE = 持仓期间最高 High − fill（点）。
  - MFE = 可执行的最大有利波动（`ts_v1_grades._room`）。

## 七、正例（老师示范）

- **必须满足：** 在 2R 规则下 Native 实际成交到 TP。即没有止损、没有其他出场、没有被数据截断；TP 与 SL 同 bar 时 Native 按 SL 处理，因此也不算正例。
- **分级：**

| 等级 | MAE / R | 到 TP 的市场分钟 |
|---|---|---|
| EXCELLENT | ≤ 0.5 | ≤ 120 |
| GOOD | ≤ 0.8 | ≤ 345（一个交易日） |

- **连续质量分：** `quality` = ½ (1 − MAE/R) + ½ (1 − min(持仓分钟, 345) / 345)。
- **不入库，只进审计：** 其他候选按原因计数，原因包括：
  - `MASKED`、`NOT_FILLED`、`FILL_OUT_OF_ZONE`
  - `STOP_LOSS`、`OTHER_EXIT`、`RIGHT_CENSORED`
  - `TP_SLOW`、`TP_DEEP_MAE`
- 不生成 BAD/POOR 教材，也不为凑数降低标准。

## 八、WAIT 教学过程（positive_timeline）

只为至少有一个正例的 touch 生成：
1. 起点是 touch 开始前最后一个 Close ≤ H − 10 的游标，并且不早于这条线连续作为 active 线的那段游标。对于第 2 次及以后的 touch，就是上一次 touch 远离之后。
2. touch 开始前各阶段：
   - `WAIT_FAR`：远离 H。
   - `WAIT_APPROACH`：正在反弹，但 High < H − 5。
   - `WAIT_NEAR_NO_VOLUME`：已接近 H，但这条线还没有放过量，且 RV < 2。
   - 每个阶段只保留首行、末行，以及不超过 8 行均匀抽样的行。
3. `ACTIVATION`：touch 开始的游标。如果它本身也是正例，就记为 `ACTIVATION_ADD`。
4. `ADD`：各个正例，按时间顺序。
5. V4（`timeline.wait_in_rule`）：第一个正例之前的每个候选分钟标 `WAIT_IN_RULE`，MASKED 的候选除外，因为那一分钟不能 ADD，没有决策可教。如果 touch 开始的游标也是这样的候选，它代替 ACTIVATION 行。

**touch 开始后不是正例的候选，不标为 WAIT。** 那等于用未来结果教学生等待，因此它们只出现在审计中。V4 的例外是第一个正例之前的候选（用户选择），第一个正例之后不是正例的候选仍然只进审计。

## 九、数据合同

输出目录：`/data/logdir/fg_single3/thinking_student/V2_1_POSITIVE_ENTRY_20261031/`

| 文件 | 内容 |
|---|---|
| `teacher/positive_events.csv` | 至少有一个正例的 touch |
| `teacher/positive_entries.csv` | 正例入场 |
| `teacher/positive_timeline.csv` | 只含正向教学过程 |
| `audit/events.csv` | 全部 touch |
| `audit/candidates.csv` | 全部候选及其原因 |
| `audit/audit_summary.json` | 层层过滤的计数和基准率 |
| `figures/` | 事件图 |
| `params.yaml` | 解析后的参数 |
| `source.json` | 两仓 SHA 与 dirty 状态、池中的行、CSV 的 sha256、命令 |
| `REPORT.md` | 报告 |

**列的划分：**
- `contract`、`cursor`：cursor 就是 env 行号，可与 `corpus/<contract>/market` 按游标关联。
- `obs_*`：决策时可见的事实，例如：
  - H、quote、H − quote、RV；
  - touch 开始时的 RV、touch 开始后的分钟数；
  - `obs_touch`：这条线的第几次 touch；
  - `obs_line_spike_rv`：这条线放量标记的 RV（尚未发生时为空）；
  - 按 quote 计划的 R 和 SL。
- 事件表另有 `touch`、`line_spike_time`、`line_spike_rv`。
- `label_*`：离线事实，例如成交、出场、PnL、MAE/MFE、等级、质量分。
- `obs_*` 中不允许出现未来价格、未来 PnL、等级或事后最高点。
- 两列说明文字（只用于人工 Review，不是新规则）：
  - `obs_explain`：只用决策时可见的事实，说明这个点为什么在这里，即位置、与 H 的距离、第几次 touch、这条线的放量、结构止损、计划 R 和计划 TP。WAIT 和 ACTIVATION 行也有。
  - `label_grade_reason`：离线说明成交、出场、MAE 和持仓时间，以及为什么是这个等级或剔除原因。

**Reader：**
- `ts_v2_positive.load_positive(folder)` 只返回 `teacher/` 下的三张表。审计文件不经过这个 Reader。
- 事件表只给出每个入场的最好值和中位数，**不给总和**。

## 十、选择偏差

- 入库正例的 TP 率是 100%，因为它们本来就是按结果筛选出来的。
- 这**不代表**规则本身有优势或盈利能力。
- REPORT 同时给出两个基准率：全部候选的，以及每个 touch 第一个候选的。
- V4 的 `WAIT_IN_RULE` 也是按结果选的：只在后面有正例的 touch 里标。它教的是"这次 touch 还会有更好的入场"，不是"这个位置不该做"。
