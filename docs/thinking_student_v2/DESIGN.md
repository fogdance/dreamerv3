# V2 正向教材：设计

日期：2026-10-31
依据：Spec V0.3。
参数集中在 `dreamerv3/configs/thinking_student/v2_positive_entry.yaml`（版本 `V2_POSITIVE_ENTRY_V1`）。
**以下规则在生成之前就已冻结。** 之后任何改动都算新版本教材，并在 REPORT 中说明；不根据结果回头调参。

## 一、范围

- 只用 9 个方向为 SHORT 的 TRAIN 合约：FG809、FG901、FG905、FG005、FG201、FG205、FG209、FG301、FG405。
- 方向来自用户配置的区间，每行都标为 `direction_source = DIRECTION_ASSUMED`。
- 不用 LONG 合约，也不做镜像。
- 不用 DEV、PAPER 和 TEST。

## 二、因果前高 H

1. 只用完整的 15 分钟 bucket，即 `MarketData.complete_buckets`，按结束时间排序。
2. **Pivot 定义：** 第 j 个 bucket 是 pivot high，当且仅当：
   - 它的 High 严格高于前 2 个完整 bucket 的 High；
   - 并且不低于后 2 个完整 bucket 的 High。
3. **可见时刻：** 第 j+2 个 bucket 的 `available_at`，即 bucket 结束时间。从第一个 bar 结束时间 ≥ 该时刻的游标起，这个 level 才可见。
4. **Level 的状态**（只看 pivot bucket 结束之后的 1m bar）：
   - `invalid`：某根 bar 的 High + 点差 ≥ H + 5。这时 Native 的空单止损已经会被触发，结构失效。
   - `departed`：某根 bar 的 Close ≤ H − 10，即价格已经远离前高。
   - `expired`：当前已可见的完整 bucket 数，减去 j，超过 46（约 2 个交易日）。
   - `consumed`：已经被一个事件使用过。每个 level 最多产生一个事件。
5. **Active H：** 在游标 s，取最近一个已可见、valid、departed、未 expired、未 consumed 的 level。
6. **不做的事：**
   - 不做箱体识别。
   - 不用"事后最高点"反推 H。
   - 不用未来价格决定 H 在何时出现。

## 三、量比 RV

- RV_s = Volume_s / 前 20 根有效 1m bar 的平均 Volume。
- 每个交易时段（21:00、09:00、10:30、13:30）的第一分钟含集合竞价量，所以既不能作为触发 bar，也不计入基准。
- 前面不足 20 根基准 bar 时，RV 记为未知，不能触发。

## 四、事件生命周期

每次只处理一个事件，按游标顺序扫描，只用游标 s 及以前的数据。

1. **激活：** 没有进行中的事件，并且对 active H，bar t 同时满足：
   - RV_t ≥ 2；
   - High_t ≥ H − 5；
   - High_t + 点差 < H + 5（没有失效）。
   之后 H 冻结，这个 level 记为 consumed。
2. **结束：** 激活之后的第一根满足下列任一条件的 bar s：
   - **失效：** High_s + 点差 ≥ H + 5。
   - **远离：** Close_s ≤ H − 10。
   - **超时：** 激活后超过 60 个市场分钟。
   - **边界：** 过了 Native 的 ADD 截止时间或 TRAIN 边界。
   结束的那根 bar 不再产生候选。

## 五、候选

- 激活后，事件内每个决策游标 s（包括 t 本身）都检查：决策时的卖出报价 `quote_s = Close_s − 点差`（向下取整）是否在 [H − 5, H] 内。
  - 后续候选**不要求** RV ≥ 2。
- 候选可执行的条件：
  - Native 空仓账户的 ADD Mask 打开；
  - 在 bar s+1 实际成交，成交价 `fill = Open_{s+1} − 点差`，并且 fill 也在 [H − 5, H] 内。如果因跳空落到区外，记为 `FILL_OUT_OF_ZONE`。
- **风险和目标：**
  - R = H + 5 − fill，取值 5 ≤ R ≤ 10。
  - SL = H + 5，与冻结的 H 相同。
  - 主目标 TP = fill − 2R。
  - 并行记录 TP = fill − ceil(1.5R)，只用于分析，不用于筛选。
- 同一事件的多个候选是**可以互相替代的入场方案**，各自从空仓账户单独计算。任何输出都不把它们的收益相加。

## 六、交易结果（离线标签）

- 每个 R 值构造一个真实 Native env：`stop_loss_points = R`、`take_profit_points = 2R`；1.5R 另外构造一组。
  - 每个合约共 12 个 env。
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

只为至少有一个正例的事件生成：
1. 起点是激活前最后一个 Close ≤ H − 10 的游标。如果 H 在那时还不可见，就从 H 可见的游标开始。
2. 激活前各阶段：
   - `WAIT_FAR`：远离 H。
   - `WAIT_APPROACH`：正在反弹，但 High < H − 5。
   - `WAIT_NEAR_NO_VOLUME`：已接近 H，但 RV < 2。
   - 每个阶段只保留首行、末行，以及不超过 8 行均匀抽样的行。
3. `ACTIVATION`：激活游标。如果它本身也是正例，就记为 `ACTIVATION_ADD`。
4. `ADD`：各个正例，按时间顺序。

**激活后不是正例的候选，不标为 WAIT。** 那等于用未来结果教学生等待，因此它们只出现在审计中。

## 九、数据合同

输出目录：`/data/logdir/fg_single3/thinking_student/V2_POSITIVE_ENTRY_20261031/`

| 文件 | 内容 |
|---|---|
| `teacher/positive_events.csv` | 至少有一个正例的事件 |
| `teacher/positive_entries.csv` | 正例入场 |
| `teacher/positive_timeline.csv` | 只含正向教学过程 |
| `audit/events.csv` | 全部激活的事件 |
| `audit/candidates.csv` | 全部候选及其原因 |
| `audit/audit_summary.json` | 层层过滤的计数和基准率 |
| `figures/` | 事件图 |
| `params.yaml` | 解析后的参数 |
| `source.json` | 两仓 SHA 与 dirty 状态、池中的行、CSV 的 sha256、命令 |
| `REPORT.md` | 报告 |

**列的划分：**
- `contract`、`cursor`：cursor 就是 env 行号，可与 `corpus/<contract>/market` 按游标关联。
- `obs_*`：决策时可见的事实，例如 H、quote、H − quote、RV、激活时的 RV、激活后的分钟数、按 quote 计划的 R 和 SL。
- `label_*`：离线事实，例如成交、出场、PnL、MAE/MFE、等级、质量分。
- `obs_*` 中不允许出现未来价格、未来 PnL、等级或事后最高点。
- 两列说明文字（只用于人工 Review，不是新规则）：
  - `obs_explain`：只用决策时可见的事实，说明这个点为什么在这里，即位置、与 H 的距离、量能背景、结构止损、计划 R 和计划 TP。WAIT 和 ACTIVATION 行也有。
  - `label_grade_reason`：离线说明成交、出场、MAE 和持仓时间，以及为什么是这个等级或剔除原因。

**Reader：**
- `ts_v2_positive.load_positive(folder)` 只返回 `teacher/` 下的三张表。审计文件不经过这个 Reader。
- 事件表只给出每个入场的最好值和中位数，**不给总和**。

## 十、选择偏差

- 入库正例的 TP 率是 100%，因为它们本来就是按结果筛选出来的。
- 这**不代表**规则本身有优势或盈利能力。
- REPORT 同时给出两个基准率：全部候选的，以及每个事件第一个候选的。
