# 生成端复盘导出方案：单一 review-v1 协议

状态：**修订版实施合同，尚未实施**。本文件同时约束生成端和复盘端；只支持 `review-v1`，不提供历史兼容、迁移或格式回退。

## 1. 配套关系与实施范围

两个目录组成一套产物：

- `corpus_v2_1_20261104` 提供本次训练的有效教师标签，发布后保持不变。
- `FORMAL_V2_1_1X_20261104` 引用这份 corpus，提供训练阶段及后续学生成交结果。

corpus 可以被多个训练运行复用。关联方向为 run → corpus，corpus 不写死某个 run。
运行启动时记录 corpus 清单的 ID 和 SHA-256，固定本次运行使用的教材。

首版只覆盖三件事：**有效教师 ADD/WAIT、学生实际成交及其交易结果、训练阶段状态**。
生成端新增一个**只读取训练事实的导出模块**（它只向 `review/` 写结果）；复盘端统一读取 `review/`。训练继续使用原始文件，绝不读取 review 文件作为训练标签或账户事实。

**支持范围只包括本方案指定的当前 Corpus 与 Run 原始数据结构。** 本次可为 `corpus_v2_1_20261104`、`FORMAL_V2_1_1X_20261104` 补充导出，不支持旧项目的格式探测、转换或回退；即使旧目录存在，也不承诺可用本导出器处理。

## 2. 文件结构

```text
thinking_student/
├── corpus_v2_1_20261104/
│   ├── CZCE.FG205/...                  原训练文件
│   └── review/
│       ├── manifest.json
│       └── teacher/
│           ├── CZCE.FG001.jsonl
│           ├── CZCE.FG205.jsonl
│           └── ...                     每合约一文件
└── FORMAL_V2_1_1X_20261104/
    ├── run_state.json
    ├── foundation/...
    └── review/
        ├── manifest.json
        └── student/
            └── pi_0000/
                └── DEV/
                    └── r0001/
                        └── CZCE.FG409.jsonl
```

复盘端只打开 `review/manifest.json` 中登记的文件，不递归猜测目录。
所有文件引用都相对于**包含该引用的 `manifest.json` 所在目录**解析，不依赖当前工作目录。数据文件路径必须是 `review/` 内部相对路径，不能用 `..` 逃出该目录；唯一例外是 run 清单中明确登记的 `corpus.manifest`，允许指向同级 Corpus 的 `review/manifest.json`，并须核验其 ID 和 SHA-256。
`review-v1` 是唯一支持的格式标识；`V2_1_STRUCTURAL_R10` 等策略名称只作信息显示。

`student/` 在没有结果时可以不存在。**Corpus manifest 发布后固定不变；Run manifest 可以随阶段推进更新。** 已发布的 JSONL 一律不覆盖，新的学生结果使用不同的结果目录（如 `r0002`）。

## 3. 教材清单

以下展示 FG205 一合约；生产清单需列出全部 18 合约。尖括号内容由导出程序填写。

```json
{
  "schema": "review-v1",
  "kind": "corpus",
  "id": "corpus_v2_1_20261104",
  "revision": 1,
  "label": "V2.1 训练教材",
  "strategy": "V2_1_STRUCTURAL_R10",
  "status": "ready",
  "updated_at": "2026-11-04T09:02:00+08:00",
  "timezone": "Asia/Shanghai",
  "contracts": [
    {"symbol": "CZCE.FG205", "price_tick": 1, "multiplier": 20, "currency": "CNY"}
  ],
  "datasets": [
    {
      "id": "teacher",
      "label": "本次训练的教师示范",
      "kind": "teacher",
      "scope": "TRAIN",
      "policy_id": null,
      "files": [
        {
          "symbol": "CZCE.FG205",
          "path": "teacher/CZCE.FG205.jsonl",
          "records": 266,
          "sha256": "<该 JSONL 文件字节的 SHA-256>",
          "summary": {"add": 194, "wait": 72, "excellent": 146, "good": 48}
        }
      ]
    }
  ]
}
```

导出源**只能是最终 corpus 中 `graded.valid=true` 的标签**。当前全量预期为 ADD 10,110、WAIT 2,510，合计 12,620 行；这些是待导出时核对的源数据预期值，不代表导出已经完成。
原始正例库经过风险、止损线等过滤后才形成 corpus；首版只导出最终有效标签。原始候选和 P4 Oracle 不在本次接口范围，不与教师示范混计。

Corpus 所有18合约导出、验证、发布完成后，`review/manifest.json`及其已登记JSONL视为**同一份不可变发布物**。若源Corpus发生变化，应另建一个明确的新Corpus身份，而不是修改已发布教材。

## 4. 运行清单

以此前核对的 foundation/L3 状态快照为例：此时可以查看关联教材，学生结果列表为空。

```json
{
  "schema": "review-v1",
  "kind": "run",
  "id": "FORMAL_V2_1_1X_20261104",
  "revision": 1,
  "label": "V2.1 · 1x",
  "strategy": "V2_1_STRUCTURAL_R10",
  "status": "running",
  "updated_at": "2026-11-04T10:18:10+08:00",
  "timezone": "Asia/Shanghai",
  "corpus": {
    "id": "corpus_v2_1_20261104",
    "manifest": "../../corpus_v2_1_20261104/review/manifest.json",
    "sha256": "<教材 manifest.json 字节的 SHA-256>"
  },
  "progress": {"phase": "foundation", "step": "L3", "round": 0},
  "latest_policy_id": null,
  "best_dev_policy_id": null,
  "contracts": [],
  "datasets": []
}
```

run 的 `status` 取 `running / completed / failed / cancelled`，**只记录训练程序最后一次已发布的状态**；`running` 不保证当前进程仍在线。复盘端展示 `progress.phase/step/round` 与 `updated_at`，不由文件状态猜测进程、增加心跳或提供实时监控。
`progress` 和 `status` 只能从已提交的训练状态读取，不得根据文件数量、最新修改时间推断。`updated_at` 表示该条训练/导出状态的发布时间，不表示市场时刻。
教材的 `status` 固定为 `ready`；未完整导出并校验全部18合约时不发布教材 manifest。

学生结果完成后，在 run 的 `datasets` 中增加：

```json
{
  "id": "student:pi_0000:DEV:r0001",
  "label": "学生 π0 · DEV",
  "kind": "student",
  "scope": "DEV",
  "policy_id": "pi_0000",
  "files": [
    {
      "symbol": "CZCE.FG409",
      "path": "student/pi_0000/DEV/r0001/CZCE.FG409.jsonl",
      "records": 0,
      "sha256": "<文件的 SHA-256>",
      "summary": {"entries": 0, "closed": 0, "open": 0, "partial": 0, "censored": 0},
      "market_complete": true
    }
  ]
}
```

此JSON仅演示一种**已完整执行但没有任何实际入场**的结果，所以 `records=0`、`entries=0`、`market_complete=true`；不能作为当前运行已产生 π0/DEV 的声明。若尚未执行考试，**不得**建立该 dataset。实际发布时按真实结果填写计数，并在 run 的 `contracts` 中登记合约 tick、乘数及币种。

**学生文件只记录已成交入场，未记录学生每分钟 WAIT。** 因此学生 summary 只允许 `entries/closed/open/partial/censored`：`entries = records = closed + open + partial + censored`。禁止发布 `wait: 0` 或由未导出推断WAIT次数。教师 summary 继续使用 `add/wait/excellent/good`，二者不混同。

`scope` 使用已有阶段名，例如 TRAIN、TRAIN_SMOKE、DEV、PAPER；复盘端显示该值，不用它选择解析算法。每个**策略+scope+独立结果ID**分别登记 dataset（如 `student:pi_0000:DEV:r0001`），禁止后一场考试覆盖前一场考试。`latest_policy_id` 和 `best_dev_policy_id` 仅引用已发布策略结果，允许为 null。
`market_complete` 表示是否走完合约生命周期，可为 true/false/null；**JSONL已完整发布**与**行情已经走完**是不同事实。

## 5. JSONL 记录：教师与学生共用一个结构

每行一个 JSON 对象。由所属 dataset 的 `kind` 区分教师示范与学生成交。
以下字段固定存在；不适用时用 null，`details` 用空对象，`note` 用空字符串。

| 字段 | 类型 | 定义 |
| --- | --- | --- |
| id | string | dataset 内唯一且稳定的记录 ID；教师按合约+决策cursor，学生按账户+Native entry_id，不能按时间和价格去重 |
| symbol | string | 完整合约名，必须与所属文件的 symbol 一致 |
| action | ADD / WAIT | 教师监督动作；学生首版只导出已成功入场的 ADD |
| direction | LONG / SHORT / null | 明确给出的方向，WAIT 未指定方向时可为空 |
| decision_time | 带时区时间 / null | 决策使用的最后一根已观察分钟的结束时间 |
| entry_time | 带时区时间 / null | Native 记录的入场时间 |
| bar_bob、bar_eob | 带时区时间 | 需要绘制标记的原始 1m K 线边界 |
| trading_day | YYYY-MM-DD | 该标记 K 线所属的交易日，含夜盘归属 |
| entry_price、volume | number / null | Native 成交价和手数；WAIT 为 null |
| stop_price、target_price | number / null | 本笔实际生效的止损、目标价 |
| grade | string / null | 教材档位，学生可为空；与 action 分开 |
| label_ready_at | 带时区时间 / null | 完整标签获得时间，读取已有 ready_at；未知则为空 |
| outcome | object / null | 教师 ADD 对应经验证的示范交易结果，学生 ADD 对应自身 Native 实际成交与退出事实；WAIT 为 null，两类来源不能混同 |
| details | object | 普通详情键值，键为显示名称，值为字符串、数字、布尔或 null |
| note | string | 普通文本解释，不含 HTML |

所有时间统一写 ISO 8601，例如 `2021-12-14T11:27:00+08:00`。价格和金额为有限 JSON 数字，禁止 NaN/Infinity。
数量、价格和金额**不使用 0 代表未知**（真实数值为 0 则正常记录 0）。金额币种由合约清单确定；`details` 中带单位的数值在键上写明单位。`label_ready_at` 若原始来源未知，写 `null`，绝不用 `updated_at` 或推测的收盘时间代替。
档位按记录中的实际字符串显示并生成筛选列表，不能由档位推导 ADD/WAIT。原始教师WAIT若以特殊数字档位表示，导出后 `grade=null`，保留 `action=WAIT`。

`outcome` 有值时必须含 `status`。学生状态只允许 `CLOSED/OPEN/PARTIAL/CENSORED`；闭合交易的 `exit_time/exit_price/exit_reason/net_pnl` 由已提交Native归因给出，其余情形不适用或无法确定的字段为 `null`。**不得用一根K线或固定TP/SL参数推测真实退出**。

### 教师 ADD 示例

以下数值按原方案记录，来自新 corpus 的 FG205 游标 941 及其引用的原始正例；正式实现仍须读取最终源文件核对，不把示例当作已发布JSONL。

```json
{
  "id": "teacher:CZCE.FG205:941",
  "symbol": "CZCE.FG205",
  "action": "ADD",
  "direction": "SHORT",
  "decision_time": "2021-12-14T11:27:00+08:00",
  "entry_time": "2021-12-14T11:27:00+08:00",
  "bar_bob": "2021-12-14T11:27:00+08:00",
  "bar_eob": "2021-12-14T11:28:00+08:00",
  "trading_day": "2021-12-14",
  "entry_price": 1846,
  "volume": 3,
  "stop_price": 1855,
  "target_price": 1828,
  "grade": "EXCELLENT",
  "label_ready_at": "2021-12-14T21:01:00+08:00",
  "outcome": {
    "status": "CLOSED",
    "exit_time": "2021-12-14T21:01:00+08:00",
    "exit_price": 1828,
    "exit_reason": "TAKE_PROFIT",
    "net_pnl": 1062
  },
  "details": {
    "原始正例编号": "CZCE.FG205@941:A",
    "前高 H": 1850,
    "第几次触线": 1,
    "该线放量量比": 3.165053622809312,
    "持有市场分钟": 94,
    "最大逆行（点）": 2,
    "最大顺行（点）": 19
  },
  "note": "Native 逐分钟复核通过"
}
```

该点的决策报价是1845，示范交易的已校验入场价是1846，目标价为1828。复盘端显示导出值；**教师ADD属于经过验证的教学事实，不等同于学生真实执行成交**。

### 教师 WAIT 示例

FG205 游标 956 是有效 WAIT：`grade=-1、valid=true、action=0`。

```json
{
  "id": "teacher:CZCE.FG205:956",
  "symbol": "CZCE.FG205",
  "action": "WAIT",
  "direction": "SHORT",
  "decision_time": "2021-12-14T13:42:00+08:00",
  "entry_time": null,
  "bar_bob": "2021-12-14T13:41:00+08:00",
  "bar_eob": "2021-12-14T13:42:00+08:00",
  "trading_day": "2021-12-14",
  "entry_price": null,
  "volume": null,
  "stop_price": null,
  "target_price": null,
  "grade": null,
  "label_ready_at": "2021-12-14T21:08:00+08:00",
  "outcome": null,
  "details": {},
  "note": "有效 WAIT 教师示范"
}
```

WAIT 不画开仓箭头；用户打开等待示范显示时，使用独立等待标记。
其 facts 中可能有“如果入场”的模拟价格和结果，这些不能写进 WAIT 的实际入场及 outcome 字段。

### 学生记录

同一JSONL外形用于学生**真实入场**：所属 dataset 为 `kind=student`，`action=ADD`，`grade` 通常为 `null`。学生记录中 `label_ready_at` 没有相应教师标签时为 `null`，不赋予其教师身份。

- 权威来源是Native已提交的**开仓成交、`entry_id`、平仓归因与期末真实持仓**；每个 `(account_id, entry_id)` 在一个结果dataset内恰好输出一行。建议稳定 `id="student:{account_id}:{entry_id}"`，账户身份不能仅用策略名替代。
- 提议的ADD、拒绝订单、未成交委托、真实成交不能混计；首版也**不导出学生逐分钟WAIT**。结果统计只基于已成交的entry。
- 按Native订单或成交身份关联开仓和归因；禁止按“时间+价格”拼接交易。如果同一 `entry_id` 多次部分平仓，需聚合该entry的真实成交归因，不能用最后一笔价格伪装单一退出价。
- Native记录无法确定某字段时，该字段为 `null`，禁止推测；如果连入场身份、已成交事实或标记行情bar都无法确认，**拒绝发布该结果并列明原因**，不制造一行看似完整的交易。

| `outcome.status` | 唯一判定条件 | 对退出字段的处理 |
| --- | --- | --- |
| `CLOSED` | Native已确认该entry全部平仓，包括真实终局强平 | 从全部成交归因计算已实现完整净盈亏（含费用）；`exit_time` 为完成全部平仓的真实时刻；没有唯一退出价/原因时对应的 `exit_price`/`exit_reason` 为 `null`，在 `details` 中记录多笔退出 |
| `OPEN` | 结果观察边界尚未结束，entry仍全量持仓 | `exit_*`及最终 `net_pnl` 为 `null` |
| `PARTIAL` | 结果观察边界尚未结束，entry已有部分平仓、仍留未平仓量 | `exit_*`及最终 `net_pnl` 为 `null`；已实现部分在 `details` 中以明确名称、单位记载 |
| `CENSORED` | 结果/行情观察边界已经结束，entry仍未全部平仓，后续结果不可见 | `exit_*`及最终 `net_pnl` 为 `null`；若此前部分平仓，仅在 `details` 保留已实现部分 |

判断顺序：**先按真实成交确认是否全部关闭；未关闭再判断结果是否到可知边界；仅在边界尚未结束时区分OPEN/PARTIAL。** 已真实关闭的entry即使发生在末尾也必须是CLOSED，不能为了凑完结率被标为CENSORED。

如果来源无法提供足够的已提交归因以核对 `CLOSED` 的完整净盈亏，则该结果校验失败，不得生成猜测的 `net_pnl`。

学生 `entry_price/volume/entry_time` 必须来自真实成交；`stop_price/target_price` 仅能来自该entry当时实际生效的记录。若目前的Native结果未留存这些生效价格，需要在**后续入场时**记录并持久化；对于已经发生且无法核实的记录，保留 `null`，不得从当前规则或结束时配置反推。

## 6. 时间与展示约定

1. ADD 的 bar_bob/bar_eob 来自实际入场所用的那根行情，不从 decision_time 加一分钟推算。
2. WAIT 的 bar_bob/bar_eob 来自决策时观察的分钟。
3. 复盘端按完整 symbol + bar_bob + bar_eob 查找行情，以 bar_eob 对齐现有图表。
4. replay 游标达到 `bar_eob` 后展示标记。真实 `entry_time` 单独显示，不冒充 `bar_eob`。教师ADD显示为教学标记，学生ADD显示为真实成交标记，二者在UI上可区分。
5. trading_day 直接使用生成端行情的交易日字段，不从自然日期、周末或夜盘时刻推断。
6. 不能精确匹配行情时列为“无匹配行情”，保留该记录供检查。

例如已有跨午休样本决策于 11:30、入场于 13:30，必须使用 13:30–13:31 的入场分钟。
本协议首版用于后验复盘，完整结果可以显示；label_ready_at 保留标签的可知时间，不能当作当时已知的交易信号。

## 7. details 的简单扩展方式

触线次数、量比、MFE、质量、教学解释等新增信息直接放入 details，复盘端按键值显示。
例如 `"本次触线起始量比": null` 表示无数据。
核心箭头依赖固定字段，新增详情不用修改组件，也不用修改 schema 标识。
核心字段的意义如确实改变，应另行定义新协议；本轮只实现 review-v1，不建设版本兼容层。

## 8. 生成端放在哪里、何时调用

建议新增**一个**可复用模块，例如 `gym_trading_env.research.review_export`；Dreamer 已依赖 Gym，可直接调用。该模块只从最终训练教材、Native成交、训练状态读取，绝不回写来源，也不变成新的训练入口。

| 时机 | 生成端工作 |
| --- | --- |
| 18 合约 corpus 全部生成并校验完成 | 导出每合约有效教师标签，发布教材 manifest |
| formal run 初始化 | 读取并固定 corpus manifest 的 ID/hash，发布 run manifest |
| 训练阶段切换 | 从当前训练状态更新 progress、status、updated_at 和 revision |
| collect / exam 成功结束 | 导出该次学生入场 JSONL，登记 dataset 及合约元数据 |
| best_dev/latest 确定 | 更新对应 policy_id 引用 |

结构止损、目标价、方向及成交结果由生成端从Native/最终教材导出，复盘端不复算金融规则。若实际成交时生效的止损/目标尚未留在结果中，应在后续入场发生时真实记录；既有记录无法核实则导出 `null`，不能补造。
首版按阶段结束导出，运行状态按**已确认阶段**更新；暂不引入每分钟成交流或增量事件协议。失败导出不影响原正式训练。

提供三个独立操作：导出 corpus、导出/更新 run、校验 review。建议 CLI 形状如下，具体模块待实施：

```bash
python -m gym_trading_env.research.review_export corpus --input CORPUS_DIR
python -m gym_trading_env.research.review_export run --input RUN_DIR
python -m gym_trading_env.research.review_export validate --input RUN_DIR
```

当前指定的Corpus和Run先补导出，不重新训练、不重新模拟成交。后续生成任务只在**原有阶段成功边界**调用同一导出模块；不因复盘端刷新而触发训练计算。

## 9. 确定性导出、发布与恢复

### 9.1 相同输入必须得到相同JSONL

1. 同一来源的记录按照**确定的顺序**输出：教师按完整 `symbol`、决策 cursor；学生按 `symbol`、账户ID与Native `entry_id`（同一记录不受文件系统遍历顺序影响）。
2. JSON字段顺序、UTF-8编码、换行（每行末尾一个 `\n`）、数值与时间序列化规则固定；同一数据重复导出，必须产生相同**字节内容**和SHA-256。JSON禁止 NaN/Infinity；不存在的数值写 `null`。
3. 学生记录ID来自原生身份，教师记录ID来自原始label身份；禁止用导出时间、行号重排或随机UUID生成ID。
4. 同一结果数据未变化时，不改动已发布文件，也**不递增manifest `revision` 或刷新 `updated_at`**。

### 9.2 发布顺序及冲突处理

1. 向目标目录写临时JSONL，写完关闭后校验**每行schema、ID唯一、内容、记录数与SHA-256**。
2. 如正式目标文件不存在，使用同目录**原子、禁止覆盖已有目标文件**的方式发布（不可仅用会覆盖目标的普通 `os.replace`）；已存在且**字节完全相同**，直接复用；已存在但不同，**报冲突、停止**，不覆盖文件，也不以改名偷偷代替相同结果。新一次考试必须显式使用新的 `r0002` 等结果身份。
3. 全部登记文件校验成功后，才生成新的manifest。发布新的Run状态或新的dataset时，增加 `revision`；先写 `manifest.json.tmp`，再原子替换现有 `manifest.json`。同一状态重复请求则不更新清单。
4. Corpus在所有18合约完成后**只发布一次**，以后不更新其manifest；新的有效教师教材必须使用新的Corpus身份。Run初始化时固定已发布Corpus manifest的ID与SHA-256，以后不得悄悄指向其他教材。
5. 若中途失败，旧的manifest仍有效；已发布但尚未登记的JSONL只是孤立文件，复盘端不可见。重试时同字节复用，不得因孤立文件重复统计。

每份manifest只有**一个写入者**：并行子任务可以生成文件，主进程统一登记。手工导出和自动更新不得同时写同一manifest；不需要数据库、分布式锁或新服务。

### 9.3 复盘端的读取规则

- 先校验 `schema=review-v1`、manifest的 `kind`、必要字段，再读取已登记的JSONL；核验文件SHA-256、行数、ID与合约身份。
- 打开Run时读取其固定Corpus引用，核验ID与文件字节SHA-256；不一致明确失败，不“自动找另一个Corpus”。
- 数据文件缺失、字段不全、哈希不一致时明确报错，不回退旧CSV、NPY、pickle或递归目录扫描。
- Run manifest更新时可刷新列表，但**保留用户已选合约/数据集**（只要仍存在）。`running`只显示最后已记录状态，不宣称进程在线。
- 复盘端只有读取权限，不得修改训练教材、Native账本、Checkpoint、Run状态，也不通过review文件驱动训练。

## 10. 最小验收条件

- 新 corpus 全量导出 18 合约、12,620 行：ADD 10,110、WAIT 2,510；分合约统计与源数据一致。
- FG001 导出 LONG，FG205 导出 SHORT；方向不根据价格猜测。
- FG205 游标 941 显示入场 1846、止损 1855、目标 1828、净收益 1062。
- FG205 游标 956 保持 WAIT、grade=null、entry_price=null，默认无开仓箭头。
- 跨午休和夜盘记录的 bar/trading_day 与源行情一致。
- 空量比保持 null；每条 ADD 的价格来自 Native 成交或已校验的教师交易事实。
- 每个文件的行数和 SHA-256 与 manifest 一致；ID唯一；相同输入重复导出**字节和SHA完全一致**，不重复增加revision；同名不同内容被拒绝且旧manifest完好。
- 学生 `entries=records=closed+open+partial+censored`，与Native已提交开仓的唯一 `(account_id, entry_id)` 数量一致；不得发布 `wait=0` 误导统计，也不把提议ADD/拒单算成交。
- 已关闭、仍持仓、部分平仓及观察边界删失各有原生事实样本；同一entry多笔归因能逐笔对账，缺乏唯一退出价则为 `null`，绝不猜测。
- 无学生结果时，打开run可查看配套教材、最近已确认的训练阶段及其更新时间；`running`不被当作进程实时存活。
- Corpus manifest发布前不出现不完整的 `ready`；Run引用教材ID/hash固定并通过检查，数据文件路径不逃出 `review/`。
- Schema不是 `review-v1`、必需字段不完整或SHA不匹配时明确报错；不解析旧格式。
- 从现有指定Corpus/Run可补导出，**不重训、不复算成交**；导出失败时训练数据和上一版review清单都不改变。

复盘端实施完成后的日常入口拟为：

```bash
npm run dev:electron -- --review /data/logdir/fg_single3/thinking_student/FORMAL_V2_1_1X_20261104
```

该参数尚未实施。复盘端据 run 清单加载配套教材和学生结果，切换 dataset、合约、档位；继续使用现有图表组件。

## 11. 现有代码依据与本方案边界

- formal_train 的 paths.corpus_root 已提供 run → corpus 关系；run_state.json 已提供 phase、step、round、best_dev 等状态。
- ts_v1_corpus 的 graded_teacher 路径已区分有效 ADD/WAIT，最终 corpus 包含方向 sign 和 ready_at。
- 原生账户 export_state 已提供 trade_history、attributions 和期末 entries，可作为学生成交导出的权威来源。
- 本文的 FG205 示例由当前 source、graded/facts 数组及原始 CSV 核对；清单中的 SHA-256 占位符由未来导出器计算。
- 本轮交付是接口及最小实施方案，尚未修改生成程序、复盘读取器或当前训练任务。

## 12. 实施边界与交付顺序（最终Review）

按最小纵切实现即可，不拆新平台：

1. **FG205教师样本**：先验证ADD游标941、WAIT游标956、时间/档位/null、可见性，再完成18合约12,620条Corpus导出。
2. **Run清单**：固定已发布教材ID/hash，生成端从`run_state.json`导出progress/status；复盘端只展示manifest中已发布的阶段状态；没有学生结果时保持`datasets=[]`。
3. **学生Native结果**：按`entry_id`导出入场及完整/部分/删失归因；加入独立结果dataset。没有可核实的Native事实时拒绝发布，而不是从行情猜测。
4. **复盘端**：只读manifest及其登记JSONL，复用现有图表，完成ADD/WAIT标记、详情和精确行情匹配。

**不做**：旧格式兼容、原始正例与P4扩展、数据库、服务端、实时事件流、反向写训练、重新训练、历史全量迁移、新版认证或复杂配置平台。

本方案只是**待实现的协议定义**；12,620条、FG205示例及代码来源为原方案的已核对预期，不能当作本轮程序实施与新文件验收已经完成。
