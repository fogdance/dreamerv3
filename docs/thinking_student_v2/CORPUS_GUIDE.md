# Corpus 指南：教材从生成到 review-v1 导出

日期：2026-11-04
读者：生成端（本仓库），以及对接复盘端 `review_future` 的 agent。
规范：`review_export_v1.md`（review-v1）。实现细节和验收记录见 `REVIEW_EXPORT_IMPL.md`。
范围：生成端的教材，也就是 corpus 的有效教师 ADD/WAIT。run 清单、阶段状态和学生成交同样由生成端导出，见 `REVIEW_EXPORT_IMPL.md` 第五节。

## 一、流水线

| 步 | 产物 | 命令（在 gym 仓库目录下） | 谁读 |
|---|---|---|---|
| 1 原始正例 | `V2_x_POSITIVE_ENTRY_<SIDE>_<date>/`：`teacher/` 下三张表 `positive_events`、`positive_entries`、`positive_timeline`，另有 `audit/` 和 `figures/` | `python -m gym_trading_env.research.ts_v2_positive --pool ../dreamerv3/configs/thinking_student/markets/pool.json --config ../dreamerv3/configs/thinking_student/<规则>.yaml --out DIR --workers 24` | 用户审图；第 2 步 |
| 1b 合并 | `..._BOTH_<date>/`：SHORT 和 LONG 的三张表逐字节拼接 | `python -m gym_trading_env.research.ts_v2_positive --combine SHORT_DIR LONG_DIR --out DIR` | 第 2 步 |
| 2 训练语料 | `corpus_<ver>_<date>/<合约>/`：每个合约一个目录 | `python -m gym_trading_env.research.ts_v1_corpus --contract C --pool ... --strategy ../dreamerv3/configs/thinking_student/markets/<策略>.yaml --teacher BOTH_DIR --out ROOT --workers 12` | 训练 |
| 3 教师检查 | 18 个合约用同一个教师，ADD / WAIT 与教材一致 | 示例：`/data/logdir/fg_single3/thinking_student/V2_1_DEV_20261104/checks/corpus_teacher_check.py`（V2.1 的建库脚本是同目录上一级的 `corpus_build/build.sh`） | 发布前的关卡 |
| 4 review 导出 | `ROOT/review/manifest.json` 和 `ROOT/review/teacher/<合约>.jsonl` | `python -m gym_trading_env.research.review_export corpus --input ROOT --label "V2.x teacher" --expect-contracts 18`，再 `... validate --input ROOT` | 复盘端 |

以上命令都用 `env PYTHONPATH=.:src /home/v/miniconda3/envs/dreamerv3/bin/python` 运行，只用 CPU。生产前两个仓库都必须是干净的已提交状态，因为产物会记录代码 SHA。

**生成端的约束**
- **建库时机：** 第 1 步做完之后，什么时候做第 2 步由用户决定。
- **规则变更：** 规则一旦改动就是新版本，必须用新的输出目录，不改已有教材。
- **review 发布：**
  - `review/` 只发布一次，之后不再改动。
  - 来源变了，必须用新的 corpus 身份（新的根目录名）重新导出。
  - 导出器写入的文字一律用英文。

## 二、corpus 目录（训练读，复盘端不读）

每个合约一个目录，内容如下：
- `source.json`：合约、pool 行（行情 CSV 的路径、sha256、行数）、策略、代码 SHA。
- `inputs/`：env、合约和策略配置。
- `market/`：观测和 `time.npy`，即每根 K 线的 eob。
- `facts/`：每个游标空仓 ADD 后的 Native 交易，包括 `entry_cursor`、`exit_cursor`、`exit_reason`、`net_pnl` 等。
- `graded/`：教师标签，包括 `valid`、`action`（1 = ADD）、`grade`（3 = EXCELLENT，2 = GOOD，−1 = WAIT）、`ready_at`、`entry_price`、`sign`，以及记录了教材目录和 sha256 的 `summary.json`。
- 另有 `strict/` 和 `behaviors/`。

训练只认带 `source.json` 的目录，`review/` 没有这个文件，所以会被跳过。

## 三、review-v1 教材：复盘端唯一需要读的东西

```text
ROOT/review/
├── manifest.json            schema review-v1，kind corpus，status ready（只发布一次）
└── teacher/CZCE.FG205.jsonl 每个合约一个文件，按游标顺序，每行一条记录
```

**manifest：**
- `id` 就是 corpus 的根目录名，`label` 例如 `V2.1 teacher`，`strategy` 例如 `V2_1_STRUCTURAL_R10`。
- `contracts` 每项包含 `symbol`、`price_tick`、`multiplier`、`currency`。
- 只有一个数据集：`id = teacher`，`kind = teacher`，`scope = TRAIN`。它的 `files` 每项包含 `symbol`、`path`（相对于 `review/`）、`records`、`sha256` 和 `summary {add, wait, excellent, good}`。

**每条记录的字段（顺序固定）：**

| 字段 | ADD | WAIT |
|---|---|---|
| `id` | `teacher:<合约>:<决策游标>` | 同左 |
| `direction` | `LONG` / `SHORT`，取自合约的交易方向 | 同左 |
| `decision_time` | 决策那根 K 线的 eob | 同左 |
| `entry_time` | Native 成交时刻，即成交 K 线的 bob | null |
| `bar_bob` / `bar_eob` | 成交那根 K 线；跨午休时就是 13:30–13:31 | 决策那根 K 线 |
| `trading_day` | 标记的那根 K 线的交易日，取行情的 `trading_date`，夜盘归下一交易日 | 同左 |
| `entry_price`、`volume` | 已校验的教师成交价；手数 3 | null |
| `stop_price`、`target_price` | 这笔示范交易的结构止损和 2R 目标 | null |
| `grade` | `EXCELLENT` / `GOOD` | null |
| `label_ready_at` | 标签可知的时刻（示范交易 TP 的那一分钟；WAIT 是所属 touch 最后一个保留的 TP） | 同左 |
| `outcome` | `{status: CLOSED, exit_time, exit_price, exit_reason, net_pnl}`，净盈亏含双边手续费，单位 CNY | null |
| `details` | `Positive ID`、`Prior high H` / `Prior low L`、`Touch`、`Line spike RV`、`Touch start RV`、`Decision quote`、`Risk R (points)`、`Held (market minutes)`、`MAE (points)`、`MFE (points)`、`Explanation` | `Wait type`（WAIT_FAR / WAIT_APPROACH / WAIT_NEAR_NO_VOLUME；V2.3 只有 WAIT_IN_RULE）、`Prior high H` / `Prior low L`、`Touch`、`RV`、`Explanation`；WAIT_IN_RULE 另有 `Decision quote`、`Planned R (points)`、`Minutes since touch began` |
| `note` | `Native minute trace verified` | `Valid teacher WAIT` |

**格式约定：**
- 时间都是 ISO 8601，带 +08:00。
- 数字都是有限值，整数值写成整数；缺失写 null（例如开盘第一分钟的 RV），不用 0 表示未知。
- `Risk R (points)` 是成交后的风险 SL − fill。`Explanation` 里写的是决策时按报价算出的计划，两者可能差 1 点。

**复盘端读取的约定（规范 §9.3）：**
- **入口：** 只打开 manifest 中登记的文件。
- **先核对：** sha256、行数，以及每行 `id` 唯一、`symbol` 与所在文件一致。
- **格式：** 只认 `review-v1`，不解析 CSV、NPY 或 pickle。
- **标记：** ADD 画在 `bar_eob` 上，作为教学标记，不是学生的真实成交；WAIT 用单独的等待标记。

## 四、已发布的教材

| corpus | 记录 | manifest sha256 | 说明 |
|---|---|---|---|
| `/data/logdir/fg_single3/thinking_student/corpus_v2_1_20261104` | 18 个文件，12,620 条：ADD 10,110，WAIT 2,510 | `8ce21bab…21c7f6` | V2.1 训练（`FORMAL_V2_1_1X_20261104`）使用的教材，2026-11-04 17:13 补导出 |
| `/data/logdir/fg_single3/thinking_student/corpus_v2_2_20261104` | 18 个文件，12,784 条：ADD 9,429（EXCELLENT 3,431，GOOD 5,998），WAIT 3,355 | `3df3b2e0…5ac54b` | V2.2（V3 教材）。2026-11-04 18:02 发布，同时完成教师检查（`V2_2_DEV_20261104/corpus_build/after_build.sh`）。`FORMAL_V2_2_1X_20261104` 固定使用这份教材 |

V2.2（规则 V3：每次 touch 都要由它自己的 RV ≥ 2 的 K 线开始）走同一条流水线：
- 第 1 步已完成（2026-11-04 17:29）：`V2_2_POSITIVE_ENTRY_{SHORT,LONG,BOTH}_20261104`，脚本 `V2_2_DEV_20261104/v3_material.sh`；两仓 clean（gym `cd89349`，dreamer `63f4777`）。用户不要统计和图。
- 用户的计划（2026-11-04）：V3 教材 → 1x 正式训练 → paper，期间的教材和学生成交都按 review-v1 导出。所以接着做第 2–4 步，建 `corpus_v2_2_20261104`；run 清单和学生成交由 run 导出负责（见 `REVIEW_EXPORT_IMPL.md` 第五节）。正式训练 `FORMAL_V2_2_1X_20261104`（`formal_v2_2.yaml`）启动前，脚本会检查这份教材的 review 已发布并通过 validate。

## 五、V2.3（规则闸门 + 真实成本，2026-11-05）

规则 V4 和闸门的设计见 `DESIGN.md` 的 V4 一节，计划见 `V2_3_RULE_GATE_PLAN.md`。流水线与前面相同，只换配置和目录：

| 步 | 输出 | 配置 |
|---|---|---|
| 1 原始正例 | `V2_3_POSITIVE_ENTRY_SHORT_20261105`、`V2_3_POSITIVE_ENTRY_LONG_20261105` | `--config ../dreamerv3/configs/thinking_student/v2_positive_entry_v4.yaml`（LONG：`v2_positive_entry_v4_long.yaml`） |
| 1b 合并 | `V2_3_POSITIVE_ENTRY_BOTH_20261105` | `--combine` SHORT 和 LONG 两个目录 |
| 2 训练语料 | `corpus_v2_3_20261105/<合约>/` | `--strategy ../dreamerv3/configs/thinking_student/markets/strategy_v2_3_rule_gate.yaml --teacher .../V2_3_POSITIVE_ENTRY_BOTH_20261105` |
| 3 教师检查 | 标签全部合法；STOP_LINE_DIFFERS 为 0；行为账户的 ADD 100% 在规则位置；标签覆盖率（`graded/summary.json` 的 `rule_gate`） | — |
| 4 review 导出 | `corpus_v2_3_20261105/review/` | `--label "V2.3 teacher" --expect-contracts 18` |

完整命令写在 `formal_v2_3.yaml` 的文件头里。

**和 V2.2 的区别**
- 语料的 `inputs/` 多两个文件 `rule_gate_short.yaml`、`rule_gate_long.yaml`（V4 规则，成本已经读成数值），`single3.yaml` 的 `costs` 换成成本预设的值。
- 成本只写在 `markets/costs_fg_real_20261105.yaml` 一个文件里，格式与原来的 CZCE_FG 成本预设相同；以后改成本只改它（新文件名、新版本）。
- 教材里的 WAIT 只有 `WAIT_IN_RULE`：touch 之前的 WAIT 在闸门下不是标签。
- 学生成交的 `details` 多 `Rule touch` 和 `Touch`。
