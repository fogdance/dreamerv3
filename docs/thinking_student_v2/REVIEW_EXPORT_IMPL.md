# review-v1 生成端实施说明（教材部分）

日期：2026-11-04
规范：`review_export_v1.md`，即用户的 `review-export-v1-revised.md` 原样复制，sha256 `559e95a9…ab7f0a91`。
范围：用户说“我们只处理生成端的教材”。
- **只做：** corpus 的有效教师 ADD/WAIT 导出。
- **不做：**
  - run 清单和学生成交；
  - 原始正例和 P4；
  - 复盘端 `review_future`（由其他 agent 对接）。

## 一、入口

```bash
cd gym-trading-env
env PYTHONPATH=src python -m gym_trading_env.research.review_export corpus \
  --input /data/logdir/fg_single3/thinking_student/corpus_v2_1_20261104 \
  --label "V2.1 teacher" --expect-contracts 18
env PYTHONPATH=src python -m gym_trading_env.research.review_export validate \
  --input /data/logdir/fg_single3/thinking_student/corpus_v2_1_20261104
```

- **代码与测试：** 代码在 gym `src/gym_trading_env/research/review_export.py`，测试在 `tests/entry_v1/test_review_export.py`。
- **输出：** `CORPUS/review/manifest.json`，以及每个合约一个 `review/teacher/<合约>.jsonl`。
- **不影响训练：** `review/` 里没有 `source.json`，训练发现 corpus 时会跳过它；训练也从不读取它。
- **调用时机：** 18 个合约的 corpus 全部建好、教师检查通过之后，在建库脚本最后调用 `corpus`，再调用 `validate`。

## 二、字段来源

所有字段都是已记录的事实，不重算任何金融规则。来源对不上时，导出直接失败，不发布任何东西。

导出器写入的文字一律用英文（用户 2026-11-04："像这些不要用中文，用英文"）：`details` 的键、`note`、数据集 label；manifest 的 label 由 `--label` 给出，例如 `V2.1 teacher`。规范示例中的中文键名只是示例，不照搬。

| 字段 | ADD | WAIT |
|---|---|---|
| 时间、K 线 | `decision_time` 取决策游标那根 K 线的 eob（CSV 的 `Date`）。`entry_time`、`bar_bob`、`bar_eob`、`trading_day` 取成交那根 K 线（`facts.entry_cursor`）的 `bob` / `Date` / `trading_date`；跨午休时就是 13:30–13:31 | 都取决策那根 K 线 |
| 方向 | `graded.sign`，并核对它与 env 的方向表（`contract_direction`）一致 | 同左 |
| 入场价、档位 | `graded.entry_price` 和 `graded.grade`（3 = EXCELLENT，2 = GOOD），并核对它们与教材的成交价、档位一致 | 都是 null |
| 手数 | env 的 `trading.trade_lot`（3） | null |
| 止损、目标 | 教材的 `label_sl` / `label_tp` | null |
| `label_ready_at` | `graded.ready_at` 所指那根 K 线的 eob | 同左 |
| `outcome` | CLOSED。退出时间是 `facts.exit_cursor` 那根 K 线的 eob；退出价取教材；退出原因和净盈亏取 `facts`；都与教材核对 | null |
| `details` | `Positive ID`、`Prior high H` / `Prior low L`、`Touch`、`Line spike RV`、`Touch start RV`、`Decision quote`、`Risk R (points)`（成交后的风险 SL − fill）、`Held (market minutes)`、`MAE (points)`、`MFE (points)`、`Explanation` | `Wait type`、`Prior high H` / `Prior low L`、`Touch`、`RV`、`Explanation`（不放“如果入场”的价格） |
| `note` | `Native minute trace verified`（教材的 `label_native_verified`） | `Valid teacher WAIT` |

**导出前先核对的来源：**
- **行情 CSV：** 它的 sha256 和行数要与 pool 行一致，`Date` 要与 corpus 的 `market/time.npy` 一致。
- **教材：** 每个 corpus 的 `graded/summary.json` 里记着教材目录和三张表的 sha256，文件必须与之一致。每个游标在教材里要恰好对应一行：ADD 对应 `positive_entries`，WAIT 对应 `positive_timeline` 中的 WAIT 行。
- **计数：** 每个合约的 ADD / WAIT / EXCELLENT / GOOD 数要与 `graded/summary.json` 一致。
- **一致性：** 18 个合约必须是同一个策略、同一个教师；合约数要等于 `--expect-contracts`。

**源数据里的已知异常：** FG001 在 2019-09-24 11:30 有一根 `bob = eob` 的收盘竞价 K 线（成交量 118），没有被任何标签用到。所以 bob < eob 只检查记录实际标记的 K 线。

## 三、确定性和发布

- **记录格式：**
  - 按游标顺序输出，字段顺序与规范表格相同；
  - 紧凑 JSON，非 ASCII 字符原样保留，每行以一个 `\n` 结尾；
  - 整数值写成整数，NaN 写成 null，核心价格不允许缺失；
  - 时间用 ISO 8601，带 +08:00。
- **发布 JSONL：** 先写临时文件，再用 `os.link` 发布，绝不覆盖已有文件。
  - 目标已存在且字节相同：复用。
  - 目标已存在但字节不同：报 `ConflictError` 并停止。
- **发布 manifest：**
  - 所有文件校验通过之后才写，只发布一次，同样不覆盖。
  - 再次导出、内容相同时，manifest 不动：`revision` 不增加，`updated_at` 不刷新。
  - 内容不同（包括 label 不同）：报冲突，旧 manifest 保持原样。新的教材必须用新的 corpus 身份。
- **validate 检查：**
  - `schema` 和 `kind`；
  - 路径必须在 `review/` 之内，不能用 `..`；
  - 每个文件的 sha256 和行数；
  - 每条记录的字段与类型、合约名、ID 唯一；
  - WAIT 的入场 / 止损 / 目标 / 档位 / outcome 字段全部为 null；
  - 各文件的 summary 与记录一致。

## 四、验收（2026-11-04，gym `cd89349`）

`corpus_v2_1_20261104` 已经补导出到它自己的 `review/` 下（17:13，英文版）。日志 `V2_2_DEV_20261104/review_export/backfill.log`，脚本 `backfill.sh`，各文件 sha256 在 `sha256.txt`。

17:10 先发布过一版中文键名的导出（gym `a6dff55`）。用户要求改成英文之后，那一版没有删除，原样移到 `V2_2_DEV_20261104/review_export/superseded_zh_20261104_1710/` 留作证据，然后重新发布。改英文前没有任何读取方使用过那一版。

| 项目 | 结果 |
|---|---|
| 文件 / 记录 | 18 个文件，12,620 条：ADD 10,110（EXCELLENT 3,786，GOOD 6,324），WAIT 2,510；分合约数与 `graded/summary.json` 一致 |
| 文字 | manifest 和 18 个 JSONL 中没有中文 |
| FG205 游标 941 | ADD SHORT：决策 11:27，入场 11:27，K 线 11:27–11:28，交易日 2021-12-14，入场 1846、止损 1855、目标 1828，EXCELLENT，ready 21:01；outcome CLOSED，21:01 TAKE_PROFIT，1828，净收益 1062 |
| FG205 游标 956 | WAIT：K 线 13:41–13:42，grade、entry_price、outcome 都是 null，ready 21:08 |
| 方向 | FG001 全部 LONG，FG205 全部 SHORT |
| 跨午休、夜盘 | 11:30 决策的 ADD 都标在 13:30–13:31 的成交 K 线上；夜盘记录的交易日取 CSV 的 `trading_date`（下一交易日） |
| validate | 通过 |
| 重复导出 | manifest 不变，18 个文件原样复用，字节与预先的试导出完全相同 |
| 训练发现 | `discover` 仍找到 18 个 corpus，`review/` 被跳过 |
| 测试 | `test_review_export.py` 4 项通过（规范记录与总数、K 线与交易日、重复导出与冲突、validate 的四类拒绝）；ruff F/E9 通过 |

**未做：** run 清单、学生成交和复盘端读取（都不在本次范围内）。
