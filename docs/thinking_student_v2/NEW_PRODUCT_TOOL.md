# 新品种 paper 工具（gym `research.product_paper`）

2026-11-06 加入主分支（用户："支持新品种的工具，比如导入导出的，建议写个通用的脚本，方便后续使用"）。第一次用于 RB，分支 `paper/ts-v3-rb-20261106`，TODO G-36。

## 输入

- **方向文件：** 用户提供的 schema 2 yaml，工具只读取其中的 `directions.intervals`。每段写明合约、方向（LONG / SHORT），以及 [start, end) 交易日区间。
  - 同一合约可以有多段；
  - 每个合约只在自己的区间里交易。
- **其余配置照 FG 的 paper：**
  - 主策略 `markets/strategy_quality_tp_v1.yaml`，包括交易时段、日期口径、时区、奖励和环境设置；
  - env `markets/envs/CZCE_FG501_env.yaml`；
  - 合约文件的生成方式：交易时段从实际 K 线提取，最后一根 K 线就是主力换月出场点。
- **品种登记：** 品种必须先登记在 gym `envs/entry_contract.CONTRACT_NAMES` 里，现在有 FG、SA、RB。新品种在这里加一行合约名的正则。

## 成本只配一处

- **品种成本预设：** 每点价值、手续费、点差和保证金，只写在品种自己的成本预设里，格式和 `markets/costs_fg_real_20261105.yaml` 相同，顶层键换成该品种。例如 `markets/rb/costs_rb_20261106.yaml`：

  ```yaml
  SHFE_RB:
    lot_size: 10
    leverage: 10
    trading_fee_per_lot: 2.1
    is_round_turn: true
    spread: 0
  ```

- **策略引用预设：** 策略的 `costs` 指向这个预设，env 再用它覆盖合约文件里的 `lot_size` 和手续费。合约文件里的这几项只是副本，不用手改。
- **规则文件共用：** 闸门只检查规则文件的点差和 env 是否一致，因为点差决定判断用的报价。所以所有品种共用同一份规则文件，不再复制。

## 命令

在 `gym-trading-env/` 目录下运行，带上 `PYTHONPATH=.:src`。

| 命令 | 作用 |
|---|---|
| `coverage --directions D.yaml` | 按区间列出数据库里该合约的交易日数、品种日历的天数、缺失天数，以及掘金归档有哪些月份 |
| `ingest --directions D.yaml [--contracts A,B] [--dry-run]` | 对缺数据的合约，调用掘金 `jobs/v2/ingest_archive_to_db_v2.py`（juejin 环境，归档 `/data/juejin`），把归档导入 `fut_bar_1m_v2`。已经导入过的月份会跳过 |
| `build --directions D.yaml --configs ../dreamerv3/configs/thinking_student --out DIR [--name rb] [--costs 预设] [--strategy strategy_v3_trend_trail.yaml]` | 生成 paper 输入，见下文 |

`build` 生成的文件：
- `DIR/export/` 下的 1m 数据，以及 `DIR/build_report.json`；
- `markets/<name>/` 下的合约文件、env、主策略 `strategy_quality_tp_v1_<name>.yaml` 和 paper 池 `paper_pool_<name>.json`；
- 给了 `--strategy` 时，还会生成该策略的品种版 `<策略名>_<name>.yaml`：只把 `costs` 换成品种预设，规则文件仍然共用；
- 数据库里没有数据的合约，写进报告后跳过。

## 流程

1. `coverage` 查看缺多少数据。
2. 有缺失就运行 `ingest`。归档里没有的部分补不上，例如 RB 的归档从 2017-12 才开始。
3. 写好品种成本预设，运行 `build --costs ... --strategy ...`。
4. paper 时 `exam --paper-pool markets/<name>/paper_pool_<name>.json --strategy markets/<name>/<策略>_<name>.yaml`，脚本可参照 `PAPER_RB_V3_20261106/run.sh`。

## 注意

- **点数参数照搬 FG：** 止损、区间、保本这些都按 FG 的点数设置。新品种波动不同时，可以先看 15m ATR 的比例。不要在 paper 上挑参数。
- **方向是事后标的：** 方向文件是用户事后标注的，paper 结果更接近"方向正确"时的上限。
