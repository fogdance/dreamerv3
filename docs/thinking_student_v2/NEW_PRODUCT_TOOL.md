# 新品种：品种档案、paper 输入工具和通用 paper 入口

- **2026-11-06：** 新品种工具 gym `research.product_paper` 加入主分支，第一次用于 RB（TODO G-36 / G-38）。
- **2026-11-08 改为品种档案**（TODO G-44）。用户的要求："后续品种可能增加，这些你要在这次看下怎么设计方案……不然到时候不好维护"。
- **同日第二轮：一次做完，不留兼容。** 用户："这样做不对吧，一次性改好。教材生成这次没有改"；"别急着改。目前我们是在开发阶段，尽量不做兼容，一次性做好"。教材、语料、标签、学生和观测都改成以档案为准。

## 设计

一个品种只在一个地方配置：**品种档案** `configs/thinking_student/products/<代码>.yaml`。

| 键 | 含义 | 例 |
|---|---|---|
| `product` | 品种代码，和主策略的 `product` 相同 | `SP` |
| `contract_pattern` | 这个品种的合约名 | `'SHFE\.sp\d{4}'` |
| `tick_points` | 一跳几点 | FG 1，SP 2 |
| `costs` | 成本：成本预设的路径，或者直接写预设块（`lot_size`、`leverage`、`trading_fee_per_lot`、`is_round_turn`、`spread`） | `../markets/costs_fg_real_20261105.yaml` |
| `paper_pool` | 可选，paper 池（通用 paper 入口用） | `../markets/paper_pool.json` |
| `rule_overrides` | 可选，品种自己的规则差异，单位是跳 | FG：`{entry: {zone_above_points: 2}}` |

**档案怎么被找到：** 品种的主策略（`strategy_quality_tp_v1*.yaml`，即品种的行情文件）用 `product_profile` 指向档案，env 从这里读取。主策略没有 `product_profile` 就报错。

**写入输入时：** `write_inputs` 把档案复制成 `product.yaml`，并把成本预设直接写进去，所以输入目录不依赖外部文件。

**成本只在档案：** 主策略、策略文件、规则文件都不能写 `costs`，写了就报错。成本预设由 gym `entry_product.read_preset` 读取，这是唯一的入口。

**没有主策略的 env**（测试里的合成 env、旧的 fg_entry_v1 协议）没有品种，用自己合约文件里的成本和一跳点数。

### 距离一律按"跳"
- **配置里的距离都是跳，** 名字保持 `*_points` 不变。包括：
  - 策略、规则文件、档案的 `rule_overrides`；
  - `entry_grades.yaml`（V1 评级）；
  - exam 和 corpus 命令行的 `--take-profit-points` / `--stop-loss-points`；
  - 学生的 `ts.outcome_points`。
- **计算和记录用实际点数：** 价格、R、MAE/MFE、标签、交易记录和报告都是实际点数。FG、SA、RB 一跳 1 点，数字不变；SP 一跳 2 点，全部变成 2 倍（用户："止损固定5跳，10点……其余参数按 2 倍换算"）。
- **换算点固定：**
  - **env：** 加载时用 `entry_product.in_points` 把规则、止损、原生止盈止损和出场距离乘以 `tick_points`。参数清单只在 gym `envs/entry_product.py`（`RULE_TICKS`、`STOP_RULE_TICKS`、`NATIVE_TICKS`）和 `research/swing_trail.TICKS`。
  - **教材 `ts_v2_positive`：**
    - `produce` 开头按行所属品种换算一次，一份教材只能是一个品种。
    - 每个候选的 Native 交易，止盈止损按跳交给 env，目录名 `R{跳}_TP{跳}`。
    - `params.yaml` 保留配置原文（跳），`source.json` 记 `product` 和 `tick_points`。
    - 教材规则和 env 闸门按点数比较（`material_rules_points`）。
  - **语料和教师：**
    - 语料 `source.json` 记 `tick_points`。
    - 交易事实的有利 1/3/5、不利 5/10/20/30，早期不利 10，V1 质量奖励的 10，V1 评级的 MAE / 空间阈值，单位都是跳。
  - **标签：**
    - dreamer `labels.outcome_points` 是每单位几跳，和学生的 `ts.outcome_points` 相同，所有品种一个设置。
    - `labels.outcome_scale` = 跳 × 语料的 `tick_points`，用来把 MAE/MFE 点数换成单位。
    - 学生的 MFE 回答按跳输出，采集时乘以 tick 记成点数，和教师表一致。
  - **观测：**
    - 账户特征的止盈、止损、浮盈距离按 5 / 30 跳缩放，持仓槽的距离按 5 跳缩放。
    - 行情序列、15m context 和 MACD 是对数比值，和点数无关。

### 一份策略对所有品种
- paper 策略写 `entry.product_rules: true`，套用档案的 `rule_overrides`；成本总是档案的。
  - `markets/strategy_v3.yaml` 就是这样一份。
    - FG 用它时，等于原来按品种复制的 `strategy_v3_trend_trail_zone_h2.yaml`（区间到 H+2，已删除，测试用保存下来的 FG501 闸门比对）。
    - SA、RB 用它时，等于 `strategy_v3_trend_trail.yaml`。
  - live FG701 / SA701 的 env 也是 V5 规则加 `product_rules`。
- 训练策略不写 `product_rules`，规则不变（品种差异只用于 paper 和 live）。
- **合约名：** gym 只认 `交易所.品种代码+年月` 这个通用格式；属于哪个品种由档案判断，加品种不改代码。

### 不兼容的地方（2026-11-08 第二轮）
- **旧配置的成本变了：** V1–V2.1 的 formal 配置原来没写成本，用的是主策略里的旧成本（手续费 3，价差 1）。现在统一用档案里的真实成本（2.1 / 0），这些旧配置不再复现当时的结果。V2.2 以后本来就是真实成本。
- **旧输入读不了：** 2026-11-08 以前写出的输入目录、语料、教材（没有 `product.yaml`，或者规则里带成本、`source.json` 没有 `tick_points`），新代码都读不了，需要时重新生成。
- **删除的文件：**
  - `strategy_v3_trend_trail_zone_h2.yaml` 和 v7 zone_h2 两份规则（V7 版本号一起去掉）；
  - `rb/strategy_v3_trend_trail_rb.yaml`；
  - `strategy_v2_1_structural_real_costs.yaml`，它现在和 `strategy_v2_1_structural.yaml` 只差版本名。

## 加一个新品种

1. **写档案** `products/<代码>.yaml`。
   - 成本可以写一份预设文件，也可以直接写在档案里；
   - 一跳点数；
   - 合约名的正则。
2. **查数据**（在 `gym-trading-env/` 下运行，`PYTHONPATH=.:src`）：
   ```
   python -m gym_trading_env.research.product_paper coverage --directions D.yaml
   ```
   有缺失时再运行 `ingest`：调用掘金 `ingest_archive_to_db_v2.py`，已导入的月份会跳过。
3. **生成 paper 输入：**
   ```
   python -m gym_trading_env.research.product_paper build --directions D.yaml \
       --configs ../dreamerv3/configs/thinking_student --out DIR \
       --profile ../dreamerv3/configs/thinking_student/products/<代码>.yaml
   ```
   - **方向文件：** 只读取其中的 `directions.intervals`。
   - **其余照 FG 的 paper：** 交易时段、奖励和 env 设置。
   - **生成的文件：**
     - `markets/<代码>/` 下的合约文件，一跳点数和手数取自档案；
     - env 文件；
     - 主策略 `strategy_quality_tp_v1_<代码>.yaml`，含 `product_profile`（没有成本，成本在档案）；
     - paper 池 `paper_pool_<代码>.json`。
   - 档案的 `paper_pool` 指向这个池。
4. **跑 paper**，用通用入口，见下节。

## 通用 paper 入口（dreamer `thinking_student/paper_run.py`）

```
cd gym-trading-env && PYTHONPATH=.:src:../dreamerv3 python -m dreamerv3.thinking_student.paper_run \
    --out DIR --strategy ../dreamerv3/configs/thinking_student/markets/strategy_v3.yaml \
    --products fg,sa,sp --run FORMAL_RUN --policy pi_0005 --policy pi_0009 [--review]
```

- **exam：** 对每个品种 paper 池里的每个合约，每个策略各跑一次，并行数由 `--workers` 设置，默认 10。
- **报告：** 每个品种一份 `DIR/<品种>/paper_report.md`。
- **`--review`：**
  - 每个品种在 FORMAL_RUN 里登记一个 review 结果，编号自动取下一个空号；
  - 加锁，两个 paper 同时运行也不会取到相同编号；
  - 最后运行 validate。
- **记录：** `DIR/paper_run.json` 写明两仓提交、策略、池和 review 编号；日志在 `DIR/logs/`。
- **重跑：** 已完成的 exam（`exam.json`）会保留，重跑时继续剩下的。

## 注意

- **学生没见过新品种的波动：** 策略是在 FG 上训练的。距离观测按跳缩放，但以跳计的波动各品种不同（15m 波幅中位数 SP 约 8 跳、FG 约 5 跳），观测仍会偏离训练范围。
- **按跳换算不等于按波动换算：** 例如 15m 波幅中位数 SP 约 16 点、FG 约 5 点。点数参数按跳换算，不要在 paper 上挑参数。
- **方向是事后标的：** 方向文件是用户事后标注的，paper 结果更接近"方向正确"时的上限。
