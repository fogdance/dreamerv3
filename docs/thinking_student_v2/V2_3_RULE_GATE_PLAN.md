# V2.3 规则闸门 + 真实成本：规则决定在哪里交易，大脑决定什么时候交易

> 状态：用户 2026-11-05 批准的实施计划（TODO G-26）。分支 `integration/ts-next`；证据在 `/data/logdir/fg_single3/thinking_student/V2_3_DEV_20261105/`。实施中的改动和结果在完成后补到文末。


## Context

### 用户要求
- 方向由外部判断。
- 下跌趋势里，价格反弹到 15 分钟前高附近、并且出现 2 倍以上放量，学生才允许做空；上涨趋势同理。
- 放量之后的多个有效位置都可以入场。
- 学生要学的是：在符合规则的位置里，哪个时刻 ADD 更合适，或者继续 WAIT。
- 不在教材里加大量错误示范，也不要求学生自己发明这条规则。
- 学生要尽量按规则入场；没有按规则入场的，要找出根本原因并修复。

### 根本原因（在 V2.2 上实测）
1. **env 没有闸门。**
   - `entry_env._can_add()` 只检查账户、方向和时间。
   - `strategy_v2_1_structural.yaml` 写明 "nothing here masks or gates an ADD"。
   - 结果：TRAIN 上学生 87% 的入场不在规则位置。
2. **教材只覆盖 2.6% 的分钟。**
   - ADD 9,429 条，全部在规则位置。
   - WAIT 3,355 条，全部在触线之前，规则位置上一条都没有。
   - 其余 97.4% 的分钟没有策略损失，学生只能外推：非规则分钟的 ADD 率是 6.5%。
3. **学生看不全规则。**
   - 17.6% 的规则位置用的线比 48 个 15m 桶的上下文更老（正例里是 12.3%）。
   - 触线和离开的状态也可能落在 60 分钟窗口之外。
4. **env 的交易不是规则的交易。** 无状态止损取"成交价上方最低的线"，再加 R ≤ 10，让 29% 的材料正例进不了教材：
   - RISK_ABOVE_MAX 3,831 条；
   - STOP_LINE_DIFFERS 94 条。
5. **次要原因：** L6 正确集和 L4 的 4 倍加权只看结果，不看是否守规则。

这些统计是计划期间在命令行里算的，没有留文件。第 0 步先把它们落盘作为证据。

### 修复思路
- 把规则做成 env 的 ADD 掩码：规则位置之外 ADD 不合法，合规由结构保证，不靠学生学。
- 原因 1、3、5 因此不再影响合规。
- 原因 2 在教师里修，原因 4 在止损里修。

### 用户已定
- **"正例之前的位置标 WAIT"：**
  - 同一次触线里，第一个正例之前的规则位置标 WAIT；
  - 没有正例的触线不标；
  - 不加错误示范。
- **"SL = H+5，保留 R ≤ 10"：**
  - 止损用本次触线自己的 H+5；
  - 掩码只放行计划 R ≤ 10 的规则位置，也就是决策报价在 [H−5, H] 之内。
- **"真实的交易成本，无滑点"：** 1 手玻璃 1 点盈利 20 元，开仓 2.1 元，平仓 2.1 元。
  - 现在 env 的 spread 1.0 让开仓、平仓各多付 1 点。WHY_NO_PROFIT 报告里每笔约 2.3–2.8 点的摩擦，大头就是它。
  - 改法见 B2。
- **止盈成交："保留穿价 1 跳"。** 止盈和止损的成交规则都不变。

## 规则位置（以 SHORT 为例；LONG 是镜像，价格取负）
决策分钟 s 同时满足以下三条：
- **线：** 有一条有效的 15m 前高 H。
  - pivot 2/2；
  - 不超过 115 个桶，且没有被突破；
  - 已经有过一根 1m 收盘 ≤ H−10（已离开）。
- **触线进行中：** 这次触线从一根 High ≥ H−5 且 RV ≥ 2 的 bar 开始，还没有因为离开、失效或过期而结束。
- **报价：** s 是有效的成交时钟分钟，决策卖价落在 [H−5, H] 之内。spread 改为 0 之后，决策卖价就是 Close。

这就是现有的 V3 扫描 `ts_v2_positive.scan`，只是 zone 从 7 改成 5。

## 步骤

### 0. 根因证据落盘，以及新成本下的基准
- **根因：**
  - 新建 `V2_3_DEV_20261105/root_cause/root_cause.py`，用 CPU、只读运行，对象是 V2.2 的语料、材料和 1x run；
  - 重算上面 5 条原因的数字，结果写成 `root_cause.json` 和 `root_cause.md`。
- **新成本下的基准：**
  - 在 WHY_NO_PROFIT 目录新增 `real_costs.py`，复用 `scale.py` 的 `first_touch`；旧脚本和 `scale.json` 不动。
  - 模型：开仓按 Open[t+1] 成交；止盈要越过 1 跳，开仓那根 bar 只看收盘；止损触价成交；手续费从成本预设读取（0.21 点）。
  - 计算每分钟按方向入场、止盈 2R（R = 5 / 10）时，TRAIN、DEV、PAPER 各自每笔的净点数，写成 `real_costs.json`。
  - `REPORT.md` 末尾加一节"用户 2026-11-05：真实成本"。

### A. gym `research/ts_v2_positive.py`：env 和材料共用一个规则扫描
- **拆出核心 `rule_pass(a, levels, cfg, end)`：** 从 `scan` 的因果单次遍历里拆出来。
  - 返回 events，以及每分钟的 `event_of[s]`、`in_zone[s]`、`active[s]`。
  - `end` 是 BOUNDARY 边界：
    - 材料传 `min(flat_end, n−1)`，和现在的 `s >= flat_end or s + 1 >= n` 等价；
    - env 传 `n`，这样 live 的最新一分钟也能放行。
- **`scan` = `rule_pass` + `_candidate` + `_wait_rows`：**
  - `_wait_rows` 移到遍历之后执行；
  - 它只读已经定下的 `active[<t]`，所以结果不变。
- **新增 `rule_cursors(env, cfg)`：**
  - 组成：`market_arrays` + `prior_highs` + `rule_pass(end=n)`。
  - 返回每分钟的 `open`、`H`（short space）、触线编号，以及第几次触线。
  - env 调用它，做法和已经共享的 `pivot_levels` / `departed_at` 一样。
- **V4 材料：**
  - `ACTIVATION` 加一项 `V2_POSITIVE_ENTRY_V4: touch`。
  - 配置里 `timeline.wait_in_rule: true` 时，`assemble` 对每个有正例的触线，给第一个正例之前的每个候选分钟写一行 `WAIT_IN_RULE`。
  - 如果激活那一分钟就是这样一行，它代替原来的 ACTIVATION 行。
  - `TEACH` 加这个类型，`wait_note` 加对应的英文说明。
- **防回归：** 用 V3 配置重生成 FG205 / FG401，teacher 和 audit 文件必须逐字节相同。

### B. 配置（`dreamerv3/configs/thinking_student/`）
- **新 `v2_positive_entry_v4.yaml` / `v2_positive_entry_v4_long.yaml`：** 在 V3 基础上只改四处：
  - `zone_points 5`；
  - `max_risk_points 10`；
  - `timeline.wait_in_rule: true`；
  - `costs: markets/costs_fg_real_20261105.yaml`（见 B2）。
- **为什么这样改：**
  - 掩码只放行 [H−5, H]，所以材料的候选正好就是合法位置。
  - 正例的成交价也必须在 [H−5, H]，R ≤ 10，env 的交易和材料逐笔一致。
  - RISK_ABOVE_MAX 和 STOP_LINE_DIFFERS 只可能由跳空引起。
  - V3 文件不动。
- **新 `markets/strategy_v2_3_rule_gate.yaml`：**
  - version `V2_3_RULE_GATE_R10`；
  - 在 V2.1 structural 基础上加 `entry.rule_gate: {SHORT: ../v2_positive_entry_v4.yaml, LONG: ../v2_positive_entry_v4_long.yaml}`；
  - 再加 `costs: costs_fg_real_20261105.yaml`（见 B2）；
  - 旧策略不动。
- **新 `formal_v2_3.yaml`：**
  - 相对 formal_v2_2 只改 `corpus_root`、`run_root`、`strategy`、`review.label`；
  - 其余键和 V2.2 1x 相同，所以两次 run 只差闸门和教材。
- **文档：**
  - `docs/thinking_student_v2/DESIGN.md` 加一节 V2.3；
  - `CORPUS_GUIDE.md` 加 V4 材料和语料的生成命令。

### B2. 交易成本改为真实成本（用户 2026-11-05）
- **新成本：**
  - 开仓 2.1 元/手，平仓 2.1 元/手；平今也按 2.1 元，因为用户没有区分。
  - spread 0、slippage 0：开仓按 Open[t+1] 成交，止盈按止盈价成交，止损按止损价成交。
  - 不变的部分：1 手 1 点 20 元（lot_size 20），保证金 0.1。
  - 3 手一个来回的手续费是 12.6 元，合 0.21 点；现在是 18 元，合 0.3 点。
- **成交规则不变（用户选择）：** 所以 `single3_exit_decision` 和 `fg_single3_oracle` 都不用改。
  - 止盈要越过止盈价 1 跳；
  - 开仓那根 bar 只看收盘价判断止盈；
  - 止损触价即成交，跳空时按开盘价成交；
  - 同一根 bar 同时触到止盈和止损，算止损。
- **怎么配置：沿用原来的方式，成本只写在一个文件里。**
  - **原来：**
    - 成本来自用户的成本预设，`CZCE_FG:` 格式，含 lot_size、leverage、trading_fee_per_lot、is_round_turn、spread；
    - 由 `fg_entry_v1.configured_fg_costs` 转成 master 的 `costs` 块，同时记下 source_path 和 sha256；
    - 原来那份 `1.yaml` 已经不在了，但 master 里的值还在。
  - **新预设：** 新建同格式的 `markets/costs_fg_real_20261105.yaml`。
    - 取值：lot_size 20、leverage 10、trading_fee_per_lot 2.1、is_round_turn true（开仓、平仓各收一次）、spread 0；
    - `configured_fg_costs` 本来就把 slippage 设为 0。
  - **引用：** V2.3 策略和 V4 材料配置都只写 `costs: <预设路径>`，不写数值。以后改成本，只改这一个文件。
  - **`write_inputs`：**
    - 加一个 `costs` 参数：用 `configured_fg_costs` 读预设，替换语料自带的 `single3.yaml` 里的 `costs` 块，格式和来源字段与原来相同；
    - 材料的 base env 和交易 env 也走这条路径。
  - **兼容：**
    - 没有 `costs` 的配置（V2.2 及以前）照旧用 master 里的成本；
    - 旧文件都不动，包括 `strategy_quality_tp_v1.yaml`。
  - **一致性检查：** env 读取 `rule_gate` 时，检查规则文件的成本和 env 实际成本一致，不一致就报错。
  - **奖励尺度：** `reward_scale_cny` 保持 618，和 V2.2 相同，免得奖励尺度又成为一个变量。
- **影响（随机游走，止盈 2R，R 5–10）：** 每笔摩擦从约 2.5 点降到约 0.5 点，其中穿价 1 跳约 0.31 点、手续费 0.21 点。

### C. gym env `envs/entry_env.py`：可选的闸门
- **读取 `entry.rule_gate`：**
  - 只能和 `prior_15m_extreme` 一起使用。
  - SHORT / LONG 两个键各自指向方向相同的规则文件，用 `load_config` 读入。
  - 检查 `stop_above_points` 和 `prior_high` 与 stop_rule 相同，保证用的是同一组线。
  - 检查 `stop_above + zone ≤ max_risk`。
- **缓存：**
  - `_gate(side)` 缓存 `rule_cursors` 的结果；
  - `_load_market` 时清空（和 `_lines` 一样），所以 `continue_data` 和 live 会重算。
- **掩码：** `_can_add()` 先查 `_gate(direction)['open'][current_step]`，再做原来的检查。
- **止损：** 有闸门时，`planned_exit(cursor, fill, side)` 用的线是这一分钟所在触线的 H，不再取"成交价上方最低的线"。
  - R = H + 5 − fill。
  - 正常情况 → `RULE`。
  - R > 10 → `CAPPED`，R 取 10。
  - 成交价已经越过止损（R ≤ 0，只会在跳空时出现）→ `GAP`，R 取 10。
  - plan 里加上触线编号 `rule_event` 和第几次触线 `touch`，随 `item['planned_exit']` 进入交易记录。
  - 在非规则分钟调用会报错。
- **其余：**
  - 不配 `rule_gate` 的 env 行为完全不变；
  - `_stop_rule_config` 的文档字符串同步更新。

### D. gym 语料 `research/ts_v1_corpus.py`
- **`load_strategy`：**
  - 按策略文件所在位置解析 `rule_gate` 路径；
  - 校验：要求 BOTH 加 prior_15m_extreme，SHORT 和 LONG 两份文件都在。
- **`write_inputs`：**
  - 把两份规则文件复制成 `rule_gate_short.yaml` / `rule_gate_long.yaml`，用相对路径引用（和 contract.yaml 一样），inputs 目录仍然自包含；
  - DEV 和 PAPER（exam `--strategy`）走同一条路径。
- **其余不用改：**
  - facts 的 KNOWN_MASK 会让 Outcome 事实只出现在规则位置；
  - P0–P4 行为读 `action_mask`，自然只在规则位置 ADD。

### E. gym 教师 `graded_teacher` 和 review 导出
- **有闸门时：**
  - **ADD = 正例，必须合法。**
    - SL ≠ obs_sl 直接报错，不再作为 STOP_LINE_DIFFERS 丢弃；
    - R > 10 仍然计入 RISK_ABOVE_MAX。
  - **WAIT = 保留下来的触线里的 `WAIT_IN_RULE` 行，也必须合法。** ready_at 和现在一样，是这次触线最后一个保留正例的 TP 出场。
  - **触线之前的 WAIT 行不再做标签：**
    - 闸门下这些分钟本来就不能 ADD，也不进损失；
    - 计入 `dropped.WAIT_OFF_RULE`，review 里也不出现。
  - **没有正例的触线不标。**
  - **summary 加两项：** 规则位置数；标签覆盖率 = 标签数 / 规则位置数。
- **无闸门时：** 行为不变，V2.1 / V2.2 仍可原样复现。
- **`review_export.py`：**
  - WAIT 类型改为从 ts_v2_positive 导入，包含 WAIT_IN_RULE；
  - WAIT_IN_RULE 的 details 加决策报价、计划 R、激活后分钟（英文键）；
  - 学生交易的 details 加触线 H、第几次触线和 stop_source。

### F. dreamer
- **`exam.py`：**
  - **合规率：** 学生指标加 `rule_compliance` = 带 `rule_event` 的入场 / 全部入场。闸门策略下它小于 1 就直接报错，因为那只可能是 bug。
  - **其他指标：** 加 `stop_source` 计数，以及成交价落在 [H−5, H] 之内的比例。
  - **基准：**
    - 随机账户不变，它现在就是"规则内随机时机"；
    - 新增一个"每个规则位置都 ADD"的账户（probability 1，单账户）；
    - 报告学生相对这两个基准的优势。
- **`paper.py`：** 报告里列出这三项指标和两个基准。
- **live.py：** 冻结，不改。闸门对 live 是否安全由 G2 测试保证。

### G. 测试
- **gym 新 `tests/entry_v1/test_rule_gate.py`（少而精）：**
  1. **一致性：** 真实合约，SHORT、LONG 各一个。
     - env 放行的分钟等于 V4 材料的候选分钟（边界之前），H 相同；
     - 空仓时，规则分钟 mask[1] = 1，其余分钟为 0。
  2. **live：**
     - 把市场截到 k 根后重算闸门，结果等于全量闸门的前 k 个；
     - 要覆盖第 k 根正好是规则位置的情况，即最新一分钟是放行的。
  3. **止损：**
     - 规则 ADD 的 SL 是 H+5（SHORT）/ L−5（LONG）；
     - 构造的跳空分别得到 CAPPED 和 GAP。
  4. **教师：**
     - 闸门下每个标签都合法；
     - WAIT_IN_RULE 只出现在第一个正例之前。
  5. **成本：**
     - V2.3 策略下，开仓价等于 Open[t+1]；
     - 一笔已知交易的净盈亏等于毛利减 3 × 4.2 元；
     - 换一份手续费不同的预设，env 的成本跟着变（验证可配置）；
     - V2.2 策略的 env 成本和 identity 都不变；
     - 规则文件的成本和 env 不一致时报错。
- **重跑：**
  - 原来的 11 个 gym 测试文件和 review export 测试；
  - dreamer GPU 测试 c4、s5、v2_minimal，加上 exam 新指标的测试。
- **lint：** ruff F/E9，和原来的计数对比。

## 执行顺序与验证
证据统一放在 `/data/logdir/fg_single3/thinking_student/V2_3_DEV_20261105/`。

1. **第 0 步和 A：**
   - 根因证据和新成本基准落盘，结果先报给你；
   - 然后用 V3 配置重生成 FG205 / FG401，逐字节一致。V3 配置没有 `costs`，仍然按旧成本。
2. **代码与提交：**
   - B 到 G 全部完成、测试通过后，对照最新的设计文档（`~/Documents/work/research/thinkstudent`）检查一遍，不清楚的地方先讨论再改；
   - 然后在 `integration/ts-next` 两仓提交，因为语料要求两仓都是 clean；
   - TODO.md 一起提交：已有的 G-25 加上新增的 G-26。
3. **V4 材料（CPU 后台）：**
   - 生成 9 个 SHORT + 9 个 LONG 合约的材料，合并成 `V2_3_POSITIVE_ENTRY_BOTH_20261105`；材料的 Native 交易按新成本计算，逐分钟复核全部通过；
   - 报告规则位置、正例、WAIT_IN_RULE 的数量，以及和 V3 的差别；
   - 预估：规则位置约 5.4 万个，占全部分钟的 11%；V3 在 zone 7 下"第一个正例之前的规则位置"有 5,543 个，V4 重新计数。
4. **语料（CPU 后台）：** 构建 18 个语料 `corpus_v2_3_20261105`，然后：
   - **教师检查：**
     - 标签全部合法；
     - STOP_LINE_DIFFERS 为 0；
     - 行为账户的 ADD 100% 在规则位置；
     - 报告标签覆盖率。
   - review export 和 validate。
5. **短 smoke（GPU 后台）：**
   - 学生采集到的 ADD 100% 在规则位置；
   - 统计 stop_source 的分布；
   - CF 分支存在，而且只出现在合法分钟。
6. **1x 和 paper：**
   - 我给出 1x 命令，由你在 tmux d3 启动；
   - 跑完后用同一策略在 FG501–FG605 上做 paper，报告合规率，并比较学生、规则内随机时机、每个规则位置都 ADD 三者。

## 不在这一步
- **成交规则不改：** 止盈穿价 1 跳、开仓 bar 只看收盘、止损触价、同一根 bar 先算止损，都按你的选择保留。
- **报告第 9 节剩下的两个决定**（方向怎么处理、出场尺度）仍然等你来定。成交模型这一项已经由你定了：真实成本。
- **不给学生加规则状态的观测**，例如离 H 的距离、第几次触线：
  - 有了闸门，合规不再依赖学生能不能看见规则；
  - 如果 paper 显示时机选择没有技能，这会是下一步的候选。
- **旧数据不动：**
  - 旧材料、旧语料、旧 run 都不动；
  - TEST 不读；
  - PAPER 不用于选择或调参。
