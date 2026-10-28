# Thinking Student V1 — 开发交接摘要（B1/B2/C1 + F-01/F-02）

> 来源：同目录下 A1–A6 冻结大脑、C2、C3、ABC 最终 Review，以及 `brain.md` 中 B1/B2/C1 的会话结论。本文只做压缩，不新增设计。
> 冲突时的优先级：学生能力与数学含义按 A1–A6；教学事实与权限按 B1/B2/C1；仓库取舍按 C2；开发顺序按 C3。

## 唯一主线
完整学生自主交易 → Native 账户事实 → 合法教师反馈 → A6 分目标参数更新 → 权重继承 → 下一版学生自主交易与独立考试。

## 五个逻辑接口（C1）
| 接口 | 输入 | 输出 |
|---|---|---|
| Student Policy | 因果市场、Native 账户、匹配的递归状态、完整模型版本 | Outcome、Q、思考证据、动作分布、最终动作 |
| Native Execution | Native 状态、合法动作、下一段市场 | 下一账户状态、交易记录、Reward |
| Teacher Annotation | 学生答卷、原始账户快照、TRAIN 未来、冻结 π_k | ADD 事实、同策略 CF 价值、严格 L2 参考 |
| Learning Update | 学生经验、成熟有效反馈、来源/版本 | 分目标更新、下一版完整学生 |
| Independent Exam | 冻结完整学生、DEV（VALIDATION）或合成行情、Native 账户；封存 TEST 只用于用户授权的最终评价 | 无教师参与的完整交易轨迹 |

## B1 教学题（一题六部分）
1. **Question**：因果 Obs、Native 账户、Mask、时钟、决策前状态、策略版本。只有同一账户初态才算同一题。
2. **Student Answer**：事前 Outcome、Q、动作分布、π 的提议动作、μ(a|s)、实际执行动作、是否探索。
3. **Student Experience**：主账户的真实 Native 经历。主账户只执行学生动作，老师永不改写。
4. **Teacher Evidence**：
   - ADD 后果：TP/SL 首次事件、时间、右删失、MAE、early10、Quality。
   - 同初态 ADD/WAIT 双分支：后续使用同一冻结完整 π_k，同 Reward，同评价范围。
   - 独立的严格 L2 Oracle。实现中（2026-10-28 起）L3/L5 的示范改由分档入场老师给出（`graded/`，见下文“分档教学与完全重训”），严格 Oracle 只做 P4 计划。
5. **Learning Feedback**：分别进入 `L_outcome`、`L_q_real`、`L_q_cf`、`L_reason`、`L_policy_demo`，不合并成单一 Reward；CF 偏好不再给 Actor 加第二份 Loss。
6. **Validity & Provenance**：每类反馈各自带 `(valid, ready_at, source)`。来源取值为 `STUDENT_EXECUTED_NATIVE`、`VERIFIED_CF_NATIVE`、`HINDSIGHT_ORACLE_TEACHING`、`NATURAL_CORPUS`、`MODEL_IMAGINED`。实现中的示范来源为 `HINDSIGHT_GRADED_TEACHING`（2026-10-28）。

其他规则：
- CF 状态取 `VALID_COMPLETE`、`VALID_ESTIMATED`、`NOT_COMPARABLE` 之一，允许老师弃权。
- L2 与 CF 可以不一致，两者都保留，不强行合成一个标签。
- 不允许依据单条后验路径断言“学生必然能预知”。

## B2 课程
- 教材由四部分组成：自然教材、P0–P4 多行为教材、学生自采经历、错题补课。补课只提高学习权重，不改变自然市场分布。
- 两种模式：
  - **离线 TRAIN**：可以使用全部合法后验。
  - **时间顺序实习**：只能消费 `ready_at ≤ 当前模拟时刻` 的反馈。
- 考试区间封存，从不进入训练。
- 探索时必须记录行为策略。行为策略 μ ≠ π_k 的轨迹，不能直接当作 `Q^{π_k}` 的 MC 真值。

## C1 修订
- 非法 ADD（持仓中或 mask=0）时，Outcome 不输出预测，只输出有效位 0。
- 模型升级后，RSSM 记忆要用新模型从合法历史重建，或按真实边界初始化。
- 学到的参数必须真的进入正式动作路径；训练、采集、考试走同一个 `student.policy`。

## 执行模式与策略改进（A1/A5/A6，2026-10-22 更新）
- 正式策略 π_k 为确定性 `DETERMINISTIC_MASKED_ARGMAX`：ADD 合法且 z>0 才 ADD，否则 WAIT，z==0 时 WAIT。S2 自主轨迹、S3 CF 续行、S5 考试用同一规则；MaskedSoftmax 只给概率与诊断。
- 探索另设行为策略 μ（ε 或温度采样），记录 π 提议、μ(a|s)、实际动作、探索标记和快照。μ 轨迹不能直接当作 `Q^{π_k}` 的回报。
- 策略改进只有一条路：学生 Native 经验 / 有效 CF → `Q^{π_k}` → ΔQ 锚点 → 新完整 π_{k+1}。V1 不启用 `L_policy_pref`、`L_align`、`L_policy_rl`。
- 双路径残差是 L3 用合法示范训练的行为先验。π0 的 `β_Q` 不贡献动作，Q 有了监督后逐步启用。`Q→Reasoning` 的输入在 S4 前保持中性（零初始化），之后刷新先验时使用中性 Q 锚点的教学视图。
- 冻结快照 `policies/<name>/policy.json` 记录推理设备和计算精度（S2 的 π0 为 cuda / bf16）。bf16 与 f32 在 z≈0 附近可能翻转动作，所以 CF 续行和考试都按快照记录的设备加载（`run.load_policy`）。

## F-01（门控梯度）
- 门控 `g_alpha` 的输入使用 `SG(e_t)` 和 `SG(q_t)`。Reasoning 与 ΔQ 锚点同样使用 SG。
- L3/L5 只更新决策参数组。未被授权的上游（Market、Account、Fusion）不进入该阶段的优化器。

## F-02（长期 Q 的 CF 尾部）
- 两个分支都要满足：同一起点、同一冻结完整 π_k、同一奖励合同、同一评价终点或折扣时长。
- 首笔 TP/SL 不是账户价值的终止。遇到人工截断时，只能接语义一致的 `V^{π_k}` 尾值估计，或者标记为 `NOT_COMPARABLE`。

## 工程约束（C2/C3）
- 两个仓库都不兼容旧 S3 功能。
- Checkpoint 和教材默认继承，由开发人员处理语义适配，不设代码或数据 Hash 门禁。
- 每个切片完成的标志是一段可运行的功能、真实的学习活动和一项薄验证。不以盈利或 AUC 设门槛。

## 实现状态（2026-10-23：Review 修正、教材重建、S3、S4、C4）
详见 `R_M_RECORD_20261023.md`、`S3_RECORD_20261023.md`、`S4_RECORD_20261023.md` 与 `FORMAL_TRAINING_IMPLEMENTATION.md`。

**账户输入与 Outcome**
- `account_native` 14 维，字段表为 gym `ts_v1_account.ACCOUNT_NATIVE_FIELDS`。新增 4 个字段：
  - `to_add_deadline`、`to_forced_exit`：事前已知的日历分钟；
  - `funds_cover`、`margin_cover`：资金充足时饱和，所以空仓账户的输入只取决于 cursor。
- Outcome 34 类：TP / SL / OTHER_EXIT 各 11 个时间桶，加 OPEN_AT_H（H=2760 分钟）。
  - RIGHT_CENSORED 按删失处理，不单独成类。
  - 答卷字段为 `p_tp_h` / `p_sl_h` / `p_other_h` / `p_open_h`。
  - evidence 43 维：34 类概率、2 个风险、3 个 MAE、3 个 MFE、1 个 Quality（MFE 为 2026-10-28 新增），未受监督的 latent 不在其中。

**环境**
- TS env 没有 CHUNK_END（`chunk_steps=0`）。
- 市场记忆从第一分钟连续运行到真终止。训练窗口默认从零状态起步并有 burn-in，这不代表数据断档。
- 正式 run `formal_v1_carry` 中，L2–L5 窗口从 MarketCarry 存储的连续路径 carry 起步，没有 burn-in（见下文 C4 一节）。

**教材**
- 按“合约 × cursor”分表：`market/`、`facts/`、`strict/`、`graded/` 每个合约一份；`behaviors/<P>/` 只存 P0–P4 的账户侧数据。
  - `graded/`（2026-10-28）：分档老师的示范和 MFE 目标，阈值来自 `configs/thinking_student/entry_grades.yaml`。
  - `strict/` 只做 P4 计划。
- 标签只取 P0 账户。只在合法空仓、权益不低于资金下限的行上给标签。
- `source.json` 只记来源、`ready_at` 和计数，没有 Hash 或 manifest。

**折扣与 CF（F-02 的实现）**
- 折扣：γ = 332/333，按有效市场分钟计。
  - 决策 t 的回报为 G(t) = r_t + γ^{advance(t,t+1)} · G(t+1)，终点是合约真终止。
- 分支的终点：分支与 π_k 自己的完整 TRAIN 正式轨迹在汇合点 c 汇合。汇合点需同时满足：
  - 两边同时空仓；
  - `position_state` / `action_mask` / `account_native` 逐位相等；
  - 资金不构成约束。
- 回报精确，不需要尾值：G = Σ_{t≤i<c} γ^{advance(t,i)} · r_i + γ^{advance(t,c)} · G_ref(c)。
- 汇合处由同一个编译后的策略复核一次决策。
- 状态：
  - 汇合或到达真终止 → `VALID_COMPLETE`；
  - 数据截断或超过步数上限 → `NOT_COMPARABLE`，不给数值标签；
  - 本版不产生 `VALID_ESTIMATED`。
- CF 必须在快照记录的设备和精度上运行。每个 worker 先重放主轨迹，动作或奖励与采集不一致就报错。

**Q 目标与 L4**
- Q 目标只属于产生它们的 π_k：
  - `q_real`：已执行动作的完整回报，来源 `STUDENT_EXECUTED_NATIVE`；
  - `q_cf`：fork 点另一动作的分支回报，来源 `VERIFIED_CF_NATIVE`。
- L4：
  - `opt_l4` 只更新 value 组。
  - 三项损失：`L_q_real`、`L_q_cf`、`L_q_pair`（ΔQ 回归到 ΔG）。上游全部 SG。零状态开窗带 burn-in，carry 开窗不带。
  - 不用 target copy（回报完整），没有 Actor 损失。
  - 留出评估在同样的行上给出常数答案（训练段目标均值）的 MSE，以及 `constant/sign_agree`（每对都取训练段 ΔG 均值的符号），与模型的 `q/sign_agree` 对比。

**下一版快照与加载**
- π_{k+1} 的快照写入 β_Q 和 κ_Q；`auto` 时 κ_Q = 训练段有效对 ΔG 的标准差。
- Q 只通过 β_Q · clip(ΔQ/κ_Q, ±3) 进入动作。
- 加载快照：
  - 非优化器参数必须完全一致；
  - 后加阶段的优化器状态（例如旧快照没有的 `opt_l4`）重新初始化，并逐项列出。

**S4：先验刷新与一轮策略改进**（`PLAN_S4_20261023.md`）
- L3 = `L_policy_demo` + `L_reason`，两者用同一批示范行（2026-10-28 起为分档老师的示范）、同样的类别权重和 burn-in：
  - `L_policy_demo` 在教学视图 `[0, correction]` 上算，anchor 不参与。所以 β_Q 取任何值，先验的梯度都一样，残差不会学去抵消 Q。
  - `L_reason`：`reason_pref` 回答“示范老师在这一行示范 ADD”，用 BCE。2026-10-28 前是严格 TRAIN Oracle 的“ADD 严格优于 WAIT”，现在是分档老师：GOOD 及以上示范 ADD，POOR 示范 WAIT。梯度只到 Reasoning 通路（`reason_in`、`reason_q`、`reason_norm`、`reason`、`reason_pref`）。`reason_pref` 不进入 z。
  - L3 评估同时给出两项的常数答案（`constant/policy_demo`、`constant/reason`）：p = 0.5，即同样加权下的 log 2 × 平均权重。
  - 评估还给出先验视图和 run 内正式 z 的 ADD 率（`policy/add_rate`、`policy/formal_add_rate`）。训练 run 的配置中 β_Q = 0，β_Q 和 κ_Q 只在 `freeze` 时写入快照，所以 CLI 流程里两者相等。π_{k+1} 带锚点的行为由冻结后的诊断和正式采集衡量。
- Q→Reasoning：
  - Q 受过 L4 监督后，停止梯度的 Q 上下文不为 0。`reason_q`、`gate_q` 从这两项独立示范 Loss 开始学习。
  - 没有任何以 ΔQ 或 ΔG 为目标的监督。
- 一轮改进在同一个 run 目录完成：`train-value`（从 π_k 出发，用它的 CF）→ `train-policy`（先验刷新）→ `freeze` π_{k+1}。
  - `freeze` 从该 run 所有 checkpoint 中取每类报告（`train_value`、`train_policy`）的最新一份。
  - `policy.json` 用 `reasoning.reason_pref` 说明这个头的含义和训练来源，取代原来的 `untrained`。
- FG809 实跑（`S4_RECORD_20261023.md`）：
  - pi1 → pi2 一轮跑通，CF(pi2) 已就绪，可作为下一轮的教材；
  - ΔQ 和先验在留出集上都不优于常数，所以 pi2 是可运行的候选，不是更好的策略；
  - 同一状态下换了后续策略，约 1/5 的 ΔG 改变符号，Q 教材只属于产生它的 π_k。

**C4：正式训练能力**（`FORMAL_TRAINING_IMPLEMENTATION.md`、`PLAN_C4_20261023.md`）
- **配置只在 `dreamerv3/configs/thinking_student/`：**
  - `student.yaml`、`entry_grades.yaml`（分档阈值）；
  - `formal_v1_graded.yaml`、`formal_v1_graded_10x.yaml`（2026-10-28 的新结构）；
  - `formal_v1.yaml`、`formal_v1_carry*.yaml`、`formal_v1_l1_l2_10x_continue.yaml`、`c4_smoke_2c.yaml`：旧结构的运行记录；

  - `markets/pool.json`：18 个 TRAIN 行 + FG409 VALIDATION 行，TEST 不列入，另有 env、合约、策略文件。
  - 教材 `inputs/` 自包含，源码不写死路径。
- **正式学生 size50m：** 市场脑继承自 S3-V2 E_ONLY step 1M。fusion 层为 MACD 加宽零行；TS 各头新建。
- **多合约学习：**
  - `MultiCorpusWindows` / `MultiQWindows` 先按合约抽、再在合约内抽窗口，窗口不跨合约。
  - L3 类别权重在全部合约的示范上统一计算。
  - κ_Q = 同轮全部有效 CF 对的合并 std，各合约等权；有效对少于 2 个时 β_Q 保持 0（`Q_ANCHOR_NOT_READY`），下界只替代真实成对样本的零方差或过小尺度。
  - 混入不同 π_k 的 CF 一律拒绝。
- **编排 `formal_train`：**
  - 命令：`pretrain [--until Lk]`、`iterate`、`evaluate`、`run`、`status`。
  - 每一步是一个现有 CLI 的子进程。
  - 完成与否每次都从磁盘判断：阶段剩余次数 = 预算 − `progress.json`；已完成的采集、CF、考试跳过；冻结策略不可覆盖。
  - 每轮开始前先补完最新冻结策略缺的考试，再看轮数上限或开新轮。
  - 某阶段的后续阶段或 π0 已经建立在它上面时，再给它加预算会被拒绝。
- **Teacher-free 考试 `exam`：**
  - 标签：`DEV`（VALIDATION 行，与 TRAIN 同一条 `write_inputs`）、`TRAIN_SMOKE` 或 `DEV_NOT_CONFIGURED`。
  - 随机入场基线：同样的分钟、同样的 Native 规则、相同的入场率。
  - 指标附定义；`best_dev` 按 DEV 每分钟 Reward 选取。
- **L5 = S4 先验刷新。** L6 见下文 S5 节；没开 L6 的轮记 `L6_SKIPPED_BY_CONFIG`。
- **有限预算的精确 CF（用户 2026-10-24；Astra 修复 2026-10-25，2026-10-27 集成为唯一正式 CF）：**
  - 耦合：EXACT_STATE 要求未来相关的 Native 状态和策略状态（carry、`n_actions` 等）都相同；FUNDS_SAFE 沿用原资金不等式，并要求策略状态和控制项相同。
  - 每个合法分叉先探测至多 `cf_probe_steps`（64）步；`cf_budget` 是未闭合探测的续算配额（分层抽样），从保存的状态接着算，合计不超过 `cf_max_steps`。
  - 状态 `UNAVAILABLE_BUDGET/PROBE_UNFINISHED_BUDGET`：探测没有结束、也没被抽中续算，所以没有回报、TERMINAL 和 Q 标签。
  - 局部补算：`cf --cursors … --base-cf OLD --out NEW`，旧的有效标签原样继承。
  - `meta.json` 增加 `forks.undecided`、`budget`（各层数量）、`end_status`（含 MARGIN_CALL）和 `snapshots`（输出完成后删除的已结束分叉快照数、字节数，以及保留的未续算探测快照数）。
  - 某合约的有效对覆盖率低于 `cf_min_coverage` 时，本轮跳过 L4，`freeze --value-anchor-of π_k` 保留 π_k 的 β_Q、κ_Q（`L4_SKIPPED_ANCHOR_KEPT`）。
  - 探索配置 `iteration.cf_coverage: exploratory`：覆盖不足时 L4 仍用已有的有效对训练，轮记录写 `S3_Q_TARGETS_EXPLORATORY` 和 `L4_coverage`。

**S5：离线教学循环 L6**（`PLAN_S5_20261024.md`；只实现 OFFLINE_TRAIN）
- **B1 教学题**（`feedback.py`）：
  - π_k 的一份 collection 中，每个决策行一道题。
  - 四类反馈各带 source、ready_at（cursor）和 valid：
    - TEACHER_ADD_FACT；
    - STUDENT_EXECUTED_NATIVE；
    - VERIFIED_CF_NATIVE：NOT_COMPARABLE 和 UNAVAILABLE_BUDGET 弃权；
    - HINDSIGHT_GRADED_TEACHING（2026-10-28 起，替代 HINDSIGHT_ORACLE_TEACHING）：分档老师的 ADD/WAIT 示范；ready_at 为出场分钟，日内 TP 为止盈后空间窗口的末尾。
- **错题与正确题**（正确集：用户 2026-10-28）：
  - 后果错：答卷给实际结果的概率低于 `outcome_error_prob`（0.1）。后果对：这一概率不低于 `outcome_correct_prob`（0.5）。
  - 决策错：执行的动作比另一个动作差，超过 `decision_error_margin`（1.0）× 同轮 κ_Q。决策对：执行的动作好出 `decision_correct_margin`（1.0）× κ_Q 以上。
  - 示范错 / 示范对：决策错 / 决策对之中，分档老师示范的正是 CF 判断的较优动作（teachers_agree）。
  - 错与对互不重叠：`check_thresholds` 要求 `outcome_error_prob ≤ outcome_correct_prob`，两个 margin 都不小于 0。
- **放行：**
  - 只有 OFFLINE_TRAIN：放行全部有效反馈。ready_at 照样保存，但不参与放行。
  - TIME_ORDERED 在放行函数、CLI 和配置三处都报 `TIME_ORDERED_NOT_SUPPORTED`。
  - 离线结果不是严格时间顺序的在线学习。
- **B2 正确集与错题集窗口：**
  - `MultiCorpusWindows(correct=…, correct_fraction=f, remedial=…, remedial_fraction=g)`：每批先放 round(g × B) 个含错题行的窗口，再放 round(f × B) 个含正确行的窗口，其余为自然窗口；f + g ≤ 1，标签与自然窗口相同。
  - `MultiQWindows(correct_boost=…, remedial_boost=…)`：决策对 / 决策错的行在起点权重里加权；同一行不能同时属于两个集合。
  - 关闭的集合（比例 0、倍数 1）不传游标。
  - 报告按集合分开记窗口、行、见过的行和占比。
- **L6 进入 `iterate`**（从 `iteration.l6.from_round` 开始，默认关）：
  - 顺序：CF → feedback → L2 复习 → L4 → L5 → freeze → exam。
  - 不训练 L1，MarketCarry 存储保持有效。
  - CF 没有代表性的轮次：跳过 L4，L5 不带这份 CF 的正确集和错题集（`decision_feedback: SKIPPED_CF_NOT_REPRESENTATIVE`）。
- **实跑（开发 run 第 7–9 轮，2026-10-25）：** 见 `S5_RECORD_20261024.md`。
  - 固定比例的后果错补课会让后果脑在自然分布上校准变差。
  - E-7 方案 A（用户 2026-10-28）：`l6.remedial_fraction: 0`，L2 复习和 L5 只用自然窗口；教学题、L2 自然复习和 L4 决策错加权保留。
  - 同日起 `l6` 默认改为正确集开（`correct_fraction` 0.25、`correct_boost` 4.0）、错题集关（`remedial_fraction` 0、`remedial_boost` 1.0），见下文。
- **教学题每轮重建：** 开启 L6 的轮次每次都按当前 CF 和同轮合并 κ_Q 重写 feedback，已有的 `summary.json` 不再使跳过生效，所以补算的 CF 会进入教学。
- **训练窗口的市场状态（R3，用户决定 2026-10-24）：**
  - 问题：零状态 64 行窗口训练出的 `pi_0000`，在 FG005 的训练窗口里 ADD 240 次，连续路径只有 99 次（WAIT→ADD 148，ADD→WAIT 7）。
  - 解决：窗口改为从连续路径的因果市场 carry 起步。
    - `run market-carry` 为每个合约记录游标 t 处理该分钟观测之前的 carry。每个合约从零状态和策略自己的种子计数器开始，不接上一个合约。
    - 市场脑只读市场观测，L1 之后冻结，所以一份存储就是此后每个 `pi_k` 部署时的市场记忆。
    - `--window-start carry --carry-root` 让 L2–L5 和 state-check 读这份存储。
    - 编排的 `student.window_start: carry` 在 L2 或 L3 之前补齐存储；已有存储时拒绝再训练 L1。
  - state-check 报告存储与部署路径的差（`store_vs_path_max_abs_diff`），以及全部行上的有符号偏差（`mean_diff`）。
- **size50m 冒烟（FG005 + FG405）：**
  - pretrain → 中断续跑的一轮 → `pi_0001`；
  - DEV FG409：`pi_0000` 16.90、`pi_0001` 18.62，随机基线分别为 13.12、16.86。只是工程贯通，不说明交易能力。

**分档教学与完全重训（2026-10-28，用户决定）**
- **分档入场老师**（gym `ts_v1_grades`，教材 `graded/`）：
  - 对每个空仓有资金的 cursor，给“现在 ADD、然后持有”的 Native 交易评档：EXCELLENT / GOOD / FAIR / POOR。
  - 三个维度，总档取最差的一档：
    - 止盈时间：同一官方交易日内 ≤15 / ≤120 市场分钟 / 当日更晚；
    - 回撤 `mae_full`（可执行报价，含双边点差）：≤3 / ≤5 / ≤10 / >10；
    - 空间：成交后到止盈后 60 个市场分钟（截到交易日结束）之间的最大可执行有利空间，≥10 / ≥7.5 / ≥6 / <6。
  - SL30、其他出场和跨日 TP 为 POOR。RIGHT_CENSORED、未成交的 ADD、空间窗口被数据截断的日内 TP 不给档。
  - GOOD 及以上示范 ADD，POOR 示范 WAIT，FAIR 不标。这是 L3 和 L5 唯一的示范来源。
  - 成交分钟的收盘价也计入止盈判断：`env.step` 可在成交分钟止盈，条件是该分钟成交量 ≥ 6 手（开仓 3 手后仍余 3 手可平）。
  - 18 个 TRAIN 合约：成交 484,249 笔，EXCELLENT 30,106、GOOD 74,263、FAIR 99,875、POOR 279,389、不给档 616。示范 ADD 104,369（strict 24,926）、WAIT 279,389。止盈分钟、MFE 与止盈时间的等价关系逐笔核对全部一致。
- **Outcome MFE 头：** 目标 `mfe_15/60/120`（点）。成交分钟计入窗口；窗口不因出场截断，只截到交易日结束。平方误差，尺度 30，`loss_scales.outcome_mfe: 1.0`。答卷增加 `answer/mfe_15|60|120`。
- **不兼容旧结构：** 参数形状变了（evidence 40 → 43，新增 `outcome/mfe`）。r2r3 迁移和 `--migrate` 已删除，旧 run 和旧 checkpoint 不能在新代码上续跑。新 run 从 F1（`paths.init_f1`）重新训练；`inherit_student`、`inherit_f1` 保留。
- **正式配置：** `formal_v1_graded.yaml`（1x 开发，L1 10k / L2 12k / L3 12k）和 `formal_v1_graded_10x.yaml`（L1 100k / L2 120k / L3 120k，CF 和 L4 设置同旧 10x）。L2–L5 用 carry 开窗，9 轮，第 7–9 轮带 L6，正确集开、错题集关。
