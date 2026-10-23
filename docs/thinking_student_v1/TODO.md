# Thinking Student V1 — TODO（完成一项勾一项）

分支：`feat/thinking-student-v1-20261022`（dreamerv3 + gym-trading-env，worktree 在 `/home/v/Documents/work/research/fg-s3-thinking-student-v1-20261022/`）
规则：每项完成并有运行证据后才打勾；每个切片结束时补一份极短记录。
设计来源：`~/Documents/work/research/thinkstudent/`（本目录为其副本）。2026-10-22 23:15 的 A1–A6/C3 更新后，S2–S5 条目已按新版改写。

## 0. 准备
- [x] 读取 thinkstudent 全部文档，理解 A1–A6 / B1–B2 / C1–C3 / ABC Review
- [x] 两仓从 `integration/s3-next` 建 worktree 和新分支
- [x] 设计文档复制到 `docs/thinking_student_v1/`
- [x] `INTERFACE_SUMMARY.md`：B1/B2/C1 与 F-01/F-02 的交接摘要

## S0 — 最小完整学生前向 + 训练底座
- [x] Gym：`account_native`（10 维账户事实）与 `StudentEnv` 包装器
- [x] Market Brain：纯市场 Encoder 与外生 RSSM（动作槽固定为零，F1 参数形状不变）
- [x] Brain：Account、Fusion、Outcome、Value（ΔQ=0 起步）、Decision（A5 公式，SG 按 F-01）
- [x] StudentModel：唯一 `policy`、答卷、L1 训练、参数组表
- [x] `transfer.py`：继承 F1 权重
- [x] `run.py`：rollout、train-market、save/resume
- [x] 测试：`test_ts_v1_s0.py` 5 项加上已有的 env/obs 测试，共 14 passed
- [x] 真实 FG809 烟测：F1 继承、20 次 L1 更新、300 步 Native rollout、resume
- [x] S0 记录：`S0_RECORD_20261022.md`
- [x] 提交 gym 仓库（`44c9b19`）
- [x] 提交 dreamerv3 仓库

## S1 — Outcome Brain 真实学习
- [x] 把 teacher 的 `trade_facts` 映射为 `OUTCOME_BINS` 竞争风险标签（RIGHT_CENSORED 和 OTHER_EXIT 只走生存项；`labels.py`，3 项语义测试通过）
- [x] `L_outcome`：竞争风险 NLL、右删失、MAE、early10、Quality（`brain.outcome_loss`，NLL 为 −log Σp[允许类]；无信息行损失和梯度都为 0；共 5 项测试通过）
- [x] `opt_l2`：只训练 outcome 组，可选加 fusion 适配；Market/Account 冻结（`ts.stage` 选择阶段；初始化时建好全部阶段优化器；`ts.l2_fusion` 声明 Fusion 适配；测试证实一次 L2 更新只改 outcome（+fusion）和 `opt_l2`，S0+S1 共 12 passed）
- [x] 把 corpus 标签（teacher_sidecar 的 L1 facts）接入训练批次（`corpus.py`：按 stepid 对齐，标签只允许出现在合法空仓 ADD 上，否则报错；空仓行精确补 `account_native`；采样按标签数加权；以每个策略最后 2 个 chunk 作留出集。6 个策略的标签：P0 28240、P4 22754、P3 832、P2_low 505、P2_mid 121、P1 53；首次解析 55 s，缓存后 1 s）
- [x] 最小验证：TP、SL、删失三例的语义，1 次梯度更新，ckpt 继承（新 ckpt 严格恢复时带着训练后的 Outcome 继续推理；结构变化的旧 ckpt 用 `transfer.inherit_student` / `--init-from` 继承，逐项列出 reused/reshaped/initialized/dropped，并写入 `inherit.json`；S0+S1+env/obs 共 23 passed）
- [x] 真实 FG809 短训，并写记录（R2 从 S0 ckpt 继承：reshaped 5、initialized 42（均为 opt_l2）；GPU 上 3000 次 L2 更新；留出集事件 NLL 2.587→2.151（最好）/2.183（最终），常数答案为 2.208；TP 行 / SL 行的 p_tp 为 0.757 / 0.576；从新 ckpt 跑 1000 步 rollout；共享回归 20 passed；`S1_RECORD_20261022.md`）
- [x] 提交 dreamerv3 仓库（gym 在 S1 无改动）

## S2 — 首个自主学生 π0
- [x] corpus 持仓行补算 `account_native`：沿原始动作重放 Native，消除训练与推理的账户输入偏差（S1 只能精确补算空仓行）（gym `replay_account_native` 用录制器的 `execute_observed` 逐行重放，并要求每行 observation 与录制值相等，不设哈希门禁；dreamerv3 `prepare-corpus` 每个策略起一个 CPU 进程，6 个策略共 2 分 20 秒，约 4 ms/行；账户与 `position_state` 逐行一致；持仓行 P1 28237、P2_low 27784、P2_mid 28169、P3 27458、P4 5492；缓存在 `/data/logdir/fg_single3/thinking_student/corpus_cache/`；S0+S1+账户+env/obs 共 25 passed）
- [x] `L_policy_demo`：只用合法严格 L2 示范训练行为先验，类别平衡/成对监督（不强制 ADD 比例、不改 Reward）；`opt_l3` 只更新 decision 组；`Q→Reasoning/Gate` 改为零初始化的中性投影；π0 的 `β_Q` 不贡献动作（示范：训练集 WAIT 46603 / ADD 440，类别权重 0.505 / 53.46。从 S1 R2 继承后在 GPU 上做 3000 次 L3 更新，约 2.5 分钟。留出集类别加权 NLL 1.722→0.898（最好）/1.036（最终），常数答案 0.826；最终 ADD 示范上的 p_add 0.397、WAIT 示范上 0.289。测试证实一次 L3 更新只改 decision 和 `opt_l3`，Q 投影保持 0；β_Q=0 时随机 Q 不改变 z）
- [x] 正式 π 用确定性 Masked Argmax（ADD 合法且 z>0 才 ADD，平局 WAIT），与 CF/考试同一规则；探索另设行为策略 μ，记录 π 提议、μ(a|s)、实际动作、探索标记；自主 Native 采集落盘（`brain.formal_action` 是唯一实现。GPU 上的 TRAIN 采集：正式 π0 跑完整个 TRAIN 时段 28263 步，ADD 50 次，35 TP / 14 SL / 1 换月；μ 采集 3000 步。中断后在新进程续采，结果与一次跑完逐位一致）
- [x] 冻结完整 π0 快照（`S2_POLICY_FG809_20261023/policies/pi0`：完整参数，`policy.json` 记录动作选择方式、β_Q=0、μ 的 ε、RSSM 初始化、推理设备 cuda/bf16、Reward 版本、teacher 来源和两仓 SHA）
- [x] 最小验证（短自主回放 + 一次 Actor 梯度更新；正式决策与 CF/考试规则一致；Mask、TP/SL 正确；记录 ADD 比例，不设门槛），并写记录（`run check`：34263 行，规则、Mask、TP/SL 点数都无违规。50 笔交易与 teacher“现在 ADD 后持有”的独立事实 50/50 一致，Native reward 之和与 teacher 的逐位相等。正式 π0 的 ADD 比例 11.8%。π0 与随机入场相当，没有优势。S0+S1+S2+账户+env/obs 共 32 passed。见 `S2_RECORD_20261023.md`）
- [x] 提交 dreamerv3 和 gym 两个仓库（gym `fb7481d`；dreamerv3 见包含本行的提交）

## 2026-10-23 计划：Review 修正 + 教材重建 → S3
计划全文：`PLAN_R_M_S3_20261023.md`（编号与下面一致）。依据：`FG_S3_TS_S0_S2_Source_Review_20261023.md`、C3_20261023 §5/§5.9。
- [x] 第 0 步：同步设计文档（Review、C3_20261023 拷入，删除 C3_20261022 副本）；写计划文档；本 TODO 新增 R/M 节并改写 S3

## R — Review 修正
- [x] R1：TS 环境不走旧 E-Data 认证 / Hash 链（新 `native_env`；env 去掉行情源 SHA、每步 fingerprint / array hash、快照 prefix_hash 与 identity 比对；同一进程复用行情与 env 实例）（`entry_env.py` 删除 `_validate_e_market_source`、`e_observation_identity`、`e_snapshot`、`_prefix_hash`；`ts_v1_corpus.native_env/corpus_env` 只用 `contracts` + 常量 identity 的轻量 E profile；gym `test_ts_v1_corpus.py` 8 passed、`test_ts_v1_account.py` 3 passed）
- [x] R2：Outcome 34 类（TP / SL / OTHER_EXIT 各 11 桶 + OPEN_AT_H），RIGHT_CENSORED 独立删失；答卷字段 `p_tp_h` / `p_sl_h` / `p_other_h` / `p_open_h`，快照记录 `outcome_horizon_minutes`（`labels.py` 34 类、`allowed_classes`；`policy.json` 的 `outcome.horizon_minutes=2760` 与 34 个类名；`test_ts_v1_s1.py` 11 passed，含 OTHER_EXIT、删失、未成交 ADD）
- [x] R3：`account_native` 14 维（`to_add_deadline`、`to_forced_exit`、`funds_cover`、`margin_cover`）（`ts_v1_account.ACCOUNT_NATIVE_FIELDS`；Dreamer 观测空间直接取 `ACCOUNT_NATIVE_SHAPE`；教材生成时直接记录，删除 `replay_account_native`）
- [x] R4：TS env `chunk_steps=0`；自主运行不因 CHUNK_END 重置市场 RSSM；训练窗口零状态起步 + burn-in，注明不是真实断档（`Autonomous` 遇到 `chunk_steps != 0` 的 env 直接报错；`policy.json` 记录 `rssm_initialization`；`CorpusWindows` 文档写明 burn-in 不是数据断档；s2 采集测试断言只有首行 `first`）
- [x] R5（低优先级）：未监督的 latent 移出 evidence；`reason_pref` 在记录与答卷说明中标注“S4 前未训练”（evidence 40 维 = 34 + 2 + 3 + 1；`policy.json` 的 `untrained.reason_pref`）
- [x] R6：S2 ckpt 迁移 `--migrate r2r3`（逐项列出 reused / remapped / initialized / dropped）（`transfer.r2r3`；s1 迁移测试；FG809 实跑 `S3_FG809_20261023/inherit.json`：reused 311、remapped 6、reshaped 10（优化器矩）、initialized 0、dropped 6（latent））

## M — 教材重建（去审计、无兼容）
- [x] M1：Gym `ts_v1_corpus.py` 生产者（market / facts / strict / behaviors P0–P4，`source.json` 只记来源与计数）+ 合成行情测试（facts = 原生逐分钟，P4 = 严格计划，无重复 cursor）（另：生产时抽 32 个 cursor 原生逐分钟核对，不一致即报错；严格平局用精确规则；`source.json` 记录各标签来源与 `ready_at`；gym `test_ts_v1_corpus.py` 8 passed，含精确 DP 的 ADD/WAIT/平局与搬目录后可读）
- [x] M2：Dreamer 新 Reader / 标签 / run.py 接入（删除 `prepare-corpus` 与旧 corpus 路径），S0–S2 测试改用新生产者（`corpus.py` 的 `TSCorpus` / `CorpusWindows`；标签只取 P0 账户（C3 EDU-R10）；Reader 检查表长、有限值与 ADD 合法性；s0 5、s1 11、s2 8 passed）
- [x] M3：生成 FG809 新教材（`/data/logdir/fg_single3/thinking_student/corpus/CZCE.FG809/`，记录耗时和大小）（169 MB、2 分 42 秒：facts 75.8 s、原生核对 23.7 s（32/32）、strict 0.2 s、behaviors 61.5 s；28264 行；严格示范 valid 28263：ADD 304、WAIT 27283、平局 676）
- [x] M4：一次性对照：新 facts / strict 与旧 P0 sidecar 按 cursor 比较；64 个 cursor 原生逐分钟核对（`CZCE.FG809.m4_compare.json`：L1 比较 28240 笔全部字段一致，另 27 行是旧 CHUNK_END 重复行；L2 28263 行全部一致；原生 64/64 一致。对照发现并修掉两个生产者问题：输入 YAML 写了 `.tmp` 绝对路径、零反馈交易误判为平局）
- [x] M5：新教材上重训 L1 → L2 → L3（GPU），冻结 `pi0r`，正式 TRAIN 采集 + `check`（`S3_FG809_20261023/`，整链 10 分 55 秒：L1 1000、L2 3000、L3 3000 次更新；pi0r 采集 28263 步到 MAIN_ROLL 真终止，合法 1211、ADD 53（4.4%），53 笔 TP 39 / SL 13 / MAIN_ROLL 1；规则、正式、Mask、点数违规均 0；与新 facts 逐笔 53/53 一致。留出评估如实记录：L2 留出集是合约最后 2048 分钟，539 个 OTHER_EXIT 全在这里、训练段为 0；按类型 NLL（`l2_heldout_by_event.json`）TP5 2.72（常数 2.48）、SL30 2.00（常数 2.99）、OTHER_EXIT 8.18（常数 10.17）。L3 留出示范损失 1.86，差于常数 0.84）
- [x] M6：删除旧审计代码闭包（两仓），剩余模块可导入、剩余测试通过（Gym 删 195 个文件、Dreamer 删 38 个（含旧 C3_20261022 副本）；剩余模块逐个导入通过。全量 150 个测试文件逐文件并行跑：132 通过；其余已分类，见 `R_M_RECORD_20261023.md` §M6：11 个在 S2 基线（`fb7481d`/`46763e2`）同样失败（其中 1 个文件里没有可收集的测试），属旧问题；embodied 的 2 个 perf 文件是 `while True` 基准，被终止；1 个（`test_fg_s3_v2_artifacts`）缺真实 GPU checkpoint 工件，未在基线复现；4 个由本轮删除/R1 引起，已修。修法：补删 R1 遗留的 E 来源引用（`entry_market_cache`、`fg_replay`、`quality_teacher_v2`、`native_account`、无用的 `NativeEObservationCache`），并删掉断言已删审计或已删旧入口的测试片段。按用户 2026-10-23 指示，与 TS 无关的旧测试不再维护；TS 相关测试全部通过：gym `test_ts_v1_{account,corpus,cf}`、`test_s3_e_native_env`、`test_wp02_macd`、`test_entry_market_cache`，dreamer `test_ts_v1_s0`–`s3`）
- [x] M7：删除旧教材数据（`/data/s3_corpus/` 全部 6 个目录；`thinking_student/{corpus_cache,label_cache}`），删除前 `lsof` 确认无占用（2026-10-23：`lsof +D` 三处均无占用；确认新 FG809 教材与 TS 代码不引用旧数据；删除 64 GB + 60 MB + 24 MB，`/data` 可用由 310G 变为 375G；保留行情 CSV、合约 YAML、`fg_s3_market_minute/.../inputs/C1/` 与全部旧 Run）
- [x] M8：写 `R_M_RECORD_20261023.md`，提交两仓（记录含 R1–R6、M1–M7 的命令、计数、路径和测试分类；`INTERFACE_SUMMARY.md` 新增实现状态一节；两仓各一个提交，未推送；S3 代码与单元测试同在此提交）

## S3 — 策略条件 Q + 同策略 CF 老师
- [x] S3-1：Gym `ts_v1_cf.py`：同初态 ADD/WAIT 双分支，注入冻结 π_k 回调；与 π_k 参考轨迹汇合得到精确回报；状态 VALID_COMPLETE / NOT_COMPARABLE（VALID_ESTIMATED 保留、本轮不生成）；合成测试（汇合 G = 跑到终止的 G；截断 → NOT_COMPARABLE）（gym `test_ts_v1_cf.py` 4 项通过：规则策略下汇合 G 与跑到真终止的 G 相对误差 ≤1e-9、两种首动作都有、长汇合分支、CAP 与 REFERENCE_INCOMPLETE、汇合时必须经策略见证；Dreamer `test_ts_v1_s3.py` 用真实小学生再验一次）
- [x] S3-2：F-02 统一折扣 γ=332/333 每有效市场分钟、终点为合约真终止；Dreamer `value.py`：参考 G_ref、GPU 多进程 CF、`targets.npz`（FG809 实跑 `S3_FG809_20261023/cf/pi0r/`：pi0r 正式采集 28263 行到 MAIN_ROLL，1211 个合法空仓 fork 全部 VALID_COMPLETE（汇合 1199、真终止 12），NOT_COMPARABLE 0；5 个 GPU worker 重放主轨迹 z 差异全 0，48 分钟；ΔG 均值 −0.056、标准差 0.344）
- [x] S3-3：L4：`L_q_real`、`L_q_cf`、`L_q_pair`；`opt_l4` 只更新 value 组；`train-value`；不用同一 CF 差距训练 Actor 偏好（`S3_PI1_FG809_20261023`：从 pi0r 继承，L4 3000 次约 3 分钟。留出评估如实记录：最后 2048 行只含 08-03 一天的 88 对；q_real 0.135 优于常数 0.179，但 q_cf 1.92、q_pair 0.87 差于常数 1.62、0.80，ΔQ 符号一致率 0.20 低于常数符号 0.236。8192 行留出诊断同样如此：符号一致率 0.52，常数 0.71。结论是 ΔQ 在 FG809 单合约上不外推。评估新增 `constant/sign_agree`）
- [x] S3-4：候选 π1：β_Q=1、κ_Q=std(ΔG) 记入快照；正式 TRAIN 采集；决策依赖检查（z − z(β=0) = anchor）（`S3_PI1_FG809_20261023/policies/pi1`：β_Q 1、κ_Q 0.29054（训练段 1123 对）。在 pi0r 的 28263 个状态上，z_pi1 − z_pi0r 与 anchor 最大偏差 2.4e-7，不合法行 z 逐位相等；翻转 17 次（WAIT→ADD 14、ADD→WAIT 3）。pi1 自己的采集 28263 步到 MAIN_ROLL：ADD 54、TP 40 / SL 13 / MAIN_ROLL 1，奖励 −11.93（pi0r −12.22）；`check` 违规 0，与 facts 54/54 一致。仅作诊断）
- [x] S3-5：最小验证（一个完整对 + 交易、一个 NOT_COMPARABLE、真实数据 2 个汇合分支跑到终止核对），写 `S3_RECORD_20261023.md`，提交两仓
  - 证据：
    - `cf/pi0r/cases.json`：两个完整对，分支分别交易 13 笔和 9 笔；
    - `cap_example.json`：fork 10693 在 1000 步上限下为 NOT_COMPARABLE/CAP，无标签；
    - `spot_check.json`：fork 10693 和 20982 的汇合 G 与跑到真终止的 G 相对误差 1.7e-16；
    - `test_ts_v1_s3.py` 4 项通过（143 s）。
  - dreamerv3 一个提交，未推送；gym 本轮无改动。

## S4 — Reasoning 与策略改进（计划：`PLAN_S4_20261023.md`；只用 FG809，pi1 → pi2）
- [x] S4-1：代码
  - `L_reason`：`reason_pref` 学严格 Oracle 的“ADD 严格优于 WAIT”，与示范同行、同权重；
  - L3 改为 `L_policy_demo`（中性 Q 锚点的教学视图 `[0, correction]`）+ `L_reason`；
  - Q→Reasoning（`reason_q` / `gate_q`）在 Q 受过监督后由这两项示范 Loss 开始学习；
  - `freeze` 跨 checkpoint 汇总报告；
  - V1 不实现 `L_policy_pref`、`L_align`、`L_policy_rl`。
  - 证据（代码改动）：
    - `student.py` 的 `l3_loss`：`[0, correction]` 教学视图，`L_reason`，新增 `loss/reason`、`reason/*`、`policy/formal_add_rate` 指标；
    - `config.py`：loss 系数 `reason: 1.0`；
    - `run.py`：L3 汇总与 `constant/reason`；新增 `reports(run)`；`policy.json` 用 `reasoning` 取代 `untrained`；
    - `brain.py`：只改 docstring，前向不变。
- [x] S4-2：测试 `test_ts_v1_s4.py`：
  - `L_reason` 的梯度范围；
  - 锚点中性（β_Q 0 / 1 梯度一致）；
  - Q 不为 0 时 `reason_q` / `gate_q` 开始学习；
  - 合成数据上的 CLI 链 L4 → L3 → freeze。
  - 回归 s0–s3。
  - 证据：五个文件在 CPU 上并行运行，全部通过：
    - s4：4 项，113 s；
    - s3：4 项，141 s；
    - s2：8 项，83 s；
    - s1：11 项，81 s；
    - s0：5 项，53 s。
- [ ] S4-3：真实数据：对 pi1 做 CF（`cf/pi1`），L4 学 Q^{pi1}（从 pi1 继承），留出集对比常数答案。
- [ ] S4-4：L3 先验刷新（示范 + `L_reason`）→ 冻结 pi2（β_Q 1，κ_Q auto）→ pi2 正式采集与 `check` → 决策依赖与受控敏感性诊断。
- [ ] S4-5：重新评估（对 pi2 做 CF，当前 Q 在 Q^{pi2} 目标上只评估），写 `S4_RECORD`，更新 INTERFACE_SUMMARY 与 memory，提交 dreamerv3。

## S5 — 自主教学循环 + Teacher-free 考试
- [ ] B1 教学题生命周期，含 `ready_at` 成熟反馈
- [ ] B2 课程循环：自然教材、多行为教材、学生自采、错题补课
- [ ] 在 DEV（VALIDATION）或合成数据上做 Teacher-free 工程短考，使用同一个 `student.policy`；封存 TEST 只在用户明确授权的最终评价中使用
- [ ] 完整闭环演示一次，并写记录
