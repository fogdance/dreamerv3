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
- [x] S4-3：真实数据：对 pi1 做 CF（`cf/pi1`），L4 学 Q^{pi1}（从 pi1 继承），留出集对比常数答案。
  - 证据（`S4_RECORD_20261023.md` 第 1、2 节）：
    - CF：`S3_PI1_FG809_20261023/cf/pi1`，1357 个 fork 全部 VALID_COMPLETE（汇合 1345、真终止 12），NOT_COMPARABLE 0；5 个 GPU worker 重放 z 差异全 0；2817 s。ΔG 均值 −0.094、标准差 0.372；训练段 1236 对（std 0.2664），留出段 121 对只在 08-03 一天。
    - L4：`S4_FG809_20261023`，从 pi1 继承（reused 349），3000 次更新约 3 分钟。2048 行留出：q_real 0.144、q_pair 0.893、符号一致率 0.241 好于常数（0.158 / 1.019 / 0.149），q_cf 0.455 差于常数 0.409；这一天 ΔG 与训练段符号相反。
    - 干净的 8192 行诊断（从 pi0r 继承，Q = 0）：符号一致率最高 0.761，始终低于常数 0.777；3000 次时 q_real、q_cf 差于常数。ΔQ 仍不外推。
- [x] S4-4：L3 先验刷新（示范 + `L_reason`）→ 冻结 pi2（β_Q 1，κ_Q auto）→ pi2 正式采集与 `check` → 决策依赖与受控敏感性诊断。
  - 证据（`S4_RECORD_20261023.md` 第 3–6 节）：
    - L3：同一 run 从 19020 恢复，2000 次更新到 21020。留出示范 NLL 2.26、L_reason 1.06，都差于常数 0.84；训练批次分别为 0.29、0.50。run 内 β_Q = 0（锚点在 `freeze` 写入），所以 `formal_add_rate` 等于 `add_rate`。
    - pi2：β_Q 1，κ_Q 0.26636；`teaching` 含 `train_value`、`train_policy`、`value_anchor`。
    - 在 pi1 的状态上：z 差 = Δanchor + Δcorrection（偏差 4.8e-7）；`reason_q` 范数 0.437、`gate_q` 0.0138；Q 头置零后，correction 平均变 0.684，先验翻转 72/1357；正式动作翻转 50 次，ADD 54 → 76。
    - pi2 采集：28263 步到 MAIN_ROLL，合法 1614，ADD 57，TP 43 / SL 13 / MAIN_ROLL 1，奖励 −11.973；`check` 违规 0，facts 57/57 一致。与 pi1 第一处不同在 cursor 38，来自先验刷新。仅作诊断。
- [x] S4-5：重新评估（对 pi2 做 CF，当前 Q 在 Q^{pi2} 目标上只评估），写 `S4_RECORD`，更新 INTERFACE_SUMMARY 与 memory，提交 dreamerv3。
  - 证据（`S4_RECORD_20261023.md` 第 7 节与小结）：
    - CF(pi2)：`S4_FG809_20261023/cf/pi2`，1614 个 fork 全部 VALID_COMPLETE（汇合 1611、真终止 3），NOT_COMPARABLE 0；5 个 worker 重放 z 差异全 0；2823 s。
    - `train-value --updates 0`（`S4_NEXT_FG809_20261023`）：当前 Q（Q^{pi1}）在 Q^{pi2} 的 2048 行留出集上，q_real 0.162 好于常数 0.234，q_cf 1.877 差于常数 1.716，q_pair 1.000 与常数 1.036 相当；只有 08-03 一天的 62 对。
    - Q 目标随策略的变化：在 927 个两策略都空仓的同一 cursor 上（Outcome 答案逐位相同），ΔG^{pi1} 与 ΔG^{pi2} 相关 0.65，符号一致 82%。下一轮必须用 `cf/pi2` 重做 L4。
    - 文档：记录、TODO、INTERFACE_SUMMARY 与计划补充；`brain.md` 与 thinkstudent 目录同步；memory 已更新；dreamerv3 一个提交，未推送；gym 无改动。

## C4 — 正式训练能力（计划：`PLAN_C4_20261023.md`；实跑只用 FG005 + FG405，不启动 18 合约正式训练）
- [x] C4-0：同步 C4 文档，写计划文件，新增本节（C4 文档已拷入本目录，其余 thinkstudent 文档与项目副本逐字节相同；计划写入 `PLAN_C4_20261023.md`）
- [x] C4-1：F0 多合约教材读取（`discover`、`MultiCorpusWindows`、market mmap、全局示范权重、holdout 0）
  - 证据：
    - `corpus.py`：`discover` / `corpus_sources`（只收 TRAIN，reward/clock/γ 不一致或合约重复时点名拒绝）、`MultiCorpusWindows`（先抽合约再抽窗口）、market 映射读取、`window(market=False)` 计数、`holdout=0`。
    - `test_ts_v1_c4.py`：发现与拒绝、混合采样（每行与本合约窗口逐键相等，batch 内两合约混合，全局类别权重）通过。
    - 18 合约只读检查（`C4_SMOKE_2C_20261023/load18.json`）：488,160 行、market 磁盘 2.3 GB；L1 选择 8.4 s、RSS 778 MiB；L3 合并示范 WAIT 423,749 / ADD 24,926，权重 (0.529, 9.000)。
- [x] C4-2：F1 多合约 L1→L2→L3（可重复 `--corpus`、分段保存与 `progress.json`、续训、`freeze` 不可覆盖并记录多合约）＋ 测试 ＋ 实跑 R1（pretrain、中断续训、pi_0000、TRAIN_SMOKE、18 合约读取检查）
  - 证据：
    - `run.py`：`--corpus`/`--cf` 可重复、`--mix`、`--save-every`、`progress.json` 与 `progress(run)`、`_train`（holdout>0 标 `TRAIN_INTERNAL_DIAGNOSTIC`）、`freeze` 拒绝覆盖并写 `corpora` / `lineage`、`train_market.json`。
    - 测试：`test_ts_v1_c4.py` 3 项通过（98.9 s，含中断后从第 2 次保存恢复、各阶段只改授权组、重复冻结被拒）；回归 s2 8 项、s3 4 项、s4 4 项通过。
    - R1（`/data/logdir/fg_single3/thinking_student/C4_SMOKE_2C_20261023`，FG005 + FG405，GPU，`c4_r1_commands.sh`）：从 pi0r 继承；L1 40 次；L2 在第 20 次保存（ckpt 13080）后被杀，重跑补 40 次，进度 L2 = 60；L3 40 次；`policies/pi_0000`（β_Q 0，两合约，lineage = pi0r）。
    - TRAIN_SMOKE：FG005 2000 步（合法 2、ADD 1）；FG405 续采到 6000 步（合法 62、ADD 34，TP 24 / SL 9 / 未平 1，奖励 −9.84）；check 违规 0，facts 1/1、34/34 一致。
- [x] C4-2a：TS 配置统一到 `dreamerv3/configs/thinking_student/`（用户 2026-10-23 要求，只限 TS）：
  - `student.yaml`（原 `config.py` 的 TS_DEFAULTS / SMALL）；
  - `markets/`：pool.json，含 18 个 TRAIN 和 1 个 VALIDATION（FG409）；19 个 env、合约文件和策略 master，env 用相对路径引用；
  - `formal_v1.yaml`、`c4_smoke_2c.yaml`、README；
  - gym 删除写死的 `TRAIN_POOL`（改为必填 `--pool`），`write_inputs` 把合约文件拷进教材 inputs（教材自包含），18 份已有教材同样补齐。
  - 证据：
    - `config.py` 改读 `student.yaml` 后，`make_config` 的结果与改动前的快照逐项相同（small 与 size12m/size50m）。
    - `markets/`：envs 19 个、contracts 19 个；用 pool 写出的 19 份 inputs 与原 env 内容一致；`train_row` 拒绝 VALIDATION 行，`dev_row` 拒绝 TRAIN 行，pool 中没有 TEST 行。
    - 18 份教材的 `inputs/` 都有 `contract.yaml`（`contract_config: contract.yaml`），内容与原合约文件逐字节相同，行数与环境加载已核对。
    - 测试：
      - gym `test_ts_v1_corpus.py` 9 项通过，新增三项：pool 行分离；教材 inputs 搬目录后仍可加载；带策略 master 的 pool inputs 搬目录后仍可加载。
      - `test_ts_v1_c4.py::test_thinking_student_configs_live_in_one_place` 通过，检查内容：
        - `student.yaml` 的位置；
        - 两份配置都能加载，size50m、DEV FG409、markets 指向同目录；
        - pool 为 18 TRAIN + 1 VALIDATION；
        - env 对合约和策略文件用存在的相对引用；
        - 未知键报错。
    - grep 两仓 TS 源码：没有其他 worktree 或 `/home/v` 路径，唯一的 `/data/logdir` 是 `ts_v1_corpus.py` 用法说明里的示例输出目录。
- [x] C4-2b：正式模型改用 size50m（用户 2026-10-23 要求），市场脑从 S3-V2 `E_ONLY_MARKET_MINUTE_FORMAL_1M` step 1M 继承（size50m、E 观测、18 个 TRAIN 合约）；fusion 层为 MACD 加宽零行：
  - 冒烟用 size50m 重跑，新运行目录 `C4_SMOKE_2C_50M_20261023`；
  - 实测每次更新、采集、CF、考试的耗时与显存，据此提出正式预算；
  - size12m 的 R1 只保留为 F0/F1 机制的记录。
  - 证据：
    - `transfer.plan` 新增 `widened`。真实形状下 reused 49、sliced 1、widened 1、initialized 60，写入 `foundation/inherit.json`；冻结时记入 `lineage.inherited_from`。
    - 测试 `test_f1_from_before_the_macd_branch_is_widened_exactly` 与 s0 `test_f1_market_weights_inherited` 通过。
    - size50m 冒烟 pretrain → R2 → R3 全部跑通，`pi_0000` 和 `pi_0001` 都是 size50m、cuda/bf16。
    - 18 合约测速（`C4_SPEED_50M_20261023/speed.json`）：
      - L1 0.199 / 0.204 s/次（batch 8 / 16），L2 0.095 / 0.096 s/次；
      - 采集约 100 步/s；CF 每个 worker 重放约 4.5–5 分钟；DEV 考试 330–370 s；每份 checkpoint 388 MB。
    - 正式预算写入 `formal_v1.yaml`：batch 16×64，L1 10,000 次，L2 12,000 次，每 1000 次保存。依据见 `FORMAL_TRAINING_IMPLEMENTATION.md`。
- [x] C4-3：F2 多合约同策略 Q（`MultiQWindows`、同策略检查、合并 κ_Q 及下界、多个 `--cf`、RSSM 短窗诊断 `state-check`）＋ 测试
  - 证据：
    - 测试 `test_q_targets_of_one_policy_on_two_contracts` 通过，覆盖：
      - 合并 κ 与手算一致；
      - 另一策略的 CF 被拒绝；
      - NOT_COMPARABLE 行没有 Q 监督；
      - 退化时无下界报错、有下界取下界；
      - `train-value` 两个 `--cf` 只改 value 组；
      - 冻结后锚点生效。
    - 真实数据（size50m 冒烟 R2，π_0000 的 CF）：
      - FG005 76 对（68 ADD + 8 WAIT），75 对有效，1 对 NOT_COMPARABLE/CAP；ΔG std 0.309。
      - FG405 315 对，全部有效；ΔG std 0.651。
      - 每个 worker 的重放与采集逐位一致（`z_differs` 0）。
      - L4 用两个 `--cf` 训练 40 次。
      - `pi_0001` 的 κ_Q = 0.5098：390 对、两合约等权的合并 std，未触及下界 0.05；β_Q = 1。
    - state-check（FG005，16 个 64 行窗口，burn-in 16）：
      - 从第 0 步起的对照窗口与连续路径完全一致（0.0）；重放与采集记录一致（0.0）。
      - burn-in 之后：`p_tp_h` 平均差 0.00013（连续路径 std 0.048）；`z` 平均差 0.49（std 2.20），48–64 行仍有 0.44。
      - 结论：短窗口没有复现连续路径的决策状态。
      - π_0000 的 Q 为 0，ΔQ 一项要到下一轮才有意义。
- [x] C4-4：F3 编排器 `formal_train.py`（pretrain / iterate / evaluate / run / status、`run_state.json`、跳过已完成的 collect/CF）＋ 测试 ＋ 实跑 R2（一轮 pi_0000 → pi_0001、中断续跑）
  - 证据：
    - 测试 `test_orchestrator_rounds_resume_from_the_disk` 通过（182 s），覆盖：
      - `pretrain --until L2` 只训练 L1、L2；
      - 第二次 `pretrain` 只训练 L3 并冻结；
      - 冻结后加 L1 预算被拒绝；
      - L4 失败后重跑，不再调用 collect、check、cf、state-check，experience 文件不变；
      - `status` 只读。
    - R2（`C4_SMOKE_2C_50M_20261023`，size50m，GPU）：
      - 23:13:18 起两合约并行正式采集约 4 分钟：FG005 20504 步、ADD 68；FG405 26429 步、ADD 307。
      - check 违规 0，facts 一致 68/68、307/307。
      - 中断：23:26:04 FG005 的 CF 写出 `meta.json` 时，对编排器进程组发 SIGKILL。被杀的有编排器、FG405 的 cf 命令、3 个 spawn worker 和 resource tracker；退出码 137，无残留进程，GPU 回到桌面占用。
      - 23:26:32 重跑：只调用了 FG405 的 cf。collect、check、FG005 的 CF 没有再调用；中断前的 103 个 experience 文件字节和 mtime 都不变，只新增 `state_check.json`。
      - 之后依次完成 state-check、L4 40 次、L5 20 次、冻结 `pi_0001`、DEV 考试，23:42:55 退出码 0。
    - `run_state.json` 第 1 轮记录：
      - 合约、collected、check `ok`、各合约 CF 摘要；
      - `L5: S4_PRIOR_REFRESH`、`L6: L6_NOT_IMPLEMENTED`、`updates {L4: 40, L5: 20}`；
      - `best_dev = pi_0001`。
    - `pi_0001` 的 lineage：`inherited_from` 为 pi_0000，`q_teacher_policy` 为 [pi_0000]，cf 为两合约。
- [x] C4-5：F4 Teacher-free 评估（gym `dev_row` + `write_inputs`、`exam.py`、随机基线、DEV / TRAIN_SMOKE / DEV_NOT_CONFIGURED）＋ 测试 ＋ 实跑 R3（DEV FG409）
  - 证据：
    - 测试：
      - `test_exam_is_teacher_free_and_names_its_market`，覆盖：
        - TRAIN_SMOKE 标 `teacher_free`、不训练、不完整；
        - 随机基线与学生同分钟数，入场概率 = ADD / 合法，单个 seed 可复现；
        - DEV 只接受 VALIDATION 行，train 和 test 行被拒绝；
        - DEV 跑到真终止。
      - `test_exam_numbers_have_defined_denominators`：每项指标与手算一致。
      - gym `test_pool_rows_keep_train_and_dev_apart`。
      - 编排测试中，没有 DEV 时写 `DEV_NOT_CONFIGURED`。
    - R3（DEV CZCE.FG409，整段 31034 个市场分钟、91 个交易日，size50m）：
      - `pi_0000`：Reward 16.90（每分钟 0.000545），ADD 290 / 合法 1257，TP 234 / SL 55 / 其他 1，占仓 0.959，最大回撤 36,321 元，净盈亏 −34,611 元。随机基线 13.12 ± 2.02（5 seeds）。用时 368 s。
      - `pi_0001`：Reward 18.62（每分钟 0.000600），ADD 292 / 合法 1460，TP 236 / SL 55 / 其他 1，占仓 0.953，最大回撤 36,009 元，净盈亏 −34,047 元。随机基线 16.86 ± 3.22。用时 329 s。
      - 两者 Reward 为正、净盈亏为负：Quality TP Reward 不等于账户盈亏，两项分开报告。
      - 两次考试都跑到真终止；`best_dev` 按每分钟 Reward 取 `pi_0001`。
    - TRAIN_SMOKE（`pi_0000`，前 6000 步，`complete` false）：FG005 Reward −7.31（随机 −6.33 ± 0.66）；FG405 Reward −7.86（随机 −9.56 ± 2.51）。
- [x] C4-6：F5 接入（S4 先验刷新作为 L5；`L6_NOT_IMPLEMENTED`）
  - 证据：
    - `iterate` 在 L4 之后用全部 TRAIN 教材跑 `train-policy`（S4：`L_policy_demo` + `L_reason`，中性锚点教学视图），R2 中 20 次；`prior_refresh_updates: 0` 时记 `L5_SKIPPED_BY_CONFIG`。
    - 每轮记 `L6: L6_NOT_IMPLEMENTED`。S5 的接口位置是 `iterate` 中 L5 与 freeze 之间，没有写空循环。
- [x] C4-7：`FORMAL_TRAINING_IMPLEMENTATION.md`、INTERFACE_SUMMARY、memory，两仓提交（不推送）
  - 证据：
    - 代码提交：gym `0056296`，dreamerv3 `89597ec`（C4 代码、configs、测试），文档随本条提交。
    - 最终回归（CPU，`JAX_PLATFORMS=cpu`），共 57 项通过：
      - c4 9 项（368 s）；s0 5、s1 11、s2 8、s3 4、s4 4；
      - gym corpus 9、account 3、cf 4。
    - 回归之后改了 `formal_train`（`--until` 与阶段先后保护），只影响编排测试和配置测试，这两项随后重跑通过。
- [x] C4-8：正式训练第一段（用户 2026-10-23 授权，开发全部完成后启动）：18 合约 L1 + L2，`formal_train pretrain --config formal_v1.yaml --until L2`，在 tmux 会话 `d3` 中启动，后台跑完，不监控
  - 证据：
    - 2026-10-23 23:53:08 在 tmux `d3` 的窗口 `ts-formal` 中启动 `FORMAL_V1_20261024/formal_l1_l2.sh`，代码为 dreamerv3 `0faf1b5`、gym `0056296`。
    - L1 日志：从 E_ONLY 继承（reused 49、sliced 1、widened 1、initialized 60）；第 100 次更新 loss 48.4；GPU 2.8 GB，利用率 76%。
    - 结果：L1 于 00:27:08 完成 10,000 次，L2 于 00:46:29 完成 12,000 次，exit 0。最新为 `foundation/ckpt/000000022000`，22 份 checkpoint 共 8.4 GB；`status` 为 `L2_done`，`pi_0000` 未冻结。
    - 训练批次损失（详见 `FORMAL_TRAINING_IMPLEMENTATION.md`）：L1 市场重建 20.9 → 9.1；L2 `outcome_event` 2.78 → 2.37，常数答案为 2.775；真实 TP 行与 SL 行上的 TP 概率只差 0.022 ± 0.003。
    - 中断后重跑同一脚本即可接着训练；进度：`formal_train status --config …/formal_v1.yaml`。
  - L2 之后的 L3（配置中暂填 12,000 次）、`pi_0000` 和后续轮次，等用户决定。

## C4 源码 Review 修复（`FG_S3_TS_C4_gy34_d334_Source_Review_20261024.md`；计划：`PLAN_C4_20261023.md` 第 4 节）
- [x] C4R-1：R1 冻结后中断时，重跑先补完该候选缺的考试，再看轮数上限；`exam.json` 原子写入 ＋ 测试
  - 证据：
    - `formal_train.iterate` 每轮开始前用 `_exam_missing` 检查最新冻结策略的考试文件，缺了就只补缺的部分（`evaluate` 按合约跳过已完成的）。k ≥ 1 时这一步计入本次调用的轮数，之后才检查 `max_rounds`。
    - `_takes_smoke` 统一了 pretrain 与各轮的考试组成；`exam.json` 改用 `run._write` 原子写入。
    - `test_orchestrator_rounds_resume_from_the_disk`：L4 失败后的重跑在冻结 `pi_0001` 后，第一次考试失败，调用序列为 train-value、train-policy、freeze、exam。再重跑只调用 `['exam', 'exam']`（两份 TRAIN_SMOKE，未配置 DEV），`step` 为 done；再跑一次没有任何调用。
    - 新测试 `test_orchestrator_finishes_the_exam_of_a_frozen_candidate_first`：两个合成 DEV 合约，DEV1 已考完。在 `max_rounds` 2、`--rounds 1` 下重跑：
      - 只调用一次 exam（DEV2）；
      - DEV1 的文件字节和 mtime 不变；
      - `best_dev` 为 `pi_0001`；
      - 没有生成 `pi_0002`，也没有 experience。
- [x] C4R-2：R2 β_Q > 0 至少需要 2 个有效 CF 成对样本，否则 β_Q = 0 并标 `Q_ANCHOR_NOT_READY`；κ_Q 下界不再顶替缺失的监督 ＋ 测试
  - 证据：
    - `run.value_anchor`：β_Q > 0 或 κ 为 `auto` 时必须给 `--cf`。有效对少于 2 个时，返回 β_Q 0，同时记 `requested_beta_q`、`status: Q_ANCHOR_NOT_READY` 和对数。
    - `MultiQWindows.kappa`：少于 2 对时，给不给下界都报错。
    - 编排器把锚点（β_Q、κ_Q、对数、状态）记入该轮。
    - `test_q_targets_of_one_policy_on_two_contracts`：
      - 0 对时，`kappa()` 和 `kappa(0.05)` 都报错；
      - 所有有效对的 ΔG 都等于 0.5 时，不给下界报错，给 0.05 则使用下界（std 0）；
      - 用 0 对的 CF 加 `--kappa-floor 0.05` 冻结，得到 β_Q 0、`Q_ANCHOR_NOT_READY`、请求值 1.0、0 对，40 步 rollout 的 anchor 全为 0；
      - β_Q 1、固定 κ、不给 `--cf` 时报错。
    - 编排测试中第 1 轮为 60 对，`pi_0001` 的 β_Q 为 1.0、κ_Q 为 0.853，与轮记录一致。
- [x] C4R-3：R3 在 1024 行窗口上测量短窗口与连续状态的差距（正式 L2 诊断快照、冒烟 `pi_0000`），给出 L3 窗口与预热建议
  - 证据：`C4_STATE_CHECK_20261024/`（`state_check.sh`，源码 dreamerv3 `4c67480`，GPU）。FG005 上 16 个 1024 行窗口，对照窗口差 0。
    - 正式脑子（ckpt 22000 的诊断快照，决策头未训练）：`p_tp_h` 在 16 行之后的差距不到其标准差的 1%。`z` 的差距占其标准差的比例：16–32 行 22%、48–64 行 16%、112–128 行 10%、1008–1024 行 6%。
    - 冒烟 `pi_0000`（市场脑接近 E_ONLY 原样）：`z` 的差距在所有位置都约为 20%，不衰减。
    - 结论与建议见 `FORMAL_TRAINING_IMPLEMENTATION.md` 的“短窗口与连续状态”一节：L3 先按 64/16 训练，冻结 `pi_0000` 后再测一次训练后决策头的差距；是否加长窗口或改从真实状态起步，由用户决定。
- [x] C4R-4：R4 正式配置 `state_check_windows: 0`，诊断改为手动运行 `run state-check`
  - 证据：`formal_v1.yaml` 与 `formal_train.DEFAULTS` 的 `state_check_windows` 为 0。冒烟配置保留 16，编排测试显式设为 4，仍覆盖这一步。
- 回归（CPU，`JAX_PLATFORMS=cpu`，在定稿代码上运行）：
  - C4：10 项通过，368 s，含新增 1 项。
  - S0–S4 共 32 项，加 gym corpus 9 / account 3 / cf 4，合计 48 项通过；pytest 退出码 0，junitxml 记录 0 失败、0 跳过，466 s。
  - 本次没有改 gym 源码。

## C4 R3 决定：正式 L3 → `pi_0000` → 状态一致性 → L4（用户 2026-10-24；计划：`PLAN_C4_20261023.md` 第 5 节）
- [x] D-1：正式 L3（从 foundation ckpt 22000 续训，64 行窗口、16 行预热，配置 12,000 次；L1/L2 不重训）→ 冻结 `pi_0000` → 考试（DEV FG409 + TRAIN_SMOKE）
  - 证据：`formal_l3_pi0.sh`（tmux `d3:ts-formal`），07:49:39–08:22:03，退出码 0。
    - L3：从 ckpt 22000 恢复，22001–34000 共 12,000 次，每次 0.11 s。示范 ADD 24,926、WAIT 423,749，类别权重 0.53 / 9.0。
    - 训练批次指标（每 2000 次取均值，每 100 次记一个 batch）：
      - 加权示范 NLL 从 0.674 降到 0.52–0.56（均匀答案为 log 2 = 0.693），约 6000 次后不再下降；
      - ADD 示范上 p(ADD) 0.56–0.62，WAIT 示范上 0.35–0.40；
      - `reason` 准确率：ADD 0.66–0.73，WAIT 0.65–0.69。
      - holdout 0，没有留出评估。
    - `pi_0000`：08:12:06 冻结，来自 ckpt 34000。`repos` 记录 dreamerv3 `18b2d50`，dirty 为 True（当时 `formal_v1.yaml` 只改了注释，尚未提交）。
    - 考试（`run_state.best_dev` 为 `pi_0000`）：
      - DEV FG409（完整，31,034 分钟）：Reward 6.42；随机入场基线（同 ADD 比例，5 个 seed）16.92 ± 11.9。ADD 275 / 合法 3223，TP 221、SL 54，占仓 0.90。
      - TRAIN_SMOKE（各 6000 步）：FG001 为 −3.33，基线 −0.61；FG005 为 −0.66，基线 −4.72。
- [x] D-2：state-check 扩展（合法 ADD 行上正式动作的翻转率、按与决策边界距离分组的翻转、归一化差异、`q_add`/`q_wait`）＋ `iterate --until cf` ＋ 测试
  - 证据：
    - `value.path_agreement` 是纯函数，`window_check` 调用它。两条路径都记录正式 Masked Argmax 动作，比较的量加入 `q_add`、`q_wait`。报告包括：
      - `by_position`：分段为 16 行到第 64 行，之后长度逐段翻倍；每段给出各量与标准差之比，以及合法行数、翻转数和翻转率；
      - `actions`：预热之后合法行上的翻转，以及 ADD→WAIT、WAIT→ADD、`masked_flips`；
      - `boundary`：按 |z|/std 分组的翻转；
      - `control_action_flips`、`replay_vs_record_action_flips`。
      mask 强制为 WAIT 的行不进入任何比率。
    - `formal_train iterate --until cf`：一轮在 CF（和 state-check）之后停下，`step` 记为 `cf_done`。下次 `iterate` 从磁盘接着做 L4。`pretrain` 不接受 `cf`。
    - 测试（CPU，2 项通过，160 s）：
      - `test_state_check_counts_action_flips_on_legal_rows_only`：两个手算窗口；被 mask 的行 z 变号也不算翻转；近边界与远边界各翻转 1 次；
      - 编排测试：`--until cf` 依次调用 collect ×2、check ×2、cf ×2、state-check，停在 `cf_done`，没有 learner 目录；对照窗口与重放的差和翻转都是 0，`masked_flips` 为 0；下一次 `iterate` 只调用 train-value，experience 文件不变。
- [x] D-3：`pi_0000` 在 FG005 上做与 R3 相同的 state-check（16 × 1024），另加训练位置上的 1000 × 64；据此决定：接受 64/16 进入 L4，或实现 carry 继承
  - 证据：先采集 `pi_0000` 在 FG005 上的完整正式路径：20,504 步，合法 6095，ADD 48，TP 39。报告在 `C4_STATE_CHECK_20261024/pi_0000/`。所有报告中对照窗口和重放都是 0 差异、0 翻转。比率只在合法行上计算（mask 允许 ADD），被 mask 的行翻转数为 0。
    - 零状态开窗，16 × 1024（与 R3 相同）：
      - z 差异 / std 按位置：16–32 行 0.29，32–48 行 0.15，48–64 行 0.16，64–128 行 0.12，之后降到 0.07；
      - 翻转 10 / 4526（0.22%），全部发生在 |z| < 0.3 std；ADD 数：窗口 43，连续路径 35。
    - 零状态开窗，1000 × 64（训练的窗口位置）：
      - z 差异 / std：预热内 0.80，16–32 行 0.31，32–48 行 0.22，48–64 行 0.19；
      - 预热后翻转 155 / 14,532（1.07%），分段为 2.1%、0.66%、0.43%；
      - ADD 数：窗口 240，连续路径 99；WAIT→ADD 148，ADD→WAIT 7，偏差单向，ADD 放大约 2.4 倍；
      - 按 |z|/std 分组的翻转率：< 0.1 为 37–38%，0.1–0.25 为 25%，0.25–0.5 为 11%，0.5–1 为 1%，> 1 为 0.03%；翻转点距边界中位 0.24 std，最远 1.04 std。
    - **决定：** 零状态开窗使 ADD/WAIT 决策明显、单向地偏移，适用用户决定第 4 条，进入 D-4。
    - 预先验证 carry 方案（`--window-start carry`，窗口从连续路径自己的 carry 开始）：
      - 用路径自己的策略种子，100 个窗口：差异 0，1503 个合法行上翻转 0。
      - 换一组策略种子（offset 1,000,003），1000 个窗口：
        - z 差异 / std 按 16 行分段为 0.22、0.24、0.25、0.26；16 行之后翻转 105 / 14,532（0.72%）；
        - ADD 数：窗口 140，连续路径 99；WAIT→ADD 73，ADD→WAIT 32；
        - 翻转点距边界中位 0.18 std，最远 0.66 std。
      - 解读：单向偏差基本消失，剩下的是随机隐变量采样本身的噪声。残留的 73 对 32 不对称，可能来自重放沿用路径自己的 mask：路径一旦 ADD，后续行被 mask 掉，所以合法行只出现在路径自己 WAIT 的时段，这些行上路径的 z 天然偏低。之后的报告增加 `mean_diff`，在全部行上计算有符号偏差，不受这种选择影响。
- [x] D-4：（仅当 ADD/WAIT 决策受显著影响时）训练窗口继承完整连续市场路径的因果 RSSM carry：预计算并复用，不加长窗口，不重训 L1/L2
  - 证据：dreamerv3 `8bf5bc1`，做法见 `FORMAL_TRAINING_IMPLEMENTATION.md`“训练窗口继承连续市场 carry”。
    - 测试（CPU）：新增 carry 测试 6 项通过（226 s）。随后修复多合约 `market-carry` 的种子计数器沿用问题，加上第二个合约逐位一致和换种子的反例，2 项通过（106 s）。TS 回归 `test_ts_v1_s0`–`s4` 加 `c4` 共 48 项通过（22 分 48 秒）。
    - 真实数据：`FORMAL_V1_CARRY_20261024/market_carry/` 下 18 个合约的存储，6 个进程并行，08:53–09:12，每个合约约 230–290 s，共 4.7 GB；种子计数器 0，cuda/bf16。
      - 正式 L3 读这份存储，冻结后的 `pi_0000` 在 FG005 上部署路径的 carry 与存储逐位相同（`store_vs_path_max_abs_diff` 0.0）。
    - 正式 L3：`formal_v1_carry.yaml`，从 ckpt 22000 继承（349 个键全部复用），09:12–09:34，12,000 次，carry 开窗，没有预热。
      - 训练批次指标（每 2000 次取均值）：示范 NLL 0.64 → 0.52–0.56；ADD 示范上 p(ADD) 0.57–0.63，WAIT 示范上 0.36–0.39；`reason` 准确率 ADD 0.65–0.75、WAIT 0.65–0.69。与零状态 L3 相近，holdout 0。
      - `pi_0000` 于 09:34:24 冻结，来自 ckpt 34000，`teaching.train_policy.window_start` 为 `carry`。`repos.dreamerv3.dirty` 为 True：当时只有文档改动未提交，代码就是 `8bf5bc1`。
    - 考试：
      - DEV FG409（完整）：Reward 13.41；随机基线 15.49 ± 4.4（同 ADD 比例）。ADD 285 / 合法 2268，TP 230、SL 54，占仓 0.93。零状态 `pi_0000` 为 6.42，基线 16.92 ± 11.9。
      - TRAIN_SMOKE：FG001 为 −0.32，基线 −1.93；FG005 为 −5.60，基线 −5.56。
- [x] D-5：第 1 轮 collect / check / CF（`iterate --until cf`）
  - 证据：`formal_carry_r1.sh`，09:43–10:09（考试 09:34–09:43），`run_state.step` 为 `cf_done`。合约 FG001、FG005、FG009，采集到真终止，check 全部 ok。
    - 采集：
      - FG001：27,463 步，合法 7261，ADD 53，TP 43，Reward −2.35；
      - FG005：20,504 步，合法 2041，ADD 58，TP 46，Reward −3.38；
      - FG009：27,599 步，合法 9201，ADD 203，TP 179，Reward 70.44。
    - CF（全部 ADD 加 100 个 WAIT 分叉）：有效对 153 / 158 / 303，共 614 对，全部 `VALID_COMPLETE`；ΔG 的 std 分别为 0.45 / 0.59 / 0.61。
- [x] D-6：正式 L4 前的小样 Q 状态一致性检查（Q_ADD、Q_WAIT、ΔQ；诊断用 L4，不写入正式 run）；必要时用同一 carry 方案
  - 证据：`C4_STATE_CHECK_20261024/carry_pi_0000/` 与 `carry_pilot_l4/`。窗口从存储起步，换一组策略种子（offset 1,000,003），在 FG005 上取 1000 个 64 行窗口，所有行都比较（没有预热）。存储与路径的差为 0.0。
    - 新 `pi_0000`（Q 为 0）：
      - z 差异 / std 为 0.24–0.27，有符号偏差 −0.0007 std，按位置在 −0.0035 到 +0.0004 之间；
      - 合法行 6098 中翻转 120（2.0%）；ADD 数：窗口 194，路径 164；WAIT→ADD 75，ADD→WAIT 45；
      - |z| < 0.5 std 的 510 行中翻转 90 次，|z| ≥ 0.5 std 的 5588 行中翻转 30 次。
      - 对照：同种子窗口差异 0、翻转 0（D-3 和测试）。剩下的是随机隐变量的采样噪声，没有偏向。
    - 诊断 L4：从 `pi_0000` 出发，用本轮 CF 训练 3000 次，carry 开窗，TRAIN_INTERNAL_DIAGNOSTIC holdout（每个合约最后 2048 行，135 对）。按 β_Q 1、κ_Q auto 冻结为 `pilot_l4`（κ_Q 0.551，来自 479 对）。
      - 留出曲线：q_pair 在 500 次时最好（0.354，常数 0.352），1000 次 0.398，2000 次 0.508；q_cf 在 500 次时最好（3.34，常数 3.29）；q_real 最好 3.04，出现在 1000 次（常数 3.01）；符号一致率在 500–2000 次之间为 0.49–0.58（常数 0.46）。Q 没有在留出集上超过常数，500 次之后转为过拟合。
      - 状态一致性（Q 状态检查）：
        - 差异 / std：ΔQ 0.085、Q_ADD 0.18、Q_WAIT 0.16；
        - 有符号偏差：ΔQ +0.0004、Q_ADD +0.0024、Q_WAIT +0.0026 std；
        - 6098 个合法行上 ΔQ 符号一致率 87.4%；
        - 带锚点的 z 翻转 245 / 6098（4.0%），ADD 数：窗口 414，路径 419；ADD→WAIT 125，WAIT→ADD 120，两个方向对称。
      - 结论：Q 的窗口状态没有偏差，正式 L4 用同一 carry 方案。正式 L4 预算定为 500 次，即 q_pair/q_cf 留出值最好的位置，写入 `formal_v1_carry.yaml`。
- [x] D-7：正式 L4 及本轮其余步骤；记录、提交
  - 证据：`formal_carry_r1_l4.sh`（dreamerv3 `375cf83`），`formal_train iterate --rounds 1`，`run_state.step` 为 `done`。
    - 训练与冻结：
      - L4：11:43:52–11:45:54，500 次，carry 开窗，使用全部 614 对；
      - L5（S4 先验刷新）：100 次，carry 开窗；
      - `pi_0001`：11:46:58 冻结，来自 learner ckpt 34600。β_Q 1.0，κ_Q 0.559（614 对的合并 std）；`lineage.q_teacher_policy` 为 `pi_0000`；L4 和 L5 的 `window_start` 都是 `carry`。`repos` 为 dreamerv3 `375cf83`，dirty 为 True（当时只有文档改动未提交）。
    - DEV FG409（完整，31,034 分钟）：
      - Reward 17.69；随机基线（同 ADD 比例，5 个 seed）15.24 ± 5.98。
      - ADD 304 / 合法 1479，TP 246、SL 58，占仓 0.95。
      - `best_dev` 为 `pi_0001`：每分钟 0.00057，`pi_0000` 为 0.00043。
      - 与基线的差在基线自身的波动之内，而且只有一个 DEV 合约，不能说明交易能力更好。
    - `pi_0001` 的 state-check（`C4_STATE_CHECK_20261024/carry_pi_0001/`，11:54–12:05）：FG005，1000 个 64 行窗口，从存储起步，换一组种子。
      - 存储与路径的差为 0.0。
      - 差异 / std：ΔQ 0.049、Q_ADD 0.10、Q_WAIT 0.10、z 0.25；有符号偏差分别为 −0.0011、−0.0027、−0.0022、−0.0021 std。
      - 6098 个合法行上 ΔQ 符号一致率 94.7%；正式动作翻转 220（3.6%），ADD 数：窗口 345，路径 309。
      - ΔQ 本身很小（std 0.015），锚点 clip(ΔQ/κ_Q) 对 z 的影响也很小。
    - 本轮记录：`L5` 为 `S4_PRIOR_REFRESH`，`L6` 为 `L6_NOT_IMPLEMENTED`。
    - 可续训：基础学生 ckpt 34000，learner ckpt 34250、34500、34600，以及冻结的 `pi_0000`、`pi_0001`。再次运行 `iterate`，会在 `max_rounds` 1 处停下。

## 开发 run 与 10x 正式 run（用户 2026-10-24）
- [x] E-1：10x 正式配置 `formal_v1_l1_l2_10x_continue.yaml`：与 `formal_v1_carry.yaml` 只差步数（各阶段 ×10，保存间隔 ×10）和 WAIT 分叉（×10，dev 的 100 是为了快）；用独立的 dreamerv3 + gym worktree；用户手动启动的 L1 不重跑
  - 配置：去掉注释后与开发配置只差 `run_root`、L1/L2 累计目标 100,000/120,000、L3 120,000、L4 5000、L5 1000、保存间隔 10,000/2500、`cf_wait_forks` 1000。用户草稿（L3 12,000、L4 3000、L5 100）按 10x 统一。
  - worktree：用户已建 dreamerv3（`/data/research/fg-s3-thinking-student-l1-l2-10x-continue-20261024/dreamerv3`，分支 `run/thinking-student-l1-l2-10x-continue-20261024`）；同名分支的 gym worktree 新建在同一目录下，起点 `0056296`。
  - L1 不重跑：手动启动的 `train-market --updates 90000` 与编排器的调用参数相同；它用 `8bf5bc1`，之后 L1 代码没有改动。速度 0.215 s/次，到 100,000 约在 15:30。
- [x] E-2：测试 10x run 将走的 L2 carry 路径：L2 只改 outcome 组；拷贝的 ckpt 上手动续 L1 之后，pretrain 用 L1 的新脑建 store，L2/L3 读 store；store 等于 `pi_0000` 部署时的市场记忆
  - `test_carry_windows_train_only_their_stage` 增加 L2（只改 outcome 组）；新测试 `test_orchestrator_trains_l2_on_the_carry_of_an_l1_continued_by_hand`：调用顺序为 market-carry ×2 → train-outcome → train-policy → freeze → exam ×2，两步都带 carry 参数；存储与 `pi_0000` 的部署记忆逐位相同，与续 L1 之前的不同。
  - `test_ts_v1_c4.py` 17 passed（CPU，12 分钟）。
  - 真实数据冒烟 `C4_L2_CARRY_SMOKE_20261024`：18 合约 size50m，从 ckpt 22000 起步、使用开发 run 的存储，60 次，exit 0，49 s（含启动），RSS 3.6 GB，每个 batch 1024 行都计损失。
- [x] E-3：开发 run 第 2–6 轮（`max_rounds: 6`，18 合约轮转一遍），快速跑完以验证代码；2026-10-25 00:38 完成（`formal_carry_r2_r6.exit` = 0）
  - 第 2 轮完成；第 3 轮 FG205、FG209 的 CF 完成。FG201 的 CF 原样要跑 19–24 小时，19:53 按用户决定停下（每个 worker 跑到 250/532 个分支，结果只在结束时写出，所以没有保存），修复见 E-5，修好后从磁盘续跑。
  - 20:38 从磁盘续跑（`formal_carry_r3_resume.sh`，tmux `d3`）：
    - FG201 的 CF 按编排器的参数加 `--budget 48` 手动运行，用 6 个 worker 而不是 3 个。每个 worker 都重放整份 collection，所以成对结果与 worker 数无关。
    - 实际模拟 1596 个分叉中的 298 个：250 个能在上限内到达终点，另 48 个是预算样本；其余 1298 个记 UNAVAILABLE_BUDGET。
    - FG205、FG209 的 CF 保留不动。
    - 接着由 `formal_carry_r2_r6.sh` 跑完第 3–6 轮，最后做 `pi_0006` 的 state-check。
  - FG201 的 CF 于 21:54 完成，用时 76 分钟，分支共 1,515,824 步。
    - 250 对 VALID_COMPLETE/TERMINAL，都以 MAIN_ROLL 结束，cursor 23071–31229，全在合约最后 8192 分钟内；其中 ADD 242、WAIT 8。
    - 48 个预算样本全部跑满 8192 步，记 NOT_COMPARABLE/CAP（end_status ACTIVE），没有 MARGIN_CALL。
    - 1298 个记 UNAVAILABLE_BUDGET/BELOW_FLOOR_BEYOND_CAP。
    - `couple_from` = 31259，即整条参考轨迹上没有可耦合的 cursor。
  - 第 3 轮按规则跳过 L4：`L4_SKIPPED_CF_NOT_REPRESENTATIVE`，FG201 覆盖率 0.1566 < 0.8。
    - L5 照常。
    - `pi_0003` 冻结时保留 `pi_0002` 的 β_Q 1.0、κ_Q 0.7255（`L4_SKIPPED_ANCHOR_KEPT`）。
    - 与 `pi_0002` 相比只有 decision 组的参数变了，value、market、outcome、fusion、account 逐位相同。
  - 到第 3 轮为止的 CF 覆盖：
    - 9 个合约共 5696 个分叉，有效 4350 对，占分叉的 76.4%，占空仓合法决策（28,512 个）的 15.3%。
    - 除 FG201 外，每个合约的分叉都 100% 有效。FG201 的 250 对是后段的选择性样本，不能代表全部状态。
  - 第 4–6 轮的每个合约，分叉全部有效，也没有只能完整模拟判定的分叉：

    | 轮 | 合约（有效对 = 分叉） | κ_Q | 有效对合计 |
    | --- | --- | --- | --- |
    | 4 | FG301 284，FG305 307，FG309 580 | 0.594 | 1171 |
    | 5 | FG401 635，FG405 409，FG809 146 | 0.600 | 1190 |
    | 6 | FG901 182，FG905 135，FG909 157 | 0.457 | 474 |

    这三轮都正常训练了 L4 500 次、L5 100 次。
  - DEV FG409 每分钟 Reward：

    | 策略 | π_0 | π_1 | π_2 | π_3 | π_4 | π_5 | π_6 |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Reward | 0.000432 | 0.000570 | 0.000476 | 0.000497 | 0.000279 | 0.000291 | 0.000386 |

    `best_dev` 仍是 `pi_0001`。开发 run 预算很小，这些数只说明闭环跑通，不说明策略变好。
  - `pi_0006` 的 carry state-check（FG005，1000 个窗口，种子偏移 1000003）：
    - 存储与部署路径的差为 0。
    - 合法行的动作翻转：

      | 策略 | 翻转率 |
      | --- | --- |
      | `pi_0000` | 2.0% |
      | `pi_0001` | 3.6% |
      | `pi_0006` | 6.1%（375 / 6098） |

    - 翻转集中在决策边界附近。|z| 在 0.25 std 以内的合法行，`pi_0000` 有 227 个，`pi_0006` 有 749 个；离边界 1 std 以上的 3044 行只翻转了 8 次。
    - ΔQ 符号一致率 0.965。
- [x] E-5：有限预算的精确 CF（用户 2026-10-24 方案 3）；提交 gym `2b4e894`、dreamerv3 `8be867b`
  - 回归（CPU，20:35–21:02）：gym `test_ts_v1_cf.py` 加 dreamerv3 `test_ts_v1_s0`–`s4`、`test_ts_v1_c4.py`，56 passed，0 failed。
  - 起因：π_2 在 FG201 上 ADD 1499 次，权益从 300,000 降到 36,987（最低 36,423，始终为正；每份 collection 都正常结束于 MAIN_ROLL，没有 MARGIN_CALL，也没有被拒的 ADD，符合 Native 规则：3 手保证金最多 18,048）。耦合下限是 90,336，参考轨迹每一行都低于它，所以不可能耦合。1596 个分叉里有 1346 个在 8192 步内到不了真实终点，只能跑满上限：没有标签，耗时 19–24 小时。
  - 定义：无法耦合 ≠ 不会终止。分支自己的账户仍可能在上限内被 MARGIN_CALL，这是有效的 TERMINAL。所以这类分叉（`Reference.decidable` 为假：上限窗口内不可能耦合，真实终点又在上限之外）只能靠完整模拟来判定。
  - 预算 `iteration.cf_budget: 48`（每个合约）：在这类分叉里按 参考动作 × 权益四分位 × 60 分钟波动率高低 分层，做系统抽样（固定种子，所用信息在决策时都已知，不看结果），精确模拟抽中的；其余标 `UNAVAILABLE_BUDGET/BELOW_FLOOR_BEYOND_CAP`，没有 Reward、回报、TERMINAL 或标签，不进 Q 监督。能耦合或能到终点的分叉照旧全部模拟。`cf_max_steps` 仍为 8192，回报定义不变。
  - `meta.json` 增加 `forks.undecided`、`couple_from`、`budget`（各层的分叉数和模拟数）、`end_status`（含 MARGIN_CALL 次数）；每个结果带 `end_status`。
  - L4 代表性：`iteration.cf_min_coverage: 0.8`。只要有一个合约的有效对数不足其分叉数的 80%，本轮就跳过 L4（`L4_SKIPPED_CF_NOT_REPRESENTATIVE`）：π_{k+1} 保留 π_k 的价值脑，freeze 时用 `--value-anchor-of π_k` 保留 β_Q、κ_Q（`L4_SKIPPED_ANCHOR_KEPT`，lineage 记 `value_kept_from`）。
  - 测试：gym `test_only_a_full_branch_decides_a_fork_that_cannot_couple_or_end_in_the_cap`，s3 `test_a_cf_budget_leaves_unsimulated_forks_unavailable`，c4 `test_orchestrator_keeps_the_value_brain_when_the_cf_is_not_representative`。
- [ ] E-6（后续研究，用户 2026-10-24）：亏到下限以下的合约，CF 只能覆盖一部分有选择性的状态（FG201 约 16%，都是上限内能到终点的后段），Q 可能主要从容易完成的轨迹学习，错过高风险状态。如果多个合约持续出现这种情况，再优化 Native 快照的完整历史复制和 CF 的重复推进，必要时研究估计目标；不为 FG201 重做 Q 学习算法。
- [ ] E-7（待用户决定，2026-10-25；10x 跑到第 7 轮之前定下来）：L6 固定比例补课的副作用
  - **后果错补课让后果脑在自然分布上变差**（`S5_L2_REVIEW_DIAG_20261025/`）：
    - 从 `pi_0006` 起做 200 次 L2 复习，在留出数据上评估（18 个合约各最后 2048 个 cursor）：

      | 留出指标 | 起点 | 补课组 | 只用自然窗口的一组 |
      | --- | --- | --- | --- |
      | 事件 NLL | 2.325 | 2.435 | 2.33–2.40（波动） |
      | quality | 1.800 | 2.146 | 1.800（不变） |
      | 平均 p_SL（实际 SL 占 14.4%） | 0.141 | 0.198 | — |

    - 开发 run 中同一 cursor 的部署答卷结论相同：`pi_0007` 在没补过课的 FG101、105、109 上，log loss 从 0.42–0.47 升到 0.47–0.52。
  - **原因：** 后果错多半是校准正常的低概率尾部，无法逐条与偶然区分。补课窗口又集中在 SL 密集的时段，改变了后果脑看到的类别频率。
  - **错题少时同几道题被反复补：** 第 8 轮 6 道后果错配了 800 个补课窗口，每道约 133 次。10x 的 L2 复习是 2,000 次，重复次数也会乘 10。
  - **L5 示范错补课：** 100 次后，补课组的留出示范损失为 0.608，只用自然窗口为 0.584（起点 0.656）。自然改进少了约三分之一。
  - **没测的部分：** 补课能否让学生在自己的同类错误上泛化得更好。
  - **可选方案：**
    - A（建议）：`l6.remedial_fraction: 0`，L2 复习和 L5 只用自然窗口；保留教学题、L2 自然复习和 L4 的决策错加权。只改两份配置。
    - B：保留补课，但不改变自然分布：补课窗口按抽样概率反加权，并限制每道错题的补课窗口数；再用下一轮新错题检验效果。需要改代码和测试。
    - C：保持现状。
- [x] E-4：同步：开发分支的新提交快进到 10x 的 run 分支（dreamerv3、gym）
  - 2026-10-24 13:2x：用户的未跟踪草稿备份到会话临时目录后删除，dreamerv3 run 分支快进到 `37cdd85`；gym run 分支即 `0056296`，两个 worktree 都干净。
  - 从 10x 的 gym worktree 运行 `formal_train status`（只读）：配置可读，L1 60,000 / L2 12,000，latest ckpt 72000。
  - 以后开发分支每有新提交都这样快进；编排器的步与步之间生效，不改正在运行的那一步。
  - 2026-10-25 00:3x：10x 的 pretrain 在 L2 = 120,000 处按 `--until L2` 停下（exit 0，00:30:55），没有步骤在运行。两个 run 分支快进到 dreamerv3 `fbd6f59`、gym `2b4e894`（带上 CF 预算修复），worktree 干净。L3 由用户手动启动。

## S5 — 自主教学循环 + Teacher-free 考试（计划：`PLAN_S5_20261024.md`；用户 2026-10-24：先做 OFFLINE_TRAIN 的 L6，TIME_ORDERED 明确标为未支持）
- [x] S5-1：B1 教学题 `thinking_student/feedback.py`：一道题 = 一个决策行；四类反馈（TEACHER_ADD_FACT、STUDENT_EXECUTED_NATIVE、VERIFIED_CF_NATIVE、HINDSIGHT_ORACLE_TEACHING）各带 source、ready_at、valid；后果错、决策错、teachers_agree；`release` 只放行 OFFLINE_TRAIN，TIME_ORDERED 报 `TIME_ORDERED_NOT_SUPPORTED`；只读 TRAIN 教材；用第 1 轮真实数据定阈值
  - 实现：`thinking_student/feedback.py`，CLI 与 `exam.py` 同样写法。
  - 预算 CF 未模拟的分叉记为 UNAVAILABLE_BUDGET，与 NOT_COMPARABLE 一样弃权；summary 里两者分开计数（`abstained.cf_not_comparable`、`cf_unavailable_budget`），并记 `cf_coverage`。
  - 阈值的真实数据依据（诊断目录 `S5_FEEDBACK_DIAG_20261024/`，不写入 run）：
    - 第 1 轮 `pi_0000`：后果错 398（FG001 253、FG005 104、FG009 41；阈值 p < 0.1），决策错 101，示范错 34，κ_Q 0.5587，CF 覆盖率 1.0。
    - 第 2 轮 `pi_0001`：后果错 182，决策错 362，示范错 284，κ_Q 0.7255。
- [x] S5-2：B2 课程：`CorpusWindows` / `MultiCorpusWindows` 的补课窗口（每批 round(fraction × B) 个窗口含错题行，标签与自然窗口相同），`QWindows` / `MultiQWindows` 的决策错加权；报告分开记唯一 cursor、补课曝光、优化器曝光
  - 用第 1、2 轮的真实教学题核对，按开发配置每批 16 × 64、每种取 20 批（`scratchpad/s5/remedial_check.py`）：
    - 后果错：398 / 182 行，都带标签、没有被丢弃；补课窗口占 25%（80 / 320）。
    - 示范错：34 / 284 行，补课窗口占 25%。
    - 决策错：101 / 362 行，Q 起点权重 ×4 后，覆盖错题行的窗口占 13% / 34%。
- [x] S5-3：L6 接入 `formal_train iterate`：配置 `iteration.l6`（默认关）；一轮为 collect → check → CF → feedback → L2 复习 → L4 → L5 → freeze → exam；L2 复习只动 outcome / fusion，不训 L1，MarketCarry 存储保持有效；run_state 与 lineage 记 OFFLINE_TRAIN 和 TIME_ORDERED_NOT_SUPPORTED；续跑跳过已有教学题
  - 与 E-5 的规则一致：某轮 CF 没有代表性（`L4_SKIPPED_CF_NOT_REPRESENTATIVE`）时：
    - L5 也不接这份 CF 产生的示范错补课，记 `l6.decision_remediation: SKIPPED_CF_NOT_REPRESENTATIVE`；
    - L2 复习只用事实的后果错，照常进行；
    - freeze 同时带 `--value-anchor-of π_k` 和 `--feedback`。
- [x] S5-4：测试 `test_ts_v1_s5.py` 与回归（c4、s1–s4）
  - 合入开发 worktree 之前，在 scratchpad 的副本（HEAD `8be867b` + S5）上测试（CPU）：

    | 测试文件 | 结果 |
    | --- | --- |
    | `test_ts_v1_s5.py` | 6 passed |
    | `test_ts_v1_c4.py` | 18 passed |
    | `test_ts_v1_s0`–`s4` | 全部 passed |

    共 57 passed，0 failed。
  - 唯一一次失败是测试写错了：regret 的期望值符号反了。已执行 WAIT、ΔG = +1e-17 时 regret 应为 +1e-17。改正后该测试通过，代码没有改。
  - S5 编排测试新增第 3 轮：CF 覆盖不足时，L2 复习照常，跳过 L4，L5 不带补课，`pi_3` 保留 `pi_2` 的价值脑和锚。
  - 拷入开发 worktree 后，用 `diff -rq` 核对与副本逐字节相同；`test_ts_v1_c4.py` 第 547 行的期望改为 `L6_SKIPPED_BY_CONFIG`。
- [x] S5-5：配置并实跑：开发 `formal_v1_carry.yaml` 的 `max_rounds` 9、`l6.from_round` 7；10x 配置同一块、L2 复习 ×10；开发 run 第 7–9 轮（与第 1–3 轮同合约），DEV Teacher-free 考试；`pi_0009` 在 FG005 上做 carry state-check
  - 配置随 S5 提交（`fbcb368`）。
  - 开发 run 于 2026-10-25 00:42–03:45 跑完第 7–9 轮和 state-check（`formal_carry_r7_r9.sh`，exit 0）。
  - 结果：`S5_RECORD_20261024.md`；各轮的表：`FORMAL_V1_CARRY_20261024/s5_rounds_report_7_9.md`。
  - 每轮依次为：教学题 → L2 复习 200 次 → L4 500 次 → L5 100 次 → freeze → DEV。
  - 第 9 轮：
    - FG201 的 CF 覆盖率 0.169：1591 个分叉中 1274 个 UNAVAILABLE_BUDGET。
    - 按规则跳过 L4 和决策错补课。
    - `pi_0009` 保留 `pi_0008` 的价值脑和锚（`L4_SKIPPED_ANCHOR_KEPT`）。
  - 错题数（第 7 / 8 / 9 轮）：

    | 类别 | 第 7 轮 | 第 8 轮 | 第 9 轮 |
    | --- | --- | --- | --- |
    | 后果错 | 432 | 6 | 91 |
    | 决策错 | 81 | 347 | 255 |
    | 示范错 | 38 | 284 | 182 |

  - 9 份采集都没有违规，成交与事实逐笔一致。
  - 参数：L2 复习只改 fusion、outcome、`opt_l2`；三轮中 market、account、`opt_l1` 都不变。
  - DEV 每分钟 Reward：

    | 策略 | `pi_0006` | `pi_0007` | `pi_0008` | `pi_0009` |
    | --- | --- | --- | --- | --- |
    | Reward | 0.000386 | 0.000335 | 0.000394 | 0.000371 |

    都低于同入场率随机基线的均值；`best_dev` 仍是 `pi_0001`。
  - `pi_0009` 的 state-check：
    - 存储与部署路径的差为 0；
    - 合法行翻转 6.9%，方向对称，集中在边界附近；
    - ΔQ 符号一致率 0.911。
  - 补课诊断（`S5_L2_REVIEW_DIAG_20261025/`）发现固定比例补课有副作用，见 E-7。
- [x] S5-6：记录 `S5_RECORD_20261024.md`；更新 INTERFACE_SUMMARY、FORMAL_TRAINING_IMPLEMENTATION、配置 README、memory；提交、同步 10x 分支
  - 已写：
    - 记录 `S5_RECORD_20261024.md`；
    - INTERFACE_SUMMARY 和 FORMAL_TRAINING_IMPLEMENTATION 的实跑结果、checkpoint 表与待办；
    - memory。
  - 配置 README 不需要改：E-7 未定，配置没有变化。
  - 提交 dreamerv3 `90bfc1d`（只有文档）；gym 没有新提交，仍为 `2b4e894`。
  - 2026-10-27：10x 没有步骤在运行（`pi_0000` 已冻结，iterate 未开始），dreamerv3 run 分支快进到 `90bfc1d`，gym 仍为 `2b4e894`，两个 worktree 都干净。
