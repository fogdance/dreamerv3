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
- [x] E-7（用户 2026-10-28 选方案 A）：L6 固定比例补课的副作用
  - 实施（集成分支 dreamer `86597da`）：`formal_v1_carry.yaml`、`formal_v1_l1_l2_10x_continue.yaml` 和探索配置都设 `l6.remedial_fraction: 0`；L2 复习和 L5 只用自然窗口；教学题、L2 自然复习和 L4 决策错加权（`remedial_boost: 4.0`）保留。补课代码和 S5 架构不变。
  - 实跑核对见 E-8 的小预算编排器验证：L2 复习和 L5 的 `remedial_windows` 为 0，L4 的决策错行照常加权。
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

## E-8：归档 Claude CF，Review 并集成 Astra 修复（用户 2026-10-27）
- [x] 归档 Claude 方案：两仓 `archive/ts-v1-cf-claude-20261027`，gym `21105bf`，dreamer `25c7a42`（含开发 worktree 中未提交的流式修改，未测试）。开发 worktree 未清理、未重置。
- [x] 集成分支 `integration/ts-v1-cf-astra-20261027`（worktree `fg-s3-ts-v1-cf-integration-20261027/`），基于 S5 最新提交，没有冲突：
  - gym：`08e3464` 回退 Claude，`c359c29` 引入 Astra `7596566`；
  - dreamer：`54a7184` 回退 Claude，`a9388cc` 引入 `4c1916f`，`e64eb5f` 引入 `b403091`。
  - 只保留一套 CF 生产器；S5 代码与记录保留。
- [x] Review（读源码和已有结果），没有发现必须修的 Bug：
  - EXACT_STATE：
    - `native_continuation` 包含 Claude 独立核对过的全部未来相关状态，还多了若干项（只会更保守）。
    - 策略只读市场、is_first 和三个账户输入，不读 reward；carry 和 `n_actions` 逐项比较。env 的 RNG 只在 reset 时使用。
    - FG201 t=7 的 G = 0.21897048173250155，与 Claude 跑完整 Native 的值相差 −1.39e−16。
  - FUNDS_SAFE：
    - 资金不等式没有改，只增加了策略状态相同和控制项相同两个条件，没有放宽。
    - TS env 的 `max_entries_per_day` 为 int64 最大值，所以每日上限保护不会触发。
  - 标签：
    - 旧 250 条与补算文件逐条相同；新增 664 条全部是 EXACT_STATE，权益差为 0。
    - 639 条 UNAVAILABLE 和 43 条 CAP 都没有 g 和 delta_g；`branch_rewards` 没有被任何读取方使用。
    - `cf_targets` 与 `pair_items` 只认 VALID_COMPLETE。
  - 产物：开发 run 的 pi_0002、CF 和 market_carry 在 10-25 之后没有被改动。
  - 运维提示：每个合约 CF 的 `work/` 目录约 4.5 GB（FG201）。/data 剩余 291 GB，/ 剩余 69 GB。
- [x] GPU 定向回归（集成代码 gym `c359c29`、dreamer `86597da`；`CF_ASTRA_INTEGRATION_20261027/regression.py`，2026-10-28 00:06–00:13）：11/11 通过。
  - FG201 pi_0002 `--cursors 7 20789 31153 31229 --budget 2`：
    - 7：EXACT_STATE，4 步后在 11 处合并，G 与补算、完整 Native 相同；
    - 31229：探测内 TERMINAL；31153：探测 64 + 续算 42 步后 TERMINAL，G 与旧值相同；
    - 20789：8192 步 CAP，没有 g、delta_g。
  - FG205 pi_0002 `--cursors 243 3932 --budget 2`：在 244、4350 处 FUNDS_SAFE 合并，G 为 0.7659586225999391、0.027956332827719615，与旧值相同；3932 在续算中完成。
  - 参考重放与采集一致；`q_cf_valid` 只在有效对上为 1；已结束分叉的快照在输出完成后删除。
- [x] 显式探索开关 `iteration.cf_coverage: exploratory`（`86597da`）：`cf_min_coverage` 保持 0.8；覆盖不足时 L4 照常训练，轮记录写 `S3_Q_TARGETS_EXPLORATORY` 和 `L4_coverage`，标签与覆盖率不变。π₂→π₃ 探索配置 `formal_v1_carry_pi2_cf_explore.yaml`：新 run 用符号链接接入开发 run 的 pi_0002 经历，FG201 的 CF 用补算结果，L6 从第 3 轮开始。E-7 已由用户定为方案 A。
- [x] TS 测试（CPU，2026-10-28 00:07 起，gym `c359c29`、dreamer `86597da`，`CF_ASTRA_INTEGRATION_20261027/tests/`）：
  - gym（cf、account、corpus、市场分钟、Native env、MACD、行情缓存）61 passed；dreamer s0–s3 29 passed；s4–s5 10 passed；c4 17 passed、1 failed。
  - 失败的是集成时新加的探索断言：夹具中 pi_0 没训练过 L4，L4 优化器从学习率 0 线性预热（`linear_schedule(0, lr, 1000)`），所以唯一一次 L4 更新只改优化器状态、不改权重。生产代码不变；测试的探索分支改为 2 次 L4 更新后通过（`c4_fixed_test.log`）。
- [x] 小预算编排器验证 π₂→π₃′（只用于验证，不是正式轮次；`verify_pi2_r3/`，`verify.json` 14/14 通过）：
  - `--until cf` 后从磁盘续跑，第三次调用报 max_rounds；不重采集、不重算 CF。
  - FG201 914/1596 原样进入，`L4 = S3_Q_TARGETS_EXPLORATORY`，`L4_coverage.below = {FG201: 0.5727}`。
  - 教学题：FG201 的 CF 914 条；三份反馈与 Astra 独立重建的逐列相同；合并 κ_Q 0.6244509768525569。
  - L4：2340 对（914 + 818 + 608），决策错 289 行 ×4；价值锚 β_Q 1.0 来自这 2340 对。
  - E-7 方案 A：L2 复习、L5 的补课窗口为 0。
  - 只改 outcome、fusion、value、decision；从 pi_0002 继承（35200 → 35216）；DEV FG409 完整；开发 run 和补算的源文件指纹前后相同。
- [x] 磁盘：删除 FG201 补算中 707 个已结束分叉的快照（2.48 GB，`clean_supplement_snapshots.json`）；保留 639 个未续算探测的快照、参考重放、全部标签和 checkpoint。正式训练的估算见 `CF_ASTRA_INTEGRATION_20261027.md` 第 6 节。
- [x] 文档：报告 `CF_ASTRA_INTEGRATION_20261027.md`；INTERFACE_SUMMARY、FORMAL_TRAINING_IMPLEMENTATION、配置 README。
- [x] 开发分支合并（2026-10-28，用户同意 `reset --hard`）：开发 worktree 的未提交修改（dreamer 两个文件，Claude 的流式分叉快照）与归档提交 `25c7a42` 逐文件相同，没有未跟踪文件。`reset --hard` 后两仓 `git merge --ff-only integration/ts-v1-cf-astra-20261027`，`feat/thinking-student-v1-20261022` 到 dreamer `f9f67c6`、gym `c359c29`。
  - 流式修改不进主线：它减少的是 Claude `_cf_job` 在内存中同时保留的分叉状态；Astra 在参考重放时就把每个分叉写入 `work/<t>.pkl`，没有这个内存问题。

## E-9：正式 L6 第 3–9 轮（π₂ → π₉，用户 2026-10-28）
不再改 CF 算法，不再研究 FG201 剩余 682 条缺失标签；只在真实运行阻断时修复。
- Run：`FORMAL_V1_CARRY_PI2_CF_EXPLORE_20261028`，配置 `formal_v1_carry_pi2_cf_explore.yaml`，代码 `feat/thinking-student-v1-20261022`。启动脚本 `explore_rounds.sh ROUNDS TAG` 在 run 目录中。
- 结果表：`rounds_report.py` → `rounds_report.md`（同目录，只读）。每轮与开发 run 同轮、同合约对比：
  - ADD/WAIT 行为；
  - Outcome（实际结果的 NLL、后果错、TP 对 SL 的 AUC）；
  - Q（ΔQ 与 CF ΔG 的符号一致率和 Spearman、平均遗憾、决策错）；
  - L4 和 L2 复习的训练批次指标；
  - DEV FG409。
  - 开发 run 的教学题在 `S5_BASELINE_ITEMS_20261028/`，只作诊断。
- [x] 第 3 轮：π₂ → π₃（2026-10-28 06:24–06:33，用补算后的 CF，不重新采集）。
  - 记录：L4 = `S3_Q_TARGETS_EXPLORATORY`，`L4_coverage.below = {FG201: 0.5727}`，L6 = `S5_FEEDBACK_CURRICULUM`。
  - 更新数：L2 复习 200、L4 500、L5 100；补课窗口 0/0。
  - 价值锚：β_Q 1.0，κ_Q 0.6245，2340 对。
- [x] 核对第 3 轮产出（`verify_round.py 3`，结果在 `verify/round_3.json`）：11/11 通过。
  - learner：`learners/pi_0003/ckpt/000000036000`，由 35200 加 800 次更新得到，继承本 run 的 pi_0002。
  - 冻结的 π₃：权重哈希 `76ddca40…`，与 π₂ 的 `2da8671c…`、π₃′ 的 `edca0f5a…`、开发 run π₃ 的 `59a44f93…` 都不同；lineage 记为 OFFLINE_TRAIN。
  - 冻结时代码为合并后的干净版本。DEV FG409 考试完整（31034 分钟，Teacher-free），没有 TEST 考试。
  - π₃′ 和 Astra 独立 L4 不算正式轮次。
- [x] 第 4–9 轮：每轮重新采集、计算 CF、生成教学题、L2 复习、L4、L5、冻结、DEV。
  - 2026-10-28 06:34–10:10，`explore_rounds.sh 6 explore_r4_r9`，退出码 0。
  - 每轮 `verify_round.py` 都是 11/11 通过（`verify/round_4.json` … `round_9.json`）。π₉ 的 learner 是 `learners/pi_0009/ckpt/000000040800`。
  - 各轮 L4 都是 `S3_Q_TARGETS_EXPLORATORY`，补课窗口都是 0/0。
- [x] 各轮结果与对比，写入记录文档：`S5_FORMAL_ROUNDS_3_9_20261028.md`。
  - Outcome 和 DEV 都好于开发 run；Q 在两条线上都没有学到；DEV 都在随机基线附近。
  - 发现：L2 复习和 L5 每轮采样同一批窗口（种子只看本 learner 的更新数）。
- [x] 修复每轮采样同一批窗口（用户 2026-10-28：先修复，验证后再启动 10x 的 9 轮）：
  - 采样种子加入 agent 的更新数（继承自 π_k），每轮的 learner 因此各自采样；同一 learner 续跑仍然确定。
  - 新测试 `test_round_learners_draw_their_own_windows`：在修复前的代码上失败（下一轮抽到完全相同的窗口），修复后通过。
  - 验证：dreamer 6cc8f61 加本修改、gym c359c29，CPU 运行 `test_ts_v1_s0`–`s5` 与 `test_ts_v1_c4`，58 项全部通过（29 分 23 秒）。
  - 主线第 3–9 轮是修复前训练的；10x 第 1–9 轮使用修复后的代码。
- [x] 用满主机（用户 2026-10-28）：
  - 实测（10x π₀，FG205）：每个 CF worker 约 1.6 GB 内存、1.1 GB 显存；1、3、6 个并发分别为每秒 131、348、545 个决策。
  - 先按空闲内存（约 18 GB）把 `cf_workers` 从 3 改为 4：2 个合约 ×（4 个 worker + 1 个父进程），约 17 GB。
  - 用户随后要求再提高：不考虑本机的虚拟机，内存不够时用 swap。现在的上限是显存（每个 worker 约 1.1 GB，共 16 GB），GPU 也接近饱和。
  - 开发、主线和 10x 配置都改为 `parallel_contracts: 3`、`cf_workers: 4`：每轮 3 个合约同时采集、同时算 CF，共 12 个 worker。
- [x] 10x 配置加 `cf_coverage: exploratory`（用户 2026-10-28）：Astra CF 覆盖率低于 0.8 时，L4 照常学有效对，记为探索。
- [x] 10x 的 `cf_budget` 从 48 改为 480（用户 2026-10-28）：预算 48 时，第 1 轮 FG005 的 1114 个分叉只有 248 个有标签（0.22）。第 1 轮已算完的 CF 在原输出上加大预算续算，探测和重放不重做。
- [ ] 主线第 9 轮完成后启动 10x（用户 2026-10-28）：
  - [x] 第 1 轮跑到 CF 为止（`--until cf`）；CF 预算改为 480 后在原输出上续算，有效对 FG001 635/1125、FG005 680/1114、FG009 942/1318。
  - [x] 用该轮 CF 的留出对试跑 L4（`FORMAL_10X_L4_PILOT_20261028`，2005 对训练、252 对留出，8000 次）：
    - 留出 pair 损失在 1500 次最低（1.2475，常数 1.5115），2000 次 1.2826，2500–8000 次在 1.35–1.46；
    - 符号一致率始终不高于常数基线的 0.8386；
    - `value_updates` 从 5000 改为 1500。
  - [ ] 第 1–9 轮（`formal_10x_iterate.sh rounds_1_9 --rounds 9`）由用户启动（用户 2026-10-28：L4 的正式启动由用户来做）；第 1 轮从磁盘上的 CF 继续，从 L4 开始。

## G：教材调整与完全重训——分档多答案示范、有利空间、正确集（用户 2026-10-28）

起因：用户把 FG001（2019-08-12）的 P4 严格老师画成图，发现每段行情只有一个 ADD 答案，相邻同样好的入场都标成 WAIT。

用户的决定：
- 分档老师代替严格老师，作为唯一的示范：
  - 止盈时间：≤15 / ≤120 / 日内 / 跨日；
  - 回调 `mae_full`：≤3 / ≤5 / ≤10 / >10；
  - 有利空间：≥10 / ≥7.5 / ≥6 / <6，从成交看到止盈后 60 分钟，不跨日；
  - 总档取最差的一档；≥良 标 ADD，差 标 WAIT，一般 不标。
- Outcome 增加 MFE（15 / 60 / 120 分钟）预测，并放进 evidence。
- 正确集默认开启，错题集可选。
- 完全重训，不做兼容。

计划：`~/.claude/plans/thinkstudent-s3-next-worktree-temporal-fountain.md`。

- [x] G-1：gym `research/ts_v1_grades.py` 生成 `graded/`（成交价、有利空间、MFE、分档、示范标签），`produce()` 同步写出；配置 `configs/thinking_student/entry_grades.yaml`
  - 止盈可以发生在成交分钟本身：env 用该分钟收盘价的可成交价判断，并要求开仓后剩余的成交量还够平 3 手（Volume ≥ 6）。有利空间和 MFE 也计入这一价格，此后各分钟用可止盈分钟的最高（最低）价。
  - `ts_v1_corpus` 的 CLI 必填 `--grades`；`LABELS` 增加 `graded`；`source.json` 记录分档阈值和计数。
- [x] G-2：gym 测试 `test_ts_v1_grades.py`：成交价和方向与 `candidate(t)` 一致；止盈分钟等于第一次达到 +6 的分钟；MFE 与止盈时间等价；分档映射
  - 9 项通过（18 秒）。用 400 分钟、跨两个交易日的合成行情；有利空间和 MFE 的窗口与逐分钟走查的结果逐一相等；分档用正式阈值和一组能产生全部四档的阈值各验一遍；`wp02_fixture.ts_corpus` 显式给出阈值 `GRADES`。
- [x] G-3：18 个 TRAIN 合约补写 `graded/`，做全量核对；FG001 2019-08-12 逐分钟对照 strict
  - 补写：FG001 单独 24 秒，其余 17 个并行 50 秒，单进程峰值约 290 MB。证据目录：`/data/logdir/fg_single3/thinking_student/GRADED_TEACHER_20261028/`（脚本、日志、`check_graded.json`、`FG001_2019-08-12.txt`）。
  - 全部通过：391,979 笔 TP5 的止盈分钟一致；MFE 15/60/120 与"日内且 h 分钟内止盈"100% 等价；日内 TP5 的有利空间都 ≥6；18×64 个抽样成交的价格和方向与 `candidate(t)` 一致。
  - 合计（484,249 个成交 cursor）：优 30,106，良 74,263，一般 99,875，差 279,389，无档 616。示范 ADD 104,369（strict 24,926），WAIT 279,389（strict 423,749），不标 100,491。
  - strict ADD 中 77.7% 仍是 ADD，2.4% 变 WAIT（跨日或回调 >10），20.0% 不标；strict WAIT 中有 63,482 个变为 ADD。
  - FG001 2019-08-12：strict ADD 11 个，graded ADD 105 个、WAIT 10 个。11:26 的 strict ADD 之后 11:27–11:29 原为 WAIT，现为良或优，标 ADD。
- [x] G-4：dreamer 读 `graded/`，作为唯一示范和 MFE 标签；Outcome 加 MFE 头，evidence 43 维；删除 r2r3 和 `--migrate`
  - `labels.outcome_labels(facts, kinds, graded)` 只在窗口完整的行上给 `outcome_mfe`（点 / 30）；`demo_labels(graded)` 是唯一的示范来源。`TSCorpus` 拒绝没有 `graded/summary.json` 的教材，报错里给出补写命令；多教材必须用同一组阈值。
  - OutcomeBrain 加 `mfe` 头（3 个输出），evidence = 34 概率 + 2 风险 + 3 MAE + 3 MFE + 1 Quality = 43；`outcome_loss` 加 `outcome_mfe`（掩码平方误差），`student.yaml` 的 `loss_scales.outcome_mfe: 1.0`；答卷加 `answer/mfe_15|60|120`。
  - `transfer.py` 删除 r2r3、MIGRATIONS 和证据常量，`inherit_student` 只复制形状相同的参数，其余逐项列出；run.py 删除 `--migrate`；对应测试删除。
  - feedback 的示范来源改为 `HINDSIGHT_GRADED_TEACHING`，列名 `demo_*`；L2、L3 报告记 `grades` 和 `teacher: GRADED_ENTRY_TEACHER`。
- [x] G-5：正确集：`feedback.correct()`，采样器的 `correct_fraction` 和 `correct_boost`，run.py 和 formal_train 的传参（默认正确集开、错题集关）
  - 标记：`outcome_correct`（给实际结局的概率 ≥ 0.5）、`decision_correct`（执行动作好出 1.0 × κ_Q 以上）、示范对 = 决策对且分档老师示范的正是较优动作。`check_thresholds` 保证错与对不重叠（`outcome_error_prob ≤ outcome_correct_prob`，两个 margin ≥ 0），`feedback.build` 和 `formal_train.load_config` 都检查。
  - `MultiCorpusWindows(correct, correct_fraction)`：每批先放错题窗口、再放正确窗口、其余自然；两个比例之和 ≤ 1。`MultiQWindows(correct, correct_boost)`：同一行不能同时属于两个集合，倍数必须为正。
  - run.py：`--correct-fraction`、`--correct-boost`，与错题参数一样需要 `--feedback`；关闭的集合不传游标。formal_train `l6` 默认 `correct_fraction 0.25`、`correct_boost 4.0`、`remedial_fraction 0`、`remedial_boost 1.0`；轮记录改名 `decision_feedback`、`feedback_windows`，并记 8 个设置。
  - 主线 `pi_0008` FG205 用新代码重建教学题（证据目录 `feedback_pi_0008/`）：示范 655（strict 713）；错题：后果 6（全 SL30）、决策 75（全 ADD）、示范 58；正确：后果 662（TP5 661）、决策 88（ADD 83、WAIT 5）、示范 57。后果正确集几乎都是 TP，风险写进配置注释。
- [x] G-6：dreamer 测试（s1、s2、s5、c4）和 TS 全套回归，ruff
  - 最终代码上每个测试文件各开一个 CPU 进程：gym account 3、cf 6、corpus 9、grades 9；dreamer s0 5、s1 12、s2 8、s3 5、s4 4、s5 8、c4 18，全部通过。日志在证据目录 `logs/tests/final_gym/`、`final_dreamer/`。
  - 复查改动发现两处问题并已修正：`student.yaml` 缺少 `loss_scales.outcome_mfe`，L2 第一次更新就会 KeyError；错题和正确集的阈值可以设成重叠，同一行同时算错和算对。后者由 `check_thresholds` 拒绝，`QWindows` 也拒绝非正的倍数。
  - 18 个合约的 graded 表全部满足不变式：止损和其他出场为差，删失无档，跨日止盈为差，三个维度和总档相符，MFE 15 ≤ 60 ≤ 120，h 分钟内止盈的 mfe_h ≥ 5（`graded_invariants.py`，共 484,249 个成交 cursor）。
  - ruff F,E9：生产代码没有新问题；测试中只有原有的 fixture F811 写法，gym 只有原有的 numpy F401（`lint/final_*.txt`）。
- [x] G-7：新配置 `formal_v1_graded.yaml`（1x）、`formal_v1_graded_10x.yaml`，用 `status` 核对；GPU 冒烟
  - 两个配置都通过 `load_config` 和 `status`：18 个教材的分档阈值相同；1x 和 10x 除步数、CF/L4 设置外相同。
  - 与计划的差异：1x 的 `cf_coverage` 用 `exploratory`（与主线和 10x 一致）。预算 48 时每轮覆盖率约 0.5–0.6，representative 会让每一轮都跳过 L4，L6 的决策正确集也就无从使用。
  - 10x 的基础阶段按旧 10x 的实测速度估计约 15 小时（计划中写的 21 小时有误）。
  - GPU 冒烟（size50m，FG005 + FG405，L2 和 L3 各 4 次更新，然后冻结）通过：`gpu_smoke/`。
- [x] G-8：文档（FORMAL_TRAINING_IMPLEMENTATION、接口说明、configs README），gym 和 dreamer 各一个原子提交
- [x] G-9：启动：旧 10x 已按用户要求停止（2026-10-28 17:27:56，SIGTERM，退出码 143，停在第 5 轮 CF；pi_0000–pi_0004 已冻结，best_dev 是 pi_0001）。新训练放在单独的训练 worktree 对里。GPU 空出来后由我启动 1x 开发 run，10x 由用户启动。
  - 提交（未 push）：gym `7d5e6af`，dreamer `4221bfa`。
  - 训练 worktree 对：`/data/research/fg-s3-thinking-student-graded-20261028/{gym-trading-env,dreamerv3}`，分支 `run/thinking-student-graded-20261028`，两仓都是 clean。
  - 1x 开发 run `FORMAL_V1_GRADED_20261028`：2026-10-28 17:50:50 由 `graded_1x.sh graded_1x` 启动，即 `formal_train run --rounds 9`。
    - F1 继承：reused 49、sliced 1、widened 1、initialized 62。
    - L1 约 0.24 秒/次更新，占 GPU 2.2 GB。
    - 日志在 `logs/`，结束时写 `logs/graded_1x.exit`。
  - 10x run `FORMAL_V1_GRADED_10X_20261028`：启动脚本 `graded_10x.sh TAG [--after-1x]` 已放在 run 目录，由用户启动。`--after-1x` 会等 1x 结束、GPU 空出后再开始。
- [x] G-10：按代码画出学生大脑，并与 A1–A6 设计对比：`FG_S3_Thinking_Student_V1_Brain_From_Code_vs_A1_A6_20261028.md`（本目录和 `~/Documents/work/research/thinkstudent/` 各一份，内容相同）。同时补拷 thinkstudent 中缺少的 `FG_S3_TS_FG201_Astra_and_S5_gy35_d335_Review_20261025.md`。
- [x] G-11：1x 分档重训 `FORMAL_V1_GRADED_20261028` 第 1–9 轮的结果报告：`S5_GRADED_1X_ROUNDS_1_9_20261029.md`
  - 2026-10-28 23:55:14 结束，退出码 0；每轮 `verify_round.py` 都是 11/11（run 目录 `verify/`）。对比表 `rounds_report.md` 由 `rounds_report.py 1 9` 生成。
  - 诊断用教学题（本 run 第 1–6 轮、开发 run π₀–π₈，用同一版代码和分档阈值重建）：`GRADED_1X_ITEMS_20261029/`。
  - DEV：10 个策略的平均 0.000946/分钟（开发 run 0.000403），8/10 个高于同 ADD 率随机基线；best_dev 是 π₆（0.001449）。Q 仍未学到；执行的 ADD 中约一半仍是“差”档。E-7：π₇、π₈ 的新采集上没有看到 p_TP 被推高。
  - 10x `FORMAL_V1_GRADED_10X_20261028` 由用户于 2026-10-29 启动（`graded_10x.sh` 已按用户要求去掉 GPU 检查）。
- [ ] G-12：模拟盘（用户 2026-10-29）：冻结策略在 FG501、FG505、FG509、FG601、FG605（2024-08-19 → 2025-12-31）上 Teacher-free 交易，只看盈利，不用于训练或选策略
  - 代码（dreamer `0922fa2`）：`markets/paper_pool.json`（role `paper`，不可训练）与 5 个合约的 env/合约文件（合约条款取自 gym `dataset_v3_345_audited`，FG409 的条款与它逐字节相同）；`exam --paper-pool`（标签 PAPER）；`paper.py` 把各合约串成一个账户（合计净盈亏、整体回撤、月度盈亏、对随机进场）。
  - 核对：用 paper 路径重跑 1x π₆ 的 FG409，学生和随机账户的结果与原 DEV 考试完全相同；c4 测试 19 项通过；ruff 无问题。
  - 运行目录 `/data/logdir/fg_single3/thinking_student/PAPER_FG501_FG605_20261029/`：1x π₆ 于 18:30 开始；10x 的 best_dev 和 π₉ 由 `after_10x.sh` 在 10x 结束后自动运行（tmux `d3:paper_10x`）。
  - 结果（2026-10-30 04:02 全部完成）：净盈亏 1x π₆ −72,420（随机 −83,170，胜 4/5 个种子），10x π₄ −84,774（随机 −106,435，5/5），10x π₉ −88,332（随机 −101,441，4/5）；三者都亏钱，但都少于随机进场。
- [x] G-13：模拟盘的 TP/SL 可配置（用户 2026-10-30：1x 模拟盘止损 30 改为 5 再跑）
  - [x] gym：`entry.take_profit_points / stop_loss_points`（默认 5 / 30，仅 Single3）进入 SL/TP 价、TP 成交检查、Quality TP 打分器（TP 检查与势函数尺度）、其他退出的亏损尺度；学生的持仓输入保持训练尺度（/5、/30）。
  - [x] `exam --take-profit-points / --stop-loss-points` 只用于 PAPER，写入考试的 `inputs/env.yaml`，学生和随机账户同规则；`paper.py` 报告每个策略的 TP/SL，同一策略各合约规则不同则拒绝。
  - [x] 核对：默认规则重跑 1x π₆ FG605，学生与随机结果与原模拟盘完全相同；gym env/reward/account/CF 测试与 c4 paper 测试通过；FG501 SL5 实测止损恰为 5 点。
  - [x] 1x π₆ SL5 / SL16、10x π₄ / π₉ SL5 模拟盘已跑完（对比表 `PAPER_FG501_FG605_20261029/comparison_tp_sl.md`）。注意：模型按 TP5/SL30 训练，SL5 下它的胜率预测与 Q 不再对应真实规则。

- [x] G-14：E0 诊断（`TS_V1_Integrated_Review_20261030.md` §6，用户 2026-10-30：不改交易规则、不改训练参数、不重训）
  - [x] E0.1 Reward 对 PnL：单笔同号 99%，但盈亏平衡胜率 Reward 约 80%、Q 折扣约 74%、PnL 86.6%；所有运行 Reward 为正、PnL 为负。
  - [x] E0.2 入场前可预测性：DEV FG409 和 PAPER 事实（`dev_facts.py`，Native 核对通过，不进语料目录）；学生 Outcome 样本外 AUC 0.39–0.54；独立探针按合约留出 0.50。
  - [x] E0.3 Q 对自己 CF：1x 秩相关约 0、翻转 41% 有利；10x 只在样本内 ADD 分叉上有相关；ΔG 与 ΔPnL 同号 91–99.8%。
  - [x] E0.4 口径：SL5 的 −6,060 元差额为滑点；入场 bar 即 SL 占 26%；1x 对随机的优势几乎全部来自 FG501。
  - [x] 点差规则（用户 2026-10-30 发现 FG501 @1317 低于 bar 最低价）：spread 1.0 让入场按收盘价 ±1 成交，出场报价也含点差，行情几何实为 TP8/SL28；1x PAPER 有 140/722 笔成交在 bar 最低价之下。用户决定只记录、不改规则。
  - 结论与建议见 `docs/thinking_student_v1/E0_DIAGNOSTICS_20261030.md`：下一轮先改 Reward（对齐净 PnL），Outcome / Q / Actor 不动；先做 E1 β_Q = 0 冻结消融。
- [x] G-15：E1 冻结权重 β_Q = 0 消融（e1.md Review，用户 2026-10-30 开始；判定规则先写在 `E1_BETA_Q_20261030/PREREGISTRATION.md`）
  - [x] DEV exam `--beta-q` 覆盖（只改决策设置，不改权重；只限 DEV）；β_Q = 1 重跑与原正式 DEV 考试逐项相同。
  - [x] 1x π₆：β_Q = 0 净 PnL −29,112，对比 β_Q = 1 的 −15,918（d = −13,194 > s = 2,232），**Anchor 有帮助**：948 次翻转中 927 次 ADD → WAIT；10x π₄：d = +1,536 < s = 4,997，不确定。保留 β_Q = 1。
  - 报告见 `docs/thinking_student_v1/E1_BETA_Q_20261030.md`；下一步 E2（经济目标对齐的最小训练，先离线重新标注 R0/R1/R2）。
- [x] G-16：E2 经济目标对齐的最小训练（e1.md 第五节；用户 2026-10-30：R2 = H 内不折扣，只做 1x）
  - [x] 离线重新标注 π₆ 第 7 轮 Q 目标（R0 / R1 净 PnL 折扣 / R2 净 PnL 在 H=2760 内不折扣），精确性核对全部通过；盈亏平衡胜率 R1 82.3%、R2 86.6%。
  - [x] 第 7 轮按原命令重跑三次并做 DEV 考试：R0 与正式 π₇ 完全相同；R1 无差别（+711）；R2 改善 +18,417（5.8σ），ADD 273 → 246，SL 53 → 41。起作用的是折扣。
  - 报告见 `docs/thinking_student_v1/E2_ECONOMIC_REWARD_20261030.md`；下一步待用户决定：训练种子稳健性、R2 后续轮次。
- [x] G-17：E3 R2 稳健性与连续学习（e2.md；1x；R0 对照 = 正式运行的 π₈/π₉）
  - [x] 判定规则写入 `E3_R2_CONTINUE_20261030/PREREGISTRATION.md`（先于结果）
  - [x] E3-A：R0 / R2 π₇ 的 β_Q = 0 DEV 考试（R0：Anchor 有害 +14,880；R2：Anchor 有帮助 −9,030；β_Q=0 时 R2 比 R0 少 5,493）
  - [x] E3-A：R2 标签核对与 2760 分钟边界（EXACT_STATE 133 对 ΔG 全为 0，占 44%；窗口截断只改变 1 对，TRUNCATION_MINOR；E2 的“ADD 更好 55.6%”把浮点噪声当成正数，实为 39.4% / 平 44.4% / WAIT 16.2%）
  - [x] E3-B：R2 第 8 轮（π₈ DEV −25,218，R0 −29,394）
  - [x] E3-B：R2 第 9 轮（π₉ DEV −8,646，R0 −25,764；ADD 127 对 268）
  - [x] E3-C：`E3_R2_CONTINUE_20261030.md`：R2_SUSTAINS（2/3 轮 d > s，平均 +13,237），但主要靠少交易：同频随机不更差，Q 对 ΔG 的符号预测约 50%，R2 标签持平占比 44% → 59% → 76%，κ_Q 0.29 → 0.15，Anchor 被放大
- [x] G-18：1x-R2 完整训练（e2-r2.md；用户 2026-10-30）
  - [x] `iteration.q_return: net_pnl_h2760`（默认 native 不变），`q_return.py` 重新标注每轮 CF；窗口结束时仍持仓的交易计入其实现净盈亏（不丢弃未实现亏损）；分支 `feat/ts-r2-q-target-20261030` dd9daa2
  - [x] `FORMAL_V1_GRADED_R2_1X_20261030`：与原 1x 共用 foundation 和 π₀（硬链接），9 轮，tmux d3:r2_1x，12:31 启动
  - [x] 9 轮完成（16:29；12:44 OOM 后 cf_workers 3 重启；14:47 起换性能优化代码 ffb9d28，已验证结果相同）；每轮 EXACT_STATE ΔG=0 占比 100%（占全部配对 9.5–79%），κ_Q 0.13–0.45
  - [x] 对比报告 `R2_1X_FULL_20261030.md`：没有明显优于原 1x（π₁–π₉ 平均净 PnL −27,189 对 −28,054，好 3 轮差 3 轮；学生 − 随机 0/9 为正，原 1x 4/9）；β_Q = 0 考试 Anchor 有帮助 2 轮、有害 2 轮；R2 Q 对 ΔG 无排序能力（Spearman −0.02），偏向逐轮摆动；π₃ 因 FG201 99% 持平只交易 33 笔，下一轮恢复。下一步待用户决定（Q 监督 / 表征 / 策略更新）
- [x] G-19：E4 同策略价值可学习性与策略迁移（TS_V1_R2_Full_Architecture_Review_20261030.md 第六节；π_A = R2 π₈，π_B = R2 π₉；不训练、不发布策略）
  - [x] 判定规则写入 `E4_VALUE_LEARNABILITY_20261030/PREREGISTRATION.md`（先于结果）；按合约 6 折
  - [x] π_A / π_B 在 18 个 TRAIN 合约上采集 + CF + R2 重标注（33 个新 CF 核对全部通过）；重放对照逐字节复现记录
  - [x] E4-A：模型 1–7（常数、价格/回答探针、记录的 ΔQ、L4 同策略重训 500/3000 次）全部无 SIGNAL → `NO_SIGNAL`；L4 同策略 ρ +0.05（跨策略 −0.085），"永远 ADD" 符号一致 62.7% 高于所有模型；不持平 ΔG 80% 为 ±282 元（一笔止盈）
  - [x] E4-B（按预注册不解读）：信号迁移到 π_B 后再降 25–50%
  - 报告 `E4_VALUE_LEARNABILITY_20261030.md`；按 E4-C 指向"新的因果输入 / 标签分解"，下一步待用户决定
- [x] G-20：V2 高质量反弹做空正向教材（Spec V0.3，用户 2026-10-31；只生成教材并人工 Review，不训练、不接入）
  - [x] 第 0 步：TP5 / SL30 可配置（用户 2026-10-31“在这次直接改掉”）：Oracle、Grades、Outcome 标签、语料 CLI 读取 env 的距离；种类改名 TP / SL；默认 5/30 时 FG809 教材逐数组相同。学生的持仓输入（账户观测 /5、/30，env 持仓 /5）保留训练时的固定尺度（beab62f 纸面考试依赖它），只是改成具名常量
  - [x] ENTRYPOINT_AUDIT.md
  - [x] DESIGN.md + `configs/thinking_student/v2_positive_entry.yaml`（规则先于生成冻结）
  - [x] FG809 切片：事件、正例、至少一张真实图
  - [x] 9 个 SHORT TRAIN 合约全部生成 + figures（≤ 12 个事件）
  - [x] 测试与自 Review（因果截断、合成事件、风险/费用、Native 逐分钟核对、Reader、范围）
  - [x] REPORT.md、复现命令、参数 / Git 身份；停止等待用户 Review
  - 结果（2026-10-31）：`/data/logdir/fg_single3/thinking_student/V2_POSITIVE_ENTRY_20261031`，192 个事件、1,548 个正例（EXCELLENT 538）；全部候选 2R TP 率 24.1%、平均净 −157 元，低于盈亏平衡点 34.8%，没有显示出优势；正例高度重复（382 个不同出场）；GOOD 的 MAE 条件不起作用。见 `docs/thinking_student_v2/REPORT.md`；是否使用、用哪部分由用户决定
- [x] G-21：V2 教材规则 V2（用户 2026-10-31 Review：FG205 沿横线的多次接近被遗漏）
  - [x] 逐点诊断用户标出的 4 处：一条 level 只用一次、每次都要放量、区间 H−5 / R ≤ 10 太窄；12-29 21:01 是开盘第一分钟的冲高，市价单卖不到
  - [x] 规则 V2：线可被多次 touch（远离后重新 armed）、取消 60 分钟窗口、放量属于这条线（从它自己的 15m bucket 起）、取最低的 armed 线、区间 H−7 / R ≤ 12、回看 115 个 bucket；DESIGN.md 版本记录 + 配置
  - [x] FG205 切片：用户标出的可执行点全部成为正例（12-14 第 2 次 touch 11 个，12-28 5 个，01-04 第 3 次 touch 15 个）
  - [x] 测试（多次 touch、高点自身的放量、最低线、因果截断、真实 FG809）+ ruff：13 项通过
  - [x] 提交后 9 个合约全部生成到 `V2_1_POSITIVE_ENTRY_20261031` + figures + REPORT_V2.md；停止等待用户 Review
  - 结果（2026-10-31，gym 0e1ef3b / dreamer 869405e）：1,991 次 touch、7,247 个正例（EXCELLENT 2,377），V1 的 1,548 个正例保留 1,478 个；用户标出的可执行点全部入库，12-29 21:01 是开盘第一分钟冲高，市价单卖不到。全部候选 2R TP 率 25.8%、平均净 −147 元，低于盈亏平衡点 34.8%。新情况：97% 的线放过量（放量几乎不再筛选）；横盘一整天的 touch 最多有 278 个正例；70 个 V1 正例因为改用更低的线而止损。见 `docs/thinking_student_v2/REPORT_V2.md` 第十一节，等待用户决定
- [ ] G-22：V2 最小改造（V0.5，用户 2026-10-31；分支 `feat/ts-v2-minimal-20261031`，worktree `/data/research/fg-s3-thinking-student-v2-20261031`；风险上限 R ≤ 10）
  - [x] 第 1 步：`docs/thinking_student_v2/minimal/CHANGE_MAP.md`
  - [x] 2.1 策略：结构止损 SL = 前高 + 5、TP = 成交 − 2R（`entry.stop_rule`），定长距离的地方改读每笔交易自己的价格
  - [x] 2.2 Reward：净 PnL / 618（`FG_SINGLE3_REWARD_NET_PNL_V1`），语料输入、facts、R2 `q_return` 同一尺度
  - [x] 2.3 教师：V2_1 正例（R ≤ 10、Native 复现）→ `graded/` 的 ADD；必要 WAIT；其它分钟无监督
  - [x] 2.4 Observation：学生不读 MACD；15m 24 → 48（`entry.context_15m_buckets`、`ts.market`）；`inherit_f1` 零行加宽
  - [x] 2.5 配置：`markets/strategy_v2_short_structural.yaml`、`formal_v2_minimal.yaml`（1x，不运行）、`trial_v2_minimal.yaml`
  - 第 2 步测试（2026-11-03）：gym V2 + 教材 22 项通过（1 项待语料）、gym 已有 87 项通过、dreamer V2 5 项、dreamer 已有 63 项通过；止损线改为“已离开过的线”后与教材一致 5,163/5,164（FG405 1 个触碰中新确认的线 → STOP_LINE_DIFFERS）
  - [x] 对照设计文档 Review 全部改动（用户 2026-11-03）：差异清单 → 与用户讨论（A–E 按建议）→ 统一修复（gym `dcf89b7`、dreamer `3abffab`）→ 重跑测试：gym 112 项通过（含语料教师测试）、dreamer 69 项通过（全套 68 项；新加的 MAE/MFE 单位测试在 float32 比较修正后单独重跑通过）
  - [x] V2 语料：9 个 SHORT TRAIN 合约（`corpus_v2_minimal_20261031`，2026-11-03 20:20）；ADD 5,163（GOOD 3,484 / EXCELLENT 1,679）、WAIT 1,183、RISK_ABOVE_MAX 2,083、STOP_LINE_DIFFERS 1；语料教师测试通过
  - [x] 第 3 步：最小验证 + `TEST_REPORT.md`（教师 mask、真实交易、Reward/Q 对账、Obs、V1 回归；L1–L5 GPU smoke 按用户 2026-11-03 决定不做，由 1x 跑这条链；手算 G(t) 放到 SMOKE_REPORT）
  - [ ] 第 4 步：`SMOKE_REPORT.md`，取自 1x 的 pi_0000 / 第 1 轮（用户 2026-11-03：重启后直接开 1x `FORMAL_V2_MINIMAL_1X_20261103`，不另跑短试训）
  - 用户 2026-11-04：V2 1x 已跑完（6 轮），但只做 SHORT 不合适（V0.5 spec 写成了 SHORT_ONLY，是 spec 的错误），改做 V2.1（G-23）；V2 1x 的第 4 步 SMOKE_REPORT 是否还写，待用户定
- [ ] G-23：V2.1 = V2 最小改造做多空两个方向（用户 2026-11-04；分支 `feat/ts-v2-1-20261104`，从 V2 分支 gym `dcf89b7` / dreamer `d9bc7dd` 拉出；新的训练 worktree）
  - 用户决定：与早期 V1 1x（`FORMAL_V1_GRADED_20261028`）结构一致，便于对比：18 个 TRAIN（9 SHORT + 9 LONG）、DEV 只用 FG409、9 轮、L6 从第 7 轮开始；LONG 教材先给用户审，再建语料和训练
  - [x] 代码：教材生成器 LONG 镜像（价格取负在“空头空间”里跑同一套扫描）；env `stop_rule: prior_15m_extreme`（SHORT 前高 + 5，LONG 前低 − 5）；LONG 的 TP 守卫和 `check_collection` 距离取绝对值；策略 `direction: BOTH`；教材合并 `--combine`
    - 另修两处：K 线图标题被坐标轴循环变量覆盖成 “direction bottom”；paper 报告在策略文件下误写 “5 / 30”（V2 1x paper 报告已重新生成，数字不变）。改动清单：`docs/thinking_student_v2/v2_1/CHANGE_MAP.md`
  - [x] 校验：SHORT 用新代码重生成 FG205，与 V2_1 教材逐字节相同；LONG 单合约（FG401）合理性检查；测试与 lint
    - FG205：events 74 行、entries 310 行逐字节相同；timeline 466 行数值相同，列顺序随第一行而变（旧代码同样，`combine` 遇表头不同即报错）
    - FG401 LONG：692/692 正例全部通过；`planned_exit` LONG 445 STRUCTURAL + 247 CAPPED 与教材一致（`V2_1_DEV_20261104/checks/long_FG401.json`）
    - 测试：gym 116 项通过（V2 + 已有 TS / env / e_data），dreamer 18 项通过（V2、c4 paper、s2）；ruff F/E9 全部通过（日志 `V2_1_DEV_20261104/tests/`）
  - [x] LONG 教材：9 个 LONG TRAIN 合约（FG001、FG009、FG101、FG105、FG109、FG305、FG309、FG401、FG909），统计 + K 线图 → 用户审
    - 2026-11-04 已生成 `V2_1_POSITIVE_ENTRY_LONG_20261104`（7,571 正例，R ≤ 10 的 5,040 个），检查通过，等用户审（统计见 CHANGE_MAP §4，图 12 张在 `figures/`）
    - 用户 2026-11-04 审过：“没问题，准备训练”
  - [x] 合并教材（V2_1 SHORT 原样 + LONG）→ 18 个语料（`corpus_v2_1_20261104`）→ 语料教师测试
    - 合并完成：`V2_1_POSITIVE_ENTRY_BOTH_20261104`（events 1,060、entries 14,818、timeline 18,613 行 = SHORT + LONG）
    - 18 个语料 2026-11-04 08:35–09:03 建好（CPU，2 个并行 × 12 workers，`V2_1_DEV_20261104/corpus_build/`）
    - 语料教师检查（`checks/corpus_teacher.json`）18/18 通过，同一个教师：ADD 10,110（SHORT 5,163 / LONG 4,947）、WAIT 2,510（1,183 / 1,327）；丢弃 RISK_ABOVE_MAX 4,614、STOP_LINE_DIFFERS 94（SHORT 1 + LONG 93）
    - `formal_v2_1.yaml` 预检（训练 worktree 的代码，CPU）：18 个语料、DEV FG409、9 轮、L6 从第 7 轮、策略版本与语料一致
  - [x] 训练 worktree + 运行脚本 + 命令（`formal_v2_1.yaml`，`FORMAL_V2_1_1X_20261104`），用户在 d3 启动
    - worktree `/data/research/fg-s3-thinking-student-v2-1-1x-20261104`（分支 `run/ts-v2-1-1x-20261104`，从 dev 快进）；脚本 `FORMAL_V2_1_1X_20261104/v2_1_1x.sh TAG`
  - [x] 1x 训练：start1 在第 2 轮 FG101 的 CF 因 GPU 显存不足退出（12 个 CF 进程同时启动的峰值，无 Xid），start2 从断点接着跑，2026-11-04 13:28 跑完 9 轮
    - DEV FG409 best_dev = pi_0000（没有一轮比基础策略好）；各轮 DEV 净亏 −27.7k ~ −33.8k，随机 −30.8k ~ −41.9k，胜率 28–29%（`PAPER_V2_1_1X_20261104/checks/dev_table.txt`）
  - [x] paper trading：best_dev pi_0000 和最终 pi_0009，FG501–FG605（全 SHORT），`PAPER_V2_1_1X_20261104`
    - pi_0000 −126,210（随机 −117,414 ± 6,772，0/5）；pi_0009 −114,630（随机 −118,863 ± 7,633，3/5）；胜率 27.3% / 27.4%，2R 盈亏平衡需 34.3%
    - 入场里 STRUCTURAL 只有 18–19%，R = 10（CAPPED / NO_LINE）占 83%，与 V2 paper 相同；仍没有优势
  - [x] paper 固定出场（用户 2026-11-04）：同两个策略，TP 5 / SL 10 和 TP 5 / SL 30（`markets/strategy_v2_1_fixed_tp5_sl{10,30}.yaml`，仅 paper 用，dreamer `2ab5e8e`），`PAPER_V2_1_1X_20261104/run_fixed.sh`
    - 5 / 30：pi_0000 −97,068（随机 −90,923，2/5）、pi_0009 −87,774（随机 −76,658，0/5）；V1 1x pi_0006 同规则 −72,420（随机 −83,170，4/5）
    - 5 / 10：pi_0000 −343,584、pi_0009 −316,998，都 0/5；胜率 52%，盈亏平衡需 68.7%
    - 对比表 `checks/compare_fixed.md`；出场检查 `checks/fixed_exits.txt`（TP 都是 +282，SL 都是 −618 / −1,818，另有 10–29 笔跳空成交或换月平仓）
- [ ] G-24：V2.2 = V2.1 + 恢复教材的高位放量（用户 2026-11-04；分支 `feat/ts-v2-2-volume-20261104`，worktree `/data/research/fg-s3-thinking-student-v2-2-20261104`）
  - [x] 教材规则 V3：每次 touch 都要由它自己接近线位且 RV ≥ 2 的那根 bar 开始（`v2_positive_entry_v3.yaml` / `_long.yaml`，`volume.activation: touch`；gym `1fa8fea`，dreamer `0bbcd3e`）
    - 新代码按 V2 规则重新生成 FG205 / FG401，与 V2.1 代码的结果逐字节相同；gym 11 个测试文件全部通过（`V2_2_DEV_20261104/`）
  - [x] review-v1 教材导出（用户的 `review-export-v1-revised.md`，只做生成端的教材，复盘端由其他 agent 对接）：gym `review_export.py`（`cd89349`），规范原文与实施说明在 `docs/thinking_student_v2/review_export_v1.md`、`REVIEW_EXPORT_IMPL.md`
    - `corpus_v2_1_20261104/review/` 已补导出：18 个文件，12,620 条（ADD 10,110、WAIT 2,510），validate 通过，重复导出不变；导出文字全部用英文（用户要求）；先发的中文版原样移到 `V2_2_DEV_20261104/review_export/superseded_zh_20261104_1710/`
  - [x] Corpus 指南：`docs/thinking_student_v2/CORPUS_GUIDE.md`（教材 → 合并 → corpus → 教师检查 → review-v1 导出的命令、目录、字段和已发布清单）
  - [x] V3 教材（SHORT + LONG，合并）：`V2_2_POSITIVE_ENTRY_{SHORT,LONG,BOTH}_20261104`（17:29，两仓 clean）；用户不要统计和图（“生成好，放着就行”）
  - [x] 规范 §12 第 2 项按复盘端建议改写（Downloads 原件和项目副本，sha `659134e6…`）；分工：run 清单、学生成交、阶段状态都由生成端导出
  - [ ] 用户计划（2026-11-04）：V3 教材 → 1x 正式训练 → paper，期间教材和学生成交都用 review-v1
    - [x] 18 个 V2.2 语料（`corpus_v2_2_20261104`，18:01，两仓 clean）→ 教师检查通过（ADD 9,429 / WAIT 3,355，SHORT 4,832 / 1,614，LONG 4,597 / 1,741）→ review 发布并通过 validate，12,784 条，manifest sha `3df3b2e0…`（`V2_2_DEV_20261104/corpus_build/after_build.sh`）
    - [x] run 导出：`review_export run` / `paper` / `validate`（run 清单固定教材 id+sha，progress/status 取自 run_state，学生记录取自 Native 的成交、交易和归因并逐笔核对）；formal_train 每次写 run_state 后调用，导出失败只记 `logs/review_export.log`；run_state 写 status，结束时 step 回到 done。在 V2.1 run 上试导出：20 个 dataset、13,250 条，与 run 自己的汇总一致（`REVIEW_EXPORT_IMPL.md` 第五节）
    - [ ] formal_v2_2.yaml + 训练 worktree `fg-s3-thinking-student-v2-2-1x-20261104` + `FORMAL_V2_2_1X_20261104/v2_2_1x.sh`，用户在 tmux d3 启动 1x
    - [ ] run 自动瘦身（用户 2026-11-04：“每一步完成后，自动删掉它不再读取的中间文件”）：CF 完成后删 work/ 和 TRAIN 采集的旧快照（保留 state.pkl 的硬链接）；policy 冻结后删 learner 各 checkpoint 的 agent.pkl（json 留着）；考试写完 exam.json 后删 replay/ 和旧快照。foundation 的 checkpoint、所有 policy、采集的 steps/replay/state.pkl、CF 结果、feedback、market_carry 都保留；删了什么记在 `logs/prune.log`，失败只记录不中断训练
    - [x] 旧 run 的多余文件清理（`CLEANUP_20261104/cleanup.py`，规则 R1–R4）：用户 18:23 执行，删 3,786 个路径、147 GB；thinking_student 306 GB → 159 GB，/data 剩 189 GB（`deleted.tsv`、`delete.log`）
    - [ ] paper，并登记到 run 清单
