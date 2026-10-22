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

## S3 — 策略条件 Q + 同策略 CF 老师
- [ ] Gym：同初态 ADD/WAIT 双分支 runner（`export_state`/`import_state`，注入冻结 π_k 回调）
- [ ] F-02：统一评价终点和折扣；状态分为 VALID_COMPLETE / VALID_ESTIMATED / NOT_COMPARABLE
- [ ] `L_q_real` 与 `L_q_cf`；`opt_l4` 只更新 value 组；Q 有了策略条件监督后逐步启用 `β_Q`（记入快照），得到候选 π1；不用同一 CF 差距训练 Actor 偏好
- [ ] 最小验证，并写记录

## S4 — Reasoning 与策略改进
- [ ] `L_reason`（L2 示范 / Outcome 证据的辅助理解）与 `L_policy_demo` 先验刷新（中性 Q 锚点的教学视图）；Q→Reasoning 投影开始学习；V1 不实现 `L_policy_pref`、`L_align`、`L_policy_rl`
- [ ] 跑通一轮迭代：冻结 π_k → Q^{π_k} → ΔQ 锚点改进完整 Policy → 发布 π_{k+1} → 重新评估
- [ ] 最小验证，并写记录

## S5 — 自主教学循环 + Teacher-free 考试
- [ ] B1 教学题生命周期，含 `ready_at` 成熟反馈
- [ ] B2 课程循环：自然教材、多行为教材、学生自采、错题补课
- [ ] 在 DEV（VALIDATION）或合成数据上做 Teacher-free 工程短考，使用同一个 `student.policy`；封存 TEST 只在用户明确授权的最终评价中使用
- [ ] 完整闭环演示一次，并写记录
