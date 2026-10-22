# Thinking Student V1 — TODO（完成一项勾一项）

分支：`feat/thinking-student-v1-20261022`（dreamerv3 + gym-trading-env，worktree 在 `/home/v/Documents/work/research/fg-s3-thinking-student-v1-20261022/`）
规则：每项完成并有运行证据后才打勾；每个切片结束时补一份极短记录。

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
- [ ] 把 teacher 的 `trade_facts` 映射为 `OUTCOME_BINS` 竞争风险标签（RIGHT_CENSORED 只走生存项）
- [ ] `L_outcome`：竞争风险 NLL、右删失、MAE、early10、Quality
- [ ] `opt_l2`：只训练 outcome 组，可选加 fusion 适配；Market/Account 冻结
- [ ] 把 corpus 标签（teacher_sidecar 的 L1 facts）接入训练批次
- [ ] 最小验证：TP、SL、删失三例的语义，1 次梯度更新，ckpt 继承
- [ ] 真实 FG809 短训，并写记录

## S2 — 首个自主学生 π0
- [ ] `L_policy_demo`（严格 L2 示范）加自然 WAIT/ADD 样本；`opt_l3` 只更新 decision 组
- [ ] 自主 Native 采集：答卷、行为策略、执行动作落盘
- [ ] 冻结完整 π0 快照
- [ ] 最小验证，并写记录

## S3 — 策略条件 Q + 同策略 CF 老师
- [ ] Gym：同初态 ADD/WAIT 双分支 runner（`export_state`/`import_state`，注入冻结 π_k 回调）
- [ ] F-02：统一评价终点和折扣；状态分为 VALID_COMPLETE / VALID_ESTIMATED / NOT_COMPARABLE
- [ ] `L_q_real` 与 `L_q_cf`；`opt_l4` 只更新 value 组；ΔQ 锚点生效后得到候选 π1
- [ ] 最小验证，并写记录

## S4 — Reasoning 与策略改进
- [ ] `L_reason`、`L_policy_pref`、`L_align`、`L_policy_rl`
- [ ] 跑通一轮迭代：π_k → Q^{π_k} → 改进 Actor → π_{k+1}
- [ ] 最小验证，并写记录

## S5 — 自主教学循环 + Teacher-free 考试
- [ ] B1 教学题生命周期，含 `ready_at` 成熟反馈
- [ ] B2 课程循环：自然教材、多行为教材、学生自采、错题补课
- [ ] 在封存区间做 Teacher-free 考试，使用同一个 `student.policy`
- [ ] 完整闭环演示一次，并写记录
