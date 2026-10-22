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
| Independent Exam | 冻结完整学生、封存行情、Native 账户 | 无教师参与的完整交易轨迹 |

## B1 教学题（一题六部分）
1. **Question**：因果 Obs、Native 账户、Mask、时钟、决策前状态、策略版本。只有同一账户初态才算同一题。
2. **Student Answer**：事前 Outcome、Q、动作分布、提议动作、实际执行动作、是否探索。
3. **Student Experience**：主账户的真实 Native 经历。主账户只执行学生动作，老师永不改写。
4. **Teacher Evidence**：
   - ADD 后果：TP/SL 首次事件、时间、右删失、MAE、early10、Quality。
   - 同初态 ADD/WAIT 双分支：后续使用同一冻结完整 π_k，同 Reward，同评价范围。
   - 独立的严格 L2 Oracle。
5. **Learning Feedback**：分别进入 `L_outcome`、`L_q_cf`、`L_reason`、`L_policy_*`，不合并成单一 Reward。
6. **Validity & Provenance**：每类反馈各自带 `(valid, ready_at, source)`。来源取值为 `STUDENT_EXECUTED_NATIVE`、`VERIFIED_CF_NATIVE`、`HINDSIGHT_ORACLE_TEACHING`、`NATURAL_CORPUS`、`MODEL_IMAGINED`。

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
