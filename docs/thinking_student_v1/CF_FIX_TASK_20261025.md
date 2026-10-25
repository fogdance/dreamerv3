# Thinking Student V1：FG201 精确耦合修复任务

状态：实现与小规模验收完成；待 Claude 独立 Review。下方未勾选项为明确暂缓的可选优化。交付提交与完整结果见 [交付报告](/data/research/fg_ts_cf_fix_20261025/DELIVERY.md)。

## 目标与依据

让已经达到未来相关真实状态等价的 FG201 分支能够精确耦合，并让正式 CF 教材生产路径实际使用该能力。保持 Native 金融规则、Reward、MarketClock、冻结学生 πₖ 与原始长期 Q 定义。

研究依据：`/data/research/fg201_cf_investigation_20261024/REPORT.md`。报告中的少量样本等价不能直接推广为通用规则；实现必须有源码证明与负例测试。

已报告的故障：旧 `Reference.couples()` 的未来最低权益 / 全局 `coupling_floor` 拒绝 FG201 耦合；`decidable()` 与上游预算筛选又在模拟前排除候选。FG201 共 1,596 个合法分支，旧结果只有 250 个 TERMINAL 完整标签，1,298 个分支被标记 `UNAVAILABLE_BUDGET`。必须同时修复 Gym 耦合与 Dreamer 预算路径。

关键回归：FG201 `t=7` 在第 4 步、`c=11` 真实状态等价；报告中的完整 Native 与拼接 Q 差约 `1.39e-16`。净盈亏 `-263022` 加初始资金 `300000` 对应权益 `36978`，不是权益 `-37000`。

## 开发隔离与交付边界

| 仓库 | 独立 Worktree | 起始 Commit |
| --- | --- | --- |
| Gym | `/home/v/Documents/work/research/fg-ts-cf-fix-20261025/gym-trading-env` | `2b4e89422111c695a1e290f5d195dd23f58a2199` |
| DreamerV3 | `/home/v/Documents/work/research/fg-ts-cf-fix-20261025/dreamerv3` | `fbcb368e30ce8b6fe8478d1c44b96f3bf2710ba9` |

两仓修复分支均为 `fix/ts-v1-cf-exact-coupling-20261025`。

- [x] 从当前开发所需的最新已提交代码创建两仓独立 Worktree。
- [x] 检查原开发树状态：Gym 干净；Dreamer 有未跟踪的 `docs/thinking_student_v1/S5_RECORD_20261024.md`，不复制或修改。
- [x] 所有源码修改仅发生在以上独立 Worktree；不 commit / stash / reset / clean 原开发树。
- [x] 正式运行目录、Checkpoint、Value Brain、18 合约 Corpus、已生成教材和 CF 结果保持不变。
- [x] 分别提交少量清晰 Commit，完成自查与简短交付报告。
- [x] 停在独立 Worktree，等待 Claude 独立 Review；未经 Review 和用户同意不得合并、cherry-pick 或 push。

遵守适用 AGENTS.md 的 Thinking Student V1 优先规则；不引入旧 S3 Hash 认证、Admission、审计或新数据准入系统。

## A：精确状态耦合

- [x] 在 Gym `ts_v1_cf.py` 的现有路径增加严格的未来相关状态等价判断。
- [x] 核对真实资金账本、持仓、活动 Reward、退出、时间、订单和账户规则读取字段。
- [x] 核对策略账户输入、MarketCarry、必要的策略 / 随机状态。
- [x] 仅在证明不会影响未来行为后排除历史日志；不以现金差为零、双方空仓或 Observation 相同单独判等。
- [x] 保留原高资金安全耦合作为另一条合法充分条件。
- [x] 保持 `VALID_COMPLETE`、真实 Native TERMINAL、CAP 语义，不伪造完整 Q。
- [x] 不降低保证金，不改强平、结算、Reward、市场时间或冻结策略；不根据未来结果倒推学生动作。

## B：64 步短探测加续算配额（用户最终选择）

每个合法候选都有最多 64 步短探测机会；旧 `coupling_floor` 不再充当“未来没有合法 Q”的预筛选依据。候选仍须满足真实 fork 合法性。

- [x] 同时检查并修复 `Reference.decidable()`、`undecided_sample()`、`run_cf()`、预算分配和不可用结果路径。
- [x] 探测中真实 TERMINAL 或合法精确 / 高资金耦合直接产生 `VALID_COMPLETE`。
- [x] 探测未闭合不是不可比较；对其沿用现有分层抽样，分配后续续算配额。
- [x] 未入选续算者准确标记 `UNAVAILABLE_BUDGET`，无伪造 Q / 成对监督。
- [x] 探测与续算共享同一分支的 Native 状态、MarketCarry、策略状态、累计奖励和折扣时间；不重复执行前 64 步或首次强制动作。
- [x] 探测加续算总步数不超过每分支 `cf_max_steps=8192`；到上限仍无合法闭合才是 CAP。
- [x] 保留计算预算与可恢复机制；ADD / WAIT 后续始终使用同一冻结策略，不能以严格 L2 Oracle 替代 Q。

令候选数为 N、续算配额为 B、每分支上限为 H，则分支模拟步数上界为：

`N * min(64, H) + B * max(H - 64, 0)`

`B=None` 表示所有未闭合分支续算，`B=0` 表示只有短探测。该式不包含参考重放、保存 / 恢复、神经网络推理等额外成本，必须另测真实开销。

## C：限定范围的性能与恢复要求

- [x] 参考轨迹按合约准备并共享，不能为 1,596 个候选重复重放完整参考。
- [x] 完整分支快照有界保留在 RAM，暂停状态落在本次独立输出中；不能长期保存全部历史快照于内存。
- [x] 共享市场特征 / MarketCarry、复用环境、减少重复拷贝仅在行为等价成立时实施。
- [ ] 仅 WAIT 合法时跳过策略推理：本次暂缓；保留每步完整冻结策略调用。跨 worker 的市场特征 mmap 也未实施。
- [x] 记录参考 / 探测 / 续算各阶段实际 Native step、Policy 调用、保存恢复开销、耗时与峰值内存；区分预算步数与真实计算成本。

不实现 `VALID_ESTIMATED`、有界尾值、区间 Q、完整事件驱动 Native 替代、逆向 DP、价值 DAG 或 S5 TIME_ORDERED；不修改 Student Brain 网络和已有 Checkpoint。

## D：教材与 S5 兼容

- [x] 新合法 CF 可被 Q Reader、C3/C4 与 S5 消费；没有合法 Q 时不生成成对监督。
- [x] 不重训 L1/L2/L3，不改 Value Brain 权重或基础教材。
- [x] 现有合法 `VALID_COMPLETE` 可以复用；受影响合约仅补算需要更新的 CF，输出到新目录。
- [x] CF 更新后，S5 局部重生成当前相关反馈，不错误复用旧 `summary.json` 错题；不建立复杂缓存 / Hash 系统。

## 验收清单

### Gym 与真实 Native

- [x] FG201 `t=7`：4 步、`c=11` 精确耦合，Q 与完整 Native 一致（记录误差）。
- [x] FG201 `t=20789`：非零现金差对照，不能仅凭空仓或相同输入错误合并。
- [x] FG205 少量真实正常耦合对照，沿用同一冻结 π₂。
- [x] 非零现金差、账户输入不同、保证金边界不能错误精确耦合。
- [x] TERMINAL / CAP / 非真实数据结束语义不回退。
- [x] 探测续算与一次性执行一致，前 64 步不重复，上限不突破。

### Dreamer 集成

- [x] 旧全局资金门槛不能阻断合法候选探测，`budget=0` 仍可获得短探测完整标签。
- [x] 短探测未闭合且无配额时为 `UNAVAILABLE_BUDGET`。
- [x] 合法标签进入 Q Reader，正确冻结策略与真实账户保持一致。
- [x] 无合法标签不产生假 delta-Q 或 S5 成对监督。
- [x] 当前 CF 更新后反馈重生成；既有训练进度与历史正式结果保留。

### 小规模执行与交付

- [x] 只运行 FG201 `t=7`、`t=20789` 与 FG205 少量分支，不重算全部 1,596 分支、不做完整 18 合约 CF、不启动正式 L6 长训练。
- [x] 人工资金扰动若使用，仅作隔离因果诊断，绝不作为真实 π₂ / 正式 Q 教材。
- [x] 逐实验记录首次双方空仓时刻、账户输入差异、资金门槛、耦合条件与 Native 终止状态。
- [x] 记录修复前后步数、有效 Q 数、耗时；未测量项明确标记“未测”。
- [x] 单列短探测新增成本、探测标签数、续算标签数、剩余不可用数。
- [x] 报告测试命令、测试数量、结果、修改文件、主要 Diff、两仓起始 / 最终 Commit 和未解决问题。
- [x] 完成自查，交 Claude Review，不自行合并。

## 当前进度

2026-10-25：两仓独立实现与自查完成。36 项唯一相关测试通过，最终 12 项定向复验通过；真实 FG201 / FG205 四分支通过，FG201 t=7 完整 Native 对照误差 -1.39e-16；旧 FG201 250 条完整标签保留，叠加一条新标签后 Q Reader / S5 均读取 251 条。正式资产与原开发树保持原状。未执行全量 CF 或正式 L6；不合并、不推送，交 Claude Review。
