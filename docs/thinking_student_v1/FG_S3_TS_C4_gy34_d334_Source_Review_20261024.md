# Thinking Student V1 — C4 正式训练源码 Review（2026-10-24）

## 审查范围及证据边界

- 代码：上传 `gy34.tar.bz2`（Gym TS 分支 `0056296`）和 `d334.tar.bz2`（Dreamer TS 分支 `69fd0b8`）。从各仓 Git 对象还原 `feat/thinking-student-v1-20261022` 分支。
- 合同：`FG_S3_Thinking_Student_V1_C4_Formal_Training_Implementation_Plan_20261023.md`，聚焦 F0–F5，不重复审查旧 S3。
- 代码与测试：`dreamerv3/thinking_student/{formal_train,corpus,value,run,exam}.py`、`dreamerv3/tests/test_ts_v1_c4.py`、Gym `ts_v1_corpus.py`、C4配置与 `FORMAL_TRAINING_IMPLEMENTATION.md`。
- 本地验证：Python `compileall` 通过；尝试执行 `pytest dreamerv3/tests/test_ts_v1_c4.py`，缺少 `ninjax`，测试收集阶段停止，**没有**在此环境实际复跑 GPU/Native。以下 GPU、DEV 数字是仓库随附开发记录，不是本次亲自复测。
- 针对“冻结后、DEV评估前被打断”做了一个不依赖训练库的真实Python最小逻辑复现：`iterate()`立即因 `max_rounds` 返回，未执行缺失DEV评估。

## 对照C4结果

| 切片 | 结论 | 主要代码 |
|---|---|---|
| F0：18合约混合采样 | 已实现 | `corpus.py:110–150, 270–345`；`MultiCorpusWindows`先选合约，再选同合约窗口；`source.json`检查TRAIN与Reward/时钟 |
| F1：L1–L3同模型训练与继承 | 已实现 | `run.py:49–144, 628–659, 689–870`；`formal_train.py:207–266`；统一参数树、分阶段优化器、定期Checkpoint、预算续训 |
| F2：同策略多合约CF与Value | 已实现 | `value.py:320–403`；`formal_train.py:286–372`；相同冻结策略路径、混合Q样本、跨合约κ_Q |
| F3：自动编排及恢复 | **核心已实现，存在中断恢复逻辑Bug** | `formal_train.py:153–190, 267–376`；见R1 |
| F4：Teacher-free DEV | 已实现 | `exam.py:1–182`、Gym `ts_v1_corpus.py:116–123`；VALIDATION环境、Native政策、同规则随机基线 |
| F5：S4/S5集成 | S4已接入，S5/L6未实现，代码明确声明 | `formal_train.py:349–359`、S4 `student.py:243–302`；不得误称正式L6闭环全部完成 |

C4提交记录中的50M双合约Smoke包括L1/L2/L3 (40/60/40 updates)、π0→π1 的L4(40)+L5(20)、390个有效CF对，以及FG409 DEV运行31,034分钟。报告称π0/π1的DEV Quality Reward分别16.90和18.62；负净收益依然存在。这些是工程贯通证据而非择时有效性证明。正式18合约L1(10,000)与L2(12,000)在代码记录里被**启动**，归档未包含运行目录，不据此声称已全部完成。

## 必须修复的两个问题

### R1 — P1：冻结后崩溃会跳过本轮尚未完成的DEV（确定性复现）

- `formal_train.py:349–373`：先`freeze(pi_{k+1})`，随后`evaluate(...)`。
- `formal_train.py:267–277`：续跑只按`_newest(lay)`找最大已冻结版本；若`k >= max_rounds`直接return。
- 场景：`pi_0001/policy.json`已产生，但随后DEV考试未完成/中断，重新`iterate --rounds 1`时`_newest`返回1；`max_rounds=1`立即返回，缺失考试不会重试。`max_rounds>1`时还有可能跳到下一轮。
- 本地最小测试创建pi_0000、pi_0001的`policy.json`且状态`step=exam`，得到`iteration.max_rounds = 1 reached ...`和DEV不存在。**违反C4任一步中断可恢复**。
- **最小修复：** 在进入下一轮前检查上一候选冻结策略的本轮考试作业是否已完成。缺失则仅补做该策略的考试、完成标记，并复用此前采集/CF；再按上限判断是否进入下一轮。分别测试“freeze刚完成即中断”“DEV一合约完成而另一合约未完成”“DEV配置为空”的恢复。

### R2 — P1：零有效CF成对标签仍可带启用的Q锚点发布策略

- `value.py:368–403`：`MultiQWindows.kappa(floor)`在`pairs < 2`甚至`pairs=0`时，只要给了正`floor`，就返回floor。
- `run.py:662–680,782–786`：`freeze --beta-q 1 --kappa-q auto`接受该结果，并没有要求本轮实际拥有可监督ADD/WAIT**差值**的CF对。`formal_train.py:357–372`照常发布`pi_{k+1}`。当前正式配置`kappa_floor: 0.05`。
- `L4`仍可能只有已执行动作的`q_real`标签，但对未执行动作价值及`ΔQ`缺乏识别：启用非中性Q锚点会在未学习可靠行动差值的情况下改动正式策略。
- **最小修复：** 若`β_Q>0`，发布前要求至少存在非零且可用的**成对CF监督**（推荐至少2对，满足尺度计算）；不足时保留`β_Q=0`且显式标为`Q_ANCHOR_NOT_READY`，或中止本轮**策略发布**并继续补教材。`kappa_floor`只允许修正有合法监督时的零方差/尺度过小，不能代替监督。无需增加收益门槛。

## 正式扩大训练前建议明确的问题

### R3 — P1（模型输入一致性）：训练短窗口与正式连续RSSM有显著差别

- `corpus.py:187–233`、`value.py:240–265`：随机窗口第一步重置Market RSSM，L2/L3/L4只在前若干burn-in步免计Loss；正式收集和DEV是整个合约连续carry。
- `value.py:404–465`有专门对照工具。开发记录在64步窗口、16步burn-in下报告`z`平均绝对差0.49，连续路径std 2.20；48–64步仍约0.44。不是小到可以忽略的数，但它是已知训练近似，**尚不足以断言模型功能Bug**。
- **建议：** 用户当前L1/L2已经在训练，不要求回滚；在正式长时L3/L4开始前，以现有state-check做一次不同窗口长度/预热（比如64 vs 128或256）的同cursor对照，选择显著减小失配且显存可负担的方案。优先在Reader侧提供更长因果前缀或carry，切勿把后验教材送入Actor。

### R4 — P2（资源）：每轮默认执行全轨state-check，耗时不低

- `formal_train.py:333–341`、`DEFAULTS iteration.state_check_windows=32`，正式配置也是32。代码每轮对选定合约运行state-check。
- 开发记录一次20,504行的state-check约117秒；这项是模型训练一致性诊断，不是每轮必需的Native教师计算。
- **建议：** 修复R3或完成针对性测量后，将正常多轮正式训练默认设为`state_check_windows: 0`，需要诊断时手动打开或按策略迭代阶段抽样执行。避免重现用户不希望的重复审计开销。

## 不建议改变的部分

- 不重做大脑、18合约基础教材、旧S3 Admission、Gate、Hash审计和G3。
- 保持 `pi_k` 冻结后同策略CF；`q_real/q_cf`合法性与 `NOT_COMPARABLE`缺失监督；真实Native账户和TRAIN/DEV/TEST隔离。
- 不将单合约/两合约Smoke的短期Reward、Q误差或盈利阈值变成开发门禁。
- DEV只有一个FG409合约，评估结果可供开发比较，但不能由此推断跨时代泛化；后续应在不接触TEST的前提下按正式研究计划设计更多独立滚动评估。

## 最小修复顺序与复核

1. **先修R1**：冻结之后和DEV途中崩溃能在同一策略版本补完考试，不启动新轮。
2. **修R2**：没有有效成对Q监督不启用`β_Q>0`；保留κ_Q尺度下界。
3. **在正式L3/L4长时训练前处理R3**：做一次有结果的state-check对照后选择正确的因果预热方案。
4. 关闭常态化state-check的重复开销（R4），保留显式诊断命令。

**最终判断：F0–F4的主链路真实存在，S4先验刷新也已接入。C4不需要重新开发；修两个确定性逻辑漏洞和一个已测出的训练状态偏差，再投入较长的多合约L3/L4最合适。**
