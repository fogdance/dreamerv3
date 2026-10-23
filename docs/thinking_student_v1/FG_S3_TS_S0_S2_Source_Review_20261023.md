# FG-S3 Thinking Student V1 — S0/S1/S2 源码 Review

日期：2026-10-23。审查范围：上传的 `gy32.tar.bz2`、`d332.tar.bz2` 中 Git 分支 `feat/thinking-student-v1-20261022` **已提交的代码**；未覆盖用户本机未提交工作树或 `/data/logdir` 权重文件。对照 `dreamerv3/docs/thinking_student_v1/FG_S3_Thinking_Student_V1_Brain_Architecture_A1_A6_Frozen_20261022.md`、C3设计和 INTERFACE_SUMMARY。

基线：Gym `fb7481d`，DreamerV3 `46763e2`。代码可从上传 Git 对象重建，实际新增 Gym `ts_v1_account.py` 与 Dreamer `thinking_student/*.py`。完成AST语法解析；本容器无 `ninjax`，未复现项目GPU训练与完整单测。仓库中的 S0–S2 RECORD 报告 32 个测试通过、3000次L2更新、3000次L3更新及28263步自主TRAIN采集，均按**开发记录**引述。

## 0. 总结

**S0/S1/S2的核心工程链已真实落地，能够进入S3；但不宜把S1的所有输出直接当成完整、校准的Native首次退出模型。** 用户要求减少审计，故本次只建议针对真实偏差做4项小修或明确合同，旧S3兼容、旧Gate及大规模效果验证不在范围内。

### 已实现（符合冻结设计）

- **S0**：市场专用Encoder与动作全零RSSM（`market.py:22–99`），独立Account Encoder与Fusion（`brain.py:29–53`），Q同值初始化（`brain.py:138–155`）；完整Student Policy（`student.py:141–179`）；L1训练、F1部分权重继承及Checkpoint保存/恢复（`student.py:206–238`、`transfer.py`、`run.py:40–87`）。
- **S1**：23类TP/SL离散时间输出、早期逆行、MAE与Quality监督（`labels.py:16–109`、`brain.py:56–135`），L2仅更新Outcome与可选Fusion（`student.py:40–46,279–296`）；Outcome evidence进入Value与Reasoning（`student.py:148–153`），没有旧S3预测头旁路。
- **S2**：正式策略 `z>0` 且合法才ADD、否则WAIT（`brain.py:213–216`）；epsilon混合另作行为策略μ（`student.py:155–179`）；仅更新decision组的示范Loss（`student.py:240–277`）；Native学生自采、日志、冻结模型和恢复（`run.py:116–314`）；Gym账户原样记账（`ts_v1_account.py`）。
- 目前无 S3 的 `L_q_real/L_q_cf` / 双分支Q教材 / S4的独立`L_reason` 是**按既定里程碑尚未实现**，不得作为S0–S2验收失败。

## 1. 优先处理事项（最少改动）

### R1 — P0（教材与运行成本）：TS仍沿用旧E-Data全量认证、Source Hash链

**证据**：`dreamerv3/thinking_student/corpus.py:63–73` 的 `corpus_factory()` 每次建立Native环境都调用 `ETrainAdmission.from_train_pool()`；而 `gym/e_data_v1/provider.py:84–135` 调用 `verified_collection()`、schema hash、`stable_identity()`验证市场与配置文件内容，`provider.py:23–43` 明确首次执行 `verify_collection` 全量验证，`e_data_v1/io.py:19–27` 使用源文件哈希。每个独立进程需自行执行这些流程。它们是旧S3认证链，而非TS本身的金融教学计算。

**影响**：与用户已经批准的C2/C3“新TS不使用旧Corpus/Hash审计链”不一致；S3如果每道CF分支频繁新建Env，成本会明显放大，甚至受源码/来源身份修改阻断。

**最小修改**：只改TS的Native环境工厂调用，不改旧S3代码。读取当前TRAIN行情及必要Env配置，直接建立功能等价的Native Env；保留最基本TRAIN边界、市场格式、Native金融规则检查，不调用旧 `verified_collection` / `stable_identity` / `SourceAdmission`。**环境工厂复用同一市场数据实例或进程级缓存**；不要因每个反事实分支重新加载整个市场。无需旧系统兼容。

### R2 — P1（A3语义）：OTHER_EXIT没有作为原生首次结束事件学习

**证据**：`labels.py:26–27`只把`TP5/SL30`作为终止，`OTHER_EXIT`被列为右删失；`labels.py:56–70` 对OTHER和RIGHT_CENSORED生成相同的竞争事件剩余集合；`OutcomeBrain`的事件头只有 TP桶、SL桶、最后生存类（`brain.py:56–84`）。直接运行纯Python的`allowed_classes('OTHER_EXIT',1034)`产生 `[9,10,20,21,22]`，而不是一个OTHER终止事件。`test_ts_v1_s1.py:64–65`也把该语义写成了测试预期。开发记录显示训练教材确实包含539个P0 OTHER_EXIT样本。

**影响**：真实Native账户在OTHER_EXIT已经强制结算，S1却将它视为“TP/SL未来还可能发生”；这对于“预测TP/SL潜在风险”可算统计删失，但**不足以声称预测真实账户的全部首次结束事件分布**。尤其S3不能将它视为完整Native结束/占仓期限预测。

**最小修改**：在进入S3的完整Q建模前，明确F2定义：建议事件集合扩为`{TP5,SL30,OTHER_EXIT,OPEN_AT_H}`，真实历史右删失独立处理；对于TP/SL超过最大预测H的情况，最后一类只能称为“**H时仍未结束**”，不能解释成永远不发生TP。调整标签、事件头和一次训练；按约定尽量继承其他已训练参数，不整体丢弃S1/S2权重。如果S3暂时不依赖F2事件结束分布，也可以先独立补充Native事实标签后再更新F2，但要承认当前F2的边界。

### R3 — P1（A2状态）：Native账户输入未覆盖所有未来交易/奖励相关字段

**证据**：`gym/research/ts_v1_account.py:11–22,28–59`只提供10个字段（仓位、方向、TP/SL距离、盈利、持仓分钟、reward bars、adverse10、跨日），没有距`last_open/main_exit_at`时间，也没有资金/保证金余量等可能影响下一次ADD合法性的账户变量。Native的 `_can_add()` 显式依赖`last_open/main_exit_at`与资金预检（`entry_env.py:340–370`）；Mask只代表**当前**能否ADD，无法自动替代未来资金与截止信息。

**影响**：与A2的“完整决策相关账户快照”合同有差距，S3完整长期Q可能面对未显式表示的账户限制和强平时刻。

**最小修改**：S3之前列出真实环境下影响未来动作与Reward的最小账户状态，并按需要扩充`account_native`。优先补充距last_open/main_exit的有效市场分钟及必要资金/容量事实；如果某信息确定可由现有因果市场/账户输入表达，简短注明即可。由开发人员负责旧Checkpoint对应输入层的部分权重继承，不建Hash门禁。

### R4 — P1（A2市场记忆）：人为CHUNK_END也重置RSSM

**证据**：`run.py:231–235`在`CHUNK_END`后令`self.first=True`；`student.py:134–139`看到`is_first`就重置MarketRSSM；Gym `entry_env.py:455–467,728–733`说明`CHUNK_END`是按`chunk_steps`人为截断，账户继续，同一历史市场继续，而非必然断档。`CorpusWindows._stack()`也在每个采样窗口强行设置首行`is_first=True`（`corpus.py:274–279`）。

**影响**：这与A2“市场记忆只在行情不连续时重置”存在语义偏差。训练与正式自主采集都这样重置，因此S2内部自洽、不是立即导致交易错误，但未来长时Q/CF的状态和长历史Market Belief定义会受影响。

**最小修改/取舍**：明确“历史连续行情”与“人为采样/进程/Env chunk”的不同。在线自主运行时，除真实市场断档/合约切换外，不要因普通CHUNK_END清空市场RSSM；训练随机窗口用burn-in重建上下文，不把随机截断宣称为真实市场重置。此项若需修改已有π0的行为，应保留旧π0作S3反事实可复现基线，新版从可继承Checkpoint继续训练并重新冻结，不篡改旧π0身份。

## 2. 低优先级（不阻止S3）

- `brain.py:72,77`的32维`OutcomeBrain.latent`进入evidence，但`brain.py:91–135`的`L_outcome`没有监督latent输出，S1时该投影不因预测损失而学到专门的知识。它是**未训练特征**，不是已验证的Outcome隐知识。可暂时屏蔽这部分或将其作为未来Value训练可适配的特征，不用创建新审计。
- `brain.py:191` 的`reason_pref`是辅助头，但`student.py:240–277`只训练最终动作NLL，没有独立的`L_reason`；S4本来才实现该教学。S2目前的`reason_pref`数值不应当被当作可信独立推理结论。
- S2日志按**有效空仓决策**计算ADD比例（正式π0为50/423=11.8%）；这是正确口径，但只有一段TRAIN历史和少量示范。没有证明择时优势（S2记录本身也承认），属于训练效果研究，**不构成工程拒收**。
- `labels.py`最后一个桶`SURVIVE`包含预测终点H之后才TP/SL的历史。`p_tp`,`p_sl`当前实为**H以内的概率**而非无限期TP/SL概率；建议答卷字段在S3使用时显式注明预测期限。
- `run.py:124–129`冻结时计算一次参数SHA256，纯记录不阻断；与旧每batch/每启动Hash门禁性质不同，不值得作为当前整改重点。

## 3. C3-S3衔接审查

S3可依赖的已完成资产：①唯一`StudentModel.policy()`；②F2真实监督的参数与既有账号/市场；③正式π0快照`policies/pi0`；④学生TRAIN原生自主轨迹（L_q_real合法来源）；⑤可复制的Native账户状态`export_state/import_state`与Native教师单笔历史事实。尚需完成：**同起点合法ADD/WAIT双分支、同完整冻结π0续行、相同评价期限、末端处理、`L_q_real/L_q_cf`训练、更新Q后的新完整策略发布。**

绝不可把旧L2/Oracle后验价值、单笔Quality或者探索μ整段回报当成`Q^{π0}`真值。旧π0虽未盈利，其功能意义不影响推进S3。

## 4. Review范围与判定

- **通过**：S0一套真实前向、S1真实后果监督与完整调用、S2自主Native学生π0；构架没有退回旧裸F3。
- **须修的真正问题**：R1旧教材认证仍进入新TS热路径；R2 Native OTHER_EXIT事件语义；R3未来账户约束字段；R4人工chunk重置的市场记忆。
- **不处理**：旧S3兼容性、旧Gate、旧测试全量审计、模型盈利/准确率门槛、DP加速、额外想象Actor。
- **证据限制**：目前审计针对上传的已提交代码和仓库内开发记录；未在当前容器运行完整项目测试或读取用户本机保存的`agent.pkl`。除了纯Python标签投影和AST解析之外，真实环境/GPU结果未独立复现。

结论：**建议允许S3继续，同时用最小范围修复上述问题；优先R1，R2/R3在将Outcome与账户状态正式用于长期Q之前处理。R4可明确版本过渡后解决，不必因此丢掉已冻结π0。**
