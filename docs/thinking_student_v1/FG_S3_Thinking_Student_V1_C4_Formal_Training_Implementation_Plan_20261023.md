# FG-S3 Thinking Student V1 — C4 正式训练实施方案（交付 Claude 开发）

> 日期：2026-10-23  
> 状态：**新增工程实施合同**，补足 C3 的「S0–S5 功能已开发后，如何使用完整 TRAIN 教材持续训练」缺口。  
> 代码依据：本轮已经检查的 Gym `gy33` / `066dece` 与 Dreamer `d333` / `c3ad03b` 中 Thinking Student 专线源代码；开发时 Claude 应以本地最新 HEAD 为准核对差异。  
> 训练资产前提：**用户已确认18个 TRAIN 合约的 Thinking Student 教材生成完成**；不再安排全量重生成旧教材。  
> 对齐：A1–A6 冻结大脑、C2「不维护旧 S3 兼容」原则、C3 S0–S5 与教材生产/消费合同。本文件不改变单仓位、3手做空、WAIT/ADD、TP5、SL30 或 Quality Reward。

---

## 0. 目标与范围

**目标：** 将目前已工作的单合约 L1–L4 教学、Native 自主交易、同策略 CF、Checkpoint，组合为一条能够**使用18合约 TRAIN 教材、定期冻结学生并持续成长、能够中断续训、进行独立 Teacher-free DEV 评估**的正式训练流程。S4、S5 在完成后接入同一流程，不需要重新训练一套学生。

核心原则：

1. **不重写学生大脑、不重造18合约教材、不重做旧S3训练系统。** 在 Thinking Student 新专线上作最少改动。
2. 只有一个 `StudentModel` / 正式 `student.policy`；目标策略 `π_k` 为**确定性 Masked Argmax**（ADD合法且 `z>0` 才ADD，平局WAIT），探索行为 `μ` 独立标记。
3. 训练时参数可以更新，**用于采集和CF的完整 `π_k` 一旦冻结就绝对不变**；新策略只能发布为 `π_{k+1}`，不得将旧标签更名为新策略的真值。
4. L1–L3先形成跨合约基础学生；L4/L5/L6以「冻结策略→Native实习→合法CF→Q/Reasoning训练→新策略→评估」循环。**每次优化器step不是一个新策略版本**。
5. TRAIN可使用合法未来生成老师标签，**学生任何一次正式动作前向只能读取因果Obs**；DEV只用于Teacher-free评价，TEST继续封存。
6. 记录实际训练进度和恢复位置，但**禁止Hash准入、SourceAdmission、全量重播审计、逐Batch文件核验、旧S3 Gate/实验平台**。
7. 不以策略盈利、ADD比例、Q优于常数等效果阈值作为工程开发阻断；保留诊断以判断正式训练是否值得继续投入。

### 本轮明确不开发

- 分布式训练平台、云训练队列、复杂实验/模型注册中心、自动超参数搜索、完整DP替代Native、旧S3代码迁移与历史回归修复。
- 重新生产已完成的18合约基础教材；除非确实缺失/语义不适用，才针对缺少字段按需补算。
- 额外Actor侧`L_align`、`L_policy_pref`、未定义的直接`L_policy_rl`；策略价值学习仍经Q锚点进入唯一Actor。
- 无上限的自动后台训练、无限轮循环。实际长时间/GPU任务必须遵守用户的显式授权和机器资源约束。

---

## 1. 当前代码与必须补齐的缺口

| 能力 | 当前代码现状（`d333`/`gy33`） | 正式训练应补什么 |
|---|---|---|
| 基础教材 | Gym `ts_v1_corpus.py` 已生成单合约 `market/facts/strict/behaviors/source.json` | 将18个已完成合约作为统一TRAIN池读取；不重算 |
| L1/L2/L3 | Dreamer `run.py` 已有 `train-market/train-outcome/train-policy` | **同一学习阶段从多个合约采样窗口**，而非一次仅一个 `--corpus` |
| L4 | `value.py` 的 `QWindows` 消费**一个**冻结策略、**一个**collection/CF | 同轮同一 `π_k` 的**多合约**collection/CF混合Value训练 |
| 学生实习与CF | `collect`、`cf` 可处理单合约，CF生产遵循冻结策略 | 循环18或本轮选定的合约，逐合约采集、增量生成；记录完成状态，避免重做 |
| 策略发布 | `freeze()` 保存完整policy及锚点 | 多合约来源元数据；策略版号、所依赖CF和训练批次；后续候选不能覆盖旧政策 |
| Checkpoint | `run/ckpt/<updates>/agent.pkl` + `latest`；同run可恢复、`--init-from`可继承 | 正式训练**阶段/轮次/合同作业进度**和训练中期Checkpoint，恢复时不误用别的策略CF |
| 训练评估 | L2/L3/L4各自有单合约尾部窗口内部`eval`报告 | 区分**TRAIN内部诊断**与真正**独立DEV Teacher-free**；多合约聚合报告 |
| L5/L6 | C3定义了S4/S5能力，`d333`尚非正式训练编排器 | S4/S5完成后注册到同一流程；无实现时只跑L1–L4而不伪称已完成L6 |

**实际代码入口：**

- Dreamer：`dreamerv3/thinking_student/run.py`：`build / resume / train_stage / freeze / collect / corpus_env / _train_with_eval`。
- Dreamer：`dreamerv3/thinking_student/corpus.py`：`TSCorpus`、`CorpusWindows`。
- Dreamer：`dreamerv3/thinking_student/value.py`：`QWindows`、`run_cf`、`load_collection`、`choose_forks`。
- Dreamer：`dreamerv3/thinking_student/student.py`、`brain.py`、`config.py`。
- Gym：`src/gym_trading_env/research/ts_v1_corpus.py`、`ts_v1_cf.py`、`ts_v1_account.py`。

**重要：** 下文所有建议的新命令和配置格式均为**待Claude实现的接口草案**，不是当前代码已经支持的选项。

---

## 2. 18合约TRAIN教材：直接复用与混合采样

### 2.1 资产发现与一次性轻量读取

- 由配置显式给出`corpus_root`或18个教材目录列表；遍历每个`source.json`读取`contract/fold/rows/reward_schema/clock/gamma`，列出实际可用教材。
- 18个合约必须满足`fold=TRAIN`，共用一致的Native Reward、MarketClock、动作/账户字段语义。发现不同口径只指出该合约的问题、拒绝混合不兼容数据，不要求重新签名整个Corpus。
- 直接读取已经存在的`.npy`市场、账户、Outcome与strict示范；**不执行旧SourceAdmission、G3、批量Hash、认证出版流程**。
- 不能因为文件名符合`CZCE.FG*`就默认可训练；检查关键文件存在、行数/shape和目标有效位即可。真实18合约名单以本地文件为准，不在源码中硬编码。
- 读取内存：优先`mmap`或按需加载，使用可配置的小缓存；不要一次将18份全部复制为多个完整数组，也不在每个Batch扫描文件系统。

### 2.2 新增最小`MultiCorpusWindows`

可以新建`thinking_student/multicorpus.py`，也可在现有`corpus.py`扩展；不要为了模块名重构已工作的`TSCorpus`。

逻辑接口建议：

```python
class MultiCorpusWindows:
    def __init__(self, corpora, stage, seed, batch_length, burnin,
                 mix='uniform_contract', contract_weights=None,
                 demo_balance=1.0, cache_size=2, train_selector=None): ...
    def sample(self, batch_size, batch_length): ...  # 返回现有agent.train所需完整Batch
    def summary(self): ...                           # 各合约真实样本/标签数
    def state_dict(self): ...                        # RNG状态与混合器配置
    def load_state_dict(self, state): ...
```

**抽样语义：**

1. **先抽合约，再抽合约内有效训练窗口**；每个Batch可以含多个合约的独立窗口，但单个时间序列窗口**严禁跨合约拼接**。
2. 初始默认`uniform_contract`：每合约近似同等曝光，避免数据量大的合约长期覆盖短合约；提供`proportional_rows`及手动权重作为实验选项，不在V1引入复杂自适应课表。
3. **L1**从该合约P0–P4账户行为窗口采样，保留自然市场/账户状态；**L2**沿用既有`LABEL_ACCOUNT`，按真实Outcome有效标签采样；**L3**仍使用`LABEL_ACCOUNT`的strict示范，按有效示范标记采样。禁止把相同市场游标的教师标签因为P0–P4重复而计成多道独立教学题。
4. L3类别平衡权重应基于**所选TRAIN合约整体的实际WAIT/ADD有效示范频率**（或明确声明的分合约权重），不是每个Batch不断改变类别权重；不要新增固定ADD比例门槛。
5. 保留现有`market_seq_e_shared / market_context_15m / market_macd_5m / position_state / action_mask / account_native`等张量合同，优先复用单合约`CorpusWindows`的切片/组装逻辑。
6. 窗口开始时的`is_first=True`是**学习窗口的临时初态/截断预热**，不能反过来变成Native真实合约中的行情中断。现有Outcome/Demo `burnin`继续生效；不跨休市/换月制造额外人为市场重置。

### 2.3 TRAIN内部诊断和真正DEV必须区分

- 现有`CorpusWindows(..., split='train'/'eval', holdout=2048)`是在**一个TRAIN合约内部**截取尾段用于开发期Loss诊断。它**不是独立DEV**。
- 新多合约训练允许对18个合法TRAIN合约使用**全部TRAIN可用窗口**；如仍需要内部尾段诊断，必须在报告中写明`TRAIN_INTERNAL_DIAGNOSTIC`，并保证该选择与实际训练排除范围一致。
- **独立DEV**由明确配置的另一批VALIDATION行情/Native环境承担Teacher-free评估。现有Gym `ts_v1_corpus.py`生产入口限于TRAIN，**不得为了DEV考试去调用TRAIN-only严格教材生产者**；如果最新代码尚无VALIDATION环境适配器，则F4只补一个从合法DEV原始行情构建因果观察与Native账户的轻量Reader/Env，不生成Oracle、ADD后果真值或CF教材。未提供DEV时可以用TRAIN Teacher-free smoke，但报告必须标记`TRAIN_SMOKE`，**不能伪装成样本外成绩**。
- 若TRAIN与DEV属于连续时间片，TRAIN教学后缀（Outcome/Oracle/CF）不得跨入DEV；不允许用DEV真实后果生成训练标签。对于TRAIN尾端未来不足的样本保留真实删失/不可比语义，不能补零或借DEV补齐。

---

## 3. 正式训练的执行顺序

### A期：跨18合约基础教育（L1→L2→L3）

**从现有Checkpoint继续，不强制从零训练。**

1. 建立独立正式run，优先继承**S2已经有合法中性Q行为先验**的学生Checkpoint；已有S3快照也可继承可复用参数，但基础教育阶段的`β_Q`和`Q→Reasoning`必须采用已确定的中性训练视图，不能让单合约尚不可靠Q误导L3。若模型结构/张量改变，开发人员做最小迁移说明；不以SHA不匹配拒绝继承。
2. **L1**：18合约混合市场+多行为Native账户窗口，更新Market/Account基础；保留外生市场RSSM，不把学生动作加入Market动力学。
3. **L2**：18合约合法ADD Outcome（TP5/SL30/OTHER_EXIT/删失、风险、MAE、Quality），只更新其授权模块与明确的Fusion适配部分。
4. **L3**：18合约合法strict示范、适当类别平衡，训练Intuition/Reasoning行为先验/Final Actor；正式动作仍按确定性Masked Argmax，Q锚点保持中性。
5. 每课程按配置新增`updates`次梯度更新，保存同一完整参数树、阶段日志和Checkpoint；不能将某合约训练完成错误地理解为新建一个Agent。
6. **冻结 `π_0`**，用2个不同TRAIN合约短程Teacher-free运行确认它们都调用同一正式Student；然后用需要的合约做**完整至真实终止**的Native采集。短程Smoke不得冒充可用于Q的完整Native回报。

基础教育可以在S4/S5代码开发期间先启动，**不需要等全系统完成才开始多合约训练**。

**正式预算**（多少L1/L2/L3更新、batch、窗口长度）由配置决定；不要把此前单合约3000次工程验证当作18合约正式训练预算。首轮先用低预算真实Smoke验证，再按用户授权增加。

### B期：按冻结策略版本进行L4/L5成长循环

一次完整循环只能对应一个**明确冻结、可调用的完整策略 `π_k`**：

| 顺序 | 操作 | 输入 | 产物与合法性要求 |
|---|---|---|---|
| B1 | `freeze(π_k)` | 完整Student+推理配置 | 唯一冻结策略目录；此后不被训练覆盖 |
| B2 | `collect` | 同一`π_k`在本轮选定TRAIN合约，`mode='policy'` | 每合约真实终止的完整Native学生经历（正式策略，不含探索/强制动作） |
| B3 | `cf` | `π_k` + 对应合约B2完整collection | 每合约合法ADD/WAIT反事实；同初态/后续同策略函数/同Reward与时钟 |
| B4 | 混合`L4` | 同一`π_k`的所有已可用TRAIN CF + 自主回报 | Value参数训练，合法CF差监督Q；不拿旧Oracle/P0–P4当`Q^{π_k}`真值 |
| B5 | `L5`（S4交付后） | Q价值锚点 + 独立示范Reasoning目标 | 先验按需补课；不能再次用同一`ΔG_CF`监督Actor偏好 |
| B6 | `freeze(π_{k+1})` | B4/B5结果及A5锚点参数 | 新完整学生候选，不覆盖`π_k` |
| B7 | Teacher-free评估 | TRAIN smoke；有独立DEV时使用DEV | 新旧策略/随机基线同条件比较，保存结果但不设收益门槛 |
| B8 | 下一轮 | 冻结的`π_{k+1}` | 重新采集/重新计算其策略条件CF，不延用旧`π_k`的长期Q标签 |

**纠正一个容易造成严重Bug的细节：** B4做很多optimizer更新时，虽然**正在训练的Value网络及由它形成的候选策略**不断变化，B2/B3的教学目标依然属于**被冻结的 `π_k`**。训练过程中**不得**用正在变化的Learner继续生成标成`π_k`的CF。整轮学习结束后才发布新候选并进入下一轮。

**L6（S5交付后）**在此循环之上加入：学生答卷、真实账户经历、反馈`ready_at`成熟、Outcome/Reasoning专项复习、课程覆盖调度。离线TRAIN可使用合法已知后缀教学，但要保留标签来源、时间和有效状态；DEV/TEST不得被反向加入训练池。

### 3.1 多合约L4的新消费层

建议新增`MultiQWindows`，**复用当前`value.py`的`QWindows`**：

- 一个样本的`collection/targets.npz`来自**同一合约、同一冻结 `π_k`**；两个动作的回报仍使用该合约的完整Native时钟和Reward，不跨合约拼接时间序列。
- Q batch可以混合同轮多个TRAIN合约的窗口，默认合约近似等权；对合法CF行保留当前`cf_boost`抽样能力。当前单合约`QWindows`强制`0 < holdout < n`；正式训练若使用独立DEV，应增加明确的**全TRAIN窗口消费模式**，不要每个合约默认砍掉2048行，更不能让短合约在默认holdout下被排除。TRAIN内部尾段诊断如仍需要则单独配置并如实标记。
- `q_real_valid`、`q_cf_valid`分别保留。`NOT_COMPARABLE`必须缺失Q监督，**绝不能当成0收益**；探索/强制动作不能冒充`π_k`自身回报。
- `κ_Q`的`auto`估计应在**同轮合法TRAIN CF成对样本**上统一计算一次，按规定的训练加权方案明确统计口径；不能取某单合约的标准差冒充18合约尺度。若样本数不足或尺度退化为0，明确报告并使用预先配置的正下界或暂不启用锚点，不能默默除以0。
- 若当前轮CF只先覆盖2–4合约，可先训练该小批并在日志写明覆盖；**不要求18合约每轮全量CF**。可以逐轮扩大或轮转，不能长期无意只训练同一合约。
- **关键待核查：市场RSSM潜在状态**。当前`QWindows`从随机窗口零状态开始、前16步burn-in，而CF标签来自连续行情下的冻结策略。做少量同cursor比较；差异明显时在Q训练窗口提供足够因果前缀/正确carry，不得为了提速把教师后验输进网络或把真实Chunk边界改成reset。

### 3.2 CF耗时与增量复用

- 18合约的`market/facts/strict/behaviors`不因为策略变化而重生成；策略条件`collection`和`cf`随着`π_k`变化而新增，目录按`policy_id/contract`保存。
- 正在重跑同一冻结策略、同一合约的CF作业，**已经完整落盘的结果可跳过**；中断的作业允许从已完成cursor继续，或以简单的单合约重跑恢复。绝不重新调用旧审计链。
- CPU/GPU worker数量用显式配置控制。已有`value.py`的CF多进程会在worker加载冻结策略并回放参考轨迹；不要默认18合约×多个GPU worker无限并行，优先以单机真实显存/内存约束运行。
- 首批每轮可选2–4个有代表性的TRAIN合约、采样部分WAIT反事实；保留全部真实策略ADD反事实。**前提是B2参考轨迹完整且CF状态有效**，否则该合同不进入可比较Q训练。
- 不为CF速度立刻重新设计逆向DP、分布式集群或索引证书。先以实际生成耗时衡量瓶颈。

---

## 4. 新增的薄训练编排器（不要替换现有算法）

建议在Dreamer新增`dreamerv3/thinking_student/formal_train.py`及必要的采样小模块；它应直接调用既有`run.py`/`value.py`逻辑而不是复制一套Student、Native或者LOSS。

### 4.1 逻辑命令（新接口，由Claude实现）

```text
python -m dreamerv3.thinking_student.formal_train pretrain --config formal_v1.yaml
python -m dreamerv3.thinking_student.formal_train iterate  --config formal_v1.yaml --rounds 1
python -m dreamerv3.thinking_student.formal_train evaluate --config formal_v1.yaml --policy /.../pi_0001
python -m dreamerv3.thinking_student.formal_train run      --config formal_v1.yaml --rounds 3
python -m dreamerv3.thinking_student.formal_train status   --config formal_v1.yaml
```

- `pretrain`：L1→L2→L3→首次freeze；若已有基础学生可用配置跳过已完成阶段。
- `iterate`：以最近指定冻结策略为起点，执行**有上限**的一轮B2–B8；如果S4尚未实现，先用现有L4→价值锚点产生下一策略，日志明确`L5_NOT_IMPLEMENTED`，不要造假成功。
- `evaluate`：使用已冻结policy，不从训练中途权重临时构建Actor；DEV未配置就明确报`DEV_NOT_CONFIGURED`，可以单独执行TRAIN smoke。
- `run`：顺序组合前述阶段；每次`--rounds`固定上限，不启动隐式后台循环。
- `status`：只读当前阶段、策略版号、已完成合约与训练更新数。

上述名字可按本地最新入口调整，但行为不能丢失。**不要另外建设任务平台、训练服务或全局数据库。**

### 4.2 一份权威训练配置（示意，非当前代码现有选项）

```yaml
# formal_v1.yaml：所有路径在此配置/CLI指定，源码不硬编码历史worktree
paths:
  corpus_root: /data/logdir/fg_single3/thinking_student/corpus   # 以本机实际目录为准
  run_root: /data/logdir/fg_single3/shared/thinking_student/formal_18c
  dev_data: null               # 独立DEV存在后填写；null不得假装DEV评估
  test_data: null              # 日常正式训练绝不读取TEST

corpora:
  fold: TRAIN
  contracts: discover         # 仅发现已生成且合法的18份教材；最终列表落入run状态
  mix: uniform_contract
  cache_size: 2

student:
  preset: size12m
  init_from: /path/to/existing/S2/checkpoint    # 真实存在的继承来源；不猜路径
  seed: 20261023
  batch_size: 8
  batch_length: 64

foundation:
  market_updates: 100         # 示例仅用于小规模Smoke，正式预算需另行确认
  outcome_updates: 100
  policy_updates: 100
  checkpoint_every: 50

iteration:
  max_rounds: 1               # 显式、有限轮次；非无限持续任务
  contracts_per_round: 3      # 初期2–4个不同市场，后续按需扩大
  cf_wait_forks_per_contract: 100
  cf_workers: 1               # 初期保守，视显存与资源再调大
  value_updates: 100
  beta_q: 1.0                 # 仅在Q有有效监督且尺度合法时启用
  kappa_q: auto_train_pairs

metrics:
  internal_train_diagnostics: true
  teacher_free_dev: false     # DEV数据未提供之前不得自动打开
  random_baseline: true
```

**关键：** 配置中的`100`步、`3`合约、`batch_size=8`只是帮助Claude做第一次真实贯通的示例，**不是用户已批准的大型GPU训练预算**，不得自动替换成百万步或一次跑完18合约全量CF。

### 4.3 恢复合同与最小运行状态

建议保存`run_state.json`（或等价简单JSON），仅包含实际恢复所需字段，不做SourceAdmission或签名：

```yaml
run_id: formal_18c
phase: foundation|iteration|evaluation|done
stage: L1|L2|L3|collect|cf|L4|L5|freeze|dev
round: 0
frozen_policy: /.../policies/pi_0000
candidate_policy: null
selected_train_contracts: [CZCE.FGxxx, CZCE.FGyyy]
completed_collections: [CZCE.FGxxx]
completed_cf: [CZCE.FGxxx]
latest_checkpoint: /.../ckpt/...
updates_by_stage: {L1: 0, L2: 0, L3: 0, L4: 0, L5: 0}
seed: 20261023
sampler_state: /.../sampler_state.pkl
```

**恢复优先级：**

1. 同run存在合法`latest`时，恢复**完整参数及训练优化器状态**；不能因为有`init_from`就在resume时从头覆盖。
2. 从最新保存的**阶段进度**继续：完成的合同采集/CF若对应冻结policy及目录正确则跳过；未完成作业只重做最小部分。
3. `train_stage()`目前主要在整条命令结束后保存Checkpoint；正式长训练增加按`checkpoint_every`分段保存，并持久化随机采样器进度。允许最后少量未提交Batch因中断重做；避免同一数据标签跨策略误复用。
4. 每个新候选使用**新run或可区分的策略版本目录**，`π_k`快照不可原地被更新。模型权重、优化器状态、配置、课程进度应明确哪份是最新；最新训练快照与DEV较优候选是**两个不同指针**。
5. 若代码修改引起部分参数shape/语义变化，优先按现有`transfer.py`做一次性明确迁移或针对性重建优化器；不能把加载失败包装成静默从零训练；更不能用源码SHA阻止合理继续训练。
6. 训练过程中使用原子写入Checkpoint/策略/作业完成标记的基本机制；不强制建设事务数据库、工作流引擎或复杂状态机框架。

### 4.4 冻结策略元数据补充

当前`freeze()`的`corpus`参数偏单合约，正式训练需要记录**TRAIN合约列表、每个合约的Reward/Clock口径、来源Checkpoint、当前Q教师策略版本、完整策略执行参数、κ_Q计算口径**。只是普通溯源信息，不是哈希准入。

独立正式policy版本目录至少含：`agent.pkl`、`policy.json`、清晰唯一的`policy_id`。`collect`/`cf`/DEV加载的必须是这个不可变化的完整快照（不是正在训练的`latest`）。可保留当前非阻断`params_sha256`用于记账，但不需要每次读取重算，也不让它阻止训练。

---

## 5. DEV评估、研究指标与模型选择

### 5.1 严格 Teacher-free

- 每次使用**同一个冻结完整student.policy**在Native账户执行，和TRAIN自主实习一致；teacher、Oracle、未来Outcome/CF标签不得参与输入或考试期梯度更新。
- DEV行情必须是**独立合法VALIDATION来源**。若尚没有DEV只做`TRAIN_SMOKE`；TEST从不作为开发期自动评估来源。
- 市场RSSM从真实合约起点连续维护状态；不因为输出日志chunk、采集暂停或换batch进行人为重置。
- 随机入场基线必须与学生在同一数据段使用**同一Native金融约束**，每次只允许空仓合法ADD、TP5/SL30、交易时钟和固定仓位；随机基线可多个seed，不能通过未来筛掉失败的随机轨迹。

### 5.2 主要指标（不当作通过门槛）

1. **单位可交易时间累计速度/Quality Reward**：总奖励/实际可交易日或有效市场时间，报告分母定义；重点衡量快速TP5而非静态胜率。
2. **相同条件相对随机入场优势**：相同合约、窗口、交易规则和资本条件下学生与随机基线的差值（多seed时展示分布）。
3. 辅助：ADD/WAIT率、TP5首次完成时间分布、SL30、OTHER_EXIT、未完成仓位、占仓率、最大浮亏、账户净盈亏、每合约及整体表现。
4. 教学诊断：L2事件Loss/常数分布基线、L3 demo表现、L4成对Q误差/符号一致率/常数Q基线、不同合约曝光比例、单轮CF成功率与耗时。**Q指标好不代表策略好**。

### 5.3 候选策略选择

- 保存**最新候选 `latest_candidate`**和**当前DEV较优 `best_dev`**两个状态；新策略并非必然提高成绩。
- DEV只有少量数据时不要过度反复挑选超参造成过拟合；模型选择使用事先明确的主指标和风险报告，不从TEST反向调参。
- 不要求连续每轮进步才能保存或继续开发；有问题时按教学目标和数据覆盖分析，不自动回退全架构。

---

## 6. 最小开发切片及Claude交付要求

按以下顺序**写代码而不是写审计报告**，每片实际训练/推理一次并可续训；没有必要一次重构全部已有CLI。

| 切片 | 具体开发内容 | 最小实际结果 |
|---|---|---|
| **F0：18合约读取** | TRAIN教材目录发现；兼容现有`TSCorpus`的懒加载/缓存与`MultiCorpusWindows`；全局L3示范类别权重；不跨合约窗口 | 两合约混合Batch可直接送已有L1/L2/L3训练，显示真实来源 |
| **F1：正式基础训练** | 同一Agent执行L1→L2→L3，继承现有S2模型；按阶段持久化Checkpoint/进度；首次冻结π0 | 两合约少量GPU或CPU真实更新、保存/恢复、正式π0 Native动作 |
| **F2：多合约策略条件Q** | 对固定πk逐合约collect/CF，新增`MultiQWindows`，L4多合约同策略Q训练，跨合约κ_Q | 至少两个不同合约产生合法CF，更新Value并发布新π1；无伪造0回报 |
| **F3：有限轮自动编排与续训** | `formal_train`最小CLI及`run_state`；一轮B2–B8；单作业完成后增量记录；中断续训 | 同一πk CF不误重算、不把旧πk的Q教材当成πk+1标签 |
| **F4：Teacher-free评估** | 复用Native正式policy执行，TRAIN smoke与独立DEV严格区分；主要指标与随机基线 | Teacher-free短考报告包含单位时间Reward、随机基线和必要风险字段 |
| **F5：接入S4/S5** | S4 Reasoning独立教学、S5反馈成熟/补课及L6调度加入相同编排器 | 同一Student从自主练习→合法反馈→更新→新policy→再次练习，不切Actor |

**开发与训练并行方式：** F0/F1完工后就可以开始18合约L1–L3正式训练；F2完成后可提前启动有上限的多合约L4迭代；F3–F5继续完善，不要求等待全套系统才训练。GPU长任务仍需用户明确授权。

### 6.1 必要且足够的自检（不再加审计链）

1. **混合采样**：batch同时来自两个合约；每条时间窗只来自同一个合约；正确`is_first/burnin`和合法标签。
2. **真实梯度**：L1、L2、L3以及L4每个至少一次非空监督更新；仅修改该阶段授权参数组。
3. **Checkpoint**：小规模训练中断并恢复；update计数与参数确实继承，未偷偷随机初始化。
4. **同策略CF**：同一冻结policy在两个合约生成有效完整Native参考及ADD/WAIT成对回报；策略、Reward、Clock版本匹配，无非法后验输入。
5. **Q策略更新**：`β_Q=0`时L3不受Q影响；训练Q后新policy显式锚点起作用，候选策略重新编号。
6. **复用**：已经产出的合法基础教材不重造，已完成同policy CF无需重新生成；与不同policy的CF不会混用。
7. **隔离**：无DEV时报告明确TRAIN smoke；DEV考试无teacher/更新；TEST不被程序默认读取。
8. **资源**：两个合约的小预算完整走通、没有在每batch运行Hash/G3/旧准入，也没有把18合约全量复制到内存。

> 只有真实失败才扩大测试范围。不要再形成几十份验收与认证报告。一个简短的`FORMAL_TRAINING_IMPLEMENTATION.md`记录入口、成功的最小运行命令、Checkpoint位置与尚未完成功能即可。

### 6.2 明确不要求的验收条件

- 不要求18合约全量CF一次生成完、不要求S4/S5没开发完就宣称全生命周期完成。
- 不要求学生获利、AUC达到某值、Q严格超过常数、每轮策略单调提升。
- 不要求旧S3实验脚本、旧Gate、旧E实验产物、旧Reader审计、Hash认证与历史worktree保持兼容。
- 不要求任何长时GPU训练在未经用户授权时启动。

---

## 7. 可以直接发给Claude的执行指令

> 以下指令基于此文档执行，**不要额外再做设计审计**。

**开发任务：Thinking Student V1正式训练能力（C4）。**

请先读取本文件、当前权威A1–A6与C3，核对本地Gym/Dreamer两仓当前Thinking Student实际实现。18个TRAIN合约教材已生成，不重造教材、不维护旧S3兼容。

1. 先实施**F0与F1**：构建轻量多合约训练窗口，从18份合法TRAIN教材中混合采样，沿现有L1→L2→L3真实Learner训练，不跨合约拼序列；共享同一模型Checkpoint；可从已有S2模型继承且能中断继续。实现后做两个合约、很少更新次数的最小真实测试。
2. 接着实施**F2与F3**：沿用`collect/run_cf/QWindows/freeze`，为冻结πk逐合约自主Native实习、生成有效同策略ADD/WAIT反事实、混合训练L4、通过显式Q锚点发布新πk+1，并保存round/合约/CF进度。不得混用不同πk的Q标签。
3. 然后实施**F4**：真实Teacher-free DEV评估接口与TRAIN smoke区分，给出核心择时Reward、随机基线、账户风险报告。尚无DEV数据时不得把TRAIN内部holdout冒充DEV。TEST保持封存。
4. S4/S5若已在本地实现，再按**F5**接入现有编排器；若尚未实现，保持接口清楚并标明未完成，不要为了形式上的完整写空壳假循环。
5. 每个切片做**一次最小功能测试+一次真实训练/Native调用**，报告修改的文件、入口、实际测试结果、可恢复Checkpoint以及剩余待开发项。**到达需要大量CPU/GPU资源或长时间训练时先停止，不擅自启动18合约大规模正式训练。**

代码改动要最小、接口尽可能复用，避免为一次训练搭建新的配置平台、源代码Hash准入、审计/认证体系；优先让完整正式训练早日开始。

---

## 8. 反向Review：这份方案是否偏离最初设计？

| 核查 | 结论 |
|---|---|
| 是否仍然只有一套完整学生、一个正式下单口？ | 是；阶段性训练更新同一Student、冻结完整policy用于采集/CF/考试 |
| 是否把Oracle最佳值冒充学生Q？ | 否；L2 Oracle演示与`Q^{π_k}`始终分离 |
| 是否把每次Q更新立刻混进后续同版本CF？ | 否；CF只由冻结πk生产，Learner候选更新后另立版本 |
| 是否无意重做18合约基础教材？ | 否；基础教材直接读，策略变化只新生成相关Q教材 |
| 是否把L1/L2/L3在不同合约变成18个互相覆盖的学生？ | 否；一个run、一个混合batch/优化器阶段，统一权重与Checkpoint |
| 是否再次引入旧审计导致生产/读取变慢？ | 否；只做关键张量与真实Native合法性检查，不做Hash/准入链 |
| 是否用TRAIN尾部当真正样本外DEV？ | 否；两者明确命名和数据界限 |
| 是否把S4/S5未完成说成可以一键实现正式持续学习？ | 否；F1/F2可先训练，F5只有实现在场时才接入 |
| 是否保证最终稳定盈利或策略每轮提升？ | **不保证**；正式训练工程与交易能力效果验证严格区分 |

**结论：** C3负责“学生应该具备什么能力以及各阶段如何开发”；本C4负责“已有18合约教材后，如何正式训练同一个学生直到多轮自主成长”。优先实现F0/F1并启动小范围多合约实训，随后完成F2/F3/F4/F5。不要倒退重构已成功的S0–S3。
