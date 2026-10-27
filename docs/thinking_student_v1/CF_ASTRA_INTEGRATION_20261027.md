# Thinking Student V1：归档 Claude CF，Review 并集成 Astra 修复（2026-10-27/28）

任务：用户 2026-10-27 的“归档Claude CF方案，Review并集成Astra修复”，以及 10-28 的补充（归档不再开发或测试；E-7 选方案 A；磁盘管理）。逐项记录见 `TODO.md` 的 E-7 和 E-8。验证产物在 `/data/logdir/fg_single3/thinking_student/CF_ASTRA_INTEGRATION_20261027/`。

## 1. 归档分支与提交

| 仓库 | 分支 | 提交 |
| --- | --- | --- |
| Gym | `archive/ts-v1-cf-claude-20261027` | `21105bf31fcae9bb058c4525fefc181fe18af655` |
| Dreamer | `archive/ts-v1-cf-claude-20261027` | `25c7a42854b2f72c37103c8b3cbffa6b6200d135` |

- Dreamer 的归档提交包含开发 worktree 当时未提交的流式修改（未测试）。
- 归档只保存，不再开发或测试（用户 10-28）。
- 集成分支先回退了该方案，所以正式训练代码中不再包含它。
- 开发 worktree `fg-s3-thinking-student-v1-20261022` 没有清理，也没有重置。

## 2. Astra Review 结论

**七项都通过，没有必须修的 Bug，按 Astra 版本集成。**

| # | Review 点 | 结论与证据 |
| --- | --- | --- |
| 1 | EXACT_STATE 保证未来 Native 和策略等价 | <ul><li>合并条件：两边都空仓、输入相同、策略状态相同（mode、first、done、n_actions、carry），未来相关的 Native 状态也相同。</li><li>Native 比较的字段包含 Claude 独立核对过的全部字段，还多几项，只会更保守。</li><li>策略不读 reward；env RNG 只在 reset 时使用。</li><li>FG201 t=7 重算：G = 0.21897048173250155，与补算文件逐位相同，与完整 Native 运行相差 1.4e−16。</li></ul> |
| 2 | FUNDS_SAFE 没有被放宽 | <ul><li>资金不等式不变，只增加了“策略状态相同”和“控制项相同”两个条件。</li><li>每日入场上限保护在 TS env 中不会触发。</li><li>FG205 t=243、3932 重算：仍分别在 244、4350 处 FUNDS_SAFE 合并，G 与旧值相同。</li></ul> |
| 3 | 64 步探测、分层预算和续算不改变 Q 定义 | <ul><li>暂停时保存 Native、策略状态和折扣回报进度；续算按原步序接着走，不重复也不跳步。</li><li>探测加续算合计不超过 8192 步。</li><li>FG201 t=31153：探测 64 步、续算 42 步后到达真实终点，G 与旧的完整运行相同。</li></ul> |
| 4 | 旧 250 + 新 664 标签的继承和有效性 | 旧 250 条与补算文件逐条相同；新增 664 条全是 EXACT_STATE，权益差为 0。 |
| 5 | 639 条 UNAVAILABLE 和 43 条 CAP 不产生标签 | 都没有 g 和 delta_g；`q_cf_valid` 只在 VALID_COMPLETE 行为 1。重算的 t=20789（CAP）也是这样。 |
| 6 | CF Reader 和 S5 反馈使用补算结果 | <ul><li>Astra 已去掉以 `summary.json` 跳过反馈的逻辑：开启 L6 的轮次每次按当前 CF 和同轮合并 κ_Q 重建教学题。</li><li>实跑见第 4 节 (c)。</li></ul> |
| 7 | Checkpoint、MarketCarry 和策略版本保持不变 | <ul><li>全部验证中，开发 run 的 `pi_0002`、MarketCarry 和补算 CF 只被读取，前后文件指纹相同。</li><li>学习器从 `pi_0002` 继承；三份 CF 的策略都是 `pi_0002`。</li></ul> |

集成时新增了以下内容，它们都不是对 Astra 代码的 Bug 修复：
- 探索开关 `iteration.cf_coverage: exploratory`；
- CF 输出完成后删除已结束分叉的快照；
- E-7 方案 A 的配置；
- 文档。

## 3. 最终集成提交

两仓的分支都是 `integration/ts-v1-cf-astra-20261027`，worktree 为 `fg-s3-ts-v1-cf-integration-20261027/`。

**Gym**，最终提交 `c359c293fa8c3776dfebd2872e96498f7ef6a7a4`：
- `08e3464`：回退 Claude；
- `c359c29`：引入 Astra `7596566`，树与之相同。

**Dreamer**，最终代码提交 `eef52f50e896c889a23063b81c5bf21bc38cfaa0`，其后只有本报告的提交：
- `54a7184`：回退 Claude；
- `a9388cc`：等于 Astra 的 `4c1916f`；
- `e64eb5f`：等于 Astra 报告 `b403091`；
- `7f713ee`：文档；
- `86597da`：探索开关、E-7 方案 A、快照清理；
- `eef52f5`：c4 测试修正和文档。

- **只有一套 CF 生产器**，即 `thinking_student/value.py` 的 `run_cf`。`tools/ts_cf_fix_smoke.py` 是 Astra 的 DEV_ONLY 诊断脚本，不产生训练标签。
- **没有合入开发分支** `feat/thinking-student-v1-20261022`：开发 worktree 仍有未提交的修改，其内容已在 Dreamer 归档提交中。集成分支是 feat 的直接后继；那边的修改处理掉之后，执行 `git merge --ff-only integration/ts-v1-cf-astra-20261027` 即可。合并之前，开发 worktree 里仍是 Claude 的 CF，不要从那里启动训练。
- **10x run 分支**已快进到上述最终提交。快进时没有 10x 步骤在运行，两个 worktree 都是干净的。

## 4. 必要测试的实际结果

代码版本：
- (a)、(b)、(c) 都在 Gym `c359c29`、Dreamer `86597da` 上启动，启动时 worktree 干净。
- 运行期间工作区只改了文档和 `value.py` 的 docstring（后来进了 `eef52f5`），不影响运行结果。
- c4 修正后的重跑使用 `eef52f5` 的测试文件。

**(a) GPU 定向 CF 回归（`regression.py`，11/11 通过）**

| 合约 / cursor | 结果 |
| --- | --- |
| FG201 t=7 | EXACT_STATE：4 步后在 11 处合并；G 与补算结果和完整 Native 运行都相同 |
| FG201 t=31229 | 探测中 30 步到达 TERMINAL；G 与旧值相同 |
| FG201 t=31153 | 探测 64 步、续算 42 步后到达 TERMINAL；G = 2.191495703696919，与旧值相同 |
| FG201 t=20789 | 8192 步到上限：NOT_COMPARABLE/CAP，没有 g 和 delta_g |
| FG205 t=243、3932 | FUNDS_SAFE，分别在 244、4350 处合并；G 与旧值相同；3932 在续算中完成 |

另外三项也通过：
- 参考重放与采集一致；
- `q_cf_valid` 只在有效对上为 1；
- 输出完成后，已结束分叉的快照都已删除。

**(b) CPU TS 测试**

| 组 | 结果 |
| --- | --- |
| gym：cf、account、corpus、市场分钟、Native env、MACD、行情缓存 | 61 passed |
| dreamer s0–s3 | 29 passed |
| dreamer s4–s5 | 10 passed |
| dreamer c4 | 17 passed、1 failed；修正测试后该项通过 |

c4 失败的是集成时新加的探索断言，生产代码没有问题：
- 该夹具里 `pi_0` 从未训练过 L4，L4 优化器从学习率 0 开始线性预热。
- 所以唯一的那一次 L4 更新不会改变任何权重。
- 测试改为 2 次更新后通过（`eef52f5`）。

**(c) 小预算编排器验证 π₂→π₃′（`verify_pi2_r3/`，14/14 通过）**

这只是验证，不是正式轮次。

做法：
- 按探索配置的方式建链接，接入开发 run 的 `pi_0002` 经历和 FG201 补算 CF；
- 缩小预算：L2 复习 4 次、L4 8 次、L5 4 次，DEV 只用 1 个随机种子。

| 检查 | 结果 |
| --- | --- |
| 从磁盘续跑 | `--until cf` 停下后接着跑完；第三次调用报告 `max_rounds`；没有重新采集，也没有重算 CF |
| CF 进入正式训练 | FG201 的 914/1596（0.5727）原样进入；`L4 = S3_Q_TARGETS_EXPLORATORY`，`L4_coverage.below = {FG201: 0.5727}` |
| S5 读到新的有效 Q | FG201 教学题中 CF 914 条；三份反馈与 Astra 独立重建的逐列相同；合并 κ_Q = 0.6244509768525569 |
| L4 更新 Value Brain | 训练对 2340 个（FG201 914、FG205 818、FG209 608）；决策错 289 行 ×4 加权，128 个窗口中有 36 个含决策错；价值锚 β_Q = 1.0，来自这 2340 对 |
| E-7 方案 A | L2 复习和 L5 的补课窗口都是 0 |
| 参数组 | 只有 outcome、fusion、value、decision 变了；market、account 不变 |
| 合法 checkpoint | 学习器从 `pi_0002` 继承，更新计数 35200 → 35216；每段都从上一段的 checkpoint 恢复 |
| 考试 | DEV FG409 完整、Teacher-free；没有使用 TEST |
| 源数据只读 | 开发 run 的 `pi_0002`、MarketCarry 和补算 CF 前后文件指纹相同 |

## 5. S5 与 CF 接口

**正常。**
- **CF → L4：** `MultiQWindows` 读补算目录，只取 VALID_COMPLETE，共 914 条。
- **CF → S5：** `feedback` 读补算结果，UNAVAILABLE 和 CAP 弃权，κ_Q 取同轮合并值。
- **S5 → L2、L4、L5：** `--feedback` 进入三个训练命令。按 E-7 方案 A，L2 和 L5 只用自然窗口；L4 照常对决策错加权。
- **`summary.json` 问题：** 已解决。开启 L6 的轮次每次都重建教学题，探索 run 的反馈也写在自己新的目录里。

## 6. 是否具备恢复正式 L6 训练的条件

**具备。**
- 只有一套 CF 代码，配置已就绪，上述验证都通过。
- 我没有启动正式训练。小预算的 π₃′ 不是正式轮次，不计入任何 run。
- Astra 的独立 L4 验证 checkpoint（`learner_pi_0002_fg201_cf_l4`）也不是正式 L6 轮次。

**π₂→π₃ 主线（开发规模，探索配置）：**
- 先执行 `formal_v1_carry_pi2_cf_explore.yaml` 文件头中的链接命令。
- 再在集成 worktree 的 gym 目录运行：
  ```bash
  env PYTHONPATH=.:src:../dreamerv3 /home/v/miniconda3/envs/dreamerv3/bin/python -m dreamerv3.thinking_student.formal_train iterate --config ../dreamerv3/configs/thinking_student/formal_v1_carry_pi2_cf_explore.yaml --rounds 1
  ```
- 第 3 轮复用开发 run 的经历，不重新采集，也不重算 CF。按小预算验证的实测推算，约 15 分钟：L2 复习 200、L4 500、L5 100 次，DEV 考试 5 个随机种子约 9 分钟。之后各轮按正常的 collect → CF（Astra）→ L6 进行。

### 磁盘核算

当前 /data 剩余 291 GB。以下为开发 run 的实测值：

| 项目 | 大小 |
| --- | --- |
| checkpoint | 每份 388 MB |
| 学习器（每轮） | 3 份（不开 L6，1.2 GB）或 4 份（开 L6，1.6 GB）；10x 的 `save_every` 也放大 10 倍，所以份数相同 |
| 冻结策略 | 0.39 GB |
| 采集（每轮） | 约 0.5 GB |
| DEV 考试 | 0.18 GB |
| 教学题 | 约 0.04 GB |

估计：

| 场景 | 每轮 | 合计 |
| --- | --- | --- |
| 探索 run 第 3 轮 | ≈ 2.2 GB | — |
| 探索 run 之后各轮（开发规模） | ≈ 2.7 GB，另加 CF 保留快照 0–7 GB | 第 4–9 轮最多约 60 GB，通常 20–30 GB |
| 10x 第 1–9 轮 | ≈ 2.3–2.7 GB，另加 CF 保留快照 | 约 25–60 GB |
| CF 运行中的临时峰值 | 每合约最多约 5 GB（FG201 为 4.7 GB）；10x 的 WAIT 分叉多 10 倍，可到约 7 GB，两合约并行约 15 GB | 输出完成后回落 |

**结论：** 空间足够，不需要额外清理。

**本次清理：**
- 删除了 FG201 补算中 707 个已结束分叉的快照，共 2.48 GB；文件名记在 `clean_supplement_snapshots.json`。
- 保留：
  - 639 个未续算探测的快照（2.25 GB）；
  - 参考重放；
  - 全部 CF 标签和 checkpoint；
  - Astra 的 L4 验证 checkpoint。
- 其他项目的数据没有动。

**仍需注意：**
- 开发分支要等开发 worktree 的未提交修改处理掉之后再合并（第 3 节）。
- FG201 剩下的 639 + 43 个分叉没有标签。按探索项目处理，不再为此扩大专项开发。
