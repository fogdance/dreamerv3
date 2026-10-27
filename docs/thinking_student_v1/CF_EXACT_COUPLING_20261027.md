# CF 精确状态耦合与预算探测（2026-10-27）

- 分支：`feat/thinking-student-v1-20261022`。
- 代码：gym `21105bf`（在 `2b4e894` 之上）；dreamerv3 在 `33515f8` 之上。
- 依据：FG201 CF 调查报告 `/data/research/fg201_cf_investigation_20261024/REPORT.md`，§7.3 第一步和 §7.5 验收方式。
- 用户要求（2026-10-27）：独立修复，不参考已有修改；精确状态重合判定和预算预筛调整是同一项工作，必须一起修。
- 产物目录：`/data/logdir/fg_single3/thinking_student/CF_EXACT_COUPLING_20261027/`。只用 TRAIN 合约 FG201、FG205 和开发 run 的 `pi_0002`，没有使用 TEST。

## 根因

1. **耦合只有资金通道。**
   - 旧规则：分支与参考在同一 cursor 都空仓、输入相同，并且参考此后的最低权益加上负的权益差不低于 `coupling_floor`（FG201 为 90,336），才按拼接公式合并。
   - FG201 的 `pi_0002` 把权益亏到 36,987，参考每一行都在下限之下，所以 `couple_from` 等于合约末行，没有一个分支能合并。
   - 但下限只是资金不影响后续的**充分条件**。分支账户与参考完全相同时，后续必然相同，不需要资金条件。
   - 例：t=7 的分支第 4 步（c=11）就与参考的现金、持仓和序号全部相同，旧规则仍让它跑满 8,192 步，结果是没有标签的 CAP。
2. **预筛不给这些分叉运行 Native 的机会。**
   - `run_cf` 用 `Reference.decidable` 找出参考上不能判定的分叉：上限窗口内不能 FUNDS 耦合，真实终点又在上限之外。
   - 预算只让分层抽中的 48 个运行；FG201 的其余 1,298 个直接记为 `UNAVAILABLE_BUDGET/BELOW_FLOOR_BEYOND_CAP`，没有运行 Native。
   - 所以只改耦合规则，它们仍然没有机会合并。

## 修复（两项一起）

### 1. EXACT 耦合通道（gym `ts_v1_cf.py`）

- `native_state(env)`：之后各步要读的全部账户状态，不含市场。
  - 包含：
    - Decimal 账本余额、未实现盈亏、当前价；
    - 持仓槽位和 entries；
    - 奖励记分器的活动持仓、Quality TP 的交易日记录、记分时钟；
    - 状态和 pending 原因、终止与截断标志、chunk 计数；
    - 已发布的 Mask；
    - 账户、入场序号和当日开仓数；
    - env RNG。
  - 不含只记历史、之后不会再读的部分：流水、成交记录、归因、奖励事件和累计、权益样本、info 计数。
  - 核对依据：
    - `account_native`、`position_state` 和 Mask 只读上面列出的状态和外生的合约条款、时钟。
    - `processed_fills` 的键带有成交时间，以后的成交不会与已有的重复。
    - `_tp_info_cache` 只用于 info。
    - 记分器的 `seen`、`closed` 只用来判断新 ID 是否重复。
    - 已平仓的条目会从 `_quality_*` 中删除，所以空仓时这两个字典为空。
- `Reference.record(c, native)`：CF worker 重放参考时，在每个空仓决策处记录参考的 `native_state`，dreamerv3 端同时记录策略记忆。
- `Reference.coupling(state, native)`：前提与旧规则相同，即双方空仓、`position_state`、`action_mask`、`account_native` 相同。然后依次判断：
  - **EXACT**：分支的 `native_state` 等于参考在 c 的记录。没有资金条件。
  - **FUNDS**：旧的资金下限条件，结果不变。
  - 两者都不满足时不合并。
- 合并时由见证复核策略：
  - 与旧规则相同的部分：答卷 z 和动作；
  - 新增：策略记忆 `policy_memory`，即 `n_actions` 种子计数和 carry 字节的 blake2b 摘要。
  - 不相等时报错，不会静默合并。
- 结果记录多一个 `coupling` 字段（EXACT、FUNDS 或 null）。拼接公式和回报定义都不变。

### 2. 预算探测（gym `run_branch(probe=…)`，dreamerv3 `value.run_cf`）

- 参考上不能判定的分叉中：
  - 分层抽中的 `cf_budget` 个照旧跑到上限；
  - 其余每个都跑 `cf_probe_steps` 步的 Native 探测。
- 探测内 EXACT 合并、分支自己 MARGIN_CALL 终止或数据结束时，结果与跑满完全相同，因为分支是确定的。
- 探测结束时仍未结束的，记 `UNAVAILABLE_BUDGET/OPEN_AFTER_PROBE`，`steps` 为探测步数，没有回报、TERMINAL 和标签。
- `Reference.decidable` 的代码不变，它只回答参考本身能否判定；文档改为说明 EXACT 和自身终止只有分支能告诉我们。
- 所有分叉轮流分给各 worker。job 带上 CF 的 `floor`，worker 不再各自计算。

### 入口与配置

| 位置 | 变化 |
| --- | --- |
| `run cf … --budget N --probe-steps P` | 新增 `--probe-steps`（默认 512，要求 0 < P < `--max-steps`） |
| `formal_train` 配置 `iteration.cf_probe_steps` | 默认 512；设置了 `cf_budget` 时检查 0 < P < `cf_max_steps`；随 `--budget` 一起传给 CF |
| `meta.json` | `budget` 改为 `full`、`probed`、`probe_steps`、`probe_closed`、`unavailable` 加各层数量；新增 `coupled`（EXACT 和 FUNDS 的次数） |
| 轮记录 `rec['cf'][c]` | 保留 `coupled` 和上述 `budget` 键 |
| `formal_v1_carry.yaml`、`formal_v1_l1_l2_10x_continue.yaml` | 都加 `cf_probe_steps: 512`。这是计算上限，不是训练步数，所以 10x 配置用同一个值 |

S5 教学题不用改：`UNAVAILABLE_BUDGET` 照旧弃权，`pair()` 给出 `delta_g = None`。

## 测试（CPU）

| 范围 | 结果 |
| --- | --- |
| gym `test_ts_v1_cf.py` | 8 passed |
| gym 的 TS 测试文件：`test_ts_v1_{cf,account,corpus}`、`test_s3_market_minute`、`test_s3_e_native_env`、`test_wp02_macd`、`test_entry_market_cache` | 63 passed |
| dreamerv3 `test_ts_v1_s3.py` | 5 passed（188 s） |
| dreamerv3 `test_ts_v1_s4.py`、`test_ts_v1_c4.py`、`test_ts_v1_s5.py` | 28 passed（1,428 s） |

新增和改写的测试：

- `test_an_exact_coupling_needs_no_funds_and_equals_the_full_branch`：
  - 下限为无穷大时只有 EXACT 能合并，合并后的 G 等于跑到真终止的 G，误差在 1e-12 以内；
  - 把参考记录的现金改一点（借贷平衡），合并会推迟；
  - 用真实下限时两个通道都会出现。
- `test_the_native_state_is_what_a_later_step_reads_not_history`：
  - 只改历史时 `native_state` 不变；
  - 改余额、入场序号、pending 或 RNG 时不相等；
  - `record` 拒绝非空仓的 cursor。
- `test_a_probe_ends_an_open_branch_without_a_label`：
  - 探测内合并的结果与跑满相同；
  - 仍未结束的是 `OPEN_AFTER_PROBE`；
  - 探测步数为 0 或不小于上限时报错。
- s3 `test_cf_branches_of_the_frozen_student_are_exact`：
  - 用真实小学生做 EXACT 合并，G 与关掉 EXACT 跑到终止的 G 相等；
  - 策略记忆不相等时报 `differs from the reference`。
- s3 `test_a_cf_budget_probes_the_forks_it_does_not_run_to_the_cap`：
  - 检查探测记录、`budget` 计数、`coupled` 计数和 `end_status`；
  - L4 跳过与锚点保留的部分不变。
- c4、s5 编排测试：`cf_budget: 0` 时每个未定分叉都只做探测，覆盖率仍低于 0.8，所以照旧跳过 L4。

## 真实数据回归（GPU，bfloat16，与原 CF 相同）

脚本 `regression.py`，结果 `regression.json`（检查）和 `regression_parts.json`（原始结果）。11 项检查全部通过：

| 分叉 | 原 CF（E-5） | 新 CF | 核对 |
| --- | --- | --- | --- |
| FG201 t=7 | CAP，8,192 步，无标签（预算样本） | EXACT 合并于 c=11，4 步，权益差 0 | G = 0.21897048173250155；关掉 EXACT、跑满 31,252 步到 MAIN_ROLL 的 G = 0.2189704817325017（与报告的完整 Native 值相同），差 −1.39e−16；c=11 之后 1,495 笔成交（进出 cursor、原因、价格、净盈亏）逐笔与参考相同 |
| FG201 t=20789 | CAP | CAP，未合并 | 第一个共同空仓点现金差 +564，不能合并 |
| FG201 t=31153 | TERMINAL，106 步 | 相同，G 逐位相等 | c=31229 处 `funds_cover` 不同 |
| FG201 t=31229 | TERMINAL，30 步 | 相同，G 逐位相等 | |
| FG205 t=243、t=3932 | FUNDS 合并于 244、4350 | FUNDS 合并于 244、4350，G 逐位相等 | 现金差 −282，EXACT 不成立，退回 FUNDS |

三个 worker 重放参考，z 全部与采集一致（`z_differs` 为 0）。

第一次运行时脚本有两处写错，代码本身没问题：

- 对照分支的上限仍是 8,192，没跑到终点，G 为空。改为 40,000。
- 成交后缀按 `entry_cursor >= 11` 过滤。决策在 d 时成交在 d+1 根，所以分支在 10 的 ADD（11 进、11 出）属于合并前。改为 `> 11`。

## 没有做的

- 报告 §7.3 的第二到四步：共享市场推理、事件复用和未来价值缓存、有界尾部（`VALID_ESTIMATED`）。
- 资金不同但行为等价的合并。EXACT 要求账户逐项相同，FUNDS 不变。
