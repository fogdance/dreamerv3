# Thinking Student V1 — C4 正式训练实现（2026-10-23）

- **依据：** `FG_S3_Thinking_Student_V1_C4_Formal_Training_Implementation_Plan_20261023.md`（C4）。
- **计划与实施中的变更：** `PLAN_C4_20261023.md`。
- **逐项证据：** `TODO.md` 的 C4 节。

**C4 交付的能力：** 同一个 Thinking Student 可以：
- 在全部 TRAIN 合约上学习 L1→L2→L3，冻结为 π0；
- 逐轮做自主 Native 实习、同策略 CF、L4 和 S4 先验刷新，冻结出 π_{k+1}；
- 在独立的 DEV 合约上做 Teacher-free 考试；
- 任一步被打断后，都能从磁盘接着跑。

S5（L6：`ready_at` 反馈与课程循环）未实现。每轮在 `run_state.json` 中记为 `L6_NOT_IMPLEMENTED`，代码里没有空循环。

## 入口

所有命令从 `gym-trading-env/` 运行；环境变量写法见 `dreamerv3/configs/thinking_student/README.md`。

| 命令 | 作用 |
| --- | --- |
| `python -m dreamerv3.thinking_student.formal_train pretrain --config F` | 用所选 TRAIN 合约的窗口依次训练 L1→L2→L3（`foundation/`），冻结为 `pi_0000`，然后考 DEV，再在两个合约上做 TRAIN 冒烟 |
| `… formal_train iterate --config F [--rounds N]` | 有上限的改进轮，每轮从最新的冻结 π_k 出发，依次是：<br>• 本轮合约各自正式采集到真终止；<br>• check；<br>• CF；<br>• RSSM 短窗诊断；<br>• L4（只用 π_k 的 CF）；<br>• L5 先验刷新；<br>• 冻结 π_{k+1}（带 β_Q、合并 κ_Q）；<br>• 考 DEV |
| `… formal_train evaluate --config F --policy pi_000k [--train-smoke]` | 对一个冻结策略做 Teacher-free 考试 |
| `… formal_train run --config F [--rounds N]` | 先 pretrain，再 iterate |
| `… formal_train status --config F` | 从磁盘读出进度，不写任何文件 |
| `python -m dreamerv3.thinking_student.exam …` | 单次考试：DEV 用 `--pool P --contract C`；TRAIN 冒烟用 `--corpus D [--steps N]` |
| `python -m dreamerv3.thinking_student.run <命令>` | 原有阶段 CLI，本次新增：<br>• `--corpus`、`--cf` 可以重复给出；<br>• `--mix`、`--save-every`；<br>• `state-check` 命令 |
| gym `python -m gym_trading_env.research.ts_v1_corpus --contract C --pool P --out D` | 生成一个 TRAIN 合约的教材（`--pool` 必须给出） |

## 配置（只在 `dreamerv3/configs/thinking_student/`）

| 文件 | 内容 |
| --- | --- |
| `student.yaml` | 学生设置（各头、burn-in、μ 的 ε、loss 权重），以及 CPU 测试用的 `small` 预设 |
| `formal_v1.yaml` | 18 合约正式训练：路径、合约、预算、轮次、CF、价值锚点、考试 |
| `c4_smoke_2c.yaml` | 同一流程，只用 FG005 + FG405，预算很小 |
| `markets/pool.json` | 18 个 TRAIN 行和 1 个 VALIDATION 行（FG409 = DEV）。TEST 封存，不在表中 |
| `markets/envs/`、`markets/contracts/`、`markets/strategy_quality_tp_v1.yaml` | 每个合约的 Native env、合约条款，以及带 Quality TP Reward 的 Single3 策略 master |

- **网络大小：** 正式配置用 `size50m`。
  - 市场脑从 `paths.init_f1` 继承，即 S3-V2 `E_ONLY_MARKET_MINUTE_FORMAL_1M` 的 step 1M：size50m、E 观测、在这 18 个 TRAIN 合约上训练过。
  - 继承范围：Encoder、RSSM、市场 Decoder。
  - Fusion 层的输入多了 MACD 的 16 行，这 16 行补零（`widened`）。
  - Thinking Student 自己的头全部新建。
- **配置合并：** `formal_train` 读取 YAML 后与默认值合并。
  - 未知的节或键直接报错。
  - 相对路径按配置文件所在目录解析。
  - 每次调用都把解析后的配置写入 `<run_root>/config.yaml`；如果和上一份不同，上一份改名为 `config.<时间>.yaml` 保留。
- **教材自包含：** 构建时 env、合约、策略三个文件都拷进 `<corpus>/inputs/`，此后只有行情 CSV 在教材之外。

## 运行目录

```
<run_root>/
  config.yaml  run_state.json  logs/<步骤>.log
  foundation/ckpt/<更新数>/{agent.pkl, progress.json, train_*.json}  inherit.json
  policies/pi_0000, pi_0001 …                 不可变快照（agent.pkl + policy.json）
  experience/pi_k/collect/<合约>/             π_k 的正式采集（meta.json、check.json、state_check.json）
  experience/pi_k/cf/<合约>/                  Q^{π_k} 教材（pairs.jsonl、targets.npz、meta.json）
  learners/pi_{k+1}/ckpt/…                    从 π_k 继承：先 L4，再 L5 → 冻结 π_{k+1}
  eval/pi_k/{DEV,TRAIN_SMOKE}/<合约>/exam.json
```

## 续训与不可变规则

完成与否每次都从磁盘重新判断。`run_state.json` 只是供人读的镜像，任何步骤都不依据它放行。

| 对象 | 重跑时的行为 |
| --- | --- |
| 训练阶段 L1–L5 | 剩余次数 = 配置预算 − `progress.json` 已记的次数，从最新 checkpoint 恢复参数和优化器。采样种子由（seed, 阶段, 已完成次数）决定，最多重做最后一段尚未保存的更新 |
| 采集 | `meta.json` 的 `done` 为真则跳过，否则从 `state.pkl` 接着采 |
| check | `check.json` 已存在则跳过。有违规或 facts 不一致的合约不进入本轮 CF 和 L4；一个可用合约都没有时，本轮报错停止 |
| CF | 已有同一 π_k 的 `meta.json` 则跳过。`meta.json` 属于另一个策略时报错。没有 `meta.json`（中途被打断）时，该合约整段重做 |
| Q 标签 | `MultiQWindows` 要求所有 `--cf` 来自同一个冻结策略（解析后的路径一致），否则报错。NOT_COMPARABLE 行没有 Q 监督 |
| 冻结策略 | 目标目录已有 `policy.json` 时拒绝写入；写入顺序是先 `agent.pkl` 再 `policy.json` |
| 考试 | `exam.json` 已存在则跳过 |

## Teacher-free 考试

- **DEV：** 取 pool 中的 VALIDATION 行（`dev_row`），与 TRAIN 教材走同一条 `write_inputs` 写出 env inputs。
  - Native 规则与 TRAIN 完全相同，不生成任何教师标签。
  - 学生的 RSSM 从合约第一分钟连续运行到真终止。
- **TRAIN_SMOKE：** 用 TRAIN 教材的 env，可截断步数。这是工程检查，不是样本外成绩。
- **DEV_NOT_CONFIGURED：** 配置里没有 DEV 时写出这个标记，不产生成绩，也不拿 TRAIN 内部留出段冒充 DEV。
- **随机入场基线：** 在同一段分钟、同一套 Native 规则下，每个合法决策以“学生 ADD 数 / 合法决策数”的概率入场。
  - 每个 seed 一个账户，不按结果挑选 seed。
- **指标（学生与基线同一口径，`exam.json` 内附定义）：**
  - Reward 合计、每市场分钟、每官方交易日；
  - ADD 数和合法决策数；
  - 按 TP / SL / 其他分的平仓数，以及期末未平仓数；
  - TP 持有分钟；
  - 占仓率、权益最大回撤、净盈亏、是否跑到真终止；
  - 学生减基线的差，以及基线各 seed 的分布。
- **`best_dev`：** 主指标事先定为 DEV 每市场分钟 Reward，只在全部 DEV 合约都完整考完时更新。`latest_candidate` 单独记录。
- **TEST：** 封存。考试代码没有读取 TEST 的路径，pool 中也没有 TEST 行。

## 实跑：size50m 冒烟（FG005 + FG405）

- **运行目录：** `/data/logdir/fg_single3/thinking_student/C4_SMOKE_2C_50M_20261023`，配置 `c4_smoke_2c.yaml`。
- **性质：** 工程贯通，不是训练成绩。
- **size12m 的 R1：** 目录 `C4_SMOKE_2C_20261023`，只作 F0/F1 机制的记录。

**pretrain**（23:00:53 → 23:12:21，退出码 0）
- 继承自 E_ONLY step 1M：reused 49、sliced 1、widened 1、initialized 60，清单在 `foundation/inherit.json`。
- 训练：L1 40 次、L2 60 次、L3 40 次（batch 8×64），每 20 次保存一次。
- 冻结 `pi_0000`：size50m，cuda/bf16，两个合约，lineage 为 `inherited_from` = E_ONLY。
- 考试：DEV FG409 整段；TRAIN 冒烟 FG005、FG405 各 6000 步。

**R2：一轮 π_0000 → π_0001，含中断续跑**
- **采集：** 两合约并行，约 4 分钟跑到真终止。

  | 合约 | 步数 | ADD | 合法决策 |
  | --- | --- | --- | --- |
  | FG005 | 20,504 | 68 | 437 |
  | FG405 | 26,429 | 307 | 1,246 |

- **check：** 违规 0，facts 一致 68/68、307/307。
- **CF：** 全部 ADD 加 8 个 WAIT，每合约 3 个 worker，重放与采集逐位一致。

  | 合约 | 对数 | 有效 | NOT_COMPARABLE/CAP | ΔG 均值 / std | 用时 |
  | --- | --- | --- | --- | --- | --- |
  | FG005 | 76 | 75 | 1 | −0.023 / 0.309 | 521 s |
  | FG405 | 315 | 315 | 0 | −0.052 / 0.651 | 450 s（重跑） |

- **中断测试：**
  - FG005 的 CF 一写出 `meta.json`，就对编排器进程组发 SIGKILL，退出码 137，没有残留进程。
  - 重跑时只调用了 FG405 的 cf。中断前的 103 个 experience 文件（采集、check、FG005 CF）字节和 mtime 都不变。
- **state-check**（FG005，16 个 64 行窗口，burn-in 16）：
  - 对照窗口、重放与采集记录都完全一致。
  - burn-in 之后：`p_tp_h` 平均差 0.00013（连续路径 std 0.048）；决策 `z` 平均差 0.49（std 2.20），窗口内 48–64 行仍有 0.44。
  - 即 64 行窗口没有复现连续路径的决策状态，L3/L4 窗口训练与连续推理之间存在差距。只做诊断，不设门槛。
  - π_0000 的 Q 为 0，ΔQ 比较要从下一轮开始才有意义。
- **训练与冻结：**
  - L4 40 次，两个 `--cf`；L5（S4 先验刷新）20 次。
  - 冻结 `pi_0001`：β_Q 1，κ_Q 0.510（390 对、两合约等权的合并 std；FG005 0.309，FG405 0.651；未触及下界）。
  - lineage 中 `q_teacher_policy` = [pi_0000]。
- **`run_state.json`：** `L5: S4_PRIOR_REFRESH`、`L6: L6_NOT_IMPLEMENTED`。

**R3：DEV FG409 Teacher-free 考试**（31034 个市场分钟，91 个交易日，均跑到真终止，随机基线 5 个 seed）

| 策略 | Reward | 每分钟 | ADD / 合法 | TP / SL / 其他 | 占仓 | 最大回撤 | 净盈亏 | 随机基线 Reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi_0000 | 16.90 | 0.000545 | 290 / 1257 | 234 / 55 / 1 | 0.959 | 36,321 元 | −34,611 元 | 13.12 ± 2.02 |
| pi_0001 | 18.62 | 0.000600 | 292 / 1460 | 236 / 55 / 1 | 0.953 | 36,009 元 | −34,047 元 | 16.86 ± 3.22 |

- **两点说明：**
  - Quality TP Reward 不等于账户盈亏，所以表中两项并列。
  - 两个策略都是几十次更新的工程产物，几乎一直持仓，不能说明交易能力。
- **`best_dev`：** 按每分钟 Reward 取 `pi_0001`。
- **TRAIN 冒烟**（`pi_0000`，前 6000 步）：
  - FG005：Reward −7.31，随机 −6.33 ± 0.66；
  - FG405：Reward −7.86，随机 −9.56 ± 2.51。

## 耗时与资源（size50m，RTX 4060 Ti 16 GB）

| 项目 | 实测 |
| --- | --- |
| L1 更新，18 合约（`C4_SPEED_50M_20261023/speed.json`） | 0.199 s/次（batch 8×64），0.204 s/次（16×64）；启动约 35 s |
| L2 更新，18 合约 | 0.095 s/次（8×64），0.096 s/次（16×64）；启动约 30 s |
| L4 40 次 / L5 20 次（两合约，含启动） | 25 s / 24 s |
| 正式采集（每个进程） | 约 100 步/s；内存 1.6 GB，显存 1.2 GB |
| CF（每合约 3 个 worker） | 每个 worker 都先重放整个合约，约 4.5–5 分钟，这是主要开销；分支本身很短。两合约 6 个 worker 并行：显存合计 7.9 GB，每个 worker 内存 1.7 GB |
| state-check（20504 行） | 117 s |
| DEV 考试（31034 分钟，含 5 个随机 seed） | 330–370 s |
| Checkpoint | 每份 388 MB；冒烟目录合计 5.3 GB |

- batch 16 每次更新的耗时与 batch 8 相同（GPU 没有吃满），正式配置因此用 16×64。

## 正式训练：先 L1 + L2（用户 2026-10-23 授权）

- **配置：** `formal_v1.yaml`，运行目录 `/data/logdir/fg_single3/thinking_student/FORMAL_V1_20261024`。
  - 18 个合约，batch 16×64，seed 20261024，从 E_ONLY 继承。
- **预算估算：**

  | 阶段 | 次数 | 依据 | 预计用时 |
  | --- | --- | --- | --- |
  | L1 | 10,000 | 每次 1024 行，约为 488k 行 TRAIN 的 21 遍；E_ONLY 本身约 5 遍（40k 次 × 64 行） | 约 35 分钟 |
  | L2 | 12,000 | 每次 768 个标签行，约为 ~450k 标签行的 20 遍。S1 在 FG809 单合约上留出最好的位置约在 25 遍，之后轻度过拟合；18 个合约多样得多，取 20 遍 | 约 20 分钟 |

  - 每 1000 次保存一次，约 22 份 × 388 MB ≈ 8.5 GB；`/data` 剩余 363 GB。
  - holdout 0，所以没有 TRAIN 内部的过拟合诊断；后续由 L3 之后的 DEV 考试衡量。
- **启动方式：** tmux 会话 `d3` 中新开窗口 `ts-formal`，运行：

  ```bash
  cd gym-trading-env
  env PYTHONPATH=.:src:../dreamerv3 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    /home/v/miniconda3/envs/dreamerv3/bin/python -u -m dreamerv3.thinking_student.formal_train \
    pretrain --config ../dreamerv3/configs/thinking_student/formal_v1.yaml --until L2
  ```

- **中断后：** 同一命令接着训练；进度看 `formal_train status --config …`。
- **L2 之后：**
  - 去掉 `--until`，再运行 `pretrain`，即训练 L3（当前填 12,000 次，跑之前需再确认）并冻结 `pi_0000`；
  - 之后依次是 DEV 考试和 `iterate`。

## 可恢复的 Checkpoint 与策略

| 位置 | 内容 |
| --- | --- |
| `C4_SMOKE_2C_50M_20261023/foundation/ckpt/000000000140` | size50m 冒烟的基础学生（L1 40 / L2 60 / L3 40） |
| `C4_SMOKE_2C_50M_20261023/learners/pi_0001/ckpt/000000000200` | 第 1 轮学习器（L4 40 / L5 20） |
| `C4_SMOKE_2C_50M_20261023/policies/pi_0000`、`pi_0001` | 冻结策略；`experience/pi_0000/` 中有两合约的采集和 CF |
| `C4_SMOKE_2C_20261023/` | size12m 的 R1（F0/F1 机制记录） |
| `C4_SPEED_50M_20261023/` | 测速用的 run（两份 L1+L2 checkpoint，1.6 GB，可删除） |

## 未完成 / 待办

- **S5 / L6：** `ready_at` 反馈、B2 课程、错题补课均未实现。接口位置在 `iterate` 中 L5 与 freeze 之间。
- **Checkpoint 清理：** 没有自动清理，每份 388 MB；目前靠 `save_every` 控制份数。
- **CF 重放开销：** 每个 worker 都重放整个合约，所以 worker 越多，重放的总开销越大。如果正式轮次中 CF 成为瓶颈，可以考虑让 worker 只重放到自己的 fork，或者共享一次重放。
- **短窗口与连续状态的差距：** state-check 显示 64 行窗口的决策 `z` 与连续路径仍有约 20% std 的差距。是否加长窗口或 burn-in，留到 L3/L4 正式预算前决定。
- **TRAIN 内部诊断：** 正式配置 holdout 0，L1/L2 没有留出曲线。如需要，可另开一个带 holdout 的诊断 run，并如实标注。
- **数据位置：** 行情 CSV 仍在主 checkout 的 `artifacts/fg_entry_v1/main_export_345_audited/`，属于数据而非配置；`/data/logdir/fg_single3/shared/` 已不再被 TS 使用。
- **TEST：** 封存，只用于用户明确授权的最终评价。
