# Thinking Student V1 — C4 正式训练实现（2026-10-23）

- **依据：** `FG_S3_Thinking_Student_V1_C4_Formal_Training_Implementation_Plan_20261023.md`（C4）。
- **计划与实施中的变更：** `PLAN_C4_20261023.md`。
- **逐项证据：** `TODO.md` 的 C4 节。

**C4 交付的能力：** 同一个 Thinking Student 可以：
- 在全部 TRAIN 合约上学习 L1→L2→L3，冻结为 π0；
- 逐轮做自主 Native 实习、同策略 CF、L4 和 S4 先验刷新，冻结出 π_{k+1}；
- 在独立的 DEV 合约上做 Teacher-free 考试；
- 任一步被打断后，都能从磁盘接着跑。

S5 的 L6 见下文“L6：离线教学循环”：
- 只实现 OFFLINE_TRAIN；TIME_ORDERED 明确不支持（`TIME_ORDERED_NOT_SUPPORTED`）。
- 2026-10-24 S5 之前的轮次在 `run_state.json` 中记 `L6_NOT_IMPLEMENTED`；现在没开 L6 的轮记 `L6_SKIPPED_BY_CONFIG`。

## 入口

所有命令从 `gym-trading-env/` 运行；环境变量写法见 `dreamerv3/configs/thinking_student/README.md`。

| 命令 | 作用 |
| --- | --- |
| `python -m dreamerv3.thinking_student.formal_train pretrain --config F` | 用所选 TRAIN 合约的窗口依次训练 L1→L2→L3（`foundation/`），冻结为 `pi_0000`，然后考 DEV，再在两个合约上做 TRAIN 冒烟 |
| `… formal_train iterate --config F [--rounds N]` | 有上限的改进轮，每轮从最新的冻结 π_k 出发，依次是：<br>• 本轮合约各自正式采集到真终止；<br>• check；<br>• CF；<br>• 从 `iteration.l6.from_round` 起：π_k 的教学题，以及带后果错补课的 L2 复习；<br>• RSSM 短窗诊断（`state_check_windows` > 0 时；正式配置为 0）；<br>• L4（只用 π_k 的 CF；L6 时决策错行加权；CF 没有代表性时跳过）；<br>• L5 先验刷新（L6 时带示范错补课）；<br>• 冻结 π_{k+1}（带 β_Q、合并 κ_Q；有效成对样本少于 2 个时 β_Q = 0，记 `Q_ANCHOR_NOT_READY`）；<br>• 考 DEV |
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
| CF | 已有同一 π_k 的 `meta.json` 则跳过。`meta.json` 属于另一个策略时报错。没有 `meta.json`（中途被打断）时，同一命令从 `work/` 中已完成的记录和快照接着算。每个合法分叉先探测至多 `cf_probe_steps` 步；`cf_budget` 是未闭合探测的续算配额（分层抽样），其余记 `UNAVAILABLE_BUDGET`。输出完成后删除已结束分叉的快照，只保留未续算探测的快照和参考重放 |
| Q 标签 | `MultiQWindows` 要求所有 `--cf` 来自同一个冻结策略（解析后的路径一致），否则报错。NOT_COMPARABLE 和 UNAVAILABLE_BUDGET 行没有 Q 监督 |
| L4 代表性 | 某合约的有效对数不足其分叉数的 `cf_min_coverage` 时，本轮跳过 L4（`L4_SKIPPED_CF_NOT_REPRESENTATIVE`，`L4_coverage` 记各合约覆盖率）。π_{k+1} 保留 π_k 的价值脑，freeze 用 `--value-anchor-of π_k` 保留 β_Q、κ_Q（`L4_SKIPPED_ANCHOR_KEPT`）。探索配置 `cf_coverage: exploratory` 下 L4 照常用已有的有效对训练，轮记录写 `S3_Q_TARGETS_EXPLORATORY` 和 `L4_coverage`；标签和覆盖率不变 |
| 价值锚点 | β_Q > 0 需要 `--cf` 中至少 2 个有效 ADD/WAIT 成对样本；不足时冻结为 β_Q = 0，`value_anchor` 记 `Q_ANCHOR_NOT_READY`，编排器把它记入该轮。`kappa_floor` 只替代真实成对样本的零方差或过小尺度，不顶替缺失的监督 |
| 冻结策略 | 目标目录已有 `policy.json` 时拒绝写入；写入顺序是先 `agent.pkl` 再 `policy.json` |
| 考试 | `exam.json` 最后原子写入，存在即表示考完，已存在则跳过。`iterate` 每轮开始前先补完最新冻结策略缺的考试，再看轮数上限或开新轮；对 `pi_k`（k ≥ 1）这就是完成第 k 轮，计入本次调用的轮数 |

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
- **结果：** 2026-10-24 00:46:29 结束，exit 0，中途没有中断。

  | 阶段 | 起止 | 每次更新 | 训练批次损失（每 1000 次均值，第一段 → 最后一段） |
  | --- | --- | --- | --- |
  | L1 | 23:53:09–00:27:08 | 0.204 s | 总损失 31.0 → 20.0；市场重建 20.9 → 9.1，最后一段仍在下降；dyn 8.9 → 9.9；`account_can_add` 4000 次后为 0 |
  | L2 | 00:27:08–00:46:29 | 0.097 s | `outcome_event` 2.78 → 2.37；常数答案（训练集类别频率）为 2.775 |

  - **曝光：** 各合约大致均匀。L1 每个合约 8,701–9,047 个窗口，L2 每个合约 10,531–10,868 个。
  - **继承正常：** E_ONLY 最后 2000 次更新的市场重建损失均值为 21.07，L1 第一个 1000 次为 20.9。
  - **L2 区分 TP 和 SL 的能力很弱：**
    - 标签是 5 跳止盈、30 跳止损，484,249 个标签里 TP 占 0.807。
    - 在真实 TP 行与真实 SL 行上，预测的 TP 概率在后 10,000 次平均只差 0.022 ± 0.003（100 个日志点）。
    - 按这个差距估算，方向部分只占 `outcome_event` 降幅（约 0.4 nats）中的约 0.02 nats，其余来自平仓落在哪个时间档。
  - **对 L3 的影响：** Outcome evidence 以停止梯度方式输入 Value 和 Decision。Decision 仍可直接从融合状态学，L3 不会被卡住，但 Reasoning 路径从 evidence 得到的方向信息很少。
  - **以上都是训练批次上的数字。** holdout 0 且约 20 遍，学到和记住无法区分。L1 重建降到 E_ONLY 水平的 43%，同样需要留出数据确认。
- **L2 之后：**
  - 去掉 `--until`，再运行 `pretrain`，即训练 L3（当前填 12,000 次，跑之前需再确认）并冻结 `pi_0000`；
  - 之后依次是 DEV 考试和 `iterate`。

## 短窗口与连续状态（C4 review R3，2026-10-24）

- **做法：** 用现有 `run state-check`，在 FG005（20,504 行）上随机取 16 个 1024 行窗口。每个窗口从零状态起步，与同一策略从合约第一分钟连续运行的结果逐行比较。
  - 从第 0 行开始的对照窗口两者完全一致（差 0），说明比较本身是准确的。
  - 运行目录 `C4_STATE_CHECK_20261024/`（508 MB），脚本 `state_check.sh`，源码 dreamerv3 `4c67480`；只读正式 run，不训练任何东西。
- **测的两个策略：**
  - **正式脑子：** 从 `FORMAL_V1_20261024/foundation` 的 ckpt 22000（L1 10,000 / L2 12,000）冻结出诊断快照 `diag_l2_22000`。它不是 `pi_0000`，决策头还没经过 L3。先在 FG005 上完整采集一遍，再做检查。
  - **对照：** 冒烟的 `pi_0000`。它的市场脑基本还是 E_ONLY 原样，L1 只训练了 40 次。
- **结果：** 下表是窗口状态与连续状态的平均绝对差，除以连续路径上该量的标准差。

  | 窗口内位置 | 正式：`z` | 正式：`p_tp_h` | 冒烟：`z` |
  | --- | --- | --- | --- |
  | 16–32 | 0.221 | 0.0044 | 0.219 |
  | 48–64 | 0.161 | 0.0074 | 0.184 |
  | 112–128 | 0.099 | 0.0025 | 0.199 |
  | 240–256 | 0.080 | 0.0016 | 0.176 |
  | 496–512 | 0.072 | 0.0013 | 0.207 |
  | 1008–1024 | 0.059 | 0.0000 | 0.197 |

  标准差：正式 `z` 2.69、`p_tp_h` 0.229；冒烟 `z` 2.21。
- **结论：**
  - **训练过的 Outcome 头读到的状态收敛很快：** 16 行之后，`p_tp_h` 的差距不到其标准差的 1%。
  - **状态里还有一部分收敛很慢，未训练的决策头能读到：** 正式脑子在 L3 计损失的位置（16–64 行）约为 `z` 标准差的 16–22%，128 行时降到 10%，1024 行时仍有 6%。
  - **正式 L1 已经明显改善了这一点：** E_ONLY 原样的脑子，这部分差距在 1024 行内完全不衰减，各位置都约 20%。
  - **加长窗口只能部分缓解：** 128 行能让 L3 计损失位置的差距减半，再长收益很小，任何长度都消除不了。
- **建议（用户决定）：**
  1. L3 先按现有的 64 行窗口、16 行预热训练（约 20 分钟），冻结 `pi_0000` 后立即用 `state-check` 测训练后决策头的 `z` 差距（约 5 分钟）。训练过的 Outcome 头对这部分残差不敏感，训练后的决策头可能也一样。这样做不需要改代码。
  2. 如果训练后差距仍明显（例如计损失位置超过标准差的 10%），有两个办法：
     - 从 ckpt 22000 开一个新 run，用 128 行窗口、64 行预热重训 L3（约 40 分钟）。L3 只训练 `decision` 参数组，市场脑不动，L1/L2 不受影响。这需要给 L3 单独配置窗口长度和预热。
     - 让 L3/L4/L5 的窗口从连续路径上的真实状态起步。L3 期间市场脑冻结，所以这样做是精确的，但需要开发。

## 训练窗口继承连续市场 carry（用户决定 2026-10-24 第 4 条）

- **依据：** 按 64 行窗口、16 行预热训练出的 `pi_0000`（`FORMAL_V1_20261024`），在 FG005 上用零状态开 1000 个 64 行窗口：
  - 合法行翻转率 1.07%；
  - 窗口内 ADD 240 次，连续路径 99 次；WAIT→ADD 148 次，ADD→WAIT 7 次。
  - 偏差是单向的，ADD/WAIT 决策受到显著影响，详见 TODO D-3。
  - 改为从连续路径的 carry 开窗、换一组策略种子之后：窗口 ADD 140 次，翻转率 0.72%，剩下的是随机隐变量的采样噪声。
- **做法：** 不加长窗口，也不重训 L1/L2。
  - **MarketCarry 存储：** `run.py market-carry` 让学生从合约第一分钟连续读完整条教材，记录每个游标之前的市场 RSSM carry，写到 `<root>/<合约>/`：
    - `deter.npy`、`stoch.npy` 按 bf16 原始位存储，读取时内存映射；
    - `meta.json` 最后写入，有它才算完整。
  - **为什么一份存储够用：** 市场脑只读市场观测和 `is_first`，不读账户和动作；L2–L5 不训练它。所以 L1 之后的一份存储就是此后每个 `pi_k` 部署时的市场记忆。策略种子计数器同为 0，在同一设备和精度下逐位相同。
  - **训练窗口：** 使用 `--window-start carry --carry-root ROOT`：
    - L2/L3 的 `TSCorpus` 和 L4 的 `MultiQWindows` 给每个窗口附上首游标的 carry（`carry/deter`、`carry/stoch`），`train_stage` 和 `evaluate` 从这份 carry 起步；
    - `is_first` 只在合约第一分钟为真；不需要预热，burn-in 为 0，64 行都计损失；
    - L1 训练市场脑，所以拒绝 carry 开窗。
  - **state-check：** 带上 `--window-start carry --carry-root ROOT --corpus C` 后，窗口从存储的 carry 起步，与训练一致，并报告 `store_vs_path_max_abs_diff`，即存储与部署路径 carry 的最大差。报告新增 `mean_diff`：在全部行上计算的有符号偏差。
  - **编排：** `student.window_start: carry` 时，
    - `pretrain` 在 L2 或 L3 开训之前补齐所有合约的存储（`carry_parallel` 个进程并行）；
    - 已有存储时，再给 L1 加预算会被拒绝；
    - L2–L5 和每轮的 state-check 都传入 carry 参数；`iterate` 只补算缺少的存储。
  - 测试（`test_ts_v1_c4.py`）覆盖：
    - 存储等于正式采集路径上的 carry：同种子时窗口完全复现路径，`--run` 与 `--policy` 两种来源写出相同的存储，且不写入 run；
    - 窗口附带正确的起点 carry；
    - carry 开窗时 L2 只改 outcome 组、L3 只改 `decision`、L4 只改 `value`；
    - 编排在 L3 之前生成存储，state-check 读取存储且与路径差 0，L4 收到 carry 参数。
- **配置与运行：** `formal_v1_carry.yaml`，run 根目录 `FORMAL_V1_CARRY_20261024`。
  - `init_from` 为 `FORMAL_V1_20261024` 的 ckpt 22000：L1/L2 原样继承，预算为 0；L3 12,000 次。
  - 脚本 `formal_carry_r1.sh`：存储 → L3 → `pi_0000` → 考试 → 第 1 轮到 CF → 诊断。诊断包括 `pi_0000` 在 FG005 上的 carry state-check，以及试跑 L4 加 Q 小样检查。
  - 原 `FORMAL_V1_20261024` 的零状态 `pi_0000` 保留不动，它的第 1 轮没有运行。
- **结果（2026-10-24，详见 TODO D-4–D-6）：**
  - **存储：** 18 个合约用 6 个进程并行，19 分钟算完，共 4.7 GB。
  - **L3：** 12,000 次，22 分钟。训练批次指标与零状态 L3 相近：示范 NLL 0.64 → 0.52–0.56。
  - **新 `pi_0000` 的考试：**
    - DEV FG409：13.41，随机基线 15.49 ± 4.4；零状态 `pi_0000` 为 6.42，基线 16.92 ± 11.9。
    - TRAIN_SMOKE：FG001 为 −0.32（基线 −1.93），FG005 为 −5.60（基线 −5.56）。
  - **第 1 轮到 CF：** FG001、FG005、FG009 共 614 个有效对，全部 `VALID_COMPLETE`。
  - **状态一致性（FG005，1000 个 64 行窗口，从存储起步，换一组种子）：**
    - 存储与部署路径的 carry 逐位相同；
    - z、Q_ADD、Q_WAIT、ΔQ 的有符号偏差都在 ±0.003 std 以内；
    - 剩下的差异是随机隐变量的采样噪声：z 为 0.26 std，ΔQ 为 0.085 std；
    - 合法行上翻转：`pi_0000` 2.0%，诊断 L4 加锚点后 4.0%，两个方向基本对称；ΔQ 符号一致率 87%。
    - 同种子时差异为 0。
  - **L4 预算：** 诊断 L4 在留出集上，q_pair、q_cf 在 500 次时最好，之后过拟合，Q 也没有超过常数。正式 L4 定为 500 次。
  - **第 1 轮完成：** carry 开窗，L4 500 次加 L5 100 次，冻结 `pi_0001`（β_Q 1，κ_Q 0.559，614 对）。
    - DEV FG409：17.69，随机基线 15.24 ± 5.98；`best_dev` 为 `pi_0001`。差在基线波动之内，且只有一个 DEV 合约。
    - `pi_0001` 的 state-check：存储与路径的差为 0；Q_ADD、Q_WAIT、ΔQ、z 的有符号偏差都在 ±0.003 std 以内；ΔQ 符号一致率 94.7%。

## 开发 run 与 10x 正式 run（用户 2026-10-24）

- **分工：** 开发 run `FORMAL_V1_CARRY_20261024`（`formal_v1_carry.yaml`）用小预算快速跑完，验证代码；正式 run `FORMAL_V1_L1_100K_L2_120K_CONTINUE_20261024`（`formal_v1_l1_l2_10x_continue.yaml`）的每个阶段都训练 10 倍步数，两者留作对比。
- **配置差异：** 正式配置只在步数上与开发配置不同，例外有三个：
  - WAIT 分叉数：开发配置设 100 是为了跑得快；如果 L4 乘 10 而 CF 不变，只会在同样约 600 对上过拟合得更深。
  - `cf_coverage: exploratory`（用户 2026-10-28，与主线探索配置一致）：Astra CF 在主线第 3–4 轮各合约的覆盖率为 0.51–0.61。不加这个开关，每轮都会保留价值脑、学不到 Q。
  - 续算预算 `cf_budget` 480（用户 2026-10-28，与 WAIT 分叉数同为开发配置的 10 倍）：预算为 48 时，10x 第 1 轮 FG005 的 1114 个分叉只有 248 个得到标签（覆盖率 0.22），10 倍的 WAIT 分叉几乎没有增加有效对。

  | | 开发 | 10x |
  | --- | --- | --- |
  | L1 / L2（累计） | 10,000 / 12,000（继承 ckpt 22000） | 100,000 / 120,000 |
  | L3 | 12,000 | 120,000 |
  | L4 / L5（每轮） | 500 / 100 | 5,000 / 1,000 |
  | 保存间隔（基础 / 每轮） | 1,000 / 250 | 10,000 / 2,500 |
  | 每合约 WAIT 分叉 | 100 | 1,000 |
  | 轮数 | 9：第 1–6 轮每轮 3 个合约，18 个合约轮转一遍；第 7–9 轮带 L6，合约与第 1–3 轮相同 | 9 |
  | L6 的 L2 复习（每轮） | 200 | 2,000 |

- **worktree：** 10x run 用自己的 `/data/research/fg-s3-thinking-student-l1-l2-10x-continue-20261024/{dreamerv3,gym-trading-env}`，两仓分支都是 `run/thinking-student-l1-l2-10x-continue-20261024`；开发继续在 `feat/thinking-student-v1-20261022` 上进行。
  - 同步方式：开发分支有新提交时，把 run 分支快进过去。
  - 编排器的每一步是一个子进程，启动时才 import 代码，所以同步在步与步之间生效；不在一步运行中途改它正在用的代码。
- **10x run 的 L1：** 用户把 ckpt 22000（L1 10,000 / L2 12,000）拷进新 run，手动运行 `run.py train-market --updates 90000 --save-every 10000`，参数与编排器的调用相同。L1 跑完后由 `formal_train pretrain` 继续：
  1. 用 L1 的新脑建 18 个合约的存储；
  2. 训练 L2 剩下的 108,000 次；
  3. 训练 L3 120,000 次；
  4. 冻结 `pi_0000` 并考试。
- **有限预算的精确 CF（2026-10-25 Astra 修复；2026-10-27 独立 Review 通过并集成为唯一正式 CF，Claude 方案只归档，见 `CF_ASTRA_INTEGRATION_20261027.md`）：**
  - FG201 最终权益为 `300000 - 263022 = 36978`。旧资金充分条件拒绝的是每个游标的**未来最低权益**，不表示参考每一行权益都低于 90336，也不证明分支无法精确耦合。
  - 删除 `Reference.decidable()` 及其预筛选。每个合法候选先执行至多 `cf_probe_steps: 64` 步；真实 TERMINAL、严格状态等价或原高资金安全耦合均产生 `VALID_COMPLETE`。
  - `cf_budget` 现在是**未闭合探测的续算配额**。沿用参考动作 × 权益四分位 × 60 分钟波动率分层抽样；未入选者为 `UNAVAILABLE_BUDGET/PROBE_UNFINISHED_BUDGET`，记录实际探测步数，但没有 Q。
  - 探测和续算共用 Native / MarketCarry / 策略种子 / 回报进度，合计不超过 `cf_max_steps`（8192）。到上限无合法闭合为 `NOT_COMPARABLE/CAP`。
  - N 个候选、B 个续算配额、上限 H 的分支步数上界为 `N*min(64,H) + B*max(H-64,0)`；参考重放、模型推理、保存恢复等真实成本单独记录。`budget=0` 仍执行探测。
  - 参考按合约重放一次；完整分支状态落盘，每 worker 复用一个 Native 环境。中断后可原命令恢复，也可增加同一输出的续算配额；已分配样本保持不变。
  - 局部补算使用 `cf --cursors ... --base-cf OLD --out NEW`，保留旧合法完整标签。新 CF 写入后重新生成当前相关 S5 反馈，不能以已有 `summary.json` 跳过。
  - 既有配置的 `cf_budget: 48`、`cf_min_coverage: 0.8` 数值未改；预算语义按上述更新。任务与验收清单见 [CF_FIX_TASK_20261025.md](CF_FIX_TASK_20261025.md)。正式资产未在本次修复中重跑。
  - 集成时增加（2026-10-27）：
    - 输出完成后删除已结束分叉的快照，未续算的探测保留快照，`meta.json` 记 `snapshots`（删除数、字节数、保留数）。FG201 一份 CF 的快照约 4.5 GB，其中已结束的约 2.3 GB。
    - 探索开关 `iteration.cf_coverage: exploratory`（见上表“L4 代表性”）。
- **L2 carry 开窗的验证：** 开发 run 的 L2 是继承来的，没有走过这条路径。
  - 新测试覆盖：
    - L2 只改 outcome 组；
    - 拷贝的 ckpt 在 L2 之后又手动续了 L1，编排器先用续训后的脑建存储，L2、L3 都读它；
    - 冻结后 `pi_0000` 的部署记忆与存储逐位相同，与续 L1 之前的脑则不同。
  - 真实数据冒烟（`C4_L2_CARRY_SMOKE_20261024`）：18 个合约、60 次，从 ckpt 22000 起步，使用开发 run 的存储。结果正常；每个 batch 的 1024 行都计损失，而零状态开窗只计预热之后的行。

## L6：离线教学循环（S5，2026-10-24）

计划：`PLAN_S5_20261024.md`；逐项证据：`TODO.md` 的 S5 节。

- **开关：** 配置块 `iteration.l6`，默认关闭。
  - 字段：`from_round`、`order`、`outcome_review_updates`、`remedial_fraction`、`remedial_boost`、`outcome_error_prob`、`decision_error_margin`。
  - `order` 只接受 `OFFLINE_TRAIN`；写 `TIME_ORDERED` 时 `load_config` 报 `TIME_ORDERED_NOT_SUPPORTED` 并拒绝。
- **开启 L6 的一轮：**
  1. collect → check → CF；
  2. `feedback`：每个合约在 `experience/π_k/feedback/<合约>/` 写 `items.npz` 和 `summary.json`。每次都按当前 CF 和同轮合并 κ_Q 重写（秒级），已有的 `summary.json` 不作为跳过的依据，所以补算的 CF 会进入教学；
  3. L2 复习（`train-outcome --feedback`，后果错补课），只动 outcome，`l2_fusion` 为真时含 fusion；
  4. L4（决策错行 ×`remedial_boost`）；
  5. L5（示范错补课）；
  6. freeze：lineage 记 feedback 目录、`teaching_order: OFFLINE_TRAIN`、`time_ordered: TIME_ORDERED_NOT_SUPPORTED`；
  7. 考 DEV。
- **CF 没有代表性的轮次：**
  - 跳过 L4；
  - L5 不接这份 CF 产生的补课（`decision_remediation: SKIPPED_CF_NOT_REPRESENTATIVE`）；
  - L2 复习只用事实，照常进行。
- **不训练 L1：** L6 不训 L1，所以 MarketCarry 存储仍是部署时的市场记忆。
- **记录：** `run_state.json` 每轮记 `L6: S5_FEEDBACK_CURRICULUM` 或 `L6_SKIPPED_BY_CONFIG`，以及各合约的反馈摘要、错题数、L2 复习次数和补课曝光（补课窗口、错题行、唯一行与重复曝光）。
- **结论的边界：** 这是离线 TRAIN 教学：教师可以使用 TRAIN 的完整后缀。ready_at 逐条保存，但不参与放行，所以结果不是严格时间顺序的在线学习。
- **开发 run 实跑（2026-10-25，`S5_RECORD_20261024.md`）：**
  - 第 7–9 轮跑通；第 9 轮 FG201 的 CF 覆盖率 0.169，按规则跳过 L4 和决策错补课。
  - DEV 没有看出改进，`best_dev` 仍是 `pi_0001`。
  - 诊断发现固定比例补课有副作用：后果错补课让后果脑在自然分布上校准变差；错题很少时，同几道题会被反复补课。
- **E-7 方案 A（用户 2026-10-28）：** 三份正式配置都设 `l6.remedial_fraction: 0`，L2 复习和 L5 只用自然窗口。教学题、L2 自然复习和 L4 决策错加权（`remedial_boost: 4.0`）保留。补课代码不变。

## 可恢复的 Checkpoint 与策略

| 位置 | 内容 |
| --- | --- |
| `FORMAL_V1_20261024/foundation/ckpt/000000022000` | 正式基础学生：L1 10,000 / L2 12,000，L3 未训练；每 1000 次一份。后续的 ckpt 23000–34000 和 `policies/pi_0000` 是零状态开窗的 L3，保留作对照 |
| `FORMAL_V1_CARRY_20261024/market_carry/` | 18 个合约的 MarketCarry 存储（4.7 GB），市场脑来自 ckpt 22000，L2–L5 共用 |
| `FORMAL_V1_CARRY_20261024/foundation/ckpt/000000034000` | carry 开窗 L3 之后的基础学生（ckpt 23000–34000，每 1000 次一份） |
| `FORMAL_V1_CARRY_20261024/policies/pi_0000` | 当前正式 `pi_0000`；`experience/pi_0000/` 中有 FG001、FG005、FG009 的采集和 CF |
| `FORMAL_V1_CARRY_20261024/learners/pi_0001/ckpt/000000034600`、`policies/pi_0001` | 第 1 轮学习器（L4 500 / L5 100）与冻结的 `pi_0001`（`best_dev`） |
| `FORMAL_V1_CARRY_20261024/learners/pi_0002`–`pi_0009`、`policies/pi_0002`–`pi_0009` | 第 2–9 轮的学习器与冻结策略。第 7–9 轮带 L6，教学题在 `experience/pi_0006`–`pi_0008/feedback/`；统计脚本 `s5_rounds_report.py`、`s5_param_groups.py` |
| `S5_FEEDBACK_DIAG_20261024/`、`S5_L2_REVIEW_DIAG_20261025/` | S5 诊断：第 1–3 轮的对照教学题；L2 复习和 L5 的自然窗口与补课对照（4 份 checkpoint，1.6 GB，可删除） |
| `C4_STATE_CHECK_20261024/carry_pi_0000/`、`carry_pilot_l4/`、`carry_pi_0001/` | 新 `pi_0000` 和 `pi_0001` 的 state-check；诊断 L4（run、`pilot_l4` 快照、Q state-check），只用于定 L4 预算 |
| `FORMAL_V1_L1_100K_L2_120K_CONTINUE_20261024/foundation/ckpt/`、`policies/pi_0000` | 10x 正式 run：ckpt 22000 拷自 `FORMAL_V1_20261024`；L1 100k、L2 120k、L3 120k 已完成，`pi_0000` 于 2026-10-25 10:36 冻结，iterate 还没开始 |
| `C4_L2_CARRY_SMOKE_20261024/` | L2 carry 开窗的真实数据冒烟（60 次，388 MB，可删除） |
| `C4_SMOKE_2C_50M_20261023/foundation/ckpt/000000000140` | size50m 冒烟的基础学生（L1 40 / L2 60 / L3 40） |
| `C4_SMOKE_2C_50M_20261023/learners/pi_0001/ckpt/000000000200` | 第 1 轮学习器（L4 40 / L5 20） |
| `C4_SMOKE_2C_50M_20261023/policies/pi_0000`、`pi_0001` | 冻结策略；`experience/pi_0000/` 中有两合约的采集和 CF |
| `C4_SMOKE_2C_20261023/` | size12m 的 R1（F0/F1 机制记录） |
| `C4_SPEED_50M_20261023/` | 测速用的 run（两份 L1+L2 checkpoint，1.6 GB，可删除） |
| `C4_STATE_CHECK_20261024/` | R3 诊断：快照 `diag_l2_22000`（ckpt 22000，决策头未训练，只用于诊断）、它在 FG005 上的采集、两份 state-check 报告（508 MB，可删除） |

## 未完成 / 待办

- **TIME_ORDERED 实习：** 未实现（`TIME_ORDERED_NOT_SUPPORTED`）。以后按方案 2 逐步做：
  - 单合约分段推进，段末只用已到期的反馈更新，并冻结新版本；
  - 段内 Q 只用在 `coupled_at` 到期的 ΔG 做成对监督；
  - 不采用尾值估计。
- **Checkpoint 清理：** 没有自动清理，每份 388 MB；目前靠 `save_every` 控制份数。
- **CF 重放开销：** 每个 worker 都重放整个合约，所以 worker 越多，重放的总开销越大。如果正式轮次中 CF 成为瓶颈，可以考虑让 worker 只重放到自己的 fork，或者共享一次重放。
- **CF 覆盖的选择性（后续研究，用户 2026-10-24）：**
  - 亏到耦合下限以下的合约，CF 只覆盖一部分有选择性的状态。旧 CF 下 FG201 约 16%，都是上限内能到终点的后段，加上预算样本。Astra 修复后 `pi_0002` 的 FG201 为 914 / 1596（57.27%），其余 639 个是未续算的探测、43 个到上限；用户决定按探索项目继续，不再为剩余缺口扩大专项开发（2026-10-27）。
  - 这样 Q 可能主要从容易完成的轨迹学习，错过高风险状态。
  - 如果多个合约持续出现这种情况，再优化 Native 快照的完整历史复制和 CF 的重复推进，必要时研究估计目标。
- **短窗口与连续状态：** 已改为从连续 carry 起步（见“训练窗口继承连续市场 carry”）。剩下的差异来自随机隐变量的采样：换一组种子后，近边界合法行有约 2–4% 的翻转，两个方向对称，与开窗方式无关。如果希望部署决策不依赖种子，需要另行设计，例如用隐变量的众数。本次没有改动。
- **Q 的质量：** 第 1 轮诊断 L4 在留出集上没有超过常数（与 FG809 的 S4 结论相同）。β_Q = 1 的锚点按配置启用，效果由 DEV 考试衡量。
- **TRAIN 内部诊断：** 正式配置 holdout 0，L1/L2 没有留出曲线。正式 L1 的重建降到 E_ONLY 的 43%，L2 的 TP/SL 区分只有 0.022，这两点都只在训练批次上看到。要分清学到和记住，可另开一个带 holdout 的诊断 run，并如实标注。
- **数据位置：** 行情 CSV 仍在主 checkout 的 `artifacts/fg_entry_v1/main_export_345_audited/`，属于数据而非配置；`/data/logdir/fg_single3/shared/` 已不再被 TS 使用。
- **TEST：** 封存，只用于用户明确授权的最终评价。
