下面这份是把你现在这套代码（DreamerV3 风格：Encoder+RSSM+Decoder+ActorCritic + Replay + Driver + 并行 runner）整理成一份「从上到下」的大纲/路线图。你可以按这个顺序读代码、画数据流图、逐层替换/改造（比如 trading env / MC / parallel eval policy）。

---

## 0. 先记住 4 条主线（读 DreamerV3 最省脑子）

1. **世界模型（World Model）**
   Encoder → RSSM（posterior/prior）→ Decoder + Reward Head + Continue Head
   目标：学会在 latent 里预测 obs/reward/continue。

2. **想象训练（Imagination / Actor-Critic）**
   用 RSSM 的 prior rollout（imagine）在 latent 里生成轨迹
   用 reward/continue/value/policy 做 actor-critic 的 loss。

3. **Replay & Stream**
   环境产生 transition → replay（chunk化存储）
   训练端从 replay sample batch → 送进 agent.train()

4. **Runner（单机/并行/评估/实盘/MC）**
   Driver 或 portal 并行把 env rollout、replay、learner、logger 串起来。

---

## 1. 目录级路线图（你这份工程的“部件地图”）

### A. Agent 核心（学习算法）

* `agent.py`

  * `Agent.policy()`：给环境动作（推理/rollout）
  * `Agent.train()`：一次训练 step（world model + imagination + 可选 replay value）
  * `Agent.report()`：生成可视化/诊断（open-loop video 等）
  * `imag_loss() / repl_loss()`：actor-critic & replay value 的损失

### B. 世界模型（模型结构）

* `rssm.py`

  * `Encoder`：obs → tokens
  * `RSSM`：

    * `observe()`：posterior（结合 obs tokens + prev action）
    * `imagine()`：prior（只用 latent + action rollout）
    * `loss()`：KL dyn/rep
  * `Decoder`：latent feat → obs reconstruction distribution

### C. 数据系统（经验回放）

* `replay.py`

  * chunk 组织、插入、采样、更新（priority 可选）
* `chunk.py`：chunk 存储结构 & save/load
* `selectors.py`：采样策略（uniform/recency/prioritized/mixture）
* `streams.py`

  * `Consec`：把一个长 batch 切成 consec chunks（prefix 支持 replay_context）
  * `Prefetch`：预取加速

### D. 环境执行（与 env 交互）

* `driver.py`：同步/多进程 env driver（非 portal 并行）
* `wrappers.py`：action/obs 的 normalize、clip、timelimit 等

### E. 运行器（把系统跑起来）

* `main.py`：解析 config，选择 script：train/train_eval/eval_only/live_trading/monte_carlo/parallel*
* `train.py`：单机训练 loop（Driver + Replay + Stream）
* `train_eval.py`：单机 train + eval driver
* `eval_only.py`：只评估
* `monte_carlo.py`：MC runner（不训练，写 episodes jsonl + summary）
* `parallel.py`：portal 多进程/多节点结构（actor/learner/replay/logger/env 分工）

---

## 2. 最关键的数据结构（你理解对了，后面都顺）

### 2.1 时间维度约定

* 环境 rollout 的 obs/act：通常是 **[B]**（step 维由 driver loop 推进）
* 训练 batch：**[B, T]**
* imagination 轨迹：**[B*K, H+1]**（K = imag_last，H = imag_length）

### 2.2 Dreamer 的 latent 结构（你这里是离散 one-hot）

* `deter`: [B, deter] （GRU-like）
* `stoch`: [B, stoch, classes] （离散分类）
* `feat2tensor`: concat(deter, flatten(stoch)) → policy/value 输入向量

---

## 3. 世界模型路线图（Encoder/RSSM/Decoder 怎么串）

### 3.1 Observe（看见真实观测）

路径：`obs -> enc(tokens) -> dyn.observe(posterior) -> feat -> dec(recon)`

在 `Agent.loss()`：

1. `enc(obs)`：把 vec/img 编码成 `tokens`
2. `dyn.loss()` 内部跑 `observe()` 得到：

   * `prior = _prior(deter)`
   * `post = obslogit`
   * `dyn KL` 与 `rep KL`（带 free_nats）
3. `dec(feat)`：重建观测（图像 MSE、向量 symlog_mse/categorical）
4. `rewhead(feat)`：预测 reward
5. `conhead(feat)`：预测 continue（是否终止/折扣）

> 这部分的 loss：`dyn, rep, rec(各obs key), rew, con`

### 3.2 Imagine（在 latent 里“做梦”）

路径：`starts(latent) -> dyn.imagine(prior rollout) -> imgfeat -> policy/value losses`

在 `Agent.loss()`：

1. 从真实 posterior 的最后 K 步拿 starts：

   * `starts = dyn.starts(dyn_entries, dyn_carry, K)`
2. 用 `policyfn` 在 latent 上生成动作
3. `dyn.imagine(starts, policyfn, H)` rollout 出 `imgfeat` 与 `imgact`
4. 用 `rew/con/value/pol` 在 imagination 上算 `imag_loss`

---

## 4. Actor-Critic 路线图（imag_loss 在做什么）

`imag_loss(act, rew, con, policy, value, slowvalue, ...)` 的核心：

1. **计算 λ-return**

* discount：`disc = 1 or 1 - 1/horizon`（取决于 contdisc）
* `weight = cumprod(disc*con)`
* `ret = lambda_return(...)`

2. **Advantage**

* `adv = (ret - tarval[:, :-1]) / rscale`
* 再做 `advnorm`

3. **policy loss**（最大化 advantage + entropy）

* `logpi = Σ logp(act)`
* `policy_loss = weight * -(logpi * adv_normed + actent * entropy)`

4. **value loss**（拟合 ret，并对 slowvalue 做 regularize）

* `value.loss(tar_padded) + slowreg * value.loss(slowvalue.pred())`

> slowvalue 是 target network（你用 SlowModel）

---

## 5. Replay/Stream 路线图（数据怎么变成 [B,T]）

### 5.1 env → replay.add(step)

* `Replay.add()`：

  * 把 step 写进当前 chunk（chunk.append）
  * 维护每个 worker 的 stream deque
    确认 `stream` 满足 length 后，把 (chunkid,index) 插入 sampler

### 5.2 replay.sample(batch) → batch dict

* `_sample()` 取 itemid → (chunkid,index)
* `_getseq()` 从 chunk 链取出 length 长度序列
* `_assemble_batch()` 拼成 `[B, T]`

### 5.3 Stream.Consec（把一个长 batch 切成 consec）

你这里训练常用：

* `length = batch_length`
* `consec = consec_train`
* `prefix = replay_context`
  所以 replay 实际存的是 `consec*length + prefix` 的长段，
  stream 每次吐出其中一段（含 prefix）。

---

## 6. Runner 路线图（哪条脚本做什么）

### 6.1 train.py（单机训练）

* driver rollout → replay.add
* stream_train = make_stream(replay)
* agent.train(carry, batch) 循环
* logger 写 episode/metrics
* report：定期 `agent.report()` 做 open-loop video

### 6.2 train_eval.py（单机训练+评估）

* 同时有 train driver 和 eval driver
* eval driver 不进 train replay（进 eval replay）
* reportfn 从 stream_report / stream_eval 取 batch 做 report

### 6.3 parallel.py（portal 并行）

分工很清晰：

* env 进程：只负责 step，并向 actor 请求 action
* actor：只负责 policy 推理 + 把 trans 发给 replay & logger
* replay：集中存储 & sample
* learner：集中训练 + 更新 replay priority/ctx + 写 logger
* logger：集中 episode 聚合、写盘

> **你目前的坑**：actor 没按 is_eval 传 mode，导致 eval env 行为不是真 eval。

### 6.4 monte_carlo.py（离线 MC 评估）

* 强制从 checkpoint 加载
* driver(policy, steps=chunk_steps) 收集 episode
* 每个 worker 写 episodes_w*.jsonl
* 最后汇总 summary（含 VaR/CVaR）

### 6.5 live_trading.py（实盘/仿真盘）

* 强制单 env
* 每次 steps=1（等新K线）
* 到时间 cutoff 退出

---

## 7. 推荐你的“深入理解阅读顺序”（最省时间）

1. **Agent.loss()**：把 world model + imagination 一次看懂
2. **RSSM.observe / imagine**：弄清 posterior/prior、KL、starts
3. **Replay.add / sample + Stream.Consec**：弄清 batch 是怎么来的
4. **train.py**：单机闭环（最直观）
5. **parallel.py**：把单机逻辑拆进 actor/learner/replay/logger
6. **monte_carlo.py / live_trading.py**：理解你 trading 场景的 runner 改造点

---

## 8. 你这套框架最常见的“可控改造点”（按优先级）

1. **policy 的 mode 行为**
   train=sample / eval=deterministic（你现在还没做）

2. **parallel eval**
   actor 需要按 is_eval 分 batch 调 policy(mode=eval)

3. **logger 指标分离**
   train episode vs eval episode 不混（否则看板很难信）

4. **reward/continue 设计**
   trading 环境常常 reward 很稀疏/尺度漂移，head & norm 要小心

5. **replay_context**
   用于 RNN/RSSM warm-up，trading 很关键（避免靠“开头随机”影响 posterior）

---

如果你想更像“教科书”那样理解，我也可以把上面路线图再加工成两张更直观的东西：

* **一张端到端数据流图（env→replay→train→imagine→loss）**
* **一张 parallel 分布式组件时序图（env/actor/replay/learner/logger）**

你要我按“你当前代码的真实函数名/变量名”来画这两张（用 ASCII 或 mermaid）也可以。
