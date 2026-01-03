1）两套 “mode” 概念分别是什么？

在 embodied / DreamerV3 风格里，确实有两套（更准确说：两套 + 一个 training flag），最容易混：

A. 「数据流 / 回放来源」的 mode：train / report / eval

这是 Replay / Stream / Runner 层的概念，决定 数据从哪来、用来干什么：

train：用于梯度更新的 batch（learner 真训练用）

parallel 里对应：sample_batch_train() / parallel_stream('train')

report：用于“诊断/可视化/评估训练分布上的模型质量”，不更新参数

典型：open-loop video、重建误差、KL、rew预测等

parallel 里对应：sample_batch_report() / parallel_stream('report')

eval：来自 评估环境 的回放（通常独立 replay），用于统计泛化表现

parallel 里对应：sample_batch_eval() / parallel_stream('eval')

env 侧会带 is_eval=True，replay.add_batch 会路由到 eval replay（你代码里已经这么做了）

这一套 mode 跟动作怎么选没关系，它只在“数据从哪来/用于什么阶段”这一层起作用。

B. 「策略执行（选动作）」的 mode：Agent.policy(..., mode=...)

这是 actor/环境交互时的概念，决定 同一个 policy 分布，你是“采样”还是“取确定性动作”，以及探索强度。

典型约定（你现在的 Agent.policy() 虽然有 mode 参数，但实际上没用起来）：

mode='train'：采样（exploration）

mode='eval'：确定性（exploitation，常用 mean/mode/argmax）

mode='report'：通常同 eval（为了稳定可复现的诊断视频/指标）

（补充）C. training=True/False（模块内部训练开关）

这是传给 enc/dyn/dec 的 flag，影响 norm/dropout/某些训练期逻辑。
它跟 A/B 都不是一回事（它不直接决定数据源，也不直接决定采样/确定性动作）。

2）“policy 都是 sample，为啥赛车学会正常开，还能过终点？”不矛盾

不矛盾，因为你“sample”的是 学出来的分布，不是“均匀随机”。

两个直观例子：

离散动作（categorical）：
如果网络输出 p = [0.999, 0.001]，你仍然是 sample，但 99.9% 都会抽到第一个动作，行为看起来就是确定性的。

连续动作（Normal）：
如果输出 Normal(μ, σ)，训练后经常会把 σ 学到很小（比如 0.01），那 sample 出来的动作几乎就是 μ，轨迹很稳。

此外，Dreamer 的 policy loss 里虽然有 entropy 正则（actent），但你设得很小（默认 3e-4），最终策略熵会下降，表现就是“sample 但不乱”。

所以“sample”≠“随机乱搞”，它只是“按当前策略分布取样”。分布一旦足够尖，行为就会稳定得像 greedy。