# WP08 Dreamer 接口

考试使用原 E Encoder/RSSM 和 `quality_actor`，复用原 `create_agent/load_exact/ActorPolicy`；没有新 Trainer 或优化器。Gym 交付报告、完整输入任务及原始测试索引位于兄弟仓 `docs/WP08_IMPLEMENTATION_REVIEW.md`。

新增 `embodied/run/fg_s3_e_quality_exam.py::inspect_exam_actor()`：分别核验 F3/F4 的完整来源、参数 Schema、不可变节点、账本、F4冻结组和原生 Gate 控制态。不将合成来源作为正式受训模型。Value、slowval、所有优化器 slots 都完整加载并冻结。

原 JAX Agent 新增推理状态 API：`configure_evaluation_seed(seed)`、`save_policy_state()`、`load_policy_state()`。状态协议为 `FG_POLICY_INFERENCE_STATE_V2`。考试随机种子独立于原 `config.seed` 和训练状态；仅在初始发布后、首个动作前配置。原未配置该 API 的训练/评价路径维持原种子算法。Eval 的 RSSM仍实际采样，因此比较模型时显式共用考试种子并保存动作计数。

完整训练 Checkpoint 作为只读模型来源，不做训练 resume；独立 EvalSession 保存 Env/Driver/账户、carry、MACD构建态、推理/随机基线状态和提交边界，不包含优化器或训练 Reader。G 阶段旧 V1 诊断节点不允许在新代码伪装同 Run 恢复；最终 V2 复验另列证据。

本轮仅 CPU 合成工程更新/回归及来源只读核验，禁止 TEST、正式 TRAIN 参数更新、GPU训练和开放训练 Gate。真实受训 F3/F4 的 DEV 策略考试未运行时明确 NOT_RUN。
