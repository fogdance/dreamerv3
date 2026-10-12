# S3-B 独立开发与 CPU 验收

本仓库基准 `8952315be94ed543af1b2bcb73e211cba5077fbc`；分支
`feat/fg-s3b-certified-bc`。S3-A 按已完成依赖复用；本轮只新增 S3-B 代码，
没有修改原 agent/RSSM/联合优化器/S3-A 文件，也没有启动正式 S3 GPU 训练。

完整证明设计、全量数据审计、测试收据、保护哈希漂移、可执行命令及后续配置：

- [完整实施报告](/home/v/Documents/work/gym-trading-env2/docs/FG_S3B_Implementation_and_Verification.md)
- [资金与价值证明设计](/home/v/Documents/work/gym-trading-env2/docs/FG_S3B_Value_Proof_Design.md)
- [CPU 交付审计](/home/v/Documents/work/gym-trading-env2/artifacts/fg_s3b_v1/development_20261012/B4_delivery_audit.json)

当前结果：402,096 个原 S3-A 索引完全一致，396,542 个严格优势证书，
16,171 ADD / 380,371 WAIT 正权重；5,554 个精确并列保留但权重 0。
总 ESS 186,925.66；ADD 权重贡献 1.896%，仍需监测多数 WAIT 学习风险。

16 项新 CPU 测试通过。实际 P4 数据的原神经 Agent 接线测试：892 行缓存、
六步接口更新、连续与恢复逐字节相同；MATCHED/CERTIFIED 逐 batch 同 index；
208 个参数/状态键中 203 个冻结，只有原 pol 的 5 个键改变，原 joint counters 未变。
实际 policy 使用更新后的 Actor；没有以诊断测试宣称 DEV 收益改善。

新增入口：`scripts/fg_s3b.py train/admission-audit/checkpoint-audit`；配置在
`configs/fg_s3b/`，FULL/MATCHED/CERTIFIED 除 mode 外一致。
源实验整个目录禁止作为输出；新协议 `FG_S3B_BC_V1` 与 S3-A checkpoint 隔离。

后续正式运行使用完成且版本兼容的 C1_M1_1M 源及其完整 S3-A feature cache；
Replay 尚在训练不阻碍本次代码交付。生产路径发生的并发字节漂移已保留在
完整报告中，本仓库原有源码字节未变。TEST 不访问，B5 NOT_RUN。

源码提交：Gym `52d2db6`，Dreamer `6ef6938`。进度 100%（本轮代码/CPU 范围）。
