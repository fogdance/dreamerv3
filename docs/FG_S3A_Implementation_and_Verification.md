# FG S3-A 开发交付

日期：2026-10-12。分支：`feat/fg-s3a-oracle-bc`；开发基线 `446e6fc9dbc92a1be24d0d44b840e85d4f96010e`；功能提交 `358b267`。

完整实施报告、CPU 命令、源身份、30 项测试与实际阻碍见 [Gym S3-A 实施与验证报告](../../gym-trading-env2/docs/FG_S3A_Implementation_and_Verification.md)。收据位于 [development_20261011](../../gym-trading-env2/artifacts/fg_s3a_v1/development_20261011)。

已完成原 Encoder/RSSM 教师历史提取、原 `self.pol` 等权 masked BC、独立 Optax 状态、逐键冻结证明、policy 参数同步、原子 Checkpoint、精确恢复、缓存身份和独立 CLI。所有实现为新增文件，旧 Agent / wrapper / 配置不改；[核心诊断补丁](FG_S3A_pending_integration.patch) 尚未应用，独立适配器已经覆盖该诊断问题。

CPU 证据：真实 C1_P4 前缀 BF16，小尺寸原 Agent；208 键中 203 冻结、5 个 `pol/*` 更新；10+10 恢复与连续 20 更新的模型、独立优化器、采样器和日志一致，更新后 policy 概率误差 <= `5.96e-8`。小切片分类为 `ALL_WAIT_COLLAPSE`，不能据此声称择时有效。

代码支持只读加载身份一致的完成版 C1_M1_1M 然后只训练 Actor；生产 15 项 core 与 9 项依赖已完成旧源身份的元数据核对，**尚未加载完整生产 payload**。原最终 62,500 Checkpoint 未完成；生产 gate / 阶段迁移修订后，重训源身份必须重新审核。正式 GPU BC 和独立 DEV 未执行。

从本仓库调用 CLI 前设置 `PYTHONPATH=/home/v/Documents/work/dreamerv3-dev:/home/v/Documents/work/gym-trading-env2/src`，使用原 `dreamerv3` conda Python。默认 CPU；先运行 `scripts/fg_s3a.py --help`；完整命令见上面的实施报告。个人已有 untracked 文件未触碰。
