# GPU smoke 验收（2026-10-10）

## Summary

仅执行授权的小步 GPU smoke，正式训练未启动。重启后 JAX 识别 cuda:0；真实 FG 数据、50M 配置（实际 42587207 参数）、batch=1、序列=4、ratio=1。先 20 步，再从同一金融检查点追加 10 步；共 7 次参数更新。未修改 Reward、方向、费用或成交规则。

## Evidence observed

持久记录位于 `/home/v/Documents/work/gym-trading-env/artifacts/fg_entry_v1/`：gpu_smoke_v3.log、gpu_smoke_seed0_v3/metrics.jsonl、smoke_result.json 和完整检查点。日志明确显示 Policy/Train devices cuda:0、Loading step-20 checkpoint、Saved step-30 checkpoint。12 个训练损失指标均为有限值，模型参数全部有限。账户游标 19 → 29，账户序号保持 1，原成交、账本和 Reward 记录前缀保持一致，原 2 笔成交保留。

## Root cause

第一次 GPU 保存失败：Driver 对 GPU BF16 carry 直接 np.array，引发 JAX Disallowed device-to-host transfer。针对性 GPU 回归进一步复现恢复后 NumPy carry 直接进入 JIT 的 Disallowed host-to-device transfer。

另一次 20 步运行结束时，定时保存间隔尚未触发，latest 仍为 step 0；这是结束路径缺少金融检查点保存。第二阶段在编译期间停止，未让它从零重新训练。

## Changes made

Driver 保存用显式 jax.device_get，恢复按已初始化策略 carry 的 sharding 显式 device_put。金融账户模式结束时保存最终检查点并输出尚未刷出的训练指标，旧模式不变。GPU smoke 脚本强制 CUDA，确认 step-20 完整金融检查点后才执行恢复；审计工具验证最终步数、有限损失与金融历史连续性。

重启清空 /tmp 工作树及生成数据。已从提交恢复到持久 worktrees/env、worktrees/dreamerv3，未覆盖原工作树。数据重新导出并验证 24 段、643929 根。统一配置起点改为已核实的保留数据起点 2018-04-10，仍排除 FG805，截止 2025-12-31；不放宽区间内完整性检查。

## Verification performed

GPU Driver 保存/恢复/JIT 回归 3 项通过；GPU 接入、掩码、Gym 和检查点合计 14 项通过。环境数据回归 5 项通过。实际主程序 smoke 两阶段退出码均为 0，独立 checkpoint/metrics 审计通过。shell 语法验证和 git diff --check 通过。

## Risks / not verified

保留 availability warm-up，未跳过门槛；policy/value loss 为 0，不能声称正式 masked actor 已开始训练或模型已学会等待。本次 batch=1/length=4 不能证明正式 batch=16/length=64 的显存容量、长期稳定性或择时效果。尚未进行多 seed 样本外比较。正式训练仍禁止；CPU 训练仍禁止。

## Next recommended step

保留本次代码、数据和审计记录。只有再次获得正式训练授权后才开展后续训练；不自动扩大步数。
