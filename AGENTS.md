# S3 Dreamer 开发与集成要求

工作区根目录 AGENTS.md 继续适用。测试合同、历史迁移、共享回归与验收证据遵循同工作区兄弟 Gym 仓的 [AGENTS.md](../gym-trading-env/AGENTS.md)。该文件随两仓 s3-next 的版本组合发布。

修改 Encoder、Actor、Loss、Clock、Replay 或 FullCheckpoint 时，应执行该文件指定的共享检查及直接受影响的行为测试。共享检查的命令从兄弟 Gym 目录运行。历史迁移固定两端完整 SHA；有意改变数学或恢复语义时，明确拒绝不等价的旧状态迁移，不为过时断言增加生产兼容。
