# S3 双仓库工作流

共同规范位于兄弟 Gym 仓库的 `docs/S3_GIT_WORKFLOW.md`。

- `main` 保存稳定版本，禁止直接开发或自动发布。
- `integration/s3-next` 是下一代 S3 公共集成分支；功能从最近验收的双仓版本建立，测试与 Review 后普通 merge 回 integration。
- 两仓独立提交，用工作区根目录的 `baseline-lock.json` 记录最终完整 SHA、W&B 版本和回归结果；不在仓库间互写提交 SHA。
- 正式训练由 SHA、不可变 Tag、配置、数据及准备身份锁定；运行中的 detached Worktree 不修改、不合并、不重置。
- 功能分支和 Worktree 只在实际开发时建立，生命周期结束后先验证提交已包含，再单独申请清理。
- W&B 位于兄弟 Gym 的独立 Tracker，Dreamer Trainer 不引入网络依赖。
- 当前集成保留 A2、Tutor、B2 的独立教学语义，不合入 Archive，不实现 E 或 Tutor+B2 联合教学。
