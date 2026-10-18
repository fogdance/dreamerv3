# WP04 Dreamer 实施核对

共同祖先：`446d1bc99debd2d05c313eafd7d0c414ebd2815c`；成对分支 `feat/s3-e-quality-wp04-20261018`。完整报告、原始命令、42 项对照、合同哈希与提交锁集中在兄弟 Gym 的 `docs/WP04_IMPLEMENTATION_REVIEW.md`，最终 E SHA 在其 `WP04_DELIVERY_COMMIT_LOCK.json`。

A `d7bcbcd2a189de75cef57122d61219341b11e65d`：fresh 完整 Foundation WM 合同；B `a2cb48e5aa115d7a66de41a0efd66073ced1bf4c`：原 WM/单优化器、K detached burn-in、六组白名单和精确冻结；C `09fc1223933f5aad828eb5f016958338b3787520`：原 WM 可选只读诊断及共享 posterior/Actor 检验；D `c6c35b0cb06b0947a9ebf1bc5a068f21b5b134ca`：原 FullCheckpoint 外部不可变 Reader 后端、原 Prefetch/seed、完整精确冷恢复与严格同版本身份。

实际最后选定集合包括：WP04 A/B/C 40 PASS、最终 cold restore +合法配置拒绝22 PASS、原 checkpoint/retention 15 PASS、mandatory Gate/SourceTransfer/formal guards 20 PASS、E/账户恢复 6 PASS、WP03 Teacher loss 11 PASS、其他 Gym/模型/金融/时钟/Reward 用例。最终不同用例总数按兄弟 Gym `docs/receipts/wp04/E_stage_results.json`，不表述为全项目测试通过。原失败和修复后的记录均保留。

完整349叶，205可变，144 frozen；有效合成样本六 WM 组全部参数叶有变化，原 Actor/Critic/slowval/Quality 及 slots 不变。连续4步与2+冷进程+2的全部状态/指标/下一批 bitwise 相同。旧 WP03→新 F1 不通过历史 Gate-only 迁移；新 diagnostic 参数必须默认 False，完整 AST 仍比较，数学/helper/factory/默认 True 变异拒绝。

真实 TRAIN 仅新 G3 与只读输入/Loss，更新0；不调用原公开非F1 Guard 的解锁。GPU/真实 F1 参数/正式 start-resume/旧1M/TEST/18合约/研究效果均 NOT_RUN。CODE_GO/REAL_DATA_READ_GO 不代表 TRAIN_GO/RESEARCH_GO。

DEV/DATA/FORMAL 按用户最新分级：DEV 不每次冻结代码、不每次 G3，可最小定向验证；DATA 发布和 FORMAL 保留来源与完整准入，三级均保持金融正确和因果隔离。

E 集成审查保持 FORMAL 10 GiB checkpoint reserve、合成诊断1 GiB；最终 E_restore_final 22 PASS，仍 bitwise 恢复。该 E 行为修正及本说明同提交，最终 SHA 见交付锁。
