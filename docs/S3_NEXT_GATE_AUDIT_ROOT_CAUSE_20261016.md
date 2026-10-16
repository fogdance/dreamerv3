# s3-next Gate 转移审计测试失败：根因与修复

## 实际失败与边界

未修改的 `integration/s3-next` Dreamer
`6114b557e5cc5a134a23ae5b446b351e4f95cc0a` 单独执行
`embodied/tests/test_fg_s3_gate.py::test_audited_ast_accepts_only_the_reviewed_transition`
失败，错误为 `Source network/training semantics changed: dreamerv3/agent.py`。
Gym 为 `713a70f37a6dc1d0800cc9c9cf239752848d5b75`。
同一失败也出现在 E-only 独立工作树；复现时 Agent、SourceAdmission 和该测试
均与集成版本相同，证明该项失败在 E-only 开发前已存在。

本报告只解释该项实际失败，不代表项目全部测试或所有旧模型迁移均已验收。
原始失败 XML、历史矩阵和修复后结果见 `receipts/s3_next_gate_audit_20261016/`。

## 最小根因

测试由 `00b82f8` 于2026-10-12引入，此后未修改。它取固定旧源码 `09e2e5c`
与运行时当前 `agent.py` 比较，断言二者只包含已批准的 Gate 阈值差异。
但生产 `audit_code_transition()` 只允许明确审核过的 Gate、WM函数提取等结构
变化；其余数学、网络、训练语义必须相同。

用同一生产审计函数检查历史版本的实际结果：

| 目标版本 | 相对 `09e2e5c` 的实际结果 | 变更 |
| --- | --- | --- |
| `00b82f8` | PASS | 已审核 Gate 阈值改动 |
| `dc6cca7` | PASS，同一归一化 AST 哈希 | WM函数提取及训练 loss 工厂 |
| `c18a8b5` | REJECT | 首次引入市场分钟折扣语义 |
| `c2988a7` | REJECT | 后续统一 policy 调用接口 |
| `f30d8da` | REJECT | 后续 E 编码器构造与 ObsSpace |
| `13f0393` | REJECT | Quality 合并前的集成版本 |
| `6403cd6`、`6114b55` | REJECT | Quality 合并及其文档版本 |

表中缩写便于阅读；测试与机器收据均固定完整40位 SHA。
第一处失效由 `c18a8b5917f73110f9b6ce840495f4da1d7ca1d4` 引入。旧逻辑对每条
反馈使用 `gamma = 1 - 1/333`，新市场分钟路径使用
`gamma ** market_advance`。当 reset／调度反馈不推进市场、`market_advance=0` 时，
新折扣为1，旧折扣约0.996997。该提交还改变 continuation target、Replay context
输入和 value return 的折扣处理，不能将这些差异当作 Gate-only 全状态迁移。
原测试把有意的语义升级当成仍可等价迁移的变更，预期已过时。

严格拒绝符合生产合同。不同经济／时间语义若明确允许复用参数，应使用现有经过
来源认证的参数 warm-start，并重建新 Run 状态；不能把旧模型/优化器/Replay
直接恢复到新语义。E-only 采用 fresh `direct_init`。

## 为什么先前验收没有发现

此前235项的 `final_test_index.json` 中既没有 `test_fg_s3_gate` 模块，也没有本
失败用例。那是选定测试集合的通过记录，范围没有覆盖本次失效的共享转移审计
断言。这是回归选择的遗漏；该记录不能扩展解释为全项目通过。

## 修复与验证

仅更新测试合同，生产 Agent、审计器和 Gate 策略不改动：

1. 历史正例固定为真实 Gate-only 版本对 `09e2e5c → 00b82f8`。
2. 当前源码须通过自身严格审计。helper、loss、factory 的负例均从同一份当前
   源码变异，避免因基准本来不等价而产生无效的拒绝断言。
3. 新增 `dc6cca7 → c18a8b5` 必须拒绝的时钟语义回归，保护迁移边界。

针对性两个用例通过；Gate、SourceAdmission、正式 Gate guard 的完整共享回归
实际为20项通过、0失败、0跳过，耗时120.03秒。原始 XML 保留；未标记 xfail、
skip，未增加生产兼容或自动迁移分支。上述测试不涉及正式 GPU／1M 训练。

## 后续固定要求

已按用户要求写入工作区根 `AGENTS.md` 的“测试合同与历史迁移”一节：当前行为
优先数值验证；历史迁移固定两端 SHA；语义升级同步明确接受和拒绝边界；不为
过时断言新增生产兼容；集成覆盖共享边界；报告保留准确测试范围与原始证据。

共享检查从同工作区的 Gym 目录执行：

```bash
env PYTHONPATH=.:src:../dreamerv3:tests/entry_v1 \
  JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/v/miniconda3/envs/dreamerv3/bin/python -B -m pytest -p no:cacheprovider \
  ../dreamerv3/embodied/tests/test_fg_s3_gate.py \
  ../dreamerv3/embodied/tests/test_fg_s3_source_transfer.py \
  ../dreamerv3/embodied/tests/test_action_mask_formal_guards.py
```

再执行直接受影响的时钟、Replay／恢复与本功能测试；不穷举无关历史组合。
这些检查不能替代 GPU、真实行情、完整训练、独立 Actor-only 评价或研究准入。
