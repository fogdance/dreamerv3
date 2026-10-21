# Batch3 Dreamer implementation evidence

The paired implementation review, original commands/exits/JUnit, native account
evidence, numerical probe and precise cold restore reports are maintained in the
sibling Gym [Batch3 review](../../../gym-trading-env/docs/batch3_r2_r8_r3_20261021/BATCH3_IMPLEMENTATION_REVIEW.md).

Dreamer changes are limited to `embodied/run/fg_s3_e_quality_online_run.py`,
`fg_s3_e_quality_online_gate.py`, `fg_s3_e_quality_online_rl.py` and
`fg_s3_e_quality_online_probe.py`. Existing Agent/Loss, Env, Replay, Reader,
shared Prefetch and FullCheckpoint implementations are reused unchanged.

The new OnlineRun/Gate V3 contract separates durable technical facts from
process-local external-review permission. Frozen older Runs remain on their
original code. Corpus ABI stays 3.1.0. Final paired HEADs and acceptance evidence
are recorded outside both repositories in the workspace `baseline-lock.json`.

Selected native synthetic CPU behavior is verified. Formal trained F3, real
published TRAIN G3, real operator approval, formal live RL, GPU, long training
and research performance remain **NOT_RUN**; TEST market data is **NOT_READ**.
