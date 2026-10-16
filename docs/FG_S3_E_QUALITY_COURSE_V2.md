# Course V2 and native E input

This deliberately replaces COURSE_V1 and E Observation/Replay V1. The pure course module remains standard-library only. Five float32 policy fields include market_macd_5m [4], with normalized DIF, signal, DIF-signal and valid. The exact MACD semantics are part of the canonical course hash and checked against native Gym.

No old weights/checkpoint migration. Source, schema and same-version checkpoint identities must match exactly. All four curriculum stages remain CONTRACT_ONLY / NOT_ADMITTED. A valid explicit draft is diagnostic only; formal preflight remains nonzero, with all blockers. Corpus certification never releases public E+Quality training or Prediction/Q/Decision gates.

## Model V2

The actual E Encoder strictly consumes all five fields. MACD projects three gated values to 16 units and masks projection output again, before fusion with minute and 15m branches. A zero valid bit suppresses bias as well as values. Encoder gradients include the MACD branch; Decoder still reconstructs only market_seq_e_shared and position_state. The native persisted Replay is consumed by a fresh real CPU Agent; continuous tokens and posterior logits respond to a legal MACD perturbation. Categorical samples need not change when logits change. FullCheckpoint verifies optimizer, policy, Driver, account, Replay, sampling RNG and counters under the same V2 identity.
