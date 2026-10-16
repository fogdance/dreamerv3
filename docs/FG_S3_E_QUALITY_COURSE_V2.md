# Course V2 and native E input

This deliberately replaces COURSE_V1 and E Observation/Replay V1. The pure course module remains standard-library only. Five float32 policy fields include market_macd_5m [4], with normalized DIF, signal, DIF-signal and valid. The exact MACD semantics are part of the canonical course hash and checked against native Gym.

No old weights/checkpoint migration. Source, schema and same-version checkpoint identities must match exactly. All four curriculum stages remain CONTRACT_ONLY / NOT_ADMITTED. A valid explicit draft is diagnostic only; formal preflight remains nonzero, with all blockers. Corpus certification never releases public E+Quality training or Prediction/Q/Decision gates.
