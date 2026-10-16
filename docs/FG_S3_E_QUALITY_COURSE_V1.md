# E + Quality TP course contract V1 (WP01)

`dreamerv3/fg_s3_e_quality_course_contract.py` is the single source of course
rules. It imports only the Python standard library and performs no I/O. It is
not an Agent, runtime `quality_stage`, SourceAdmission, or training configuration.

The public API is `validate_course_spec(raw)`, `expected_stage_contract(stage)`,
`inspect_course_readiness(validated)`, and `course_identity_payload(validated)`.
All errors are `ValueError`. Unknown fields, including user-supplied derived
stage matrices and approval flags, are rejected. Readiness revalidates derived
output rather than trusting it. Returned structures do not alias the input.

The raw specification must explicitly select `spec_mode: draft` or `formal`.
Only draft allows **explicit nulls** for pending production identity. Missing
keys, blank strings and empty sources are errors. Mode controls diagnostic
validation, not execution permission. The new student is always `direct_init`;
old E+Tutor parameters are not an allowed course source.

Four stages declare parameter permissions: FOUNDATION_WM trains the six WM
groups including separately named `avail/`; QUALITY_HEADS requires valid labels
for each head; ACTOR_SUPERVISED trains only `quality_actor/`, without requiring
prediction admission; ONLINE_ADAPT keeps other groups behind WP07 review and
their independent gates. Every stage has `training_enabled=false` and
`CONTRACT_ONLY / NOT_ADMITTED`. All prediction corrections are zero in WP01.
The first two stages have no selected decision head; F3/F4 select quality_actor.
F3 depends on F1; F2 research admission is not a prerequisite for label-only F3.

Gym owns native consistency checks, source/path inspection, hashing and report
I/O. It must use existing `fingerprint/signed/write_json` helpers. The identity
payload includes the normalized specification and derived four-stage matrix,
excluding the diagnostic spec_mode. Mapping key order is immaterial; ordered
lists, seeds and declared source hashes are significant. Reward numeric values
and random probabilities are normalized; no reward formula is implemented here.

This introduces a **new design identity**, not migration or same-run resume.
Legacy HEADS_ONLY/CONTROLLED_FULL and composition readiness remain unchanged.
Course manifests cannot be substituted for Prepared, corpus/teacher certification,
checkpoint provenance, or training authorization. Native Gym documentation and
an explicit draft fixture are provided by the second WP01 commit.
