# Offline judge abstention coverage contract

DEV generation remains frozen. AI judge v3 stopped on query 76447153:
Q003-A02 now correctly returns supported_abstention, but nugget_coverage is {}.
The strict validator requires N001..N004 even for an empty abstention. The two
other eligible responses and the first two questions meet the same contract.

Preserve legacy and explicit prompt bytes and all previous paid results. Add an
opt-in explicit_coverage protocol that inherits explicit refusal constraints and
supplies required_nugget_ids, with a system instruction requiring exactly those
keys for every review, including empty abstentions. Coverage values and scores
remain evaluator decisions. Do not default blank answers to missing, change the
rubric/validator, repair provider outputs, or pool different scoring batches.

1. RED tests for opt-in coverage payload, non-mutation/eligible-only inputs,
   actual provider delivery and binding, strict empty-coverage rejection.
2. Minimal analysis-only protocol/CLI change; GREEN related tests and Ruff.
3. Verify old protocol bytes against archived source and historical metrics;
   verify frozen runtime and generation48 bytes unchanged. Independent review.
4. Fresh full12 AI assessment under one protocol; old expenses retained. Then
   original four-profile DEV10 controls before runtime release or TEST selection.

No runtime, retrieval, generation, provider routing or budget change. Human0.
