# Complete-cohort context audit for final evaluation

Implements the existing approved TEST context-quality requirement. Current `context_metrics.py` only accepts historical DEV53 format; preserve it unchanged. Add an offline `analysis/evaluation_context.py` for complete modern evaluation runs.

Inputs: one or more exact `evaluate_pipeline.py` directories, requested split, optional immutable lock. No provider, credentials, balance, GPU or model operations. No actual TEST rows will be read during implementation; use synthetic TEST150 and TEST30 fixtures. The current paid runtime/config/source contract must stay unchanged.

Validate complete ordered cohorts, unique profile names, identical frozen contracts across runs, per-profile manifests, semantic source hashes for data/metrics/types/packing, corpus/query/split hashes and every `make_record` checksum/recomputed metric. Reject controls and unresolved in-flight markers. TEST requires an exact lock and full TEST150 retrieval/rerank or frozen first30 generation. Load only selected annotation rows.

For ranking-only phases replay shared raw corpus `pack_context` at 4,000 proxy tokens. For pipeline/retrieve-only phases validate observed context IDs/text/metadata against the corpus and shared packing; retain contexts even when later generation fails, reporting operational errors separately. Original evidence and gold remain unchanged. Reuse `context_quality`, `QUALITY_KEYS` and `summarize` from the historical audit, plus paired bootstrap for all method pairs. Null gold makes paired metrics unavailable; never shrink denominators. Save per-question scopes/IDs/packing/outcomes and source/input hashes. Bind JSON parsing to exact byte snapshots and recheck consumed artifacts before exclusive publication.

- [x] Synthetic real-parquet/checksummed fixtures; observed missing-feature RED.
- [x] Complete-cohort validation, shared packing and per-metric all-pair statistics.
- [x] Partial/reordered/corrupt/profile/source/input/in-flight/unknown-ID/context/TEST-lock and mutation-race checks. Independent review found mixed-phase short TEST cohort and missing manifest identity checks; real RED/GREEN regressions close both.
- [x] 34 focused and 161 combined tests; Ruff/diff clean; independent AI code re-review accepted, no remaining actionable P1/P2.
- [x] Real zero-paid DEV12 retrieval preflight reproduction: `devsupport/reports/dev-context-recovery-v3-audit-v2/summary.json`. No real TEST inference/scoring.
- [x] README, RUNBOOK, experiment/acceptance evidence and local checkpoint; current42 runtime contract checked unchanged.
