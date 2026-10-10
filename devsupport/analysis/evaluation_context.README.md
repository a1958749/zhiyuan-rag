# Complete-cohort context evaluation

`evaluation_context.py` consumes complete modern `scripts/evaluate_pipeline.py` output directories. It makes no model, network, balance or credential calls. Historical DEV53 caches retain their separate `context_metrics.py` protocol.

```powershell
& 'F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe' devsupport/analysis/evaluation_context.py --runs devsupport/reports/dev-decision-recovery-retrieval-v3 --output devsupport/reports/NEW_CONTEXT_REPORT
```

For final TEST, supply `--split test --lock PATH_TO_FROZEN_LOCK`. Ranking/retrieval/rerank/retrieve-only phases require all frozen TEST150 IDs; generation pipeline requires the exact first30. Each run is checked independently. A TEST30 pipeline cannot authorize a shortened retrieval experiment. Multiple directories need the same exact contract and ordered cohort and disjoint method names; retrieval/rerank may be combined when both cover TEST150. TEST150 rankings and TEST30 generation are reported separately.

The tool binds run/manifest/JSONL parsing to the exact consumed bytes, verifies corpus/query/split and semantic data/metrics/types/packing source hashes, matches profile/settings/variant and query/index/phase identities, and reconstructs every evaluation record with the shared evaluator. Partial, reordered, changed or in-flight inputs are rejected. Dataset loading hashes are checked before and after loading; only the requested question rows/columns are materialized. All consumed inputs and the current analysis sources are rechecked before acceptance and exclusive publication.

Ranking-only results are packed from raw corpus chunks with the shared 4,000 cl100k proxy-token policy and explicitly labeled **replayed**. Pipeline/retrieve-only contexts are validated against raw source text/metadata, evidence order, scores, citation numbers and the shared packing trace. Recorded rank/score-kind fields are retained where present; these are not calibrated confidence measures. Original source text is never rewritten.

Summaries include actual packed-document positive-qrel precision, qrel AP, positive-document recall, nugget coverage and effective annotation denominators. These are document diagnostics, not answer accuracy or faithfulness. Unjudged gold yields null values; complete-pair metrics become unavailable instead of discarding those queries. Every method pair uses the contract seed and 2,000 paired percentile-bootstrap samples; the intervals are exploratory and have no multiplicity correction.

Generation/provider failures remain separate operational counts. A later generation failure does not erase already valid retrieved context. An earlier failure with no evidence yields empty-context metrics where positive gold exists. Failures are never replayed by this utility. Existing output directories are refused; no paid runtime configuration or source file is changed. Current semantic source verification does not assert that all archived runtime code was executed again.

Development uses synthetic TEST150/30 fixtures; no real TEST scoring precedes DEV selection and the final lock. The actual DEV12 retrieval preflight reproduction is diagnostic evidence, not a new quality benchmark or an independent repeat of the original DEV dataset.
