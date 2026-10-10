# Offline context quality (DEV53)

Run from the checkout containing this utility, with data in the original project:

```powershell
F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe devsupport/analysis/context_metrics.py --project-root F:/DevSupportCopilot
```

The command exclusively creates `devsupport/reports/context-quality-dev-v1/rows.json`
and `summary.json` under the supplied project root. An existing output directory
is rejected. `--output` can select another new directory. The audit performs no
API calls, model loading, GPU inference, generation, TEST scoring or negative-control
evaluation. Only DEV rows and required columns are selected from the query Parquet;
the complete original query file is hashed as bytes for provenance.

The seven fixed inputs are B0/B1/H0/H1 from `retrieval-v1/dev`, C from
`contextual-v1/dev`, and R1/R2 from `rerank-v1/dev`. Each must contain the complete
ordered DEV53 cohort. Manifest profile/configuration, split/query/index hashes,
query embedding provenance and successful run status are checked. Reranker
manifests bind the validated H1 file and manifest; every reranked pool must retain
its H1 membership. Explicit `fallback=false`, `error=null`, matching model/revision,
nonzero forward count and matching output traces are required. Stored retrieval
metrics are recomputed with relative/absolute tolerance 1e-12 for interpreter
floating-point roundoff; drift beyond that is rejected. Duplicate or unknown ranking
IDs and unknown annotation IDs are rejected. Annotation lists use set semantics;
the same positive document may support multiple nuggets. Original corpus duplicate rows use the existing
`official_last` convention.

All rankings are repacked from original corpus evidence with shared `pack_context`,
`ProxyTokenCounter` and budget 4000. This is the cl100k proxy policy, not a claim
about the generator's exact tokenizer. Cached rankings are replayed; these are
context-selection diagnostics, not observed generated answers.

Metric definitions for ordered packed IDs:

| Metric | Definition |
| --- | --- |
| `packed_context_doc_precision` | Unique packed documents with positive qrels / actual packed document count. Unlike existing `precision@10`, missing slots are not padded to ten. |
| `context_average_precision_qrels` | Sum of precision at each positive packed position / number of positive packed documents. Measures packed retrieval order; its denominator differs from ordinary AP over all gold positives. |
| `context_doc_recall` | Positive packed documents / all unique known positive qrel documents. |
| `context_nugget_coverage` | Nuggets with at least one positive packed document / all nuggets, including unjudged nuggets, following the existing FreshStack coverage convention. |

With known positive gold, empty context or no positive packed documents yields
zero. Missing/all-unjudged gold yields null for all four quality metrics, not a
bad-quality claim. “Unjudged nugget” follows the existing dataset audit convention:
a nugget without any positive document judgments. Unjudged packed documents stay
in the precision denominator, so precision is agreement with known positive qrels,
not an assertion that every other document is irrelevant.

Per-query rows retain packed IDs, packing trace, rendered-context hash, annotation
status and denominator counts. Summaries report macro means with defined `n` and
missing counts per metric. Paired C−B0/C−H0/C−H1 percentile bootstrap intervals use
the exact same ordered 53 queries, seed 20261008 and 2000 samples. A null value
anywhere makes that paired metric unavailable rather than silently shrinking the
cohort. These are exploratory DEV comparisons, without multiple-comparison adjustment.

The summary binds every consumed ranking, configuration, manifest, original
corpus/query/split file, tokenizer and audit/packing/metric implementation by
SHA-256. Historical retrieval source hashes are preserved indirectly through
archived manifest hashes; the annotation/metric source hashes must still match
the archived project source, which must equal the executing checkout except for
CRLF/LF checkout line endings; both exact byte hashes are recorded. Consumed input bytes are rehashed before publication to
detect concurrent change. Large model/index artifacts are not loaded or reverified;
their historical provenance is bound through the index/run manifests.

These metrics do not claim Ragas parity, LLM judgment, faithfulness or answer accuracy.

Focused verification:

```powershell
F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe -m pytest -c devsupport/pyproject.toml devsupport/tests/test_context_metrics.py -q
F:/DevSupportCopilot/.venv/Scripts/ruff.exe check devsupport/analysis/context_metrics.py devsupport/tests/test_context_metrics.py
```
