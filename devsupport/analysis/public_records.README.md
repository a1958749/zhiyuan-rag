# Public per-query views

`public_records.py` publishes a whitelist-derived view of one private JSONL file.
It keeps all rows in their original order, including failed attempts. Answers,
decisions, evidence IDs, rankings, supported metric fields, validated AI reviews
and stage diagnostics retain their original values. Balances, raw provider text,
raw validation-failure responses and unknown nested fields are excluded.

The output contains `records.jsonl` and `derivation.json`. The receipt binds the
source-file bytes and projection rules; each row binds its original line and
canonical record hash as well as its derived hash. Existing embedded checkpoint
digests are validated. This is a public view, **not a resumable checkpoint**.
The utility does not verify full benchmark completeness or score semantic
quality; final summaries must be computed from complete private original cohorts.

For a new directory, run:

```powershell
$env:PYTHONPATH='F:/DevSupportCopilot/devsupport/src'
F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe devsupport/analysis/public_records.py --source devsupport/reports/dev-decision-generation-v2/H0.jsonl --output devsupport/reports/public-example/H0 --method H0
```

CLI export scans long key patterns. Final packaging must also call `publish`
with the actual credentials supplied as `forbidden_values` **in memory**; never
put credentials in CLI arguments or a public receipt. Sensitive allowed answer
or evidence text causes rejection before publication; it is not silently edited.
Media needs visual review separately. Original private records remain unchanged.

Native U records retain their actual ordered `hits`, native class/store, candidate
and token budgets, embedding provenance and full native trace through explicit
fields. Unknown nested fields remain excluded; these records are not normalized
into synthetic modern runner outputs.

Modern evaluator U records retain the same complete native whitelist, including
`variant`, at their actual `result.trace.retrieval[].native` location. Nested native
embedding/model-manifest values use the same scalar/object boundaries and credential
scans as legacy records. Requested candidate count can differ from the actual returned
count because genuine ClassicRAG applies its existing token budget; do not pad native
results or reject a valid shorter list merely to report fifty candidates.

The default remains strict. An explicit `--redact-query-keys` opt-in replaces long
`sk-` patterns only in original, retrieval and adaptive query text. Every changed
field records its path, original whole-text SHA256 and replacement count; the
derived hash is recomputed. Original query/record hashes, answers, evidence and
metrics remain unchanged. Known in-memory credentials are rejected **before**
replacement, including JSON-escaped values. Patterns in answers or evidence
still reject publication. Public redacted query text cannot be treated as the
exact original input or used to resume an evaluation.

The actual `reports/public-dev-completed-v1/summary.json` binds 21 complete
existing DEV method files and 703 records, with source/public hash rechecks,
both in-memory credential literals and zero model calls. This is a count of
method records, not 703 independent questions. DEV53 retrieval/reranking and
adaptive results, and the two DEV12 generation/assessment cohorts, stay separate.
Six query fields across three records for DEV78549270 are explicitly redacted;
the validity of the pattern-shaped string is unknown. Native hits/trace and all
original rankings/metrics/answers were verified exact. No TEST or final acceptance
is established by this publication.

`reports/public-dev-decision-generation-v2/summary.json` records the reviewed real
DEV12x4 projection: all48 rows, two in-memory credential literals checked, zero
model calls. It does not establish final answer quality, TEST results or delivery.
