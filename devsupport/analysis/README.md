# Offline analysis

Audit recorded pipeline control decisions:

```powershell
python devsupport/analysis/control_decisions.py --run PATH_TO_COMPLETE_CONTROL_RUN --output NEW_REPORT_DIRECTORY
```

The command requires controls=true, pipeline phase, and every declared profile's complete ordered DEV10 or TEST30
cohort from the frozen 40-row controls file. It rejects partial/missing/in-flight runs and changed annotations,
contracts, profile settings, runtime identities or reconstructed records. Existing output directories are refused.
It writes summary.json and summary.md with raw counts, denominators, null rates for empty denominators, confusion
matrices, query IDs and input/implementation byte hashes. Only controls, run JSON, manifests and JSONL are consumed;
it does not read FreshStack query data, credentials, indexes or provider configurations from disk, or execute models.

Frozen expected_decision labels are decision proxies, not human-verified answer faithfulness. Label verification
is not established by this tool. The unsupported control answer rate is not a hallucination rate or an evidence-level
unsupported-claim metric. Errors precede provisional decisions, and fallback events stay separate from terminal
failures. Historical source hashes remain historical; recorded byte hashes provide binding, not external authenticity.

Complete modern evaluation contexts are checked by [evaluation_context.py](evaluation_context.py);
[protocol and commands](evaluation_context.README.md) describe DEV and locked TEST150/30 requirements.
Historical DEV53 caches retain [context_metrics.py](context_metrics.py). Observed and replayed contexts are labeled
separately, and later generation failures do not erase valid evidence diagnostics. Neither tool measures answer faithfulness.

Complete recorded provider accounting is available through `provider_accounting.py`:

```powershell
python devsupport/analysis/provider_accounting.py --run PATH_TO_COMPLETE_RUN --output NEW_REPORT_DIRECTORY
```

For TEST, add `--split test --lock PATH_TO_EXACT_FROZEN_LOCK`. Main cohorts reuse the context validator;
control cohorts reuse the control validator. Partial/in-flight inputs cannot be published. Input JSONL and every
validator binding are hash-checked before acceptance and publication. The helper never calls providers, balances,
credentials or models. Publish sequentially to a new directory; an existing summary file is never overwritten.

Stage invocations can stop before transport. Missing `attempts` stays unknown, while explicit zero stays zero.
Known transports and retry extras are recorded evidence, not verified billed request counts. Usage sums include
only response counters actually present. Positive transports need both input/output counters to be marked complete;
retries remain conservatively incomplete because call-level response usage does not reconstruct the retry ledger.
Explicit input/output double zero is valid. Configured-rate costs are estimates in their configured currency, not
confirmed payment, shared-balance attribution, or total local GPU/hosting cost. Missing costs remain unavailable.
Per-query accounting and operational outcomes keep every attempted row, including failures.
