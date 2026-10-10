# Recover final evaluation from valid Unicode JSONL framing

The first final lock was created before TEST scoring. Five local retrieval
conditions each produced all 150 records without operational errors, then the
outer completeness reader failed with JSONDecodeError. Byte-delimited reading
confirms 150 valid JSON objects per file. Literal U+2028 in document text causes
Python str.splitlines() to split inside a valid JSON string; byte splitlines()
does not treat Unicode characters as physical record delimiters.

This is a serialization-reader bug, not a retrieval or provider failure. The
problem also affects frozen runner resume and rerank cache readers, so fixing
only the outside controller would leave the final workflow incomplete.

Preserve the original lock, controller exit, original 750 records and all hashes.
No paid TEST calls have started. Repair JSONL framing only; do not change DEV
selection, corpus, split, rankings, scoring, providers, prompts or parameters.
Create an explicitly versioned corrective final lock and new TEST directories
after verification, rather than rewriting the first frozen source identity.
The initial run is a retained diagnostic, not an independent replicate or
untouched first-use TEST. Selection remains the pre-TEST DEV decision.

- [x] Reproduce Unicode separators in complete checkpoint resume and context
  audit with synthetic inputs (no real TEST annotations or provider calls).
- [x] Repair all relevant JSONL readers with physical byte-line delimiters.
  Seventeen reader changes preserve JSON string values and original byte hashes.
- [x] Verify original 750 row digests and complete order; never alter them.
- [x] Six regressions observed RED for the actual parse failure, then GREEN.
  Core 833 passed, four dependency warnings; independent core also 833 passed.
  Lint and read-only source review have no remaining P1/P2.
  The corrective outside controller has 19 synthetic checks. Review identified
  a preflight-to-freeze drift gap; two new drift checks observed RED then GREEN.
  It now verifies the entire new lock against its captured contract before TEST.
  Independent re-review has no remaining P1/P2 (16 synthetic tests, excluding
  three main-invocation checks to avoid real TEST access).
- [x] Commit the narrowly scoped repair and freeze corrective v2 with exactly
  the same selection/parameters/data; retain v1 and its source evidence.
  Repair commit fa9a1859a4d341d2a2b2a033d8c5801365200330. Corrective freeze
  2026-10-10T07:10:38.346166Z, contract SHA256
  5b0e89be88826853c5cbe1808d85798a8369791d9d757773c5cc18bd334e4674.
  The v2 configuration is byte-identical to v1. Actual controller PID3308,
  session86337; launch proof is
  devsupport/reports/final-controller-launch-verification-20261010-v2.json.
- [ ] Run the full approved TEST scope in new v2 directories, serially; preserve
  old results and never replay any already billed TEST question.
  Five local methods and native U have each completed 150 ordered records with
  zero operational errors. The controller is still executing remaining stages.
  Original 750 rows also passed the repaired offline context audit in
  test-initial-framing-context-v1, with no model calls.
- [ ] Update public publication paths, documentation, final media and archive
  to the verified corrective version, with the original failure disclosed.
