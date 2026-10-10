# Native public evidence implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; independent review is read-only.

**Goal:** Preserve actual native U rankings and configuration diagnostics in the existing sanitized per-query format.

**Architecture:** Extend only the offline explicit projection, retaining the original `hits` and native `trace` shapes. Normalizing native hits into synthetic runner records would misrepresent their provenance; a separate exporter would duplicate credential boundaries. Existing private JSONL, paid runtime and active cohort remain unchanged.

**Tech Stack:** Python, pytest, existing `public_records.py`.

## Global constraints

- No provider calls, TEST scoring, credential printing or changes to frozen runtime sources/configuration.
- Every source row stays in order; unknown fields and nested account observations are excluded.
- Existing projection receipts keep their historic rules hashes; new receipts bind the extended projection.

### Task 1: Native preservation and confidentiality

Files: `devsupport/analysis/public_records.py`, `devsupport/tests/test_public_records.py`.

- [x] Add a native fixture with ordered hits, actual native budget fields, embedding provenance and unknown private fields.
- [x] Verify native hit preservation fails on the old projection, before changing production code.
- [x] Add explicit native scalar trace fields, ranked ID arrays, embedding metadata and hit scalar schema. Keep dynamic model-file values on the existing text-or-object boundary. Do not pass native dictionaries through.
- [x] Run `pytest devsupport/tests/test_public_records.py devsupport/tests/test_export_handoff.py -q -o addopts=''` and ruff for the modified files.
- [x] Obtain independent read-only review before creating real public batches.

### Task 2: Real DEV evidence publication

Files: a new exclusive directory under `devsupport/reports/public-dev-completed-v1/`; `devsupport/analysis/public_records.README.md`.

- [x] Check each existing DEV53 source against frozen DEV IDs. Check both complete DEV12 generation cohorts and complete judge rows independently.
- [x] Publish explicit source paths through `publish(source, output, method, forbidden_values)` with both DPAPI credentials only in memory.
- [x] Recheck source and public-file hashes, exact hit/answer/metric equality and all row counts. Keep completeness claims separate from projection integrity.
- [x] Record actual source/rules/output bindings and zero model calls, then commit source/tests/docs without private reports.

### Observed publication boundary: query-key strings

The real preflight rejects only the original query and retrieval query for DEV ID 78549270 in all three adaptive conditions. No known credential matched. Native rankings/embedding traces and generation answers have no such pattern. Keep all rows and add an explicit opt-in query-only redaction policy; removing the entire row would hide a difficult sample. The policy must preserve private originals, original query hashes and metrics, record every changed field/path/original text hash/replacement count, and recompute the derived hash. Default publication remains strict; answer/evidence text still rejects key patterns. Known in-memory credentials must be rejected before any redaction, including escaped string cases. Independent review and real preflight precede publication.

## Actual verification

- Native preservation and scalar-boundary tests: two observed RED failures, then 40 public/export checks passed.
- Explicit query-only policy: four observed RED failures, then 44 public/export checks passed; ruff/diff-check passed.
- Both changes independently reviewed read-only; no P1/P2. Native DEV53 hits/trace remained exact and every query-only boundary was probed in memory.
- Actual publication: `devsupport/reports/public-dev-completed-v1/summary.json`, 21 source files/703 rows, six disclosed query fields in three rows, zero known credential matches and zero model calls.
- Private sources/public hashes rechecked; active original42-source runtime contract unchanged. TEST and final acceptance remain pending.

Core regression after the completed extension: 800 passed, 4 existing dependency deprecation warnings, 24.79 seconds; no model calls.

## Modern runner native trace boundary

The final TEST U runner uses `result.trace.retrieval[].native`, unlike the older
native DEV53 file's top-level trace. Preserve the same explicit native scalar,
ranked-ID, embedding and model-manifest whitelist at this nested location; never
pass arbitrary objects through. No paid runtime source or evaluation contract changes.

- [x] Four new regressions observed RED: native trace lost; two nested scalar
  smuggling cases not rejected; retained provenance string skipped credential scan.
- [x] Add the existing NATIVE_TRACE schema to the modern retrieval trace branch.
- [x] Public/export48 tests and lint passed. Independent review found missing variant;
  complete native equality fixture failed before adding it, then 48 tests passed.
  Re-review preserved all fields for 53/53 real DEV traces in memory; no P1/P2.
- [x] Verify a real DEV query through the existing native adapter and export its
  modern result shape without generation. Preserve source, rankings and native
  provenance exactly; do not call this a TEST score or a new independent cohort.
  Actual returned45/requested50 matches the historical native order; an initial
  diagnostic's incorrect exactly50 assumption was corrected by offline verification
  of the saved original/public bytes, without another retrieval. No paid calls.
- [x] Keep prior 703-record public batch and old rules receipts immutable. Final
  public views bind the extended rules and are still pending actual complete TEST.

Real preflight evidence: `reports/modern-native-dev-export-preflight-v1/verification.json`.
The native environment first lacked pyarrow; inspection also found absent bm25s/scipy
needed by the unified runner's raw-index load. Add those three only, preserving all
previous package versions. Original runtime source/data/model/config contract stays
unchanged. The separate native DEV12 CLI check has completed with zero errors and
paid generation calls. All twelve historical native rankings match, and modern
public records preserve every nested native trace and original ranking. Original
and derived digests verify. The mixed-runtime context audit discloses three rows
with eight bounded floating-point differences; historical core context scores are
unchanged. This is a DEV compatibility diagnostic, not a TEST result or independent
quality cohort. Proof: `devsupport/reports/native-cross-python-context-verification-20261010-v1.json`.
