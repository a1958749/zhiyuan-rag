# Recorded provider accounting implementation plan

> **For agentic workers:** Execute inline using the existing approved evaluation scope, TDD and read-only review. This reports original records; it does not modify live runtime or budget policy.

**Goal:** Report request/token/cost observability without interpreting missing trace fields as free or zero-transport calls.

**Architecture:** A small offline analysis utility reuses the complete context/control auditors to validate immutable cohorts. It reads exactly hash-bound JSONL bytes and summarizes observed stage invocations, known/unknown transport attempts, response usage, retries and estimated cost. TEST requires the exact existing lock. No additional runtime source or paid calls.

**Tech Stack:** Existing Python, pytest, JSON and analysis validators; no new dependencies.

## Global constraints

- Main `src/devsupport/*.py` and `scripts/*.py` remain frozen for the active original DEV cohort.
- Failed rows remain in every denominator; missing trace is not an empty observed trace.
- Missing `attempts` means unknown, while explicit numeric zero is a recorded zero.
- Report response usage as reported usage, not total billed tokens. Retries with unknown usage make completeness false.
- Costs remain estimates in configured currency; absence/partial retries make full total unavailable. No balance GET, recharge, credentials or provider requests. Raw provider text in input JSONL is excluded from this output.
- Reject partial/in-flight/changed inputs and unknown TEST lock before publication. Bind and recheck all consumed bytes.

### Task 1: Summarize accounting evidence

**Files:** create `devsupport/analysis/provider_accounting.py`, `devsupport/tests/test_provider_accounting.py`.

**Interface:** `summarize_calls(rows: list[dict]) -> dict`, `audit(directory: Path, split: str = 'dev', lock: Path | None = None) -> dict`.

- [x] Observe RED for missing helper and accounting distinctions: missing attempts versus explicit0; retry usage unknown; missing trace versus empty observed list; partial cost; negative/bool counts rejected.
- [x] Implement `summarize_calls`: invocation count, recorded transports/lower bound, unknown counts, reported usage, completeness, retry extras and estimated cost, without inferred billable counts.
- [x] Reuse `evaluation_context.audit` for main runs and `control_decisions.audit` for controls. Validate requested split/TEST lock and bind JSONL parsing to returned hashes. Keep original validators' completeness and cohort scope.
- [x] CLI `--run --output --split --lock` publishes new `summary.json` after input recheck. Existing report files are never overwritten; the nonblocking directory-creation race is recorded below.
- [x] GREEN focused tests; actual complete historical DEV12 audit; strict partial original-run rejection; no model/network calls.
- [x] Scoped Ruff, diff-check and read-only review. Explicit commit follows; helper and exact report remain required in final handoff.

## Verification

Initial helper RED: 9 assertions failed because implementation was absent. Review found a P2: a nonempty partial usage object incorrectly implied complete accounting. Three regression tests reproduced expected-false/actual-true before the fix. Positive attempts now require both input/output dimensions; explicit double zero remains complete, and retries remain incomplete.

Related accounting/context/control suite: 95 passed under Python 3.10.11 with plugins disabled. Independent read-only reviewer also observed 95 passed and no remaining P1/P2. Scoped Ruff and diff-check passed. The review's nonblocking P3 remains: output files use exclusive creation, but the parent directory is not exclusively reserved against a concurrent creator. Do not concurrently publish to the same output path.

Actual historical DEV12 output: `devsupport/reports/dev-provider-accounting-historical-v2/summary.json`; zero model/network calls. Earlier v1 is superseded because the helper's notes changed afterward. Historical observed counts total 84 stage invocations and 85 known transport attempts; these are not verified billed request counts, and configured-currency totals remain unavailable. Missing/partial usage remains explicit. The active original cohort with an in-flight record is correctly rejected, not partially published. Main frozen runtime sources/config remain unchanged.
