# Complete ablation judge cohort implementation plan

> **For agentic workers:** Execute inline with executing-plans and TDD; review before committing. This implements the already approved full ablation scope.

**Goal:** Judge every frozen generation comparison and Full-minus condition together without the arbitrary four-method restriction.

**Architecture:** Keep the existing anonymous packet builder and all provenance checks. Require at least two unique methods; actual input remains bounded by the existing 200000-byte cap and output by the explicitly selected 2048/4096/8192 protocol. A larger method set is a different comparison cohort and must have a fresh frozen generation/judge run; do not merge separate judge cohorts into a causal leaderboard.

**Tech Stack:** Existing Python 3.10 core-repro environment, pytest, Ruff; no installation or live calls.

## Global constraints

- Active DEV generation source contract (42 files), configuration, indexes and provider remain unchanged.
- Existing four-method packet bytes, mapping, prompt, rubric and historical score reproduction must remain unchanged.
- Incomplete/duplicate methods, input overflow, missing gold, operational failures, altered artifacts and TEST subsets remain rejected or explicitly accounted for.
- No actual TEST scoring/selection; synthetic TEST fixtures only. Human ratings remain zero.
- Finish offline edits before the controller starts its judge preflight/execute stage; do not mutate an active judge.

### Task 1: Generic complete comparison cohort

**Files:** `devsupport/analysis/answer_judge.py`, `devsupport/tests/test_answer_judge.py`, `devsupport/tests/test_answer_quality.py`, two analysis README files, `devsupport/docs/FINAL_EVALUATION.md`.

**Interfaces:** Existing `build_packets(questions, variants, corpus, seed)`, `load_run(run_dir, split_name)` and `answer_quality.audit(run, assessment)`; no new runtime API.

- [x] Add parameterized packet tests for 2/5/8 methods, exact texts and sealed mapping, rejection of fewer than two methods, complete load with 2/5/8 methods, duplicate profiles and a missing fifth row.
- [x] Add end-to-end synthetic five-method assessment/audit: ten paired comparisons, two attempted rows per method, preserved failed G row and Full refusal, human count zero.
- [x] Run tests in `.venv-core-repro` with `PYTHONPATH=devsupport/src`, plugin autoload disabled, `-q -o addopts=''`; observe failures specifically from the four-method restriction.
- [x] Replace packet predicate `len(variants) != 4` with `len(variants) < 2`, and loader predicate with `len(profiles) < 2 or len(set(profiles)) != len(profiles)`. Update error messages. Preserve all remaining code and prompt constants.
- [x] Run judge/quality/prepare-human/context audit suites; run scoped Ruff and `git diff --check`.
- [x] Reproduce historical DEV12 judgment summary with the updated offline auditor in a fresh directory; compare attempted means and all six pairs to the recorded historical report.
- [x] Document dynamic method count, same-cohort comparison limits and unchanged caps; reviewer checks source diff and test evidence without implementation delegation.
- [x] Recheck active 42-source contract and original prefix bytes, then commit only the explicit edited files and plan. Never stage reports/secrets.

The existing controller's four-method judge remains valid. No new calls are caused by this change. If the final method set exceeds a byte/output budget, stop before dispatch rather than truncating evidence or dropping comparisons.

Verification: RED 12 failed on the fixed-four restriction; GREEN 151 passed. Historical DEV12 packet/mapping validation and methods/per-question/all-six-pairs statistics exactly match. Read-only reviewer found no P1/P2; stale fixed-four/six documentation was corrected. Scoped Ruff and diff checks pass. Evidence: `devsupport/reports/judge-cohort-verification-20261010-0130.json`. Active contract/prefixes remained unchanged; no new model calls.
