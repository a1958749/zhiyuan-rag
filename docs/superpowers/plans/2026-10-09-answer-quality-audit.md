# Answer Quality Audit Implementation Plan

> **For agentic workers:** Apply the existing approved DevSupport evaluation design; use executing-plans and test-driven-development for this single offline task. An independent reviewer checks the finished diff.

**Goal:** Reproduce and compare complete four-method AI assessments without discarding generation failures or accepting changed inputs.

**Architecture:** `analysis/answer_quality.py` loads the complete generation through the existing judge loader, regenerates its blinded packets and mapping, validates the saved judge manifest and each row, and checks the saved summary. It publishes all-attempt utility, response/answer counts, conditional dimensions, paired bootstrap intervals and input hashes. No model, credential, balance or HTTP call occurs.

**Tech Stack:** Existing Python 3.10 core-repro, stdlib, existing pyarrow loader and bootstrap_paired; no dependency changes.

## Global Constraints

- Approved spec: `docs/superpowers/specs/2026-10-08-devsupport-design.md`, evaluation task in the existing implementation plan.
- Do not alter `devsupport/src`, `devsupport/scripts`, any prompt, data, config or active experiment files.
- AI ratings stay AI ratings, underlying model identity unverified, no human validation claim.
- Require complete, identically ordered generation and successful complete judge recording. TEST must be the frozen first 30 generation IDs; DEV uses its frozen selected cohort.
- Failures contribute zero utility, legitimate abstentions retain AI grades, missing judges make publication fail. Show answer coverage alongside utility; neither is accuracy.
- Default comparisons are all six distinct pairs in the run's fixed method order. The seed is the recorded seed and bootstrap uses 2,000 samples. Descriptive, no multiplicity correction or causal/population assertion.
- Publish exclusively to a new directory. Recheck consumed input hashes before creating output.

## Task: validated quality report

**Files:** Create `devsupport/analysis/answer_quality.py`, `devsupport/tests/test_answer_quality.py`, and `devsupport/analysis/answer_quality.README.md`.

**Interfaces:** `audit(run: Path, assessment: Path, split: str = 'dev') -> dict` returns the validated complete report; `publish(run: Path, assessment: Path, output: Path, split: str = 'dev') -> dict` writes `summary.json` exclusively after input rechecks.

- [x] Write end-to-end synthetic fixtures using real parquet, generation records, blinded packets and the existing judge checkpoint writer with a fake in-memory provider. Assert failed G contributes zero while Q-H0 remains exactly +2; Full's legitimate false-refusal proxy lowers answer coverage but is not an operational failure.
- [x] Run `core-repro python -m pytest -c devsupport/pyproject.toml devsupport/tests/test_answer_quality.py -q`; observe failure because the report utility is absent.
- [x] Implement source/input/manifest/packet/mapping/row/summary validation using `answer_judge.load_run`, `build_packets`, `validate_review`, `summarize_judgments`, `digest` and `sha256_file`. Build per-question four-method utility and all six `bootstrap_paired` comparisons, plus valid answer-only score means with their own denominator.
- [x] Add corruption cases: generation/assessment in-flight, partial or reordered judge rows, judge_error, row checksum, packet hash, method mapping, summary, source input bytes, prompt/rubric identity, provider settings, and wrong split. Recomputed checksums must not make invalid scores acceptable.
- [x] Check publication rejects a changing input and an existing output without overwriting it. Confirm the utility never constructs a provider or reads credentials.
- [x] Run the new tests with existing answer-judge and human-packet tests; run scoped Ruff and git diff checks. Request an independent review of the narrow diff and resolve findings.
- [x] Reproduce the historical DEV12 v4 report and compare known utility means/deltas against the preserved manual audit. Integrate only the new analysis/tests/docs after verification; verify all 42 active runtime source hashes unchanged.

Minimal behavioral assertion:

```python
report = quality.audit(generation_dir, judge_dir)
assert report['methods']['G']['operational_failures'] == 1
assert report['methods']['G']['attempted_utility_mean'] == 1
assert report['paired']['Q-minus-H0']['mean_delta'] == 2
assert report['methods']['Full']['answer_coverage'] == .5
```

No product UI, generation protocol or method-selection policy is added by this task.
