# Evaluation local dependency stop implementation plan

> **For agentic workers:** Execute inline with executing-plans/TDD and request a read-only review. Implementation is staged in the existing isolated worktree; main integration remains queued.

**Goal:** Preserve the first infrastructure failure and leave later queries/profiles unattempted when the local model dependency becomes unavailable.

**Architecture:** Detect HTTPX transport errors and HTTP5xx from the evaluation callback, checkpoint the existing sanitized error record, then stop. The CLI stops later profiles when a profile returns a partial ordered prefix. Explicit resume after recovery preserves that failed row and calls only the remaining queries. Query-specific ValueError and HTTP4xx retain their existing per-query behavior; provider failures retain the existing prepaid guard behavior.

**Tech Stack:** Existing Python/core-repro, HTTPX, pytest, Ruff; no new dependencies or paid diagnostic calls.

## Global constraints

- Do not change the **main checkout's** `devsupport/src/*.py`, `devsupport/src/devsupport/*.py` or `devsupport/scripts/*.py` while the original DEV contract still needs generation, AI-judge or control work. A terminal **partial** controller does not release this constraint: it still needs the original source hashes to resume. Offline implementation/tests can use the clean existing `F:/DevSupportCopilotWork/reporter-fixes` worktree on branch `evaluation-local-outage-stop`, based on `50dad77`; this worktree is not used by active services or the paid controller.
- Earliest integration is after complete DEV48 generation, DEV12 assessment and four methods × DEV10 control attempts, with the associated controller terminal. Integrate before final TEST freeze.
- All existing record/manifest integrity, uncertain-call markers, budget continuity, query order and no replay of completed rows remain enforced.
- This protects callback failures in evaluation execution; it is not a retrieval-quality improvement. Existing reranker/adaptive fallbacks that handle exceptions internally remain unchanged and observable. Do not claim all fallback paths stop.
- No TEST scoring or model/balance calls in tests. Synthetic questions and callback failures only.

### Task 1: Stop after the durable dependency-failure row

**Files:** modify `devsupport/src/devsupport/evaluate.py`, create `devsupport/tests/test_local_dependency_stop.py` from the queued regression at `F:/DevSupportCopilotWork/test_local_dependency_stop_regression.py`.

**Interface:** `run(questions, execute, output, manifest, resume=False, guard=None)` keeps its signature and record schema.

- [x] Reproduce current behavior offline: transport/503 callbacks yield three failed rows, and CLI executes later profiles. Corrected fixture supplies dimension1024. Final RED command reports exit1, **7 failed / 1 passed**, with the failures at the intended stop assertions. Runtime sources unchanged.
- [x] Copy regression into the isolated worktree; rewrite the temporary hardcoded script path as `Path(__file__).parents[1] / 'scripts/evaluate_pipeline.py'`; add HTTP404/422 continuation coverage.
- [x] Initialize `halt_after_record = False` for each query. Share a `dependency_unavailable()` classifier between the runner and CLI:

```python
try:
    import httpx
except ImportError:
    return False
return isinstance(exc, httpx.TransportError) or (
    isinstance(exc, httpx.HTTPStatusError) and 500 <= exc.response.status_code <= 599
)
```

- [x] After the existing append, flush/fsync, marker removal and row append:

```python
if halt_after_record:
    if guard is not None:
        guard.stopped = True
    break
```

- [x] Keep the error class only; do not add raw exception messages, response bodies, new gold access or fabricated successful results. Verify resumed prefix bytes are identical and query0 is never replayed.

### Task 2: Stop remaining CLI profiles

**Files:** modify `devsupport/scripts/evaluate_pipeline.py`, same regression test file.

**Interface:** A callback wrapper records whether a dependency exception escaped. After writing the existing versioned profile summary and print, before the next profile, add:

```python
if local_dependency_failed or len(result) < len(ids):
    break
```

- [x] Run the queued CLI test against the real entry point using fake adapters; require the attempted B0 prefix only and no B1 artifact, including an outage on the final query. HTTP422 still permits the next profile. No network/model provider is involved.
- [x] Run `test_local_dependency_stop.py` and the actual `test_evaluate.py` with core-repro, isolated source PYTHONPATH and plugin autoload disabled.
- [x] Run focused context/judge/quality tests: **219 passed** after the optional-dependency regression. Complete historical DEV12 methods/paired/per-question/IDs remain exactly equal, zero API calls.
- [x] Scoped Ruff and `git diff --check` pass. Initial read-only review found P2 optional HTTPX import; reproduced RED `ModuleNotFoundError` before durable append, fixed and GREEN.
- [x] Re-review accepted with no remaining P1/P2; independent 74 focused tests passed. Commit explicit files in the isolated branch.
- [ ] After the release condition, integrate the staged commit and verify main. Record new source identity in the final TEST lock; do not edit historical manifests.

Queued initial RED evidence is at `F:/DevSupportCopilotWork/test_local_dependency_stop_regression.py`; final-query and missing-HTTPX regressions were also observed RED before the isolated fixes. Main source remains authoritative until the original controller has fully finished. Staged green verification does not mean the running main process has this fix.
