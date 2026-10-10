# DevSupport Copilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the approved developer-support RAG product, real model ablations, grounded answer workflow and career-demo artifacts.

**Architecture:** Extend the real DocsGPT application with a shared retrieval/answer service and a separate Python package for immutable FreshStack data, model inference and evaluation. Keep task-specific GPU/runtime dependencies separate from the upstream app. Preserve the original UI, native upload and citations.

**Tech Stack:** DocsGPT/Python >=3.12/React/Postgres/Redis; DevSupport Python >=3.10, FAISS CPU, BM25S, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, Jina reranker v3.5, a fixed paid Opus provider, pytest, FreshStack metrics.

## Global Constraints

- Preserve official corpus IDs, raw text, URLs and byte offsets; archive all 49,514 raw rows. Explicit audited official_last canonicalization yields 49,505 IDs matching the official loader; no distractor subset or new IDs.
- Answers, nuggets and qrels are offline-only annotations; the live pipeline accepts query text and config only.
- Keep development and holdout disjoint by near-duplicate groups; choose all thresholds/models/prompts using development only.
- Candidate pool 50 for equal reranker comparisons; 50 per retrieval leg; RRF 60; initial final-context budget 4,000 generation tokens.
- Maximum two subqueries, one additional retrieval round; retain original query; final gate is independent.
- All online/evaluation modes share the same retrieval implementation, model settings and artifact IDs.
- No credential in source, Git, logs, reports or distributable archive; only a user-identified host may receive the Opus credential.
- No invented measurements or human reviewers; identify unavailable APIs, unrun models and pending human labels explicitly.
- The user-selected work directory is F:/DevSupportCopilot, with task-only environments under F:/DevSupportCopilotWork; final handoff artifacts are exported into the original task outputs directory. No global Python upgrades or unrelated filesystem cleanup.

## Task 1: Immutable data, benchmark evaluator and shared types

**Files:** `devsupport/pyproject.toml`, `devsupport/src/devsupport/{__init__,types,data,metrics}.py`, `devsupport/tests/{test_data,test_metrics}.py`, `devsupport/tests/fixtures/mini_freshstack.json`, `devsupport/configs/data.json`.

**Consumes:** official Parquet rows with `_id/text/metadata` and question rows with `query_id/query_title/query_text/nuggets/answer_text/metadata`.

**Produces:** Evidence and Hit dataclasses; `load_corpus(path) -> dict[str,Evidence]`; `load_questions(path) -> list[Question]`; `build_split(questions, dev_count=53, seed=20261008) -> dict`; `evaluate_query(ranked_ids, query) -> dict`; `bootstrap_paired(a, b, seed=20261008) -> dict`. Canonical manifests serialize hashes/counts/revisions without annotations in corpus records.

- [ ] Write focused failing tests including this evaluator contract:

```python
def test_recall_is_not_any_hit():
    q = {'nuggets': [{'text':'a','relevant_corpus_ids':['a','b']},
                     {'text':'c','relevant_corpus_ids':['c']}]}
    m = evaluate_query(['a','x'], q, ks=(1,2))
    assert m['recall@1'] == 1/3
    assert m['coverage@1'] == 1/2
    assert m['mrr'] == 1.0
```

- [ ] Run `python -m pytest devsupport/tests/test_data.py devsupport/tests/test_metrics.py -q` and record missing-feature failures.
- [ ] Implement pure data loading and metrics; alpha-nDCG uses per-nugget novelty discount alpha=.5, union-document recall and exact official IDs. Validate against the official FreshStack evaluator on a deterministic fixture.
- [ ] Implement deterministic group-disjoint splits and original-text identity checks, missing-qrel-ID reporting, truncation metadata and paired bootstrap.
- [ ] Run focused suite, inspect output, self-review, commit only owned files and write a task report outside tracked production data.

## Task 2: Task environments, real data download and GPU baseline

**Files:** `devsupport/scripts/{bootstrap.ps1,download_data.py,environment_probe.py}`, `devsupport/configs/runtime.example.json`, `devsupport/requirements-*.txt`, `devsupport/reports/environment.json`.

**Consumes:** Task 1 data types; user's current Windows/WSL environment. **Produces:** real frozen data Parquets, manifests, working sm_120 GPU inference environment, reproducible dependency locks.

- [ ] Reproduce CUDA failure; query the official cu128 wheel index; create a new task venv and install a compatible Torch build without changing global Torch.
- [ ] Install Docker/Compose in the existing WSL distro through its available maintained apt repository; verify engine/Compose and real image execution. Prepare upstream Python >=3.12 runtime.
- [ ] Download the exact FreshStack langchain artifacts, record SHA256/revisions and run Task 1 checks across all corpus/questions. The environment probe assertion is `assert torch.ones(16, device='cuda').sum().item() == 16`.
- [ ] Download Qwen3 model artifacts into ignored cache, run actual embedding of two texts, verify 1024 dimensions, finite normalized outputs and real CUDA execution.
- [ ] Freeze actual package versions and record the original environment failure and fix evidence; commit scripts/locks/reports, never data/cache/credentials.

## Task 3: Dense, lexical, identifier-aware fusion and contextual indexing

**Files:** `devsupport/src/devsupport/{embedding,index,lexical,retrieval,config}.py`, `devsupport/tests/test_retrieval.py`, `devsupport/configs/{B0,B1,H0,H1,C}.json`.

**Consumes:** Task 1 types and frozen corpus; Task 2 GPU model. **Produces:** `Embedder.encode(texts, is_query=False)`, `CorpusIndex.build/load`, and `Retriever.search(query,k)` returning Hit; persisted raw/contextual indexes with model/dimension/tokenizer/data checksums.

- [ ] Test failure first: `assert 'jq_schema' in identifier_tokens('JSONLoader jq_schema TS-999')`; assert whole `JSONLoader` and useful subterms coexist; exact ID de-dup and candidate bounds survive overlapping legs.
- [ ] Implement FAISS normalized vector search, BM25S ordinary and identifier tokenizers, RRF fusion with stable tie-breaking, no stemming of API symbols.
- [ ] Implement optional factual metadata/heading prefix for index representation only; ensure `Evidence.text` remains byte-for-byte identical and IDs unchanged.
- [ ] Run focused tests and index the full corpus. Compare development B0/B1/H0/H1/C with real models; cache ranked IDs, scores and stage timings.
- [ ] Commit code/configs with independently verified TDD and identity evidence.

## Task 4: Pairwise and listwise rerankers

**Files:** `devsupport/src/devsupport/{reranking,model_service}.py`, `devsupport/tests/test_reranking.py`, `devsupport/configs/{R1,R2}.json`.

**Consumes:** identical 50-item candidate IDs/text and query from Task 3. **Produces:** `Reranker.rank(query,hits) -> list[Hit]`, local Qwen3 and real Jina-v3.5 implementations, annotated fallback/errors/timing; rankers must not invent or add corpus IDs.

- [ ] RED contract: a ranker response referencing an unknown ID raises; a failed remote call preserves the original candidate list with explicit fallback metadata.
- [ ] Implement actual Qwen3 yes/no scoring with pinned supported model instructions. Implement Jina v3.5 using an authorized API if one is available, otherwise its official local model code after inspecting required dependencies; do not substitute a different algorithm while labeling it R2.
- [ ] Compare both on identical cached development candidates; check listwise order sensitivity on fixed seeds, input truncation and memory behavior.
- [ ] Freeze the chosen reranker using dev metrics/cost, then run the focused tests and commit.

## Task 5: Paid-model provider, bounded adaptive retrieval and independent evidence gate

**Files:** `devsupport/src/devsupport/{llm,adaptive,gate,answer,pipeline,credentials}.py`, `devsupport/tests/{test_adaptive,test_gate,test_citations}.py`, `devsupport/configs/{Q,G,Full}.json`.

**Consumes:** query text, Retriever, Reranker, authorized Opus base URL/model and ignored credential. **Produces:** pipeline result with answer/clarify/insufficient decision, evidence/citations, missing info, trace, usage/cost and bounded retries; no offline Question fields are accepted.

- [ ] RED invariant: `assert result.trace.additional_rounds <= 1` and `len(result.trace.subqueries) <= 2`; malformed decomposition falls back to the original query.
- [ ] Implement provider protocol detection only at the user-identified host, safe secret access, bounded timeout/retry, usage recording and sanitized errors. Use one paid Opus model/prompt across answer comparisons.
- [ ] Implement never/always/selective decomposition retaining original identifiers, runtime-only triggers, 4,000-token evidence packing, and independent final gate.
- [ ] RED citation invariant: a generated unknown citation ID is rejected/repair-limited, never rendered as valid. Zero evidence results in insufficient; missing decision-critical version/language can result in clarify.
- [ ] Run real authorized API smoke, focused tests, 40 provenance-backed control cases and bounded-cost dev trials; commit code and safe records only.

## Task 6: Real DocsGPT UI and retriever integration

**Files:** `docsgpt/retriever/devsupport.py`, minimal registration/config edits, `devsupport/src/devsupport/service.py`, `devsupport/scripts/{start.ps1,stop.ps1,smoke_ui.py}`, relevant focused integration tests.

**Consumes:** shared pipeline and original source authorization. **Produces:** working original DocsGPT chat/upload/citation interface using exactly the evaluated pipeline, snapshot evidence previews and visible knowledge date, retriever configuration and trace metadata.

- [ ] Trace original Dispatcher/BaseRetriever/agent source contracts before editing. Write the failing adapter contract test for official ID/raw snapshot/source URL preservation.
- [ ] Extend native retriever registration and source creation in the smallest applicable entry points; preserve native auth, upload and chat behavior.
- [ ] Run original U and integrated modes from the same UI. Verify actual browser question, answer, citation preview, upload and insufficient/clarification behaviors.
- [ ] Verify Windows start/stop commands, model-service health and app health. Do not replace the agreed UI with a custom fallback UI.
- [ ] Record actual rendered UI evidence and focused test results, review and commit.

## Task 7: Frozen holdout ablations, review and reports

**Files:** `devsupport/src/devsupport/{evaluate,report,review}.py`, `devsupport/configs/evaluation.json`, `devsupport/reports/` summaries/per-query traces, `devsupport/review/` blank human-review templates and provenance-backed controls.

**Consumes:** development-selected frozen configs, untouched holdout IDs, real model/service outputs. **Produces:** all named U/B0/B1/H0/H1/R1/R2/Q/G/C applicable experiments, Full-minus-component removal results, paired bootstrap confidence intervals, generation/cost/latency results, failure classification and honest source of ratings.

- [ ] RED evaluator check: a configuration using annotation-containing online input is rejected; test/holdout config changes after lock are separately labeled follow-up experiments.
- [ ] Run full retrieval tests and generation B0/H1/selected-R/Full with identical answer prompts/model/context budget. Report every fallback, error, refusal and denominator.
- [ ] Inspect at least 30 question outputs independently of method labels; save AI-assisted inspection separately. Provide blinded human-review UI/templates and request actual human ratings asynchronously when available; never claim they were human-supplied without evidence.
- [ ] Generate metric deltas, intervals, side effects, both positive and negative cases; check target gains as hypotheses, not hardcoded values.
- [ ] Run the broad integration suite once after changes are frozen; fix findings with focused regressions and a final reviewed commit.

## Task 8: Career deliverables, real five-minute demo and completion audit

**Files:** top-level project README, `devsupport/docs/{PRD,ARCHITECTURE,DECISIONS,EVALUATION,FAILURES,INTERVIEW,CV}.md`, real UI captures and `devsupport-demo.mp4`, deliverable archive, `devsupport/reports/acceptance.json`.

**Consumes:** authoritative run artifacts and rendered UI. **Produces:** Chinese handoff explaining build/run/data/index/evaluation/reproduction, five-minute real-interface demonstration, truthful CV numbers and requirement-by-requirement evidence ledger.

- [ ] Show identifier-sensitive success, compound-question recovery, insufficient/clarification, experiment tradeoffs and unresolved failure. Capture the actual UI; do not fabricate screenshots or measurements.
- [ ] Verify video duration/playback and deliverable contents, exclusions of credentials/cache/private runtime, source/benchmark/model license notices and upstream provenance.
- [ ] Dispatch independent final architecture/spec/code review with the immutable diff and evidence ledger; resolve Important/Critical findings.
- [ ] Inspect every design requirement against real current evidence. Keep the goal active for missing authorized API connectivity, required model experiments, actual human ratings or any other required artifact; do not redefine completion.

## Execution ledger

Task status and authoritative handles are recorded in `work/devsupport-progress.json` in the parent task workspace. Revalidate live processes before polling or restarting. Maintain a short commentary at least every minute; user requested uninterrupted execution, so never ask whether to continue.
