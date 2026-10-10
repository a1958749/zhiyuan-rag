# DevSupport Copilot approved design

Approved for continuous implementation by the user's follow-up and attached specification on 2026-10-08. Product: an AI developer-tools troubleshooting and evidence QA assistant, based on the real DocsGPT UI and backend, not a replacement chatbot.

## Binding requirements

1. Preserve DocsGPT chat, upload, citation UI and native source authorization. Implement an adapter to a shared DevSupport retrieval package used by both UI and evaluation.
2. Use the full FreshStack `corpus-oct-2024/langchain` corpus (49,514 official chunks) and `queries-oct-2024/langchain` (203 questions). Preserve `_id`, exact evidence text, URL and UTF-8 byte offsets. Do not re-chunk, remove distractors or add answers/nuggets/qrels to the online corpus.
3. Freeze corpus/query revisions, checksums, upstream commit, models, prompts, candidate budget and generation context budget. Group near-duplicate questions into one split; aim for 53 development / 150 local holdout questions and disclose any grouping-driven count deviation.
4. Provide native upstream U, controlled dense B0, ordinary BM25 B1, ordinary hybrid H0, identifier-aware hybrid H1, Qwen3 pairwise R1, Jina v3.5 listwise R2, adaptive Q, evidence-gated G and optional contextual-index C configurations. Do not stack R1 and R2.
5. Dense model: Qwen/Qwen3-Embedding-0.6B on the user's RTX 5070 Ti; CPU FAISS for vector search and BM25S for lexical search. Documents and queries use consistent normalization/dimensions and appropriate query instructions. Preserve input truncation statistics.
6. Retrieval: 50 candidates per initial retrieval leg, RRF constant 60, deduplicate by official ID, cap reranker input to the same 50-item candidate pool for R1/R2. Final evidence context budget initially 4,000 generation-model tokens; measurement must distinguish pre-packing ranked metrics and post-packing context coverage.
7. Identifier lexical analysis preserves complete API names, snake_case, camelCase, error codes and versions and also emits controlled subterms without stemming code identifiers.
8. Adaptive retrieval keeps the original query, generates at most two subqueries, performs at most one additional retrieval round, has no access to gold annotations and records trigger/fallback/cost. Compare never / always / selective decomposition independently of final answer gating.
9. Final answer gate outputs answer / clarify / insufficient, reports missing information and citations, and never labels a raw rerank score a confidence probability. Evaluate unsupported answers, false refusals and answer coverage together.
10. Contextual indexing, if retained after dev-only selection, changes only the index representation using factual metadata/available headings. Evidence IDs and original text remain unchanged. ModernColBERT is a bounded optional challenge, not a mandatory substitute for the agreed models.
11. Answer model remains the same paid Opus model for comparable answer experiments, once the user identifies its API provider/base URL. Never send the provided credential to a guessed hostname. Credentials remain ignored/encrypted local runtime state and must be absent from Git, logs, reports and archives.
12. Test retrieval ID integrity, fusion/dedup/bounds, retry/fallback behavior, subquery bounds, citation provenance, split isolation and evaluator formulas with meaningful red/green tests. Record original upstream/environment failures separately.
13. Evaluation: candidate Recall@50, alpha-nDCG@10, Coverage@10/20, MRR, nDCG, Recall@5/10, post-budget context precision/coverage, answer correctness/completeness/citation support/faithfulness, refusal behavior, per-stage and end-to-end p50/p95 and actual cost. Use paired differences and bootstrap intervals; do not present tuning on the holdout as untouched testing.
14. Forty additional control examples: 20 supported, 10 corpus-unsupported, 10 clarification-needed; 10 development and 30 holdout. Make labels and provenance inspectable. Human blind review of at least 30 main-test questions must be distinguished from AI judge review; never fabricate human ratings.
15. Deliver a runnable modified DocsGPT repository, setup/launch/stop/data/index/evaluation commands, environment locks, per-query and summary results, ablation/removal experiments, failure analysis, concise PRD, architecture and decision records, five-minute demo video and truthful CV template.
16. Target gains are hypotheses, not completion facts: +5 percentage points Coverage@10 over B0, +0.03 alpha-nDCG@10 over ordinary hybrid. Report actual positive, negative and inconclusive results. A component is selected on development data only.
17. Scope excludes UI rewrite, new auth system, auto-executing fixes, GraphRAG, multimodal/multi-agent product features, MCP, long-term memory, fine-tuning and live crawling.

## Current environment evidence

Windows has Python 3.10.11, PyTorch 2.6.0+cu124 and transformers 4.57.6. A real CUDA tensor operation fails because that Torch build lacks sm_120 kernels for the RTX 5070 Ti. Use a task-specific newer CUDA-12.8 Torch environment; do not alter the global installation. WSL distro `Ubuntu2204-F` is Ubuntu 22.04, systemd enabled, no Docker installed; current apt metadata offers Docker 29.1.3 and Compose 2.40.3. The upstream app requires Python >=3.12.

## Architecture and interfaces

`devsupport/src/devsupport/` is a separate Python package within the upstream repository. Core evidence schema: `Evidence(doc_id: str, text: str, metadata: dict)`. Search result: `Hit(evidence: Evidence, score: float, rank: int, score_kind: str)`. Benchmark schema: `Question(query_id, query_title, query_text, answer_text, nuggets, metadata)` is offline only. The pipeline accepts only the query string and a runtime config, not a Question object.

Retrieval and evaluation share `Retriever.search(query: str, k: int) -> list[Hit]`. Online service exposes health, search, evidence-by-ID and grounded answer routes; DocsGPT adapter converts Hit to the upstream standard source structure and preserves provenance. The UI opens stored snapshot text for citations, and original URLs remain external references.

## Acceptance audit

Every binding requirement must be linked to an authoritative command output, reviewed code, dataset manifest, real run artifact or rendered UI/video. Test success alone does not prove model experiments or product behavior. API provider resolution and actual human blind ratings are external inputs, not values to invent. Keep completion unproven while a required item has missing evidence.

## Verified implementation clarifications

User selected F:/DevSupportCopilot and authorized https://www.xiaoyaoapi.com/v1. An early native Messages smoke returned DEV_SUPPORT_OK from claude-opus-5-5. The subsequent user-authorized gateway requests deepseek-v4-pro over chat completions and reports claude-opus-4-8; underlying model identity is unverified. Comparative runs must share this frozen route and must not mix the early smoke with current experiments. Raw corpus contains 49,514 rows, 49,505 unique IDs, two duplicate groups and one conflicting group. Preserve the raw archive; use explicit audited official_last selection matching the official FreshStack loader. All 203 questions and 639 nuggets remain intact. Existing Windows PostgreSQL owns port 5432; isolated project infrastructure uses 5433 and Redis 6380. A live WSL service maintains infrastructure while native Windows services run.

Before holdout, the provisional 4,000 UTF-8 byte packer was replaced by a common 4,000 cl100k_base_proxy_v1 token budget including source-table formatting. This is a documented proxy because the gateway generator tokenizer is unverified. The DEV packing diagnostic is a protocol correction, not a retrieval-method improvement. Historical byte runs remain separate. All final methods use the same proxy counter and tokenizer SHA256.
