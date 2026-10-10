# Public per-query evidence views

The approved final handoff requires complete, shareable per-query evidence while
excluding account observations and raw failed provider text. FINAL_EVALUATION.md
already specifies a whitelist-derived view. Keep private original JSONL bytes
unchanged; summaries and statistics continue to use those complete originals.

Implement an offline analysis utility with explicit nested field schemas for
pipeline, retrieval, reranking and judge records. Preserve every row in order,
original answer/decision, rankings, metrics, evidence IDs, validated judge scores,
stage diagnostics and reported usage/cost status. Remove balances, raw provider
text, validation-failure response_text, unknown fields and unknown nested fields.
Do not use blacklist-only recursion or pass arbitrary nested dictionaries.

Bind source-file bytes, each original line and original canonical record digest,
the schema/rules and derived output hashes. A derived record is explicitly not an
original runner checkpoint and cannot support resume. Validate original embedded
digest when present, finite JSON, unique IDs and all output bytes against secret
patterns/in-memory literals. Reject before publication on sensitive allowed text.
Use a fresh output directory and recheck snapshot inputs before publishing.

RED/GREEN tests: all rows/errors preserved, nested account/raw/error/unknown data
removed, exact answer and ranking retained, original and derived hashes, malformed
records/digests, nested secret failure, no overwrite, mutation before publish.
Then real complete DEV projection, independent review and ZIP selection later.

No runtime/data/index/provider configuration edits or paid calls. Human0.
