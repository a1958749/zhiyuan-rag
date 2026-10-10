# Offline answer judge

`--decision-label-protocol explicit_coverage` adds explicit required nugget IDs
to the earlier `explicit` decision-label protocol. Every eligible review,
including an empty abstention, must cover exactly those IDs. The evaluator
chooses the coverage labels and scores; none are filled or corrected locally.
The strict validator and rubric are unchanged. This defines a fresh whole-cohort
assessment, not a retry or a pool with earlier partial batches. Both legacy and
explicit prompt bytes remain supported unchanged; the protocol and prompt hash
are bound in the manifest and checked by `answer_quality.py`.

This utility assesses existing complete generation artifacts. It never calls retrieval or generation with gold answers. Its labels are **AI judgments**, not human review or independent model validation: the configured gateway reports an alias and its underlying model identity is unknown.

Use `--decision-label-protocol explicit` for the revised decision-label input contract in a fresh assessment
directory. Each eligible anonymous response receives allowed refusal labels derived only from its recorded
decision: answer allows none; clarify/insufficient allows supported_abstention or false_refusal, including empty
answer text. The judge still chooses the evidence-supported nonanswer label and all scores; no label is repaired
locally and the strict validator is unchanged. This only exposes an existing format invariant, not a new scoring
rule or guarantee of grading quality. The option is bound into the manifest, actual prompt and quality audit.
The default legacy protocol retains the exact previous prompt bytes; old manifests without the field remain
auditable. Changing protocol requires a new whole-cohort assessment, and selected scores from different runs
must not be pooled. Unknown protocols are rejected before creating paid output or invoking a provider.

Comparison cohorts require at least two unique methods, all complete and identically ordered. There is no fixed four-method limit: final baselines and every retained component's Full-minus condition can share one frozen cohort. Existing four-method packet bytes, seeded mapping, prompt and rubric are unchanged. Adding or removing methods changes the blinded comparison cohort and requires a fresh generation/judge directory and final lock; do not pool separately judged cohorts into one fair leaderboard. Larger cohorts may exceed the unchanged input-byte or selected output-token budgets: preflight rejects oversized input without truncating evidence, and output/schema failure remains a terminal recorded failure. This extension does not guarantee valid or accurate judgments.

Run from the repository root using the existing core Python environment:

```powershell
# Offline: validates the full cohort and byte caps; no credentials or network.
F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe devsupport/analysis/answer_judge.py --run devsupport/reports/dev-generation-v2 --output devsupport/reports/dev-judge-preflight

# Paid: one call per question, all blinded methods together, no retries.
F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe devsupport/analysis/answer_judge.py --run devsupport/reports/dev-generation-v2 --output devsupport/reports/dev-judge --execute
```

The default split is `dev`. After the frozen TEST generation has completed, explicitly pass `--split test` and its run directory; only the frozen generation cohort is loaded from the query Parquet. At least two unique, complete, identically ordered generation variants are required. Each manifest's profile must match its filename, its settings must match that profile in the frozen contract, and its variant must match the configured reranker or retrieval method. Missing, swapped or duplicated profile metadata fails before judging. A preflight directory is distinct from an execution directory. Add `--resume` only to resume a matching execution directory.

For TEST, the embedded split must also match the hash-bound split file, and the
generation cohort must equal that file's first 30 TEST IDs in order. Membership
in TEST alone is insufficient; a replacement or reordered subset is rejected.

The utility takes the provider, shared pacing path, and budget authorization from the generation contract. It requires at least 65 seconds shared pacing, uses the same `.accounting.json` budget state, forces transport retries to zero and defaults to a 2048-token output cap. `--output-tokens` accepts only 2048, 4096 or 8192; other values are rejected before paid work. The selected judge-only limit is bound into provider configuration, the actual request, prompt instruction, manifest and offline preflight. The shared generation provider limit remains unchanged. It checks every packet against the configured input-byte cap before paid work; no reference, nugget or evidence text is truncated. Missing gold evidence fails closed.

A larger output budget defines a new judge protocol and requires a fresh run directory; changed-limit resume is rejected. It does not improve the original answers and does not guarantee valid judgments: output may still be spent on stripped reasoning or fail schema validation. The prior failed 2048-token run remains immutable and must not be replayed automatically. For an offline 8192-token check, add `--output-tokens 8192` to the preflight command and use a fresh preflight directory. Any later paid diagnostic must use a separate fresh execution directory with the same explicit limit.

`manifest.json` binds input bytes, all local DevSupport dependency sources plus the judge source, provider settings, prompt, rubric, seed, and packet/mapping hashes. `packets.json` omits method labels, retrieval scores, and evidence metadata. `sealed-mapping.json` must stay out of the judge prompt. Exact question, reference, nugget and evidence texts remain intact. Review IDs and nugget IDs are assigned locally; source answer text may still contain stylistic clues, so this is label blinding rather than guaranteed identity concealment.

`results.jsonl` retains validated scores and only whitelisted provider output/accounting fields, using text already stripped by the provider adapter. Provider failures also retain a whitelisted local `provider_error_code` (for example `output_truncated` or `http_429`) and available already-sanitized HTTP retry-after/code/type diagnostics. Raw bodies, headers, reasoning and credentials are excluded; successful response whitelisting is unchanged. Billed schema/provider failures are checkpointed and stop the run. They never replay automatically; a crash leaves `inflight.json` requiring explicit billing/result reconciliation. Do not delete markers just to resume. Prior output and summaries are never overwritten. Numeric balance snapshots are separate files. `final_balance_reconciled` is true only when the final guarded balance check completes successfully. A stopped guard is never reactivated or bypassed: after a judge failure, the snapshot can contain only pre-call observations and records `final_balance_reconciled: false` with an accounting note. The last potentially billed call's cost may still be pending; these observations must not be treated as the final spend. Any other incomplete final check also conservatively records false, even if it obtained a balance before reaching a budget limit.

Each valid response receives integer 0–3 correctness, completeness, faithfulness, citation support and answer relevance, a concise evidence explanation (target at most 240 characters, hard maximum 1000 characters), explicit refusal classification, and a coverage label for every nugget. Operational generation failures receive no quality judgment and contribute zero attempted utility. Conditional quality excludes those failures. Missing AI judgments are counted and make full attempted utility unavailable. Utility is the unweighted mean of the five scores, not a calibrated human-quality estimate. Supported abstention may score well for faithful/relevant behavior while still missing reference nuggets. A multi-answer prompt can have position/contrast bias despite seeded shuffling; no repeated judge calls or human agreement claim is made.

Offline tests:

```powershell
$env:PYTHONPATH='F:/DevSupportCopilot/devsupport/src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe -m pytest devsupport/tests/test_answer_judge.py -q -o addopts=''
```

The 1000-character explanation maximum is a revised offline judge protocol. Reasons must contain at least one nonwhitespace character; the hard maximum counts the entire returned string, including surrounding whitespace. Accepted explanations are preserved exactly, without truncation, rewriting or normalization. All scoring and other schema constraints remain unchanged. Previous failed 2048- and 8192-token runs remain terminal and unchanged; after review, explicitly choose a fresh directory for any new run. A new run is not automatic replay, does not improve original answers, and this explanation-bound change does not establish model grading correctness.

The eligible-input correction is a revised offline judge protocol requiring an explicitly selected fresh run directory after review; previous runs remain terminal and unchanged. The archived canonical `packets.json` and sealed mapping retain every response, their operational status and full context. Only the prompt copy filters out operational failures and their answer/context contents. Its explicit `allowed_review_ids` list contains exactly the eligible response IDs; the judge must return exactly that set. Question, reference, nuggets, gold evidence and eligible response text/context remain exact and complete. The byte cap applies to the actual serialized prompt. All-failed questions still make zero provider calls. Validation continues to reject failed, unknown, extra, duplicate or missing IDs; excess reviews are never silently dropped. Filtering prompt inputs does not remove failures from the attempted denominator: each remains an operational failure with zero attempted utility and no conditional quality judgment. This is still AI judging, not human review, and the correction does not establish grading accuracy.
