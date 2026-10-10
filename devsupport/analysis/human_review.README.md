# Offline human blind review

This standalone evaluation artifact collects human judgments. It is not the
DocsGPT product UI, an AI judge, or evidence that human review has occurred.
It uses only the Python standard library and makes no provider/network calls.

For the final handoff, use the frozen-run preparation entry point after all four
generation profiles are complete (this loader additionally needs the existing
core environment's PyArrow and DevSupport data dependencies):

```powershell
& 'F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe' devsupport/analysis/prepare_human_review.py --run 'F:/path/to/completed-test-generation' --split test --output 'F:/path/to/new-human-handoff'
```

It validates profile identity, record integrity, annotations, and the exact
ordered first 30 TEST IDs in the bound split. Share only its `share/` directory,
containing `review.html`, `packets.json`, and instructions. Keep `private/` sealed:
it contains the method mapping and provenance manifest. No ratings are filled in.
Operational failures are marked explicitly and skipped by the scoring form.
`--split dev` produces a rehearsal clearly labeled as such, never TEST acceptance.
Existing output directories and incomplete generation runs are rejected.
Preparation makes no provider calls and does not execute the AI judge.

The legacy `report_evaluation.py --review` output wraps packets in an object and
does not populate the form's explicit operational-failure flag. Use the entry
point above for the human form rather than passing that legacy wrapper directly.

From the repository root (PowerShell):

```powershell
& 'F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe' devsupport/analysis/human_review.py --packets 'F:/path/to/blind-packets.json' --output 'F:/path/to/human-review.html'
```

Open the HTML in a browser. Read the Chinese instructions and rubric, enter a
name/pseudonym, score each anonymous response, and make the explicit human
declaration. The answer is displayed as exact plain text; references, nuggets,
gold evidence, retrieved evidence, and citations are expandable.

Input is the JSON list from `devsupport.review.blind_review`, without its sealed
mapping. Question/response fields are validated strictly. `ratings` and
`reviewer_notes` may be absent or null, never prefilled. Optional boolean
`operational_failure` marks service failures: these get a prominent warning,
no rating controls, and `skip_reason: "operational_failure"` in the export.
Their five ratings remain null; they are excluded from the scoring denominator.
Explicit method/model metadata keys are rejected recursively. Scoring metadata
is also rejected at every object/list depth, including expandable evidence and
gold annotations: keys containing the terms `score`, `scores`, `confidence`,
`confidences`, `probability`, `probabilities`, `logprob`, or `logprobs` are reserved.
Terms are separated by punctuation/underscores or camel case, so `retrieval_score`,
`rerank_score`, `judge_scores`, and `similarityScore` are rejected before HTML is
written. This field-name boundary leaves ordinary string values (including code
that mentions scores), corpus metadata such as URLs/byte offsets, evidence IDs,
citation numbers, and gold nugget relevance annotations unchanged.
The generator cannot detect identities or scores hidden in arbitrary prose or
unrecognized field names; the packet producer must provide properly blinded
content. It does not read or sanitize a sealed map.

All questions in the supplied packet are in scope. Each eligible response needs
five integer scores (0–3). Export uses schema `devsupport.human-review.v1`, the
SHA-256 of the exact input file bytes, reviewer, declaration, UTC timestamp,
anonymous rating rows, and computed completion counts/status. Missing scores,
name, or declaration keep status `partial`; skipped failures are counted
separately. `complete` means the review form is finished, not successful answer
generation. A packet consisting entirely of failures can be acknowledged as
complete with zero scored answers; downstream analysis must inspect skip counts.

Scores start empty. Local storage is keyed by the packet hash; clearing requires
browser confirmation and removes only that key. Byte changes to the packet
(including whitespace) create a different key. Local-file storage support varies
by browser; export JSON frequently. Import accepts only matching hash/schema,
exact anonymous IDs/order, valid integer/null scores and consistent completion
metadata, then asks the reviewer to confirm it is their own backup before
replacement. This is a local form, not authentication or proof of reviewer identity.

Embedded JSON escapes script delimiters and Unicode line separators. All
untrusted visible content uses `textContent`/form values. No external libraries,
resources, global storage clearing, runtime imports, or paid calls are used.

Validation (synthetic data only):

```powershell
& 'F:/DevSupportCopilotWork/.venv-core/Scripts/python.exe' -m pytest -c devsupport/pyproject.toml devsupport/tests/test_human_review.py -q
& '.venv/Scripts/ruff.exe' check devsupport/analysis/human_review.py devsupport/tests/test_human_review.py
```

Node is used by tests to execute the actual browser state/export/import logic and
parse the full browser script. Browser interaction/layout should also be checked
on synthetic packets before distributing a real review artifact.
