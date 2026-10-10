# Explicit judge decision labels implementation plan

> Execute inline within the approved complete evaluation scope, with TDD and read-only review. Main runtime source42 remains frozen. This changes only a versioned offline assessment protocol.

**Goal:** Expose the already-enforced decision/refusal consistency rule to the judge without inventing or repairing scores.

**Architecture:** Preserve default legacy prompt bytes. Add opt-in `decision_label_protocol=explicit` to prompt construction, the CLI manifest and run_judge. Each eligible anonymous ID receives allowed labels based only on its recorded decision. Quality audit uses the protocol from the manifest. Unknown protocols fail before paid work. Old terminal assessment is immutable; a new protocol uses a fresh whole-cohort assessment, never pools selected prior judgments.

**Tech:** Existing Python/pytest/JSON; no dependencies or runtime edits.

Actual failure: original decision DEV judge-v2 stopped after three rows; the third valid/untruncated response labeled Q003-A02 `none` despite recorded insufficient. Local validate_review reproduces `refusal label inconsistent with generation decision`. Provider transport passed. Do not normalize that label or relax validation.

Alternatives considered: relax validation (rejected: inconsistent labels accepted), replay an unchanged prompt (no identified correction), expose the existing contract (chosen). Rubric, scoring range, evidence, nuggets, blinding, generation and provider remain unchanged.

- [x] RED explicit protocol: correct label whitelist for answer/clarify/insufficient, eligible-only IDs, canonical packet unchanged; unknown protocol rejected; run binding/CLI preflight/quality auditing select recorded protocol.
- [x] Implement `system_prompt(tokens, decision_label_protocol='legacy')` and `prompt(packet, cap, tokens, decision_label_protocol='legacy')`. Legacy yields original bytes; explicit adds the consistency instruction and `allowed_refusal_labels` in the prompt copy. answer allows only none; clarify/insufficient allow supported_abstention/false_refusal. These lists constrain format, not which score or refusal judgment wins.
- [x] Bind CLI flag `--decision-label-protocol explicit` into the manifest only when opted in; run_judge validates it before making an output directory or provider calls. Quality audit validates declared protocol and its system hash. Existing strict validator is unchanged.
- [x] GREEN117 related tests, Ruff and diff-check. Actual historical DEV12 legacy report retained identical questions/methods/paired/per_question values in `reports/dev-ai-quality-decision-protocol-compat-v1/summary.json`; original-v2 prompt hash remains equal under default legacy. Runtime42 contract unchanged.
- [x] Read-only review: independent117 passed, legacy all three token-cap prompts exact, no P1/P2. Explicit commit and offline new-v3 preflight follow.
- [ ] Dispatch new-v3 whole12 judge and remaining original DEV10 controls serially through the unchanged prepaid guard/pacing. Prior3 records and expense remain terminal, no score pooling; actual complete output still required.

Run regression: `F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe -m pytest -c devsupport/pyproject.toml devsupport/tests/test_answer_judge.py devsupport/tests/test_answer_quality.py -q -o addopts=''` with PYTHONPATH=devsupport/src and plugins disabled. Synthetic TEST fixtures only; no real TEST scoring before DEV selection/final freeze.
