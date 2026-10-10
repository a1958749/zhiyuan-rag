# Offline analyses

## Complete DEV adaptive effects

`devsupport/analysis/audit_adaptive_effects.py` compares H0 (never decompose), Q
(selective) and QAlways (always) using a complete, ordered DEV53 run. It checks
the shared frozen contract, manifests, annotations, query hashes and reconstructed
metric rows before writing reports. It refuses partial or in-flight cohorts.
The original experiment files are never overwritten, and no provider or retrieval
call is made.

From the repository root, after all three profiles are complete:

```powershell
& F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe devsupport/analysis/audit_adaptive_effects.py --run devsupport/reports/dev-adaptive-h0-v2 --output devsupport/reports/dev-adaptive-h0-v2-audit
```

Use a fresh output location when reproducing. `audit.json` contains full-cohort
paired deltas, bootstrap intervals, win/tie/loss counts, prepack and packed
coverage, operational events and recorded stage timing. `audit.md` is a compact
index. The audit also checks that untriggered Q and fallback rows preserve H0
rankings and packed IDs. Operational errors take precedence over terminal
fallback status; overlapping fallback events are counted separately.

Triggered/expanded subsets are descriptive, selection-conditioned results.
Q and QAlways invoke the model separately, so their difference does not isolate
triggering from stochastic query generation. Recorded provider time and pacing
are separate from end-to-end time; unknown cost remains unknown. Source hashes
bind recorded bytes, not external authenticity. As of 2026-10-09 06:00, the real
v2 run is incomplete (53/33/0) and has correctly been refused; no complete
adaptive effect result is claimed. The current partial report is in
`devsupport/reports/dev-adaptive-h0-v2-stopped-n33-report/`.

## Citation normalization counterfactual

`citation_normalization_audit.py` tests a post-hoc hypothesis on completed B0/H0/H1
DEV12 generation artifacts. It is never imported by the production pipeline and
does not change the strict validator, original results, prompts, or runtime.

The candidate first invokes the existing strict validator. A valid input is returned
byte-for-byte unchanged. Otherwise it can only prune extra declared IDs that are
known context sources but unused by prose citations. It requires the exact parsed
answer schema, unique known IDs, at least one valid inline citation, and every inline
source already declared. It uses the strict validator's `citation_prose` semantics;
code and link labels are not citations. The candidate must pass the same strict
validator and preserve `answer_text` exactly. Unknown IDs, missing declarations,
missing inline citations, malformed numbers and invalid JSON remain rejected.

Before reporting, the audit verifies the fixed ordered DEV12 cohort, run/manifest
contract and profile bindings, query hashes, row hashes, and numbered context order
and hashes. Only B0/H0/H1 are opened; live C is excluded. Stored failure text
must be shorter than 16,384 characters: a record at the producer's
truncation boundary has no independent proof of completeness and aborts the audit.
The archived main runtime files must match their frozen source hashes. Executing `answer.py` must match the
archived file after newline normalization; `parse_json` must have the same AST.
Whole executing `llm.py` can differ because its HTTP diagnostics changed separately.
All input hashes are rechecked before an exclusive output directory is created.

Example from the isolated checkout:

```powershell
& F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe analysis/citation_normalization_audit.py --run F:/DevSupportCopilot/devsupport/reports/dev-generation-v2 --runtime-root F:/DevSupportCopilot/devsupport/reports/dev-generation-v2-sources --output F:/DevSupportCopilot/devsupport/reports/citation-normalization-reproduction
```

`audit.json` records file/source hashes, original record hashes and outcomes,
per-query/stage candidate validity, removed IDs, response hashes and unchanged
answer-text hashes. It contains no copied response or factual answer text. Actual
HTTP, truncation, model-identity and other operational failures are excluded from
recoverable counts. Final contract failures and earlier stored rejected attempts
are counted separately. Earlier acceptance means a repair might have been avoided;
it is not measured savings. This is no new model run and supports no claim about
cost, faithfulness, human quality or TEST gains. Original artifacts are preserved.

The exact 677 source files bound to the generation contract were copied and
hash-verified before integrating runtime fixes. Their archive manifest is
`devsupport/reports/dev-generation-v2-sources/archive-manifest.json`; the command
above uses that archive rather than assuming the active checkout still has the
original `llm.py`. A future changed answer validator or JSON parser must still be
compatible with the archive; otherwise this audit deliberately refuses to run.

Checks (offline):

```powershell
& F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe -m pytest -c devsupport/pyproject.toml devsupport/tests/test_citation_normalization_audit.py
& F:/DevSupportCopilot/.venv/Scripts/ruff.exe check analysis/citation_normalization_audit.py devsupport/tests/test_citation_normalization_audit.py
```
