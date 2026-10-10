# Sanitized handoff packaging

`export_handoff.py` is an offline explicit-file exporter. It verifies packaging integrity, not project completion or answer quality. Final export remains pending method selection, TEST/ablation and final video.

Prepare a **public JSON array of exact relative file names**. Include committed source, applicable source/license manifests, final selected reports and visually checked media. Do not expand all ignored report folders automatically. Preserve attribution to DocsGPT, FreshStack and models; excluded weights/data/indexes are rebuilt through the recorded download and index commands.

```powershell
& 'F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe' devsupport/analysis/export_handoff.py --root 'F:/DevSupportCopilot' --files 'F:/path/to/public-file-list.json' --output 'F:/path/to/new-handoff.zip'
```

The CLI does not read credentials, accept literal secrets in command arguments, access the network or create a project acceptance assertion. For the final credential-literal scan, call `export(root, files, output, forbidden_values=values)` in a local process with credential values already held in memory; never serialize or print them. Known values are checked even against the dummy-fixture exception.

Path restrictions reject traversal, ambiguous Windows names, resolved filesystem aliases, secret directories, environment files, runtime/index storage, full DevSupport data, model weights, databases and account-credit filenames. The only long-key pattern exception is the exact upstream redaction regression fixture with its verified whole-file SHA256. A changed fixture is rejected. File snapshots are scanned and hashed, then inputs rechecked before exclusive ZIP creation. The ZIP's CRC, members, per-file bytes and manifest are validated against one final byte snapshot; the receipt SHA256 and size use that same snapshot, followed by a disk hash recheck.

Scanning checks long `sk-` strings and supplied literal values; it is not exhaustive secret discovery or image OCR. Each media selection requires visual review. Path restrictions do not establish the semantic contents of an arbitrary public filename. Select public evidence deliberately, keep original integrity-bound evaluation records separate from any derived shareable excerpts, and record exclusions. Do not republish account screenshots or raw gateway diagnostics.

The receipt contains per-file hashes inside its `manifest`, ZIP hash/size and `zip_integrity_verified`. Save it as a separate public JSON receipt only after success. Failed exports may leave an unverified ZIP after an I/O or post-write check error; do not deliver it or rerun into the existing filename. No recursive cleanup occurs.
