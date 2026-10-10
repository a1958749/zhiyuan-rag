# Approved handoff export implementation

This implements the existing approved sanitized handoff requirement, without changing runtime, paid experiments or final method selection.

Create `devsupport/analysis/export_handoff.py` and focused boundary tests. Accept only an explicit root-relative file list; reject traversal, ambiguous Windows paths, symlink escapes, credentials, runtime/data/index/database/weight files and account-credit artifacts. Read exact file snapshots, scan a long-key pattern plus supplied in-memory literal values, and permit the upstream dummy-key fixture only by its exact approved file hash. Recheck input hashes before exclusive ZIP creation. Include per-file hashes, verify duplicate names, ZIP CRC and every archived byte hash; issue a packaging receipt without claiming project or semantic acceptance.

- [x] Write real-file behavioral tests and observe the missing-exporter RED failure.
- [x] Implement the narrow archive utility; run focused tests, Ruff and diff checks.
- [x] Independently review boundary handling and resolve findings.
- [x] Rehearse on public source/report files only; verify the current 42-source generation contract remains unchanged.
- [ ] Final export remains pending DEV selection, immutable TEST/ablation evidence and final video; never label the rehearsal as final delivery.

No network calls, DPAPI decryption, package installation, database operation or model invocation. The root agent supplies any literal secret values in memory during the final scan; never serialize them. Pattern scanning does not inspect rendered image/video pixels, so explicit media selection and visual inspection remain required.
