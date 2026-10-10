# Cross-Python context metric verification

The real native DEV12 runner completed with zero errors and paid calls, but its
context audit failed while reconstructing metric checksums in Python 3.10.
All twelve original record digests remain valid. Only three records differ on
recomputed binary nDCG, by 2.78e-17 to 5.55e-17. Python 3.12 changed floating-point
`sum` to Neumaier summation; the isolated example sum([0.1]*10) gives different
values in the existing 3.10 and 3.12 runtimes. Official reference:
https://docs.python.org/3.12/whatsnew/3.12.html#other-language-changes

Do not rewrite metrics, change the runtime metric implementation, or relax query,
cohort, source, ranking, context, or original byte-integrity checks. Extend only
the offline auditor: separately verify the original record digest, compare every
nonmetric field exactly, then require equal metric key sets, finite nonboolean
numeric values, and recomputation agreement within declared 1e-12 relative/absolute
tolerance. Disclose every accepted numerical difference in the per-query report.
This matches the existing legacy context comparison tolerance, not a new method gain.

- [x] Two ULP acceptance/disclosure regressions observed RED; five checksum,
  nonmetric, key, boolean and out-of-tolerance rejection regressions already passed.
- [x] Implement the bounded offline comparison and disclosure.
- [x] Related regressions, lint and independent read-only review: 121 related
  regressions locally, 124 independently; no remaining P1/P2.
- [x] Re-audit the unchanged real native DEV12 and historical same-runtime DEV12.
  Verify old method/paired scores remain equal; no API, TEST or label changes.
- [x] Core regression: 811 passed, four existing dependency warnings, exit 0.
  Original native ranks match the historical native run on all twelve queries;
  all original and public digests and nested native traces verify. Three rows
  disclose eight bounded roundoff differences. Historical 48 context rows,
  method summaries and paired intervals are unchanged. Both active runtime
  contracts remain unchanged. Proof:
  `devsupport/reports/native-cross-python-context-verification-20261010-v1.json`.
- [x] The verified offline-only change is included in this commit.

Native dependency prerequisites and the modern public trace boundary are separately
verified. This correction is necessary for the final mixed-runtime TEST audit; it
does not establish final test results or product completion.
