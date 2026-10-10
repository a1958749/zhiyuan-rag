# Final serial evaluation workflow

## Approved scope and current state

The original DEV controller (session 71834, PID 33164) still owns the frozen
runtime. No TEST scoring, final configuration, or final lock has been created.
Human labels and ratings remain deferred by the user and must be reported as zero.

Use the one-off controller at
`F:/DevSupportCopilotWork/final-workflow-controller-20261010.py` only after its
independent review has no unresolved P1/P2 finding. This file lives outside the
runtime contract and does not change the ongoing DEV experiment.

## Release and freeze

- [x] Add regressions for live-process release, complete ordered cohorts, selection
  evidence, preservation of configuration, H0 semantic drift, candidate contract
  drift, and missing/replaced final lock. Six added regressions failed before the
  fixes; all ten tests subsequently passed. These are outside the 800-test core suite.
- [x] Bind the entire candidate configuration and optional indexes before waiting;
  match every common runtime/data/provider parameter to the original contract.
  Recheck after the original process actually exits. Write from the captured
  configuration only after its original byte hash is rechecked following integration.
- [x] After freeze, require the recorded lock identity and a fresh-child complete
  contract verification before every command. Missing lock stops dispatch.
- [x] Finish independent re-review: both P2 findings closed, ten independent offline
  tests passed; missing lock dispatched zero subprocesses and failed fresh-child
  verification did not dispatch the target stage. No API or TEST access in review.
- [x] Launch one exclusive controller instance: session 68029, PID 29568, started
  2026-10-10 00:56:23 UTC. Actual process is live and waiting for original PID 33164.
  Source and candidate bindings, zero new model calls, and unchanged original
  runtime are recorded in `devsupport/reports/final-controller-launch-verification-20261010-v1.json`.
- [ ] Wait for the original PID to exit with the complete DEV terminal state.
  Require all four DEV control groups, ten original ordered records each, and no
  unresolved call. Recompute formal control outcomes; top-level exceptions alone
  are not an operational failure count.
- [ ] Select using separate complete DEV generation/AI-quality cohorts, DEV53
  retrieval/context evidence, and complete DEV controls. The current conservative
  candidate is ordinary H0; stop for fresh judgment if the recorded quality ordering
  changes. Control decisions are unverified proxies, not human-supported answer quality.
- [ ] Integrate reviewed commit 25b5f5f, run the complete core suite and changed-file
  lint, write a fresh final configuration, and freeze all required conditions.

## Complete evaluation, with one paid queue at a time

- [ ] TEST150 for B0/B1/H0/H1/C using the core runtime; U separately using native
  Python 3.12; R1/R2 on the identical complete H1 candidate pool. Eight complete
  local retrieval conditions, including negative DEV candidates.
- [ ] Generate the fixed first TEST30 for B0/H1/the DEV-selected method and every
  retained-minus condition. Equivalent final/minus aliases reuse identical results.
- [ ] One whole-cohort AI queue, explicit coverage protocol, 8,192 output-token cap;
  complete quality, context, operational failure, latency, usage, and accounting reports.
- [ ] Independent TEST30 product controls for H0 and G when H0 is selected. Report
  structured-decision proxies, failed attempts, and provisional-label disagreement.
- [ ] Q and QAlways each receive all TEST150 retrieval questions, even if not selected
  for the default product. Complete combined context and adaptive usage accounting.

The serial controller prioritizes generation and controls before the longer adaptive
holdout queue, while all parameters and selection remain frozen before any TEST score.
This changes queue order only. It never reselects from TEST, resets allowances, recharges,
changes routes, restarts failed questions, or overwrites existing evidence. A nonzero
stage, incomplete cohort, unresolved call, changed contract, or lost lock stops the chain.

## Remaining product delivery

- [ ] Actual selected UI configuration and query/evidence/packing/citation parity.
- [ ] Actual final five-minute demonstration using the three preselected DEV cases;
  retain failures, ambiguity, snapshot limitations, and true waiting time.
- [ ] Complete public derived views of all final cohorts, preserving failures and
  rankings; credential/account/raw-provider-text removal with source/digest bindings.
- [ ] Update PRD, architecture, decisions, failures, runbook, interview notes and resume
  using actual complete results. Negative experiments do not count as achieved gains.
- [ ] Fresh sanitized archive, receipt, and actual extraction/member/hash verification.
- [ ] Full approved-spec acceptance audit before marking the project goal complete.

The controller's terminal success means automated TEST work is complete and UI/video/
public delivery is still pending. It is never a project completion signal.
