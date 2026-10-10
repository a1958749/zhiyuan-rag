# Complete AI answer-quality comparison

This offline report verifies an existing complete generation cohort with at least two unique methods and its complete AI assessment. It does not generate, rejudge, read credentials, query balances or invoke network services. It calculates every pair within that one assessment cohort; five methods produce ten pairs. Failures and refusals keep their distinct denominators. Do not pool separately judged cohorts with different method sets or protocols into one fair leaderboard. Existing four-method historical validation remains supported.

```powershell
& 'F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe' devsupport/analysis/answer_quality.py --run 'F:/path/to/complete-generation' --assessment 'F:/path/to/complete-ai-judge' --output 'F:/path/to/new-quality-report'
```

For the final frozen first30 TEST generation cohort, add `--split test`. The generation must be the exact matching split and cohort stored in its contract. A partial judge, an unresolved generation/judge in-flight marker, changed input bytes, altered blinded mapping, invalid ratings or inconsistent summary prevents publication. Assessment artifacts are parsed from their exact hashed byte snapshots. Recorded generation hashes are checked before loading and compared afterward. The destination must be new; consumed data and analysis source hashes are rechecked at audit completion and immediately before output creation.

The utility reuses `answer_judge.load_run`, regenerates the exact blinded packets and mapping using the recorded seed, binds the judge-only provider changes (`retries=0` and its supported response cap), and revalidates all saved rows, rubric/prompt identities and summary. Recorded historical judge source hashes are retained as provenance; it does not claim to execute an archived historical implementation. Current validator/source identities appear separately in the report. Hash consistency is integrity evidence, not a digital signature or proof of semantic correctness.

`summary.json` contains:

- All methods' original conditional AI score means, attempted utility, operational-failure and refusal counts.
- Each method's valid answer count, answer coverage across every attempted question, and answer-only score means with an explicit denominator. Valid nonanswers remain distinct from operational failures.
- Per-question utility and every method pair in the run's fixed method order. Utility is the unweighted mean of five integer 0..3 ratings; generation failures contribute zero. Each pair uses 2,000 paired percentile-bootstrap samples and the recorded seed.
- Original generation/judge identities and hashes of consumed data, records, blinded packets, mapping, summary and current analysis sources.

These are AI ratings, not answer accuracy, human review or independent model validation. Legitimate abstentions can earn high faithfulness/relevance scores while reducing answer coverage; assess them together with the separate control-decision report and false-refusal labels. Conditional means use different successful subsets and do not establish paired causal effects. The paired intervals are descriptive, have no multiplicity correction, and come from single generation draws. DEV diagnostic selection, imperfect references and an unverified gateway alias further limit generalization. The `human_ratings=0` field describes this AI-only report's inputs, not a global count of unrelated human reviews.

The historical DEV12 v4 comparison can be reproduced with `--run devsupport/reports/dev-generation-v2 --assessment devsupport/reports/dev-generation-v2-ai-judge-v4` and a new output path. Other complete comparison cohorts use the same command; method names are not hard-coded.
