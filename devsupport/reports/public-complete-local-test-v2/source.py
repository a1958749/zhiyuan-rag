"""Analyze the complete eight-condition local TEST150, without changing selection."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('F:/DevSupportCopilot')
REPORTS = ROOT / 'devsupport/reports'
sys.path.insert(0, str(ROOT / 'devsupport/src'))
from devsupport.evaluate import build_contract, digest, verify_lock, write_new  # noqa: E402
from devsupport.report import paired, summarize  # noqa: E402
from devsupport.credentials import load_credential  # noqa: E402
import importlib.util  # noqa: E402


def main():
    """Retain complete denominators and publish local-ranking evidence exclusively."""
    contract = build_contract(ROOT / 'devsupport/configs/evaluation.final.v2.json', ROOT)
    locked = verify_lock(REPORTS / 'final-evaluation-lock-v2.json', contract)
    context_path = REPORTS / 'test-local-context-v2/summary.json'
    raw_context = context_path.read_bytes()
    context = json.loads(raw_context)
    if (context['complete'] is not True or context['contract_sha256'] != locked
            or context['query_ids'] != contract['split_ids']['test']
            or set(context['methods']) != {'B0', 'B1', 'H0', 'H1', 'C', 'U', 'R1', 'R2'}):
        raise ValueError('complete same-contract eight-method context audit required')
    inputs = dict(context['input_sha256'])
    inputs[str(context_path.resolve())] = hashlib.sha256(raw_context).hexdigest()
    inputs[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows, summaries = {}, {}
    for folder, methods in [('test-local-v2', ['B0', 'B1', 'H0', 'H1', 'C']),
                            ('test-native-u-v2', ['U']), ('test-rerank-v2', ['R1', 'R2'])]:
        for method in methods:
            path = REPORTS / folder / f'{method}.jsonl'
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != inputs[str(path.resolve())]:
                raise ValueError('audited original records changed')
            group = [json.loads(line) for line in raw.splitlines()]
            if [row['query_id'] for row in group] != contract['split_ids']['test']:
                raise ValueError('all ordered TEST150 records required')
            if any(row['record_sha256'] != digest({k: v for k, v in row.items() if k != 'record_sha256'}) for row in group):
                raise ValueError('invalid original record digest')
            rows[method] = group
            summaries[method] = summarize(group, contract['split_ids']['test'])
    comparisons = {f'{candidate}-minus-{baseline}': paired(rows[baseline], rows[candidate], contract['seeds'][0])
                   for candidate, baseline in [('H0', 'B0'), ('H1', 'H0'), ('R1', 'H1'), ('R2', 'H1'), ('C', 'B0')]}
    report = {'schema': 'devsupport.complete-local-test.v2', 'complete_local_scope': True,
              'complete_project': False, 'contract_sha256': locked, 'questions': 150,
              'methods': summaries, 'paired': comparisons, 'context_methods': context['methods'],
              'input_sha256': inputs, 'paid_calls': 0, 'human_ratings': 0,
              'scope': 'Complete local TEST ranking and packed qrel diagnostics only; Q/QAlways, generation and controls pending.',
              'notes': ['H0 is the unchanged pre-TEST choice; this report never reselects methods.',
                        'Original v1 local records retained; corrective rerun is not independent replication.',
                        'Ranking-only p50/p95 is not end-to-end answer latency.',
                        'All attempted records retained; exploratory paired bootstrap, no multiplicity correction.',
                        'U uses shared Qwen and native 45000 proxy-token candidate budget, not stock Granite.',
                        'Qrel coverage/precision are not AI answer quality or human validation.']}
    raw = json.dumps(report, ensure_ascii=False, allow_nan=False).encode('utf-8')
    spec = importlib.util.spec_from_file_location('public_records', ROOT / 'devsupport/analysis/public_records.py')
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)
    credentials = tuple(load_credential(ROOT / '.secrets' / name) for name in ('deepseek.dpapi', 'internal.dpapi'))
    if any(value.encode() in raw for value in credentials) or public.KEY_PATTERN.search(raw):
        raise ValueError('credential literal or pattern in local aggregate')
    for path, expected in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('consumed analysis input changed')
    output = REPORTS / 'public-complete-local-test-v2/summary.json'
    write_new(output, report)
    print(json.dumps({'path': str(output), 'questions': 150, 'methods': {
        name: {'coverage10': row['metrics_prepack']['coverage@10'],
               'alpha_ndcg10': row['metrics_prepack']['alpha_ndcg@10'], 'errors': row['errors']}
        for name, row in summaries.items()}, 'h0_minus_b0': comparisons['H0-minus-B0']['prepack']['coverage@10'],
        'paid_calls': 0, 'project_complete': False}))


if __name__ == '__main__':
    main()
