"""Offline complete DEV53 H0/Q/QAlways audit; never executes retrieval or a provider."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from devsupport.data import load_questions, sha256_file
from devsupport.evaluate import digest, make_record, read_json, text_hash, write_new
from devsupport.report import outcome, paired, summarize

MODES = {'H0': 'never', 'Q': 'selective', 'QAlways': 'always'}
METRICS = {'prepack_coverage10': ('metrics_prepack', 'coverage@10'),
           'prepack_alpha10': ('metrics_prepack', 'alpha_ndcg@10'),
           'prepack_coverage50': ('metrics_prepack', 'coverage@50'),
           'packed_coverage': ('metrics_packed', 'coverage@50')}
NOTES = [
    'Quality effects include ALL53 attempted queries, including operational errors scored by make_record; '
    'no successful-only selection. Packed coverage uses coverage@50 of the packed evidence IDs.',
    'Triggered-only and expanded-only subsets are descriptive, selection-conditioned, not causal population effects.',
    'Per-query statuses are mutually exclusive: operational_error takes precedence over fallback. '
    'Adaptive fallback counts terminal non-error fallbacks; fallback_event_rows and failure_stage_codes '
    'count recorded adaptive fallback events, including rows ending in operational error.',
    'Q and QAlways use separate stochastic provider calls, not identical generated queries; '
    'their contrast cannot isolate pure trigger causality.',
    'Provider active time, pacing wait and end-to-end elapsed time are separate recorded quantities. '
    'No GPU or all-in currency cost is inferred.',
    'Hashes bind recorded contracts and bytes, not external authenticity. Current runtime source need not '
    'equal historical source; all three profiles must share the identical recorded frozen contract.',
]


def load_complete(directory: Path) -> tuple:
    """Verify frozen manifests, annotations and reconstructed rows before analysis."""
    run = read_json(directory / 'run.json')
    if (run.get('profiles') != list(MODES) or run.get('phase') != 'retrieve-only'
            or run.get('split') != 'dev' or run.get('controls') is not False):
        raise ValueError('requires H0/Q/QAlways retrieve-only DEV, without controls')
    ids = run['query_ids']
    if len(ids) != 53 or len(set(ids)) != 53:
        raise ValueError('requires complete unique DEV53 cohort')
    contract, variants, bindings = None, {}, {'run_sha256': sha256_file(directory / 'run.json'), 'profiles': {}}
    for name, mode in MODES.items():
        path = directory / f'{name}.jsonl'
        manifest_path = path.with_suffix('.manifest.json')
        if not path.is_file() or not manifest_path.is_file() or path.with_suffix('.inflight.json').exists():
            raise ValueError('incomplete or in-flight run')
        meta = read_json(manifest_path)
        frozen = meta['evaluation_contract']
        if (digest(frozen) != run['contract_sha256'] or any(meta.get(k) != v for k, v in run.items())
                or (contract is not None and frozen != contract)):
            raise ValueError('mixed or tampered evaluation contract')
        contract = frozen
        settings = {'retrieval': 'H0', 'reranker': None, 'gate_enabled': False, 'adaptive': mode}
        if (set(contract['profiles']) != set(MODES) or contract['profiles'][name] != settings
                or meta.get('settings') != settings or meta.get('profile') != name or meta.get('variant') != 'H0'
                or meta.get('questions_sha256') != contract['artifacts']['queries']
                or meta.get('index_manifest_sha256') != contract['artifacts']['index']
                or meta.get('context_counter') != contract['context_counter']
                or meta.get('latency_scope') != 'retrieve-only'):
            raise ValueError('profile provenance mismatch')
        for key in ('queries', 'split'):
            if sha256_file(contract['paths'][key]) != contract['artifacts'][key]:
                raise ValueError(f'offline {key} annotations changed')
        split = read_json(contract['paths']['split'])
        if (ids != contract['split_ids']['dev'] or ids != split['dev_ids']
                or contract['split_ids']['test'] != split['test_ids']
                or len(split['test_ids']) != 150 or len(set(ids + split['test_ids'])) != 203):
            raise ValueError('DEV53 frozen cohort/order mismatch')
        questions = {q.query_id: q for q in load_questions(contract['paths']['queries'])}
        if meta.get('query_hashes') != [text_hash(questions[i].query) for i in ids]:
            raise ValueError('manifest query hash mismatch')
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if [r['query_id'] for r in rows] != ids:
            raise ValueError('incomplete, duplicate or reordered cohort')
        for row in rows:
            result = row['result']
            if ('variant' in result and result['variant'] != 'H0') or (
                    result.get('evaluation_phase') != 'retrieve-only'
                    and ('evaluation_phase' in result or not row.get('error'))):
                raise ValueError('result phase/variant mismatch')
            if row != make_record(questions[row['query_id']], result, row['elapsed_seconds'], row['error']):
                raise ValueError('result integrity mismatch')
        runtime_hashes = {r['result'].get('runtime_hash') for r in rows if not r['error']}
        policies = {r['result'].get('trace', {}).get('adaptive', {}).get('anchoring_policy')
                    for r in rows if not r['error']}
        if len(runtime_hashes) != 1 or None in runtime_hashes or len(policies) != 1 or None in policies:
            raise ValueError('inconsistent or unavailable runtime hash/anchoring policy within profile')
        bindings['profiles'][name] = {'rows_sha256': sha256_file(path), 'manifest_sha256': sha256_file(manifest_path),
                                      'runtime_hash': next(iter(runtime_hashes)), 'anchoring_policy': next(iter(policies))}
        variants[name] = rows
    if len({b['anchoring_policy'] for b in bindings['profiles'].values()}) != 1:
        raise ValueError('mixed anchoring policies')
    return run, contract, variants, bindings


def status(row: dict) -> str:
    """Separate terminal failure, fallback, triggered no-op and completed expansion."""
    adaptive = row['result'].get('trace', {}).get('adaptive', {})
    if outcome(row)['error']:
        return 'operational_error'
    if adaptive.get('fallback'):
        return 'fallback'
    if adaptive.get('triggered') is True:
        return 'expanded' if adaptive.get('subqueries') else 'no_op'
    return 'untriggered' if adaptive.get('triggered') is False else 'trigger_unavailable'


def effects(rows: list[dict]) -> dict:
    """Count actual retrieval attempts from trace, including failed additional calls."""
    counts = Counter(status(r) for r in rows)
    adaptive = [r['result'].get('trace', {}).get('adaptive', {}) for r in rows]
    triggers = [a.get('triggered') for a in adaptive]
    anchored = [a.get('anchored_query_count') for a in adaptive]
    known = [v for v in anchored if isinstance(v, int)]
    additional = [max(0, len(r['result'].get('trace', {}).get('retrieval', [])) - 1) for r in rows]
    failures = Counter((a.get('failure_stage', 'unknown'), a.get('failure_code', 'unknown'))
                       for a in adaptive if a.get('fallback'))
    return {**{k: counts[k] for k in ('expanded', 'fallback', 'no_op', 'operational_error', 'trigger_unavailable')},
            'fallback_event_rows': sum(bool(a.get('fallback')) for a in adaptive),
            'triggered': sum(t is True for t in triggers), 'untriggered': sum(t is False for t in triggers),
            'trigger_status_unavailable': sum(t is not True and t is not False for t in triggers),
            'failure_stage_codes': [{'stage': s, 'code': c, 'count': n} for (s, c), n in sorted(failures.items())],
            'anchored_subqueries': sum(known), 'anchored_rows': sum(v > 0 for v in known),
            'anchoring_observed_rows': len(known), 'anchoring_unavailable_rows': 53 - len(known),
            'anchored_row_rate_observed': sum(v > 0 for v in known) / len(known) if known else None,
            'actual_total_additional_queries': sum(additional),
            'additional_query_scope': 'Trace retrieval attempts beyond first, including errors; not planned subqueries.'}


def values(row: dict) -> dict:
    """Select the prespecified retrieval and packed-context metrics."""
    return {label: row[field][key] for label, (field, key) in METRICS.items()}


def delta_summary(deltas: list[dict]) -> dict:
    """Summarize wins/ties/losses without changing cohort selection."""
    return {key: {'n': len(deltas), 'mean_delta': statistics.fmean(d[key] for d in deltas) if deltas else None,
                  'win': sum(d[key] > 0 for d in deltas), 'tie': sum(d[key] == 0 for d in deltas),
                  'loss': sum(d[key] < 0 for d in deltas)} for key in METRICS}


def audit(directory: Path) -> dict:
    """Build the audit exclusively from validated recorded artifacts and offline gold."""
    run, contract, variants, bindings = load_complete(Path(directory))
    methods, comparisons, query_rows = {}, {}, []
    preservation = {'checked': 0, 'violations': [], 'unavailable': []}
    for name, rows in variants.items():
        methods[name] = summarize(rows, run['query_ids'])
        methods[name]['latency_scope'] = 'retrieve-only'
        methods[name]['adaptive_effects'] = effects(rows)
        calls = [c for r in rows for c in r['result'].get('trace', {}).get('provider_calls', [])]
        methods[name]['provider_calls'] = len(calls)
        methods[name]['provider_attempts_recorded'] = sum(c.get('attempts', 0) for c in calls)
        for base, row in zip(variants['H0'], rows):
            adaptive = row['result'].get('trace', {}).get('adaptive', {})
            if name == 'H0':
                continue
            check = adaptive.get('fallback') == 'first_round_preserved' or (
                name == 'Q' and adaptive.get('triggered') is False)
            identity = {'profile': name, 'query_id': row['query_id']}
            if outcome(row)['error'] or outcome(base)['error']:
                preservation['unavailable'].append({**identity, 'reason': 'operational error; no empty-ranking comparison'})
            elif check:
                preservation['checked'] += 1
                if any(base[k] != row[k] for k in ('prepack_ids', 'packed_ids')):
                    preservation['violations'].append({**identity, 'reason': 'first-round preservation mismatch',
                        **{f'{label}_{key}': source[key] for label, source in [('H0', base), ('candidate', row)]
                           for key in ('prepack_ids', 'packed_ids')}})
    for i, query_id in enumerate(run['query_ids']):
        query_rows.append({'query_id': query_id, 'profiles': {
            name: {'metrics': values(rows[i]), 'status': status(rows[i]),
                   'adaptive': rows[i]['result'].get('trace', {}).get('adaptive', {})}
            for name, rows in variants.items()}})
    for base, candidate in [('H0', 'Q'), ('H0', 'QAlways'), ('Q', 'QAlways')]:
        label = f'{candidate}-minus-{base}'
        deltas = [{key: values(b)[key] - values(a)[key] for key in METRICS}
                  for a, b in zip(variants[base], variants[candidate])]
        for query, delta in zip(query_rows, deltas):
            query.setdefault('deltas', {})[label] = delta
        # Reuse project bootstrap, restricting columns to those reported here.
        def compact(rows: list[dict]) -> list[dict]:
            """Select columns without reimplementing paired bootstrap."""
            return [{**r, 'metrics_prepack': {k: r['metrics_prepack'][k] for k in
                     ('coverage@10', 'alpha_ndcg@10', 'coverage@50')},
                     'metrics_packed': {'coverage@50': r['metrics_packed']['coverage@50']}} for r in rows]
        comparisons[label] = {'all53': delta_summary(deltas),
            'paired_bootstrap_all53': paired(compact(variants[base]), compact(variants[candidate]), contract['seeds'][0]),
            'selection_conditioned_on_candidate': {
                subset: delta_summary([d for d, r in zip(deltas, variants[candidate]) if (
                    r['result'].get('trace', {}).get('adaptive', {}).get('triggered') is True
                    if subset == 'triggered_only' else status(r) == 'expanded')])
                for subset in ('triggered_only', 'expanded_only')}}
    return {'run': run, 'bindings': bindings, 'methods': methods, 'comparisons': comparisons,
            'per_query': query_rows, 'preservation_checks': preservation, 'notes': NOTES}


def markdown(report: dict) -> str:
    """Render a compact index; full query metrics and costs remain in adjacent JSON."""
    lines = ['# DEV adaptive effects', '', *report['notes'], '',
             '| Profile | Planned | Attempted | Triggered | Untriggered | Expanded | Fallback | No-op | Extra queries |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, method in report['methods'].items():
        effect = method['adaptive_effects']
        cells = [method['planned'], method['attempted']] + [effect[k] for k in
                 ('triggered', 'untriggered', 'expanded', 'fallback', 'no_op', 'actual_total_additional_queries')]
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, cells)) + ' |')
    checks = report['preservation_checks']
    lines += ['', f"Preservation: {checks['checked']} checked; {len(checks['violations'])} violations; "
              f"{len(checks['unavailable'])} unavailable. IDs and both rankings are in audit.json.", '',
              'Full53 deltas, win/tie/loss counts, bootstrap intervals, descriptive subsets, query metrics, '
              'anchoring counts and provider/stage/end-to-end summaries are in audit.json.']
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Validate fully before creating exclusively new audit output files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    write_new(args.output / 'audit.json', report)
    with (args.output / 'audit.md').open('x', encoding='utf-8') as handle:
        handle.write(markdown(report))


if __name__ == '__main__':
    main()
