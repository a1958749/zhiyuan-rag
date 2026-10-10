"""Denominator-aware summaries and paired, query-aligned bootstrap comparisons."""
import math
import statistics

from .metrics import bootstrap_paired


def outcome(row: dict) -> dict:
    """Separate operational failures from legitimate knowledge refusals and repaired answers."""
    result = row.get('result', {})
    trace = result.get('trace', {})
    retrieve_only = result.get('evaluation_phase') == 'retrieve-only'
    answer = result.get('decision') == 'answer' and not retrieve_only
    failed = bool(row.get('error') or trace.get('excluded_failure') or
                  trace.get('gate', {}).get('reason') == 'gate_validation_or_provider_failure' or
                  (trace.get('generation', {}).get('error') and not answer))
    fallback = bool(trace.get('adaptive', {}).get('fallback') or
                    any(item.get('fallback') for item in trace.get('reranking', [])) or trace.get('fallback'))
    return {'error': failed, 'success': (answer or (('ranked_ids' in result or retrieve_only) and not fallback)) and not failed,
            'refusal': result.get('decision') in ('insufficient', 'clarify') and not failed and not retrieve_only,
            'fallback': fallback}


def percentiles(values: list[float]) -> dict | None:
    """Return observed p50/p95, with empty denominators unavailable."""
    if not values:
        return None
    ordered = sorted(values)
    return {'p50': statistics.median(ordered), 'p95': ordered[math.ceil(len(ordered) * .95) - 1],
            'n': len(ordered)}


def stage_failures(row: dict) -> set[str]:
    """Identify observed failure categories once per row, including recovered stages.

    Unknown legacy fallbacks stay unclassified; validation is not transport failure.
    Categories can overlap, so their counts must not be summed as a row total.
    """
    trace = row.get('result', {}).get('trace', {})
    failures = set()
    for call in trace.get('provider_calls', []):
        if (call.get('error') or call.get('truncated') or call.get('stop_reason') == 'max_tokens'
                or any(event.get('status') in ('network_error', 'transport_error', 'redirect_refused')
                       or str(event.get('status', '')).startswith('http_') for event in call.get('events', []))):
            failures.add('provider')
    if trace.get('excluded_failure') or trace.get('generation', {}).get('error') == 'provider_failure':
        failures.add('provider')
    if any(item.get('error') for item in trace.get('retrieval', [])):
        failures.add('retrieval')
    if any(item.get('error') or item.get('fallback') or any(retry.get('error') for retry in item.get('retries', []))
           for item in trace.get('reranking', [])):
        failures.add('reranking')
    if (trace.get('generation', {}).get('error') == 'citation_or_schema_validation_failed'
            or trace.get('generation', {}).get('validation_failures')):
        failures.add('validation')
    adaptive = trace.get('adaptive', {})
    category = {'decomposition_provider': 'provider', 'decomposition_validation': 'validation',
                'subquery_retrieval': 'retrieval', 'expanded_rerank': 'reranking'}.get(adaptive.get('failure_stage'))
    if category:
        failures.add(category)
    elif adaptive.get('fallback'):
        # Old rows may still have explicit evidence of a failed decomposition call or retrieval.
        known_expansion_failure = (any(call.get('stage') == 'decomposition' and call.get('error')
                                       for call in trace.get('provider_calls', []))
                                   or any(item.get('error') for item in trace.get('retrieval', [])))
        if not known_expansion_failure:
            failures.add('unknown')
    if (row.get('error') or trace.get('fallback')
            or trace.get('gate', {}).get('reason') == 'gate_validation_or_provider_failure') and not failures:
        failures.add('unknown')
    return failures


def summarize(rows: list[dict], expected_ids: list[str], allow_partial: bool = False) -> dict:
    """Report every attempted row; stopped runs disclose missing planned queries explicitly."""
    ids = [r['query_id'] for r in rows]
    if len(ids) != len(set(ids)) or ids != expected_ids[:len(ids)] or len(ids) > len(expected_ids):
        raise ValueError('output denominator identity mismatch')
    if not allow_partial and ids != expected_ids:
        raise ValueError('incomplete output denominator')
    flags = [outcome(row) for row in rows]
    summary = {'planned': len(expected_ids), 'attempted': len(rows), 'unattempted': len(expected_ids) - len(rows),
               'complete': ids == expected_ids,
               **{key: sum(f[key] for f in flags) for key in ('success', 'refusal', 'fallback')},
               'errors': sum(f['error'] for f in flags),
               'answers': sum(r.get('result', {}).get('decision') == 'answer' and not f['error']
                              for r, f in zip(rows, flags))}
    failures = [stage_failures(row) for row in rows]
    summary['operational_failure_rows'] = sum(bool(f & {'provider', 'retrieval', 'reranking'}) for f in failures)
    summary['stage_failure_rows'] = {stage: sum(stage in f for f in failures)
                                     for stage in ('provider', 'retrieval', 'reranking', 'validation', 'unknown')}
    for field in ('metrics_prepack', 'metrics_packed'):
        summary[field] = {key: statistics.fmean(row[field][key] for row in rows)
                          for key in rows[0][field]} if rows else None
    if rows and all('ranked_ids' in r.get('result', {}) for r in rows):
        summary['metrics_packed'] = None  # Local ranking-only phase does not pack any context.
    summary['elapsed_seconds'] = percentiles([r['elapsed_seconds'] for r in rows])
    stages = {}
    for row in rows:
        trace = row.get('result', {}).get('trace', {})
        for call in trace.get('provider_calls', []):
            if 'elapsed_seconds' in call:
                stages.setdefault(call['stage'], []).append(call['elapsed_seconds'])
                wait = call.get('pacing_wait_seconds', 0)
                stages.setdefault('provider_pacing_wait', []).append(wait)
                stages.setdefault('provider_active_seconds', []).append(max(0, call['elapsed_seconds'] - wait))
        for retrieval in trace.get('retrieval', []):
            for key, value in retrieval.get('timings', {}).items():
                if type(value) in (int, float):
                    stages.setdefault('retrieval_' + key, []).append(value)
        for ranker in trace.get('reranking', []):
            if 'elapsed_seconds' in ranker:
                stages.setdefault('reranking', []).append(ranker['elapsed_seconds'])
    summary['stage_seconds'] = {key: percentiles(value) for key, value in stages.items()}
    summary['usage'] = {}
    for row in rows:
        for key, value in row.get('result', {}).get('usage', {}).items():
            if type(value) is int:
                summary['usage'][key] = summary['usage'].get(key, 0) + value
    costs = [row.get('result', {}).get('cost') for row in rows]
    summary['estimated_cost_configured_currency'] = sum(costs) if costs and all(c is not None for c in costs) else None
    summary['cost_scope'] = 'provider_only'
    summary['currency_note'] = 'Unknown unless explicit configured provider rates; quota_unit=500000 is not currency.'
    summary['context_note'] = '4000 cl100k_base_proxy_v1 tokens; not exact gateway generation tokens.'
    summary['latency_note'] = 'Scope is manifest phase; first observed query may be cold, warm/cold not instrumented.'
    return summary


def paired(baseline: list[dict], candidate: list[dict], seed: int = 20261008) -> dict:
    """Bootstrap candidate-minus-baseline only on the identical ordered attempted cohort."""
    ids = [r['query_id'] for r in baseline]
    if ids != [r['query_id'] for r in candidate] or len(ids) != len(set(ids)):
        raise ValueError('paired query identities differ')
    if not ids:
        return {'prepack': None, 'packed': None}
    local_only = all('ranked_ids' in r.get('result', {}) for r in baseline + candidate)
    return {label: None if label == 'packed' and local_only else {key: bootstrap_paired([r[field][key] for r in baseline],
                                          [r[field][key] for r in candidate], seed=seed)
                    for key in baseline[0][field]}
            for label, field in (('prepack', 'metrics_prepack'), ('packed', 'metrics_packed'))}


def markdown(summaries: dict) -> str:
    """Render transparent counts without fabricated answer-quality judgments."""
    lines = ['# Evaluation', '', '| Method | Planned | Attempted | Success | Answers | Refusal | Error | Fallback |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summaries.items():
        lines.append(f"| {name} | {row['planned']} | {row['attempted']} | {row['success']} | "
                     f"{row['answers']} | {row['refusal']} | {row['errors']} | {row['fallback']} |")
    lines += ['', '| Method | Operational failure rows | Provider | Retrieval | Reranking | Validation | Unknown |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summaries.items():
        stages = row.get('stage_failure_rows', {})
        counts = ' | '.join(str(stages.get(stage, 'unavailable'))
                            for stage in ('provider', 'retrieval', 'reranking', 'validation', 'unknown'))
        lines.append(f"| {name} | {row.get('operational_failure_rows', 'unavailable')} | {counts} |")
    lines += ['', 'Failure categories count rows, including recovered failures, and can overlap. Operational failure '
              'rows are the union of observed provider, retrieval and reranking failures. Error above remains terminal '
              'failure; validation and unknown legacy reasons are separate.']
    lines += ['', 'Raw-ranking and packed-context metrics are separate JSON tables. Generation holdout n=30; '
              'local retrieval holdout n=150. Actual attempted counts above take precedence.', '',
              'Context budget: 4000 cl100k_base_proxy_v1 tokens, not exact gateway generation tokens. Provider currency costs '
              'remain unknown without explicit rates. Costs are provider-only, excluding GPU and other infrastructure; '
              'they are not all-in costs. No human or AI-judge ratings are inferred.']
    return '\n'.join(lines) + '\n'
