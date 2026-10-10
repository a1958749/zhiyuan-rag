"""Whitelist-derived public evidence; never modifies or resumes private checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))
from devsupport.evaluate import digest  # noqa: E402

KEY_PATTERN = re.compile(rb'sk-[A-Za-z0-9_-]{24,}')
METRIC_PATTERN = re.compile(r'^(mrr|(?:coverage|precision|recall|ndcg|alpha_ndcg)@(5|10|20|50))$')
USAGE = {k: 'number' for k in (
    'input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')}
HTTP = {k: 'atom' for k in ('status_code', 'retry_after_seconds', 'provider_error_code', 'provider_error_type')}
EVENT = {**{k: 'atom' for k in ('attempt', 'status', 'cost', 'pacing_wait_seconds')}, 'usage': USAGE}
PROVIDER = {**{k: 'atom' for k in (
    'stage', 'attempts', 'cost', 'cost_status', 'elapsed_seconds', 'error', 'actual_model', 'requested_model',
    'expected_reported_model', 'identity_status', 'known_response_cost', 'max_tokens', 'pacing_status',
    'pacing_wait_seconds', 'prompt_hash', 'protocol', 'reasoning_prefix_removed', 'stop_reason', 'thinking',
    'truncated', 'provider_error_code')}, 'usage': USAGE, 'http_error': HTTP, 'events': [EVENT]}
CONTEXT = {**{k: 'atom' for k in ('doc_id', 'text', 'citation_number', 'rank', 'score', 'score_kind')},
           'metadata': {k: 'atom' for k in ('url', 'start_byte', 'end_byte')}}
RERANK = {**{k: 'atom' for k in (
    'requested_model', 'revision', 'actual_model', 'variant', 'device', 'fallback', 'error', 'forward_count',
    'instruction', 'query_cap', 'document_cap', 'dtype', 'attention', 'prepared_sha256', 'elapsed_seconds',
    'peak_gpu_bytes')}, 'input_ids': ['text'], 'prompt_token_counts': ['number'],
    'model_files_sha256': ('map', ('text_or_object', {'sha256': 'text', 'bytes': 'number'})),
    'lengths': {'query': {'original': 'number', 'retained': 'number'},
                'documents': [{'original': 'number', 'retained': 'number'}]},
    'outputs': [{k: 'atom' for k in ('doc_id', 'score', 'rank', 'score_kind')}],
    'retries': [{k: 'atom' for k in ('attempt', 'error', 'batch_size', 'elapsed_seconds')}]}
NATIVE_TRACE = {**{k: 'atom' for k in (
    'profile', 'variant', 'native_class', 'native_store', 'requested_candidates', 'native_internal_fetch_k',
    'native_doc_token_limit', 'native_effective_token_budget', 'rephrase_calls', 'generation_calls',
    'canonical_corpus_sha256', 'score_kind', 'seconds')}, 'ranked_ids': ['text'],
    'embedding': {**{k: 'atom' for k in (
        'attention', 'device', 'dimension', 'dtype', 'max_length', 'model', 'pooling',
        'query_instruction', 'revision')}, 'model_files_sha256': RERANK['model_files_sha256']}}
TRACE = {
    'elapsed_seconds': 'number', 'final_candidate_ids': ['text'],
    'adaptive': {**{k: 'atom' for k in (
        'additional_rounds', 'anchored_query_count', 'anchoring_error', 'anchoring_policy', 'failure_code',
        'failure_stage', 'fallback', 'preflight_policy', 'reason', 'triggered')},
        **{k: ['text'] for k in ('appended_identifiers', 'effective_subqueries', 'model_subqueries', 'subqueries')}},
    'gate': {**{k: 'atom' for k in ('decision', 'enabled', 'reason')}, 'missing_information': ['text']},
    'generation': {'attempts': 'number', 'error': 'atom', 'validation_failures': [{'error': 'text'}]},
    'packing': {**{k: 'atom' for k in ('budget', 'counter', 'exact_generation_tokens', 'used')},
                'skipped_ids': ['text']},
    'prompt_versions': {k: 'text' for k in ('answer', 'decomposition', 'gate')},
    'provider_calls': [PROVIDER], 'reranking': [RERANK],
    'retrieval': [{'ids': ['text'], 'leg_ids': {'dense': ['text'], 'lexical': ['text']}, 'query': 'text',
                   'native': NATIVE_TRACE,
                   'timings': {k: 'number' for k in (
                       'dense_seconds', 'embedding_seconds', 'fusion_seconds', 'lexical_seconds', 'total_seconds')}}],
}
RESULT = {**{k: 'atom' for k in (
    'answer_text', 'decision', 'evaluation_phase', 'prompt_hash', 'query', 'runtime_hash',
    'search_id', 'cost', 'cost_scope', 'cost_status')},
    **{k: ['text'] for k in ('citation_ids', 'evidence_ids', 'ranked_ids', 'score_kinds')},
    'scores': ['number'], 'missing_information': ['text'], 'context': [CONTEXT], 'trace': TRACE, 'usage': USAGE}
REVIEW = {'review_id': 'text', 'reason': 'text', 'refusal': 'text', 'nugget_coverage': ('map', 'text'),
          'scores': {k: 'number' for k in (
              'correctness', 'completeness', 'faithfulness', 'citation_support', 'answer_relevance')}}
ROW_SCHEMA = {**{k: 'atom' for k in (
    'query_id', 'query_sha256', 'elapsed_seconds', 'error', 'status', 'packet_sha256', 'manifest_sha256', 'seed',
    'provider_error_code', 'profile')},
    **{k: ['text'] for k in ('ranked_ids', 'score_kinds', 'prepack_ids', 'packed_ids')},
    'scores': ['number'], 'prepack_scores': ['number'],
    **{k: ('metrics', 'number') for k in ('metrics', 'metrics_prepack', 'metrics_packed')},
    'stage_seconds': {k: 'number' for k in (
        'dense_seconds', 'embedding_seconds', 'fusion_seconds', 'lexical_seconds', 'total_seconds')},
    'leg_ids': {'dense': ['text'], 'lexical': ['text']}, 'context_sha256': ('map', 'text'),
    'hits': [{k: 'atom' for k in ('doc_id', 'score', 'rank', 'score_kind')}],
    'result': RESULT, 'trace': {**RERANK, **NATIVE_TRACE}, 'provider': PROVIDER, 'reviews': {'reviews': [REVIEW]}}
RULES = {'schema': 'devsupport.public-record.v1', 'projection': ROW_SCHEMA,
         'excluded': ['account observations', 'raw provider text', 'validation-failure response_text',
                      'all unknown fields at every explicit schema boundary'],
         'optional_transform': {'policy': 'query_sk_key_pattern_v1',
                                'scope': 'only original/retrieval/adaptive query text; every change disclosed',
                                'known_credentials': 'reject before transformation',
                                'answer_and_evidence_text': 'unchanged; key patterns still reject'},
         'not_resumable': True, 'metrics_source': 'private complete original cohorts; never derived subsets'}


def _project(value, schema):
    """Project explicit object/list boundaries; never pass arbitrary nested data through."""
    if value is None:
        return None
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError('invalid public object field')
        return {key: _project(value[key], spec) for key, spec in schema.items() if key in value}
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError('invalid public array field')
        return [_project(v, schema[0]) for v in value]
    if isinstance(schema, tuple):
        if schema[0] == 'text_or_object':
            return _project(value, schema[1] if isinstance(value, dict) else 'text')
        if not isinstance(value, dict):
            raise ValueError('invalid public map field')
        return {k: _project(v, schema[1]) for k, v in value.items()
                if schema[0] != 'metrics' or METRIC_PATTERN.fullmatch(k)}
    valid = (schema == 'text' and isinstance(value, str)
             or schema == 'number' and type(value) in (int, float) and math.isfinite(value)
             or schema == 'atom' and (isinstance(value, (str, bool))
                                     or type(value) in (int, float) and math.isfinite(value)))
    if not valid:
        raise ValueError('invalid public scalar field')
    return value


def project_record(row: dict, source_sha256: str, line_sha256: str, method: str) -> dict:
    """Preserve one attempted record and its original digest, without filling any labels."""
    if not isinstance(row, dict) or not isinstance(row.get('query_id'), str) or not row['query_id']:
        raise ValueError('invalid source record ID')
    hash_fields = [k for k in ('record_sha256', 'sha256') if k in row]
    if len(hash_fields) > 1:
        raise ValueError('ambiguous source record digest')
    field = hash_fields[0] if hash_fields else None
    canonical = digest({k: v for k, v in row.items() if k != field})
    if field and row[field] != canonical:
        raise ValueError('source record digest mismatch')
    result = _project(row, ROW_SCHEMA)
    result['method'] = method
    result['provenance'] = {'source_sha256': source_sha256, 'source_line_sha256': line_sha256,
                            'source_record_sha256': canonical, 'original_hash_field': field,
                            'rules_sha256': digest(RULES), 'not_resumable': True}
    result['derived_sha256'] = digest(result)
    return result


def _scan(raw: bytes, forbidden_values: tuple[str, ...], retained, check_key_pattern: bool = True) -> None:
    """Check retained keys/strings before JSON escaping, plus actual serialized bytes."""
    if (check_key_pattern and KEY_PATTERN.search(raw)) or any(v.encode('utf-8') in raw for v in forbidden_values):
        raise ValueError('credential in public projection')
    pending = [retained]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, str) and any(secret in value for secret in forbidden_values):
            raise ValueError('credential in public projection')


def _redact_query_keys(row: dict) -> None:
    """Redact only explicit query fields, preserving original hashes and disclosing every change."""
    result = row.get('result')
    if not isinstance(result, dict):
        return
    targets = [(result, 'query', 'result.query')]
    trace = result.get('trace') or {}
    for index, retrieval in enumerate(trace.get('retrieval') or []):
        targets.append((retrieval, 'query', f'result.trace.retrieval[{index}].query'))
    adaptive = trace.get('adaptive') or {}
    for key in ('subqueries', 'model_subqueries', 'effective_subqueries'):
        for index, _ in enumerate(adaptive.get(key) or []):
            targets.append((adaptive[key], index, f'result.trace.adaptive.{key}[{index}]'))
    changes = []
    for container, key, path in targets:
        value = container.get(key) if isinstance(container, dict) else container[key]
        if not isinstance(value, str):
            continue
        encoded = value.encode('utf-8')
        redacted, count = KEY_PATTERN.subn(b'[REDACTED_KEY_PATTERN]', encoded)
        if count:
            container[key] = redacted.decode('utf-8')
            changes.append({'path': path, 'kind': 'long_sk_key_pattern',
                            'original_text_sha256': hashlib.sha256(encoded).hexdigest(), 'replacements': count})
    if changes:
        row['redactions'] = changes
        row['derived_sha256'] = digest({k: v for k, v in row.items() if k != 'derived_sha256'})


def publish(source: Path, output: Path, method: str, forbidden_values: tuple[str, ...] = (),
            redact_query_keys: bool = False) -> dict:
    """Publish every input row in order to an exclusive, hash-bound derived directory."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('public output exists; never overwrite')
    if (not isinstance(method, str) or not method.strip()
            or any(not isinstance(v, str) or not v for v in forbidden_values)
            or type(redact_query_keys) is not bool):
        raise ValueError('invalid public method or scan values')
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            raise ValueError('blank source record')
        # Finite serialization checks unknown private fields as well as the projection.
        row = json.loads(line)
        json.dumps(row, allow_nan=False)
        rows.append(project_record(row, source_hash, hashlib.sha256(line).hexdigest(), method))
    if not rows or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate source cohort')
    if redact_query_keys:
        # Known literals must be rejected even if a later pattern replacement would hide them.
        original = json.dumps(rows, ensure_ascii=False, allow_nan=False).encode('utf-8')
        _scan(original, forbidden_values, rows, check_key_pattern=False)
        for row in rows:
            _redact_query_keys(row)
    encoded = ('\n'.join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + '\n').encode('utf-8')
    report = {'schema': RULES['schema'], 'source_name': source.name, 'source_sha256': source_hash,
              'rows': len(rows), 'ordered_query_ids': [r['query_id'] for r in rows], 'method': method,
              'rules': RULES, 'rules_sha256': digest(RULES), 'output_sha256': hashlib.sha256(encoded).hexdigest(),
              'output_bytes': len(encoded), 'not_resumable': True,
              'acceptance_scope': 'projection integrity only; not cohort completeness or semantic quality'}
    if redact_query_keys:
        report['query_key_redaction'] = {'enabled': True, 'fields': sum(len(r.get('redactions', [])) for r in rows),
                                        'rows': sum(bool(r.get('redactions')) for r in rows)}
    receipt = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    _scan(encoded, forbidden_values, rows)
    _scan(receipt, forbidden_values, report)
    if source.read_bytes() != raw:
        raise ValueError('source changed before public publication')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'records.jsonl').open('xb') as handle:
        handle.write(encoded)
    with (output / 'derivation.json').open('xb') as handle:
        handle.write(receipt)
    if (output / 'records.jsonl').read_bytes() != encoded or (output / 'derivation.json').read_bytes() != receipt:
        raise ValueError('public publication verification failed')
    return report


def main() -> None:
    """Accept public paths only; callers supply additional credential literals in memory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--redact-query-keys', action='store_true',
                        help='Disclosed query-only pattern replacement; never edits original records or answers')
    args = parser.parse_args()
    result = publish(args.source, args.output, args.method, redact_query_keys=args.redact_query_keys)
    print(json.dumps({k: result[k] for k in ('rows', 'source_sha256', 'output_sha256', 'not_resumable')}))


if __name__ == '__main__':
    main()
