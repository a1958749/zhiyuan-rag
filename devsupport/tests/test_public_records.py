"""Public projections preserve evidence while removing private nested fields."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from devsupport.evaluate import digest


def load():
    path = Path(__file__).resolve().parents[1] / 'analysis/public_records.py'
    assert path.is_file(), 'public record utility is not implemented'
    spec = importlib.util.spec_from_file_location('public_records', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(tmp_path, secret='private-opaque-value'):
    rows = [
        {'query_id': 'q1', 'prepack_ids': ['d1', 'd2'], 'packed_ids': ['d1'],
         'metrics_packed': {'coverage@10': 1.0}, 'error': None, 'elapsed_seconds': 2,
         'balance_observations': [{'remain_quota': 999, 'unexpected': secret}],
         'result': {'answer_text': 'Exact answer [1].', 'decision': 'answer', 'citation_ids': ['d1'],
                    'context': [{'doc_id': 'd1', 'text': 'Exact evidence.',
                                 'metadata': {'url': 'https://example.org', 'account': secret}}],
                    'balance_observations': [{'remain_quota': 999}],
                    'trace': {'generation': {'attempts': 2, 'validation_failures': [
                        {'error': 'missing citation', 'response_text': secret}]},
                        'provider_calls': [{'stage': 'answer', 'attempts': 2, 'usage': {'input_tokens': 8},
                                            'events': [{'attempt': 1, 'status': 'http_503', 'body': secret}],
                                            'http_error': {'provider_error_code': 'busy', 'raw_body': secret},
                                            'text': secret}], 'unknown_plugin': {'private': secret}}}},
        {'query_id': 'q2', 'error': 'ProviderError', 'result': {}, 'elapsed_seconds': 3},
    ]
    for r in rows:
        r['record_sha256'] = digest(r)
    path = tmp_path / 'Full.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    return path, rows


def test_complete_order_exact_answers_and_private_fields_removed(tmp_path):
    path, original = source(tmp_path)
    before = path.read_bytes()
    out = tmp_path / 'public'
    report = load().publish(path, out, method='Full', forbidden_values=('private-opaque-value',))
    rows = [json.loads(line) for line in (out / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['query_id'] for r in rows] == ['q1', 'q2']
    assert rows[0]['result']['answer_text'] == original[0]['result']['answer_text']
    assert rows[0]['prepack_ids'] == ['d1', 'd2'] and rows[0]['packed_ids'] == ['d1']
    assert rows[0]['metrics_packed'] == original[0]['metrics_packed']
    assert rows[1]['error'] == 'ProviderError'
    assert rows[0]['result']['trace']['generation']['validation_failures'] == [{'error': 'missing citation'}]
    assert rows[0]['result']['trace']['provider_calls'][0]['events'] == [{'attempt': 1, 'status': 'http_503'}]
    assert rows[0]['result']['context'][0]['metadata'] == {'url': 'https://example.org'}
    assert report['rows'] == 2 and report['source_sha256'] == hashlib.sha256(before).hexdigest()
    assert rows[0]['provenance']['source_record_sha256'] == original[0]['record_sha256']
    assert rows[0]['provenance']['original_hash_field'] == 'record_sha256'
    assert rows[0]['provenance']['not_resumable'] is True
    assert path.read_bytes() == before
    raw = (out / 'records.jsonl').read_bytes()
    assert b'private-opaque-value' not in raw and b'remain_quota' not in raw
    assert b'unknown_plugin' not in raw and b'response_text' not in raw
    assert report['output_sha256'] == hashlib.sha256(raw).hexdigest()
    for row in rows:
        assert row['derived_sha256'] == digest({k: v for k, v in row.items() if k != 'derived_sha256'})


def test_judge_retains_validated_reviews_not_provider_text(tmp_path):
    row = {'query_id': 'q', 'status': 'judge_error', 'reviews': {'reviews': []}, 'error': 'ProviderError',
           'provider_error_code': 'output_truncated',
           'provider': {'text': 'private-value', 'attempts': 1, 'usage': {'output_tokens': 3}}}
    row['sha256'] = digest(row)
    path = tmp_path / 'results.jsonl'
    path.write_text(json.dumps(row) + '\n')
    load().publish(path, tmp_path / 'public', method='AI assessment', forbidden_values=('private-value',))
    result = json.loads((tmp_path / 'public/records.jsonl').read_text())
    assert result['status'] == 'judge_error' and result['reviews'] == {'reviews': []}
    assert result['provider_error_code'] == 'output_truncated'
    assert result['provider'] == {'attempts': 1, 'usage': {'output_tokens': 3}}
    assert result['provenance']['original_hash_field'] == 'sha256'


def test_native_u_retains_ordered_hits_and_snapshot_protocol_without_private_fields(tmp_path):
    row = {'query_id': 'q-native', 'profile': 'U',
           'hits': [{'doc_id': 'd2', 'score': 0.8, 'score_kind': 'cosine', 'rank': 1,
                     'account': 'private-native'},
                    {'doc_id': 'd1', 'score': 0.7, 'score_kind': 'cosine', 'rank': 2}],
           'trace': {'profile': 'U', 'variant': 'classic', 'native_class': 'ClassicRAG',
                     'native_store': 'faiss', 'requested_candidates': 50, 'native_internal_fetch_k': 100,
                     'native_doc_token_limit': 45000, 'native_effective_token_budget': 45000,
                     'rephrase_calls': 0, 'generation_calls': 0, 'canonical_corpus_sha256': 'corpus-hash',
                     'ranked_ids': ['d2', 'd1'], 'score_kind': 'cosine', 'seconds': 0.4,
                     'embedding': {'dimension': 1024, 'model': 'shared-qwen', 'revision': 'model-revision',
                                   'model_files_sha256': {'config.json': {'bytes': 5, 'sha256': 'config-hash',
                                                                        'account': 'private-native'}},
                                   'account': 'private-native'},
                     'balance_observations': [{'remain_quota': 999}]}}
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps(row) + '\n')
    before = path.read_bytes()
    out = tmp_path / 'public'
    load().publish(path, out, method='U', forbidden_values=('private-native',))
    public = json.loads((out / 'records.jsonl').read_text())
    assert public['hits'] == [{k: v for k, v in hit.items() if k != 'account'} for hit in row['hits']]
    assert public['profile'] == 'U'
    assert public['trace']['ranked_ids'] == ['d2', 'd1']
    for key in ('native_class', 'native_store', 'requested_candidates', 'native_internal_fetch_k',
                'native_doc_token_limit', 'native_effective_token_budget', 'rephrase_calls',
                'generation_calls', 'canonical_corpus_sha256', 'seconds'):
        assert public['trace'][key] == row['trace'][key]
    assert public['trace']['embedding']['dimension'] == 1024
    assert public['trace']['embedding']['model_files_sha256'] == {
        'config.json': {'bytes': 5, 'sha256': 'config-hash'}}
    assert b'private-native' not in (out / 'records.jsonl').read_bytes()
    assert 'balance_observations' not in public['trace']
    assert path.read_bytes() == before


def test_native_embedding_scalar_cannot_smuggle_nested_private_data(tmp_path):
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps({'query_id': 'q-native', 'trace': {
        'embedding': {'model': {'account': 'private'}}}}) + '\n')
    out = tmp_path / 'public'
    with pytest.raises(ValueError, match='scalar'):
        load().publish(path, out, method='U')
    assert not out.exists()


def test_modern_runner_retains_nested_native_trace_with_explicit_private_boundaries(tmp_path):
    path, rows = source(tmp_path)
    native = {'profile': 'U', 'variant': 'classic', 'native_class': 'ClassicRAG', 'native_store': 'faiss',
              'ranked_ids': ['d2', 'd1'], 'native_effective_token_budget': 45000,
              'seconds': 1.25, 'balance_observations': [{'remain_quota': 999}],
              'embedding': {'dimension': 1024, 'revision': 'fixed-revision',
                            'model_files_sha256': {'weights': {'sha256': 'abc', 'bytes': 123,
                                                              'account': 'private-opaque-value'}},
                            'account': 'private-opaque-value'},
              'unknown_plugin': {'private': 'private-opaque-value'}}
    rows[0]['result']['trace']['retrieval'] = [{'query': 'Exact original query', 'ids': ['d2', 'd1'],
                                              'native': native}]
    rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf8')
    before = path.read_bytes()
    output = tmp_path / 'public'
    load().publish(path, output, 'U', forbidden_values=('private-opaque-value',))
    public = json.loads((output / 'records.jsonl').read_text().splitlines()[0])
    trace = public['result']['trace']['retrieval'][0]
    assert trace['query'] == 'Exact original query' and trace['ids'] == ['d2', 'd1']
    assert trace['native'] == {'profile': 'U', 'variant': 'classic', 'native_class': 'ClassicRAG', 'native_store': 'faiss',
                               'ranked_ids': ['d2', 'd1'], 'native_effective_token_budget': 45000,
                               'seconds': 1.25, 'embedding': {
                                   'dimension': 1024, 'revision': 'fixed-revision',
                                   'model_files_sha256': {'weights': {'sha256': 'abc', 'bytes': 123}}}}
    assert b'private-opaque-value' not in (output / 'records.jsonl').read_bytes()
    assert path.read_bytes() == before


@pytest.mark.parametrize('native', [{'native_class': {'private': 'drop'}},
                                  {'embedding': {'dimension': {'private': 'drop'}}}])
def test_modern_native_scalar_cannot_hide_nested_private_data(tmp_path, native):
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps({'query_id': 'q-native', 'result': {'trace': {'retrieval': [{'native': native}]}}}))
    output = tmp_path / 'public'
    with pytest.raises(ValueError, match='scalar'):
        load().publish(path, output, 'U')
    assert not output.exists()


def test_modern_native_retained_provenance_string_still_has_credential_scan(tmp_path):
    path = tmp_path / 'native.jsonl'
    path.write_text(json.dumps({'query_id': 'q-native', 'result': {'trace': {'retrieval': [
        {'native': {'embedding': {'model': 'opaque-secret'}}}]}}}))
    output = tmp_path / 'public'
    with pytest.raises(ValueError, match='credential'):
        load().publish(path, output, 'U', forbidden_values=('opaque-secret',))
    assert not output.exists()


def test_opt_in_query_key_redaction_keeps_row_metrics_original_hash_and_discloses_changes(tmp_path):
    key = 'sk-' + 'a' * 30
    query = 'Help with credential ' + key
    row = {'query_id': 'q-key', 'query_sha256': hashlib.sha256(query.encode()).hexdigest(),
           'metrics_packed': {'coverage@10': 0.5}, 'result': {
               'query': query, 'answer_text': 'Original answer [1].', 'decision': 'answer',
               'trace': {'retrieval': [{'query': query, 'ids': ['d1']}]}}}
    row['record_sha256'] = digest(row)
    path = tmp_path / 'query.jsonl'
    path.write_text(json.dumps(row) + '\n')
    before = path.read_bytes()
    strict = tmp_path / 'strict'
    with pytest.raises(ValueError, match='credential'):
        load().publish(path, strict, 'H0')
    assert not strict.exists()
    out = tmp_path / 'public'
    report = load().publish(path, out, 'H0', redact_query_keys=True)
    public = json.loads((out / 'records.jsonl').read_text())
    assert public['query_id'] == row['query_id']
    assert public['query_sha256'] == row['query_sha256']
    assert public['metrics_packed'] == row['metrics_packed']
    assert public['result']['query'] == 'Help with credential [REDACTED_KEY_PATTERN]'
    assert public['result']['trace']['retrieval'][0]['query'] == public['result']['query']
    assert public['result']['answer_text'] == row['result']['answer_text']
    assert public['provenance']['source_record_sha256'] == row['record_sha256']
    assert public['redactions'] == [{'path': field, 'kind': 'long_sk_key_pattern',
                                     'original_text_sha256': row['query_sha256'], 'replacements': 1}
                                    for field in ('result.query', 'result.trace.retrieval[0].query')]
    assert public['derived_sha256'] == digest({k: v for k, v in public.items() if k != 'derived_sha256'})
    assert report['query_key_redaction'] == {'enabled': True, 'fields': 2, 'rows': 1}
    assert path.read_bytes() == before
    assert key not in (out / 'records.jsonl').read_text()


@pytest.mark.parametrize('secret', ['sk-' + 'b' * 30, 'opaque"secret'])
def test_query_redaction_cannot_hide_known_credential_matches(tmp_path, secret):
    path = tmp_path / 'query.jsonl'
    path.write_text(json.dumps({'query_id': 'q', 'result': {'query': 'query ' + secret}}) + '\n')
    out = tmp_path / 'public'
    with pytest.raises(ValueError, match='credential'):
        load().publish(path, out, 'H0', forbidden_values=(secret,), redact_query_keys=True)
    assert not out.exists()


def test_query_redaction_does_not_change_answer_or_evidence_boundary(tmp_path):
    path = tmp_path / 'query.jsonl'
    path.write_text(json.dumps({'query_id': 'q', 'result': {'answer_text': 'sk-' + 'c' * 30}}) + '\n')
    out = tmp_path / 'public'
    with pytest.raises(ValueError, match='credential'):
        load().publish(path, out, 'H0', redact_query_keys=True)
    assert not out.exists()


def test_real_trace_shapes_preserve_whitelisted_nested_diagnostics(tmp_path):
    path, rows = source(tmp_path)
    rows[0]['result']['missing_information'] = ['version']
    rows[0]['result']['trace']['gate'] = {'decision': 'clarify', 'missing_information': ['version']}
    trace = {'lengths': {'query': {'original': 5, 'retained': 4},
                        'documents': [{'original': 9, 'retained': 8, 'private': 'drop'}]},
             'model_files_sha256': {'config.json': 'abc', 'weights': {'sha256': 'def', 'bytes': 4,
                                                                      'account': 'drop'}}}
    rows[0]['trace'] = trace
    rows[0]['result']['trace']['reranking'] = [trace]
    rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    out = tmp_path / 'public'
    load().publish(path, out, method='H1')
    row = json.loads((out / 'records.jsonl').read_text().splitlines()[0])
    assert row['result']['missing_information'] == ['version']
    assert row['result']['trace']['gate']['missing_information'] == ['version']
    assert row['trace']['lengths']['documents'] == [{'original': 9, 'retained': 8}]
    assert row['trace']['model_files_sha256']['weights'] == {'sha256': 'def', 'bytes': 4}
    assert row['result']['trace']['reranking'][0] == row['trace']


@pytest.mark.parametrize('mutation', ['duplicate', 'digest', 'nan', 'nested_object'])
def test_invalid_records_fail_before_publication(tmp_path, mutation):
    path, rows = source(tmp_path)
    if mutation == 'duplicate':
        rows.append(rows[0])
    elif mutation == 'digest':
        rows[0]['error'] = 'changed'
    elif mutation == 'nan':
        rows[0]['elapsed_seconds'] = float('nan')
    else:
        rows[0]['result']['answer_text'] = {'arbitrary_private': 'value'}
        rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    out = tmp_path / 'public'
    with pytest.raises(ValueError):
        load().publish(path, out, method='Full')
    assert not out.exists()


@pytest.mark.parametrize('secret', ['sk-' + 'x' * 30, 'opaque-secret', 'opaque"secret',
                                    'opaque\\secret', 'opaque\nsecret'])
def test_sensitive_allowed_text_fails_before_publication(tmp_path, secret):
    path, rows = source(tmp_path)
    rows[0]['result']['context'][0]['text'] = secret
    rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    out = tmp_path / 'public'
    with pytest.raises(ValueError, match='credential'):
        load().publish(path, out, method='Full', forbidden_values=(secret,))
    assert not out.exists()


def test_cache_creation_usage_is_preserved_at_every_recorded_level(tmp_path):
    path, rows = source(tmp_path)
    usage = {'input_tokens': 1, 'output_tokens': 2, 'cache_read_input_tokens': 3,
             'cache_creation_input_tokens': 4, 'private_unknown': 999}
    rows[0]['result']['usage'] = usage
    call = rows[0]['result']['trace']['provider_calls'][0]
    call['usage'] = usage
    call['events'][0]['usage'] = usage
    rows[0]['provider'] = {'usage': usage}
    rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    out = tmp_path / 'public'
    load().publish(path, out, method='Full')
    row = json.loads((out / 'records.jsonl').read_text().splitlines()[0])
    expected = {k: v for k, v in usage.items() if k != 'private_unknown'}
    assert row['result']['usage'] == expected and row['provider']['usage'] == expected
    call = row['result']['trace']['provider_calls'][0]
    assert call['usage'] == expected and call['events'][0]['usage'] == expected


def test_existing_output_is_never_modified(tmp_path):
    path, _ = source(tmp_path)
    out = tmp_path / 'public'
    out.mkdir()
    (out / 'keep').write_text('old')
    with pytest.raises(FileExistsError):
        load().publish(path, out, method='Full')
    assert (out / 'keep').read_text() == 'old' and not (out / 'records.jsonl').exists()


def test_source_mutation_during_projection_fails_before_publication(tmp_path, monkeypatch):
    path, _ = source(tmp_path)
    module = load()
    original = module.project_record

    def changing(*args, **kwargs):
        value = original(*args, **kwargs)
        path.write_text('changed')
        return value

    monkeypatch.setattr(module, 'project_record', changing)
    out = tmp_path / 'public'
    with pytest.raises(ValueError, match='source changed'):
        module.publish(path, out, method='Full')
    assert not out.exists()
