"""Local dependency outages stop before unattempted queries; explicit resume retains prior failure."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import httpx
import pytest

from devsupport.evaluate import run
from devsupport.types import Question


def questions():
    return [Question(f'q{i}', 'Title', f'Query{i}', '', [], {}) for i in range(3)]


@pytest.mark.parametrize('guarded', [False, True])
@pytest.mark.parametrize('kind', ['connect', 'read_timeout', '503'])
def test_local_dependency_failure_stops_and_preserves_unattempted(tmp_path, guarded, kind):
    selected = questions()
    guard = SimpleNamespace(stopped=False, observations=[]) if guarded else None
    path = tmp_path / 'rows.jsonl'
    request = httpx.Request('POST', 'http://127.0.0.1:8093/embed')
    failures = {'connect': httpx.ConnectError('offline-local-error', request=request),
                'read_timeout': httpx.ReadTimeout('offline-local-error', request=request),
                '503': httpx.HTTPStatusError('offline-local-error', request=request,
                                             response=httpx.Response(503, request=request))}
    seen = []

    def offline_failed(query):
        seen.append(query)
        raise failures[kind]

    manifest = {'scope': 'offline local dependency regression'}
    rows = run(selected, offline_failed, path, manifest, guard=guard)
    assert len(rows) == 1 and seen == [selected[0].query]
    assert rows[0]['error'] == type(failures[kind]).__name__
    assert 'offline-local-error' not in path.read_text(encoding='utf-8')
    assert not path.with_suffix('.inflight.json').exists()
    if guarded:
        assert guard.stopped is True
    prefix = path.read_bytes()
    later = []

    def offline_recovered(query):
        later.append(query)
        return {'ranked_ids': [], 'scores': []}

    fresh = SimpleNamespace(stopped=False, observations=[]) if guarded else None
    completed = run(selected, offline_recovered, path, manifest, resume=True, guard=fresh)
    assert [row['query_id'] for row in completed] == [q.query_id for q in selected]
    assert later == [q.query for q in selected[1:]]
    assert path.read_bytes().startswith(prefix) and completed[0] == rows[0]


@pytest.mark.parametrize('kind', ['value', '404', '422'])
def test_query_specific_error_keeps_later_queries(tmp_path, kind):
    selected = questions()
    guard = SimpleNamespace(stopped=False, observations=[])

    def query_callback(query):
        if query == selected[0].query:
            if kind == 'value':
                raise ValueError('offline-input-error')
            request = httpx.Request('POST', 'http://127.0.0.1:8093/embed')
            raise httpx.HTTPStatusError('offline-input-error', request=request,
                                       response=httpx.Response(int(kind), request=request))
        return {'ranked_ids': [], 'scores': []}

    rows = run(selected, query_callback, tmp_path / 'rows.jsonl', {}, guard=guard)
    assert len(rows) == 3
    assert rows[0]['error'] == ('ValueError' if kind == 'value' else 'HTTPStatusError')
    assert all(row['error'] is None for row in rows[1:]) and guard.stopped is False


@pytest.mark.parametrize('failed_query_index', [0, 2])
@pytest.mark.parametrize('kind', ['connect', '503', '422'])
def test_local_retrieval_cli_stops_other_profiles_after_outage(tmp_path, monkeypatch, failed_query_index, kind):
    import devsupport.data as data
    import devsupport.embedding as embedding
    import devsupport.evaluate as evaluation
    from devsupport.index import CorpusIndex

    script = Path(__file__).parents[1] / 'scripts/evaluate_pipeline.py'
    spec = importlib.util.spec_from_file_location('evaluate_pipeline_outage', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = questions()
    settings = {name: {'retrieval': name, 'reranker': None, 'adaptive': 'never', 'gate_enabled': False}
                for name in ('B0', 'B1')}
    contract = {'profiles': settings, 'paths': {'queries': 'offline', 'index': 'offline'},
                'artifacts': {'queries': 'offline', 'index': 'offline'}, 'config': {},
                'generation_max_tokens': 2048, 'context_counter': 'cl100k_base_proxy_v1'}
    provenance = {'offline': True, 'dimension': 1024}
    index = SimpleNamespace(manifest={'provenance': provenance})
    monkeypatch.setattr(evaluation, 'build_contract', lambda *args: contract)
    monkeypatch.setattr(evaluation, 'select_ids', lambda *args: [q.query_id for q in selected])
    monkeypatch.setattr(data, 'load_questions', lambda *args: selected)
    monkeypatch.setattr(CorpusIndex, 'load', lambda *args: index)
    monkeypatch.setattr(embedding, 'Embedder', lambda: SimpleNamespace(provenance=provenance))
    calls = []

    class OfflineRetriever:
        def __init__(self, variant):
            self.variant = variant

        def search(self, query, k):
            calls.append(self.variant)
            if self.variant == 'B0' and query == selected[failed_query_index].query:
                request = httpx.Request('POST', 'http://127.0.0.1:8093/embed')
                if kind == 'connect':
                    raise httpx.ConnectError('offline-only', request=request)
                raise httpx.HTTPStatusError('offline-only', request=request,
                                            response=httpx.Response(int(kind), request=request))
            return []

    monkeypatch.setattr(evaluation, 'build_retriever', lambda contract, variant, *args: OfflineRetriever(variant))
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['evaluate_pipeline', '--config', str(tmp_path / 'offline.json'),
                                    '--output', str(output), '--phase', 'retrieval', '--profiles', 'B0', 'B1'])
    module.main()
    if kind == '422':
        assert calls == ['B0'] * 3 + ['B1'] * 3
        assert len((output / 'B0.jsonl').read_text(encoding='utf-8').splitlines()) == 3
        assert len((output / 'B1.jsonl').read_text(encoding='utf-8').splitlines()) == 3
    else:
        assert calls == ['B0'] * (failed_query_index + 1)
        assert len((output / 'B0.jsonl').read_text(encoding='utf-8').splitlines()) == failed_query_index + 1
        assert not (output / 'B1.jsonl').exists()
