import hashlib
import pytest
from devsupport.service import PipelineService
from devsupport.bridge_client import source_mapping, select_mapping, native_document

class Pipeline:
    def __init__(self): self.calls = []
    def retrieve(self, query):
        self.calls.append(('retrieve', query))
        return {'search_id': 'prepared', 'query': query, 'context': [], 'evidence_ids': []}
    def answer(self, query, envelope):
        self.calls.append(('answer', query))
        return {'decision': 'insufficient', 'answer_text': '', 'context': []}

def test_empty_run_scoped_and_once():
    pipe = Pipeline()
    service = PipelineService({'H1': pipe}, 'revision', {})
    run = service.search('question', 'H1', 'revision', 'session')
    for query, session in [('different', 'session'), ('question', 'other')]:
        with pytest.raises(ValueError):
            service.answer(run['run_id'], query, 'H1', 'revision', session)
    service.answer(run['run_id'], 'question', 'H1', 'revision', 'session')
    with pytest.raises(ValueError):
        service.answer(run['run_id'], 'question', 'H1', 'revision', 'session')
    assert pipe.calls == [('retrieve', 'question'), ('answer', 'question')]

def row(profile='H1'):
    return {'id': 'native', 'metadata': {'devsupport_profile': profile, 'corpus_revision': 'revision',
            'devsupport_corpus': 'freshstack/langchain', 'knowledge_date': '2024-10'},
            'config': {'kind': 'classic', 'retrieval': {'retriever': 'devsupport', 'exposure': 'prefetch',
                        'rephrase_query': False}}}

def test_selection_rejects_mixed_and_invalid_config():
    mapping = source_mapping(row())
    assert select_mapping([{'devsupport': mapping}]) == mapping
    assert select_mapping([{'id': 'plain'}]) is None
    with pytest.raises(ValueError):
        select_mapping([{'devsupport': mapping}, {'id': 'upload'}])
    with pytest.raises(ValueError):
        select_mapping([{'devsupport': mapping}, {'devsupport': source_mapping(row('B0'))}])
    bad = row()
    bad['config'] = {}
    with pytest.raises(ValueError):
        source_mapping(bad)

def test_native_identity_and_snapshot():
    raw = {'doc_id': 'official', 'text': 'original 中文\n', 'metadata': {'url': 'https://example.test'}}
    mapped = native_document(raw, source_mapping(row()))
    assert mapped['source_id'] == 'native' and mapped['doc_id'] == 'official'
    assert mapped['chunk_key'] == hashlib.md5(raw['text'].encode()).hexdigest()
    service = PipelineService({}, 'revision', {'official': raw})
    assert service.evidence('revision', chunk_key=mapped['chunk_key']) == raw
    with pytest.raises(ValueError):
        service.evidence('other', doc_id='official')
    with pytest.raises(ValueError):
        service.evidence('revision', doc_id='unknown')

def test_shared_pipeline_offline_service_parity():
    from devsupport.pipeline import Pipeline, PipelineConfig
    from test_adaptive import Provider, Retriever, hits
    reply = {'answer_text': 'Use Widget.render [1].', 'citation_ids': ['0']}
    offline = Pipeline(Retriever(hits()), Provider([reply]), config=PipelineConfig(adaptive='never', gate_enabled=False))
    online = Pipeline(Retriever(hits()), Provider([reply]), config=PipelineConfig(adaptive='never', gate_enabled=False))
    expected = offline.run('query')
    service = PipelineService({'H1': online}, 'revision', {})
    prepared = service.search('query', 'H1', 'revision', 'session')
    actual = service.answer(prepared['run_id'], 'query', 'H1', 'revision', 'session')
    for key in ('evidence_ids', 'context', 'prompt_hash', 'runtime_hash', 'decision', 'answer_text', 'citation_ids'):
        assert expected[key] == actual[key]
    assert online.retriever.queries == ['query']
    assert online.provider.calls == offline.provider.calls


def test_embedding_semantic_mismatch_rejected(monkeypatch):
    import httpx
    from devsupport.service import ModelEmbedder
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'provenance': {'dimension': 1024, 'revision': 'wrong'}}
    monkeypatch.setattr(httpx, 'post', lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match='provenance'):
        ModelEmbedder({'dimension': 1024, 'revision': 'correct'}).encode(['query'], is_query=True)


def test_expired_unknown_and_extra_evidence_rejected():
    import time
    service = PipelineService({'H1': Pipeline()}, 'revision', {})
    run = service.search('question', 'H1', 'revision', 'session')
    state = service.pending[run['run_id']]
    service.pending[run['run_id']] = (*state[:4], time.monotonic() - 1801, state[5])
    with pytest.raises(ValueError):
        service.answer(run['run_id'], 'question', 'H1', 'revision', 'session')
    with pytest.raises(ValueError):
        service.answer('invented', 'question', 'H1', 'revision', 'session')
    with pytest.raises(TypeError):
        service.answer('x', 'question', 'H1', 'revision', 'session', evidence=[])

def test_http_private_token_and_unknown_lookup(monkeypatch):
    from fastapi.testclient import TestClient
    from devsupport.service import create_app
    monkeypatch.setenv('DEVSUPPORT_SERVICE_TOKEN', 'private-test-token')
    client = TestClient(create_app(PipelineService({'H1': Pipeline()}, 'revision', {})))
    assert client.get('/health').status_code == 200
    assert client.post('/v1/search', json={}).status_code == 403
    headers = {'X-Devsupport-Token': 'private-test-token'}
    assert client.post('/v1/evidence', json={'revision': 'revision', 'doc_id': 'unknown'}, headers=headers).status_code == 404
    body = {'query': 'question', 'profile': 'H1', 'revision': 'revision', 'session': 'session'}
    response = client.post('/v1/search', json=body, headers=headers)
    assert response.status_code == 200
    body['run_id'] = response.json()['run_id']
    assert client.post('/v1/answer', json=body, headers=headers).status_code == 200
    assert client.post('/v1/answer', json=body, headers=headers).status_code == 422
