"""Focused native bridge tests, with no DB/GPU/provider calls."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask, request

from devsupport.src.devsupport.bridge_client import source_mapping
from docsgpt.api.answer.services import stream_processor as sp
from docsgpt.agents.devsupport_agent import DevSupportAgent
from docsgpt.retriever.devsupport import DevSupportRetriever
from docsgpt.api.user.sources import chunks


def source():
    return {'id': 'native', 'metadata': {'devsupport_profile': 'H1', 'corpus_revision': 'revision',
            'devsupport_corpus': 'freshstack/langchain'},
            'config': {'kind': 'classic', 'retrieval': {'retriever': 'devsupport', 'exposure': 'prefetch',
                        'rephrase_query': False}}}


@contextmanager
def connection():
    yield object()


@pytest.mark.parametrize('principal,permitted', [(None, True), ('stranger', False)])
def test_unauthorized_source_never_reaches_service(monkeypatch, principal, permitted):
    proc = sp.StreamProcessor.__new__(sp.StreamProcessor)
    proc.initial_user_id = principal
    repo = Mock()
    repo.get_by_id.return_value = source()
    monkeypatch.setattr(sp, 'db_readonly', connection)
    monkeypatch.setattr(sp, 'can_access', lambda *args: permitted)
    monkeypatch.setattr(sp, 'SourcesRepository', lambda conn: repo)
    assert proc._load_request_sources(['native']) == []
    repo.get_by_id.assert_not_called()


def test_authorized_mapping_and_empty_prefetch(monkeypatch):
    proc = sp.StreamProcessor.__new__(sp.StreamProcessor)
    proc.initial_user_id = 'owner'
    repo = Mock()
    repo.get_by_id.return_value = source()
    monkeypatch.setattr(sp, 'db_readonly', connection)
    monkeypatch.setattr(sp, 'can_access', lambda *args: True)
    monkeypatch.setattr(sp, 'SourcesRepository', lambda conn: repo)
    proc.all_sources = proc._load_request_sources(['native'])
    proc.data = {}
    proc.agent_config = {'agent_type': 'classic'}
    proc.retriever_config = {'chunks': 6}
    proc.request_id = 'session'
    proc.source = {'active_docs': 'native'}
    search = Mock(return_value={'run_id': 'run', 'context': []})
    monkeypatch.setattr('docsgpt.retriever.devsupport.BridgeClient.search', search)
    proc.pre_fetch_docs('query')
    assert proc.devsupport_context['run_id'] == 'run'
    assert proc.retrieved_docs == []
    search.assert_called_once()


def test_untrusted_retriever_selection_cannot_search():
    with pytest.raises(ValueError):
        DevSupportRetriever(source={'active_docs': 'native'}).search('query')


def agent():
    instance = DevSupportAgent.__new__(DevSupportAgent)
    instance.devsupport_context = {'query': 'query', 'run_id': 'run', 'session': 'session',
                                   'mapping': source_mapping(source())}
    instance.attachments = []
    instance.earlier_attachments = []
    instance.skipped_attachments = []
    instance.retrieved_docs = []
    instance._guardrail_stage = lambda *args: None
    return instance


def test_shared_answer_once_no_native_generation(monkeypatch):
    instance = agent()
    answer = Mock(return_value={'evidence_ids': [], 'answer_text': 'Need version', 'decision': 'clarify'})
    monkeypatch.setattr('docsgpt.agents.devsupport_agent.BridgeClient.answer', answer)
    instance._llm_gen = Mock(side_effect=AssertionError('native generator must not run'))
    events = list(instance._gen_inner('query', SimpleNamespace(stacks=[])))
    assert {'answer': 'Need version'} in events
    answer.assert_called_once()
    instance._llm_gen.assert_not_called()


def test_output_guard_blocks_shared_answer(monkeypatch):
    from docsgpt.guardrails.types import Stage
    instance = agent()
    decision = SimpleNamespace(blocked=True, block_message='blocked')
    instance._guardrail_stage = lambda text, stage: decision if stage == Stage.OUTPUT else None
    instance._guardrail_block_event = lambda *args: {'answer': 'blocked'}
    monkeypatch.setattr('docsgpt.agents.devsupport_agent.BridgeClient.answer',
                        lambda *args: {'evidence_ids': [], 'answer_text': 'secret output'})
    events = list(instance._gen_inner('query', SimpleNamespace(stacks=[])))
    assert {'answer': 'blocked'} in events
    assert 'secret output' not in str(events)


def test_provider_failure_is_not_reported_as_missing_knowledge(monkeypatch):
    instance = agent()
    monkeypatch.setattr('docsgpt.agents.devsupport_agent.BridgeClient.answer',
                        lambda *args: {'evidence_ids': [], 'answer_text': '', 'decision': 'insufficient',
                                       'trace': {'gate': {'reason': 'gate_validation_or_provider_failure'}}})
    events = list(instance._gen_inner('query', SimpleNamespace(stacks=[])))
    assert any('could not complete' in event.get('answer', '') for event in events)
    assert 'does not provide sufficient evidence' not in str(events)


def test_snapshot_lookup_after_native_auth(monkeypatch):
    app = Flask(__name__)
    raw = {'doc_id': 'official', 'text': 'original', 'metadata': {}}
    load = Mock(return_value=(source(), SimpleNamespace(payload=lambda: {})))
    lookup = Mock(return_value=raw)
    monkeypatch.setattr(chunks, 'db_readonly', connection)
    monkeypatch.setattr(chunks, 'load_source', load)
    monkeypatch.setattr(chunks.BridgeClient, 'evidence', lookup)
    with app.test_request_context('/?chunk_key=0123456789abcdef0123456789abcdef'):
        request.decoded_token = {'sub': 'owner'}
        response = chunks.ChunkByKey().get('native')
        assert response.status_code == 200
        assert response.json['chunk']['text'] == 'original'
    load.assert_called_once()
    lookup.assert_called_once()

def test_citation_denial_never_calls_snapshot(monkeypatch):
    app = Flask(__name__)
    lookup = Mock()
    monkeypatch.setattr(chunks, 'db_readonly', connection)
    monkeypatch.setattr(chunks, 'load_source', Mock(side_effect=chunks.AccessDenied(403, 'Denied')))
    monkeypatch.setattr(chunks.BridgeClient, 'evidence', lookup)
    with app.test_request_context('/?chunk_key=0123456789abcdef0123456789abcdef'):
        request.decoded_token = {'sub': 'stranger'}
        assert chunks.ChunkByKey().get('native').status_code == 403
    lookup.assert_not_called()


def test_native_citation_uses_existing_store(monkeypatch):
    app = Flask(__name__)
    raw = {'doc_id': 'native_chunk', 'text': 'uploaded text', 'metadata': {}}
    store = Mock()
    store.get_chunk_by_key.return_value = raw
    lookup = Mock()
    monkeypatch.setattr(chunks, 'db_readonly', connection)
    monkeypatch.setattr(chunks, 'load_source', lambda *args: ({'id': 'upload'}, SimpleNamespace(payload=lambda: {})))
    monkeypatch.setattr(chunks, 'get_vector_store', lambda *args: store)
    monkeypatch.setattr(chunks.BridgeClient, 'evidence', lookup)
    with app.test_request_context('/?chunk_key=0123456789abcdef0123456789abcdef'):
        request.decoded_token = {'sub': 'owner'}
        assert chunks.ChunkByKey().get('upload').json['chunk']['text'] == 'uploaded text'
    store.get_chunk_by_key.assert_called_once()
    lookup.assert_not_called()


def test_retrieval_guard_never_releases_or_answers(monkeypatch):
    instance = agent()
    instance.retrieved_docs = [{'doc_id': 'official', 'text': 'sensitive', 'filename': 'doc'}]
    from docsgpt.guardrails.types import Stage
    instance._guardrail_stage = lambda text, stage: (
        SimpleNamespace(blocked=False, redacted=True) if stage == Stage.RETRIEVAL else None)
    answer = Mock()
    monkeypatch.setattr('docsgpt.agents.devsupport_agent.BridgeClient.answer', answer)
    events = list(instance._gen_inner('query', SimpleNamespace(stacks=[])))
    answer.assert_not_called()
    assert 'sensitive' not in str(events)
    assert {'sources': []} in events
