"""Native adapter exercises; only external embedding/LLM boundaries are injected."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from devsupport.src.devsupport.config import MODEL_ID
from devsupport.src.devsupport.native_baseline import export_native_index, NativeClassicRetriever


def fixture_index(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    rows = [{'doc_id': 'repo/path/doc-1', 'text': 'first original text', 'metadata': {'url': 'https://example.org/1', 'start_byte': 0}},
            {'doc_id': 'repo/path/doc-2', 'text': 'second original text', 'metadata': {'url': 'https://example.org/2'}}]
    vectors = np.zeros((2, 1024), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    np.save(raw / 'vectors.npy', vectors)
    for name, value in [('ids.json', [r['doc_id'] for r in rows]), ('corpus.json', rows)]:
        (raw / name).write_text(json.dumps(value), encoding='utf-8')
    files = {name: hashlib.sha256((raw / name).read_bytes()).hexdigest() for name in ['ids.json', 'corpus.json', 'vectors.npy']}
    manifest = {'count': 2, 'dimension': 1024, 'contextual': False, 'files': files,
                'canonical_corpus_sha256': 'fixture', 'provenance': {'model': MODEL_ID, 'dimension': 1024}}
    (raw / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return raw, rows


def test_export_and_actual_native_classic_search(tmp_path, monkeypatch):
    from docsgpt.vectorstore.faiss import FaissStore
    from docsgpt.storage.local import LocalStorage
    from docsgpt.retriever.classic_rag import ClassicRAG
    from docsgpt.core.settings import settings
    raw, rows = fixture_index(tmp_path)
    destination = tmp_path / 'indexes' / 'native-u'
    manifest = export_native_index(raw, destination)
    assert manifest['count'] == 2
    assert not (destination / 'index.pkl').exists()
    from docsgpt.vectorstore.base import RemoteEmbeddings
    calls = []
    def post(url, headers, json, timeout):
        calls.append(json)
        return SimpleNamespace(raise_for_status=lambda: None, status_code=200,
                               json=lambda: {'data': [{'index': 0, 'embedding': [1.] + [0.] * 1023}]})
    monkeypatch.setattr('docsgpt.vectorstore.base.requests.post', post)
    remote = RemoteEmbeddings('http://127.0.0.1:8093', MODEL_ID)
    # Bound tokenizer dependency without overriding native limit resolution/truncation.
    monkeypatch.setattr(remote, '_token_counter', lambda: SimpleNamespace(
        count=lambda text: len(text.encode('utf-8')), split=lambda text, limit: [text[:limit]]))
    monkeypatch.setattr(FaissStore, '_get_embeddings', lambda *args: remote)
    monkeypatch.setattr('docsgpt.vectorstore.faiss.StorageCreator.get_storage', lambda: LocalStorage(str(tmp_path)))
    monkeypatch.setattr('docsgpt.retriever.classic_rag.LLMCreator.create_llm', lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(settings, 'VECTOR_STORE', 'faiss')
    monkeypatch.setattr(settings, 'EMBEDDINGS_NAME', MODEL_ID)
    monkeypatch.setattr(settings, 'EMBEDDINGS_BASE_URL', 'http://127.0.0.1:8093')
    monkeypatch.setattr(settings, 'EMBEDDINGS_MAX_QUERY_TOKENS', 0)
    monkeypatch.setattr(settings, 'EMBEDDINGS_MAX_INPUT_TOKENS', 1)
    with pytest.raises(ValueError, match='unclipped shared local'):
        NativeClassicRetriever('native-u', destination)
    assert not calls
    monkeypatch.setattr(settings, 'EMBEDDINGS_MAX_INPUT_TOKENS', 0)
    assert remote._resolve_input_limit() is None
    adapter = NativeClassicRetriever('native-u', destination)
    query = 'exact query preserving Unicode and spaces: \u4f60\u597d  query'
    hits = adapter.search(query)
    assert calls == [{'model': MODEL_ID, 'input': query}]

    assert adapter.last_trace['native_class'] == 'docsgpt.retriever.classic_rag.ClassicRAG'
    assert isinstance(adapter.last_retriever, ClassicRAG)
    assert [h.evidence.doc_id for h in hits] == [r['doc_id'] for r in rows]
    assert [h.score for h in hits] == [0., 2.]
    assert [h.evidence.text for h in hits] == [r['text'] for r in rows]
    assert hits[0].evidence.metadata == rows[0]['metadata']
    assert all(h.score_kind == 'l2_distance' for h in hits)
    with pytest.raises(FileExistsError):
        export_native_index(raw, destination)


def test_export_rejects_corrupt_vectors_before_destination(tmp_path):
    raw, _ = fixture_index(tmp_path)
    (raw / 'vectors.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        export_native_index(raw, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_native_upload_and_query_use_service_protocol(monkeypatch, tmp_path):
    """Exercise native upload/query payloads; service validation is tested in core."""
    from docsgpt.core.settings import settings
    from docsgpt.vectorstore.base import RemoteEmbeddings
    from docsgpt.vectorstore.document_class import Document
    from docsgpt.vectorstore.faiss import FaissStore
    from docsgpt.storage.local import LocalStorage

    calls = []
    def post(url, headers, json, timeout):
        calls.append(json)
        count = len(json['input']) if isinstance(json['input'], list) else 1
        vector = [1.] + [0.] * 1023
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'data': [{'index': i, 'embedding': vector} for i in range(count)]})
    monkeypatch.setattr('docsgpt.vectorstore.base.requests.post', post)
    monkeypatch.setattr(settings, 'EMBEDDINGS_MAX_QUERY_TOKENS', 0)
    remote = RemoteEmbeddings('http://127.0.0.1:8093', MODEL_ID)
    remote.dimension = 1024
    monkeypatch.setattr(remote, '_resolve_input_limit', lambda: None)
    monkeypatch.setattr(FaissStore, '_get_embeddings', lambda *args: remote)
    monkeypatch.setattr('docsgpt.vectorstore.faiss.StorageCreator.get_storage', lambda: LocalStorage(str(tmp_path)))
    store = FaissStore('upload-fixture', '', docs_init=[Document(page_content='original uploaded text', metadata={})])
    found = store.search_with_scores('uploaded text question')
    assert calls == [{'model': MODEL_ID, 'input': ['original uploaded text']},
                     {'model': MODEL_ID, 'input': 'uploaded text question'}]
    assert found[0][0].page_content == 'original uploaded text'
    assert found[0][1] == 0.
