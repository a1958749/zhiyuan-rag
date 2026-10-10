import numpy as np
from fastapi.testclient import TestClient
from devsupport.config import MODEL_ID
from devsupport.model_service import create_app


def test_native_and_core_embedding_protocols(monkeypatch):
    from devsupport.embedding import Embedder
    calls = []
    def encode(self, texts, is_query=False):
        calls.append((texts, is_query))
        return np.ones((len(texts), 1024), dtype=np.float32)
    monkeypatch.setattr(Embedder, 'encode', encode)
    monkeypatch.setattr(Embedder, 'provenance', property(lambda self: {'model': MODEL_ID}))
    client = TestClient(create_app())
    for body, expected in [({'model': MODEL_ID, 'input': 'q'}, (['q'], True)),
                           ({'model': MODEL_ID, 'input': ['doc']}, (['doc'], False)),
                           ({'input': ['q'], 'is_query': True}, (['q'], True))]:
        response = client.post('/v1/embeddings', json=body)
        assert response.status_code == 200
        assert calls[-1] == expected
        assert response.json()['provenance'] == {'model': MODEL_ID}
        assert len(response.json()['data'][0]['embedding']) == 1024
    accepted = len(calls)
    for body in [{'model': 'wrong', 'input': 'q'}, {'input': 'q'},
                 {'model': MODEL_ID, 'input': ['q'], 'is_query': 'true'},
                 {'model': MODEL_ID, 'input': []}, {'model': MODEL_ID, 'input': [3]},
                 {'model': MODEL_ID, 'input': 'q', 'nuggets': []}]:
        assert client.post('/v1/embeddings', json=body).status_code == 422
    assert len(calls) == accepted
