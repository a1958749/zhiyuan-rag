"""Loopback GPU model service accepting plain corpus evidence only."""

from threading import Lock

from .config import MODEL_ID
from .types import Evidence, Hit


def _reject_annotations(value: object) -> None:
    """Reject offline annotation keys at every nesting depth."""
    if isinstance(value, dict):
        for key, item in value.items():
            if any(label in str(key).lower() for label in ('answer', 'nugget', 'qrel', 'relevant_corpus', 'question')):
                raise ValueError('offline annotations are forbidden')
            _reject_annotations(item)
    elif isinstance(value, list):
        for item in value:
            _reject_annotations(item)


def validate_request(body: dict) -> dict:
    """Validate rerank requests before any loading or inference."""
    _reject_annotations(body)
    if not isinstance(body, dict) or set(body) != {'model', 'query', 'documents'}:
        raise ValueError('expected model, query, documents')
    if body['model'] not in ('R1', 'R2'):
        raise ValueError('unknown model')
    if not isinstance(body['query'], str) or not body['query'].strip():
        raise ValueError('query must be nonempty text')
    docs = body['documents']
    if not isinstance(docs, list) or len(docs) > 50:
        raise ValueError('at most 50 documents')
    for doc in docs:
        if not isinstance(doc, dict) or set(doc) != {'doc_id', 'text', 'metadata'}:
            raise ValueError('expected corpus evidence')
        if not isinstance(doc['doc_id'], str) or not doc['doc_id'] or not isinstance(doc['text'], str):
            raise ValueError('invalid evidence ID/text')
        if not isinstance(doc['metadata'], dict):
            raise ValueError('metadata must be an object')
    if len({doc['doc_id'] for doc in docs}) != len(docs):
        raise ValueError('duplicate evidence IDs')
    return body


def create_app(unload_before_rerank: bool = False):
    """Create a lazy, serialized GPU service without loading weights at startup."""
    from fastapi import FastAPI, HTTPException

    from .embedding import Embedder
    from .reranking import JinaListwiseReranker, QwenPairwiseReranker

    app = FastAPI(title='DevSupport local models')
    lock = Lock()
    models = {'R1': QwenPairwiseReranker(), 'R2': JinaListwiseReranker()}
    embedder = Embedder()

    @app.get('/health')
    def health() -> dict:
        """Report liveness and loaded rankers without causing GPU work."""
        return {'status': 'ok', 'resident': [key for key, model in models.items() if model.model is not None]}

    @app.post('/v1/rerank')
    def rerank(body: dict) -> dict:
        """Return complete ranked IDs/scores plus auditable fallback status."""
        try:
            validate_request(body)
            hits = [Hit(Evidence(**doc), 0.0, i, 'input_order') for i, doc in enumerate(body['documents'], 1)]
            with lock:
                if unload_before_rerank:
                    for key, model in models.items():
                        if key != body['model'] and model.model is not None:
                            model.unload()
                    if embedder.model is not None:
                        embedder.model = None
                        import gc
                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
                model = models[body['model']]
                result = model.rank(body['query'], hits)
                return {'results': [{'doc_id': h.evidence.doc_id, 'score': h.score, 'rank': h.rank,
                                     'score_kind': h.score_kind} for h in result], 'trace': model.last_trace}
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post('/v1/embeddings')
    def embeddings(body: dict) -> dict:
        """Embed strings using the same local query/document protocol as retrieval."""
        try:
            _reject_annotations(body)
            if set(body) - {'input', 'is_query', 'model'} or 'input' not in body:
                raise ValueError('expected input and optional is_query/model')
            if 'model' in body and body['model'] != MODEL_ID:
                raise ValueError('unknown embedding model')
            native_query = 'model' in body and isinstance(body['input'], str)
            texts = [body['input']] if native_query else body['input']
            if not isinstance(texts, list) or not 1 <= len(texts) <= 50:
                raise ValueError('input must contain 1 to 50 strings')
            if any(not isinstance(text, str) for text in texts):
                raise ValueError('input must contain strings')
            if not isinstance(body.get('is_query', False), bool):
                raise ValueError('is_query must be boolean')
            with lock:
                vectors = embedder.encode(texts, is_query=body.get('is_query', native_query))
                return {'data': [{'index': i, 'embedding': row.tolist()} for i, row in enumerate(vectors)],
                        'trace': embedder.stats, 'provenance': embedder.provenance}
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
