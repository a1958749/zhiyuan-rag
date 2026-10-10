"""Shared pipeline service with scoped, bounded, one-use prepared answers."""
import hashlib
import os
import secrets
import time
from collections import OrderedDict
from copy import deepcopy
from threading import RLock


class PipelineService:
    """Serialize shared model adapters and retain only server-created evidence envelopes."""

    def __init__(self, pipelines: dict, revision: str, corpus: dict):
        self.pipelines, self.revision = pipelines, revision
        self.corpus = {key: value.to_corpus_dict() if hasattr(value, 'to_corpus_dict') else value
                       for key, value in corpus.items()}
        self.keys = {}
        for key in sorted(self.corpus):
            digest = hashlib.md5(self.corpus[key]['text'].encode('utf-8')).hexdigest()
            self.keys.setdefault(digest, key)
        self.pending = OrderedDict()
        self.lock = RLock()

    def _validate(self, profile: str, revision: str, session: str) -> None:
        """Check trusted deployment profile, snapshot, and backend request scope."""
        if profile not in self.pipelines or revision != self.revision or not session:
            raise ValueError('unknown profile, corpus revision or request scope')

    def search(self, query: str, profile: str, revision: str, session: str) -> dict:
        """Retrieve/pack once and retain exact evidence server-side, including empty runs."""
        with self.lock:
            self._validate(profile, revision, session)
            envelope = self.pipelines[profile].retrieve(query)
            run_id = secrets.token_urlsafe(32)
            self.pending[run_id] = (query, profile, revision, session, time.monotonic(), envelope)
            while len(self.pending) > 32:
                self.pending.popitem(last=False)
            return {**deepcopy(envelope), 'run_id': run_id}

    def answer(self, run_id: str, query: str, profile: str, revision: str, session: str) -> dict:
        """Validate scope/query/age then consume once without accepting client evidence."""
        with self.lock:
            self._validate(profile, revision, session)
            state = self.pending.get(run_id)
            if state is None or state[:4] != (query, profile, revision, session):
                raise ValueError('unknown or mismatched prepared run')
            if time.monotonic() - state[4] > 1800:
                del self.pending[run_id]
                raise ValueError('expired prepared run')
            del self.pending[run_id]
            result = self.pipelines[profile].answer(query, state[5])
            return {**result, 'run_id': run_id, 'profile': profile, 'corpus_revision': revision}

    def evidence(self, revision: str, doc_id: str | None = None, chunk_key: str | None = None) -> dict:
        """Lookup original snapshot bytes; duplicate MD5 maps to lexicographically first ID."""
        if revision != self.revision:
            raise ValueError('unknown corpus revision')
        if doc_id is None:
            doc_id = self.keys.get(chunk_key)
        row = self.corpus.get(doc_id)
        if row is None or (chunk_key and hashlib.md5(row['text'].encode()).hexdigest() != chunk_key):
            raise ValueError('unknown evidence identity')
        return deepcopy(row)

    def chunks(self, revision: str, page: int = 1, per_page: int = 20,
               path: str = '', search: str = '') -> dict:
        """Page through snapshot data for native Knowledge navigation."""
        if revision != self.revision or page < 1 or not 1 <= per_page <= 100:
            raise ValueError('invalid snapshot page')
        rows = [row for row in self.corpus.values()
                if (not path or path in str(row['metadata'].get('source', row['metadata'].get('path', ''))))
                and (not search or search.lower() in (row['text'] + str(row['metadata'])).lower())]
        return {'page': page, 'per_page': per_page, 'total': len(rows),
                'chunks': deepcopy(rows[(page - 1) * per_page:page * per_page]),
                'path': path or None, 'search': search or None}


def create_app(service: PipelineService):
    """Create a private loopback FastAPI app; optional shared token applies to all evidence APIs."""
    from fastapi import FastAPI, HTTPException, Request
    app = FastAPI(title='DevSupport pipeline')
    # FastAPI resolves this postponed annotation in the module namespace.
    globals()['Request'] = Request

    @app.get('/health')
    def health():
        return {'status': 'ok', 'profiles': list(service.pipelines), 'corpus_revision': service.revision}

    @app.post('/v1/{operation}')
    async def invoke(operation: str, request: Request):
        if request.headers.get('origin') or request.headers.get('content-type', '').split(';')[0] != 'application/json':
            raise HTTPException(403, 'Backend requests only')
        token = os.environ.get('DEVSUPPORT_SERVICE_TOKEN')
        if token and not secrets.compare_digest(token, request.headers.get('X-Devsupport-Token', '')):
            raise HTTPException(403, 'Forbidden')
        if operation not in ('search', 'answer', 'evidence', 'chunks'):
            raise HTTPException(404, 'Unknown operation')
        try:
            from starlette.concurrency import run_in_threadpool
            body = await request.json()
            return await run_in_threadpool(getattr(service, operation), **body)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(404 if operation == 'evidence' else 422,
                                'Invalid, expired or unsupported request') from exc
    return app


class ModelEmbedder:
    """HTTP GPU adapter enforcing the complete semantic index contract."""

    def __init__(self, provenance: dict, url: str = 'http://127.0.0.1:8093'):
        self.provenance, self.url, self.stats = provenance, url, {}
        if provenance.get('dimension') != 1024:
            raise ValueError('expected dimension 1024')

    def encode(self, texts: list[str], is_query: bool = False):
        """Validate provenance and every vector before use."""
        import httpx
        import numpy as np
        from .embedding import normalized
        response = httpx.post(self.url + '/v1/embeddings', json={'input': texts, 'is_query': is_query}, timeout=180)
        response.raise_for_status()
        body = response.json()
        if body.get('provenance') != self.provenance:
            raise ValueError('GPU embedding/index provenance mismatch')
        data = body['data']
        if [row['index'] for row in data] != list(range(len(texts))):
            raise ValueError('GPU embedding row identity mismatch')
        self.stats = body['trace']
        return normalized(np.asarray([row['embedding'] for row in data]), 1024)


class ModelReranker:
    """Bounded GPU HTTP ranker preserving candidates and observable fallback."""

    def __init__(self, variant: str, url: str = 'http://127.0.0.1:8093'):
        if variant not in ('R1', 'R2'):
            raise ValueError('unknown reranker')
        self.variant, self.url, self.last_trace = variant, url, {}

    def rank(self, query: str, hits: list) -> list:
        """Retain H1 on transport failure with an explicit failure trace."""
        import httpx
        from .types import Hit
        started = time.monotonic()
        try:
            response = httpx.post(self.url + '/v1/rerank',
                                  json={'model': self.variant, 'query': query,
                                        'documents': [hit.evidence.to_corpus_dict() for hit in hits]}, timeout=300)
            response.raise_for_status()
            body = response.json()
            source = {hit.evidence.doc_id: hit.evidence for hit in hits}
            rows = body['results']
            if len(rows) != len(hits) or {row['doc_id'] for row in rows} != set(source):
                raise ValueError('GPU ranker changed candidate identities')
            self.last_trace = body['trace']
            return [Hit(source[row['doc_id']], row['score'], row['rank'], row['score_kind']) for row in rows]
        except (httpx.HTTPError, ValueError, KeyError):
            self.last_trace = {'variant': self.variant, 'fallback': True, 'reason': 'gpu_transport_or_contract',
                               'elapsed_seconds': time.monotonic() - started}
            return hits
