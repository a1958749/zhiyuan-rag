"""Authentic upstream ClassicRAG/FAISS baseline using existing frozen Qwen vectors.

This is configured U (shared Qwen embeddings), not the stock Granite model.
Generation is deliberately outside this adapter: comparisons must use the same
Opus generator/prompt/context packer as the other answer experiments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import quote, unquote

import numpy as np

from .config import DIMENSION, MODEL_ID
from .types import Evidence, Hit


def file_hash(path: Path) -> str:
    """Hash an artifact with bounded memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_native_index(raw: str | Path, destination: str | Path) -> dict:
    """Convert verified frozen vectors into native L2/JSON files without inference.

    Destination must not exist; publish the complete directory atomically. The
    native docstore retains original metadata separately from citation labels.
    """
    import faiss
    from docsgpt.vectorstore.faiss_docstore import dump_json_sidecar

    raw, destination = Path(raw), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((raw / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['dimension'] != DIMENSION or manifest['provenance']['model'] != MODEL_ID
            or manifest['contextual'] is not False):
        raise ValueError('requires raw configured Qwen vectors')
    for name in ('vectors.npy', 'ids.json', 'corpus.json'):
        if file_hash(raw / name) != manifest['files'][name]:
            raise ValueError(f'raw artifact checksum mismatch: {name}')
    ids = json.loads((raw / 'ids.json').read_text(encoding='utf-8'))
    rows = json.loads((raw / 'corpus.json').read_text(encoding='utf-8'))
    vectors = np.load(raw / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    if (len(set(ids)) != len(ids) or ids != [r['doc_id'] for r in rows]
            or len(ids) != manifest['count'] or vectors.shape != (len(ids), DIMENSION)):
        raise ValueError('corpus/vector identity or shape mismatch')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4):
        raise ValueError('vectors must be finite and normalized')
    documents = {}
    for row in rows:
        text, metadata = row['text'], row['metadata']
        documents[row['doc_id']] = {'page_content': text, 'metadata': {
            **metadata, 'official_doc_id': row['doc_id'], 'original_metadata': metadata,
            # ClassicRAG strips slash-separated title prefixes; encoding preserves
            # the full official identity, including distinct duplicate-text rows.
            'title': quote(row['doc_id'], safe=''),
            'source': metadata.get('url') or metadata.get('source') or '',
            'chunk_key': hashlib.md5(text.encode('utf-8')).hexdigest(),
        }}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.native-export-', dir=destination.parent))
    try:
        index = faiss.IndexFlatL2(DIMENSION)
        index.add(np.asarray(vectors, dtype=np.float32))
        faiss.write_index(index, str(staging / 'index.faiss'))
        (staging / 'index.json').write_bytes(dump_json_sidecar(documents, dict(enumerate(ids))))
        exported = {'schema': 1, 'profile': 'U', 'variant': 'upstream-classic-shared-qwen',
                    'count': len(ids), 'dimension': DIMENSION, 'score_kind': 'l2_distance',
                    'native_index': 'IndexFlatL2', 'provenance': manifest['provenance'],
                    'canonical_corpus_sha256': manifest['canonical_corpus_sha256'],
                    'raw_manifest_sha256': file_hash(raw / 'manifest.json'),
                    'files': {name: file_hash(staging / name) for name in ('index.faiss', 'index.json')}}
        (staging / 'native-manifest.json').write_text(json.dumps(exported, indent=2), encoding='utf-8')
        staging.rename(destination)
        return exported
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class NativeClassicRetriever:
    """Call the actual upstream ClassicRAG class, returning original-ID Hits."""

    def __init__(self, source_id: str, index_dir: str | Path, doc_token_limit: int = 50000):
        """Verify exported snapshot and require the configured shared embedding model."""
        from docsgpt.core.settings import settings
        from docsgpt.storage.local import LocalStorage
        from docsgpt.storage.storage_creator import StorageCreator
        from docsgpt.vectorstore.faiss import get_vectorstore
        from docsgpt.vectorstore.faiss_docstore import load_json_sidecar

        self.source_id = source_id
        self.doc_token_limit = doc_token_limit
        root = Path(index_dir)
        storage = StorageCreator.get_storage()
        if (not isinstance(storage, LocalStorage)
                or Path(storage._get_full_path(get_vectorstore(source_id))).resolve() != root.resolve()):
            raise ValueError('source_id must resolve to the verified native index directory')
        self.manifest = json.loads((root / 'native-manifest.json').read_text(encoding='utf-8'))
        for name in ('index.faiss', 'index.json'):
            if file_hash(root / name) != self.manifest['files'][name]:
                raise ValueError('native artifact checksum mismatch')
        if (settings.VECTOR_STORE != 'faiss' or settings.EMBEDDINGS_NAME != MODEL_ID
                or (settings.EMBEDDINGS_BASE_URL or '').rstrip('/') != 'http://127.0.0.1:8093'
                or settings.EMBEDDINGS_MAX_QUERY_TOKENS != 0
                or settings.EMBEDDINGS_MAX_INPUT_TOKENS != 0):
            raise ValueError('native U requires FAISS and unclipped shared local Qwen embeddings')
        self.documents, _ = load_json_sidecar((root / 'index.json').read_bytes())
        self.last_trace = {}
        self.last_retriever = None

    def search(self, query: str, k: int = 50) -> list[Hit]:
        """Run native retrieval with no chat history, hence no LLM rephrase call.

        Native ClassicRAG fetches up to 2*k internally then returns up to k under
        its 90%-of-doc_token_limit budget. Distances are lower-is-better; FreshStack
        metrics consume this returned ordering, never sort distances descending.
        """
        from docsgpt.retriever.classic_rag import ClassicRAG

        if not isinstance(query, str) or not query.strip() or not 1 <= k <= 50:
            raise ValueError('nonempty query and k between 1 and 50 required')
        started = time.perf_counter()
        native = ClassicRAG(source={'question': query, 'active_docs': [self.source_id]},
                            chat_history=[], chunks=k, doc_token_limit=self.doc_token_limit,
                            model_id=None, llm_name='openai', api_key='unused-retrieval-only',
                            defer_rephrase=True, include_scores=True)
        self.last_retriever = native
        results = native.search()
        hits = []
        for rank, result in enumerate(results, 1):
            doc_id = unquote(result['title'])
            stored = self.documents.get(doc_id)
            if (stored is None or stored['page_content'] != result['text']
                    or result.get('score_kind') != 'l2_distance'):
                raise ValueError('native retrieval identity/score mismatch')
            evidence = Evidence(doc_id, stored['page_content'], stored['metadata']['original_metadata'])
            hits.append(Hit(evidence, result['score'], rank, 'l2_distance'))
        if not hits:
            raise RuntimeError('native U returned no documents; inspect native retrieval errors')
        self.last_trace = {'profile': 'U', 'variant': self.manifest['variant'],
                           'native_class': 'docsgpt.retriever.classic_rag.ClassicRAG',
                           'native_store': 'docsgpt.vectorstore.faiss.FaissStore',
                           'embedding': self.manifest['provenance'], 'requested_candidates': k,
                           'native_internal_fetch_k': min(max(k * 2, 20), 500),
                           'native_doc_token_limit': self.doc_token_limit,
                           'native_effective_token_budget': max(int(self.doc_token_limit * .9), 100),
                           'rephrase_calls': 0, 'generation_calls': 0,
                           'canonical_corpus_sha256': self.manifest['canonical_corpus_sha256'],
                           'ranked_ids': [h.evidence.doc_id for h in hits],
                           'score_kind': 'l2_distance', 'seconds': time.perf_counter() - started}
        return hits
