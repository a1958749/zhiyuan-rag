"""One bounded corpus-only retrieval API shared by offline and production callers."""

import time

from .config import LEG_CAP, RRF_K, RetrievalConfig
from .types import Hit


def _positive_k(k: int) -> int:
    """Validate request size without accepting bools/floats as candidate counts."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError('k must be a positive integer')
    return min(k, LEG_CAP)


def rrf(legs: list[list[str]], k: int = LEG_CAP) -> list[tuple[str, float]]:
    """Fuse unique exact IDs per leg using k=60 and stable official-ID tie breaks."""
    limit = _positive_k(k)
    scores = {}
    for leg in legs:
        unique = list(dict.fromkeys(leg))[:LEG_CAP]
        for rank, doc_id in enumerate(unique, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:limit]


class Retriever:
    """Search variants B0/B1/H0/H1/C while exposing shared candidate IDs and timing."""

    def __init__(self, index: object, embedder: object = None, variant: str = 'H1'):
        """Bind a loaded CorpusIndex; ordinary and identifier hybrids share raw vectors."""
        self.config = RetrievalConfig(variant)
        self.index, self.embedder = index, embedder
        if bool(index.manifest['contextual']) != (variant == 'C'):
            raise ValueError('variant requires matching raw/contextual index')
        if variant != 'B1' and embedder is None:
            raise ValueError('dense retrieval requires an embedder')
        if variant != 'B1' and any(index.manifest['provenance'].get(key) != value
                                    for key, value in embedder.provenance.items()):
            raise ValueError('query embedder/index provenance mismatch')
        self.pool_ids = []
        self.timings = {}
        self.leg_ids = {}

    def search(self, query: str, k: int = LEG_CAP) -> list[Hit]:
        """Return scored Evidence only; Question/reference annotations never enter this API."""
        if not isinstance(query, str):
            raise TypeError('query must be a string')
        limit = _positive_k(k)
        self.pool_ids, self.timings, self.leg_ids = [], {}, {}
        if not query.strip():
            self.timings = {'total_seconds': 0.0}
            return []
        started = time.perf_counter()
        variant = self.config.variant
        dense, lexical = [], []
        if variant != 'B1':
            stage = time.perf_counter()
            query_vector = self.embedder.encode([query], is_query=True)[0]
            self.timings['embedding_seconds'] = time.perf_counter() - stage
            stage = time.perf_counter()
            dense = self.index.dense_search(query_vector, LEG_CAP)
            self.timings['dense_seconds'] = time.perf_counter() - stage
            self.leg_ids['dense'] = [doc_id for doc_id, _ in dense]
        if variant != 'B0':
            stage = time.perf_counter()
            engine = self.index.identifier if variant in ('H1', 'C') else self.index.ordinary
            lexical = engine.search(query, LEG_CAP)
            self.timings['lexical_seconds'] = time.perf_counter() - stage
            self.leg_ids['lexical'] = [doc_id for doc_id, _ in lexical]
        stage = time.perf_counter()
        if variant == 'B0':
            candidates, kind = dense[:limit], 'cosine'
        elif variant == 'B1':
            candidates, kind = lexical[:limit], 'BM25'
        else:
            candidates = rrf([self.leg_ids['dense'], self.leg_ids['lexical']], limit)
            kind = 'RRF'
        if any(doc_id not in self.index.corpus for doc_id, _ in candidates):
            raise ValueError('retrieval returned unknown corpus ID')
        self.timings['fusion_seconds'] = time.perf_counter() - stage
        self.timings['total_seconds'] = time.perf_counter() - started
        self.pool_ids = [doc_id for doc_id, _ in candidates]
        return [Hit(self.index.corpus[doc_id], score, rank, kind)
                for rank, (doc_id, score) in enumerate(candidates, 1)]
