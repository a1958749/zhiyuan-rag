"""BM25S lexical retrieval with disclosed ordinary and code-aware tokenization."""

import re
from pathlib import Path

import numpy as np

from .config import BM25_PARAMETERS, LEG_CAP

_IDENTIFIER = re.compile(r'[\w]+(?:[.@:/+\-][\w]+)*', re.UNICODE)
_CAMEL = re.compile(r'[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+')


def ordinary_tokens(text: str, is_query: bool = False) -> list[str]:
    """Lowercase word tokens for documents and queries; no stopwords or stemming."""
    return re.findall(r'\w+', text.lower(), flags=re.UNICODE)


def identifier_tokens(text: str, is_query: bool = False) -> list[str]:
    """Keep case-sensitive full identifiers plus lowercase aliases and bounded subterms.

    Query and document tokenization are identical: exact case gains its own token,
    while the lowercase alias supports differently cased user questions. Versions,
    error codes, snake names and dotted/package identifiers keep their complete form.
    """
    output = []
    for token in _IDENTIFIER.findall(text):
        terms = [token, token.lower()]
        for piece in re.split(r'[_@:/+\-]', token):
            terms.append(piece.lower())
            terms.extend(part.lower() for part in _CAMEL.findall(piece) if len(part) >= 2)
        # Deduplicate aliases within each occurrence; repeated occurrences retain BM25 TF.
        output.extend(dict.fromkeys(term for term in terms if term))
    return output


class LexicalIndex:
    """Thin BM25S wrapper; serialized NumPy/JSON only and never pickle."""

    def __init__(self, ids: list[str], identifier: bool = False):
        """Bind exact corpus IDs to a chosen tokenizer."""
        self.ids = list(ids)
        self.identifier = identifier
        self.engine = None

    @property
    def tokenizer(self):
        """Return the same tokenizer for corpus and query."""
        return identifier_tokens if self.identifier else ordinary_tokens

    def build(self, texts: list[str]):
        """Compute BM25S Lucene scores with fixed shared parameters."""
        import bm25s

        self.engine = bm25s.BM25(**BM25_PARAMETERS)
        tokens = [self.tokenizer(text) for text in texts]
        # BM25S supports empty rows; preserve their actual zero document lengths.
        # Its all-empty vocabulary edge needs the library's unqueryable empty token.
        if not any(tokens):
            tokens = [[''] for _ in tokens]
        self.engine.index(tokens, show_progress=False)
        return self

    def search(self, query: str, k: int = LEG_CAP) -> list[tuple[str, float]]:
        """Return positive BM25 scores with exact-ID tie breaking."""
        if k <= 0:
            raise ValueError('k must be positive')
        tokens = self.tokenizer(query, is_query=True)
        if not tokens:
            return []
        scores = np.asarray(self.engine.get_scores(tokens))
        candidates = [(self.ids[i], float(score)) for i, score in enumerate(scores) if score > 0]
        return sorted(candidates, key=lambda row: (-row[1], row[0]))[:min(k, LEG_CAP)]

    def save(self, path: str | Path):
        """Save BM25S without corpus payloads or pickle."""
        self.engine.save(str(path), allow_pickle=False)

    @classmethod
    def load(cls, path: str | Path, ids: list[str], identifier: bool = False):
        """Load safe BM25S arrays after CorpusIndex verifies their checksums."""
        import bm25s

        index = cls(ids, identifier)
        index.engine = bm25s.BM25.load(str(path), allow_pickle=False, load_corpus=False)
        if index.engine.scores['num_docs'] != len(ids):
            raise ValueError('lexical count mismatch')
        return index
