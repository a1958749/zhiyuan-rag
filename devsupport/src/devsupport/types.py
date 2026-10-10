"""Shared offline and retrieval types."""

from dataclasses import dataclass
from copy import deepcopy
from typing import Any


def _corpus_metadata(value: Any) -> Any:
    """Copy metadata while excluding offline evaluation annotation keys recursively."""
    if isinstance(value, dict):
        return {
            key: _corpus_metadata(item)
            for key, item in value.items()
            if not any(label in str(key).lower() for label in ("answer", "nugget", "qrel", "relevant_corpus"))
        }
    if isinstance(value, (list, tuple)):
        return [_corpus_metadata(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class Evidence:
    """An official corpus chunk with its original metadata."""

    doc_id: str
    text: str
    metadata: dict

    def to_corpus_dict(self) -> dict:
        """Serialize a detached corpus-only row without offline annotation fields.

        Returns:
            A JSON-compatible corpus row containing original ID, text and safe metadata.
        """
        return {"doc_id": self.doc_id, "text": self.text, "metadata": _corpus_metadata(self.metadata)}


@dataclass
class Hit:
    """A retrieval result with an explicit score interpretation."""

    evidence: Evidence
    score: float
    rank: int
    score_kind: str


@dataclass(frozen=True)
class Question:
    """An offline evaluation query and its reference annotations."""

    query_id: str
    query_title: str
    query_text: str
    answer_text: str
    nuggets: list[dict]
    metadata: dict

    @property
    def query(self) -> str:
        """Join original query title and body."""
        return self.query_title + " " + self.query_text
