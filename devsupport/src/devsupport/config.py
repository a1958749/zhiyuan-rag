"""Frozen retrieval constants shared by indexing and live search."""

from dataclasses import dataclass
import json
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'
MODEL_REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
QUERY_INSTRUCTION = ('Instruct: Given a technical question, retrieve relevant code snippets or technical '
                     'documentation that best answer the question\nQuery: ')
MAX_LENGTH = 2048
DIMENSION = 1024
LEG_CAP = 50
RRF_K = 60
BM25_PARAMETERS = {'k1': 1.5, 'b': 0.75, 'method': 'lucene', 'backend': 'numpy'}


@dataclass(frozen=True)
class RetrievalConfig:
    """One named experiment with the same bounded production candidates."""

    variant: str = 'H1'
    leg_cap: int = LEG_CAP
    rrf_k: int = RRF_K

    def __post_init__(self):
        """Reject unsupported variants or accidental experiment drift."""
        if self.variant not in ('B0', 'B1', 'H0', 'H1', 'C'):
            raise ValueError('unknown retrieval variant')
        if self.leg_cap != LEG_CAP or self.rrf_k != RRF_K:
            raise ValueError('production retrieval requires leg_cap=50 and rrf_k=60')

    @classmethod
    def load(cls, path: str | Path):
        """Read only the small explicit JSON config contract."""
        values = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**{key: values[key] for key in ('variant', 'leg_cap', 'rrf_k')})
