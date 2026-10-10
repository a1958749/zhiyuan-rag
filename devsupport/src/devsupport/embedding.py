"""Lazy offline Qwen3 embedding inference with observable truncation and OOM retries."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import DIMENSION, MAX_LENGTH, MODEL_ID, MODEL_REVISION, QUERY_INSTRUCTION


def normalized(vectors: np.ndarray, dimension: int) -> np.ndarray:
    """Validate and normalize finite float32 row vectors, rejecting zero vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != dimension or not np.isfinite(vectors).all():
        raise ValueError('invalid embedding shape/dimension or nonfinite vectors')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('zero embedding vector')
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def last_token_pool(hidden: object, attention_mask: object):
    """Pool each row's last nonpadding token for either padding direction."""
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    last = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if torch.any(last < 0):
        raise ValueError('cannot pool an empty attention row')
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class Embedder:
    """Direct Transformers local model; never silently falls back from CUDA."""

    dimension = DIMENSION

    def __init__(self, model_path: str | Path | None = None, batch_size: int = 16, device: str = 'cuda'):
        """Set protocol without importing Torch or loading expensive model weights."""
        self.model_path = Path(model_path) if model_path else Path(__file__).resolve().parents[2] / 'model-cache/embedding'
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if device != 'cuda':
            raise ValueError('this production embedder requires CUDA')
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.tokenizer = None
        self.stats = {}

    @property
    def provenance(self) -> dict:
        """Return the pinned semantic model contract used in safe resume checks."""
        manifest_path = self.model_path / 'artifact-manifest.json'
        archive = json.loads(manifest_path.read_text(encoding='utf-8'))
        if archive['repo'] != MODEL_ID or archive['revision'] != MODEL_REVISION:
            raise ValueError('local model archive revision mismatch')
        return {'model': MODEL_ID, 'revision': MODEL_REVISION, 'dimension': self.dimension,
                'query_instruction': QUERY_INSTRUCTION, 'max_length': MAX_LENGTH,
                'dtype': 'bfloat16', 'device': self.device, 'attention': 'sdpa', 'pooling': 'last_nonpadding',
                'model_files_sha256': archive['files']}

    def validate_local_artifacts(self):
        """Verify archived weights/tokenizer bytes before local inference."""
        for name, record in self.provenance['model_files_sha256'].items():
            source = self.model_path / name
            if not source.resolve().is_relative_to(self.model_path.resolve()) or not source.is_file():
                raise ValueError('local model artifact missing or unsafe path')
            digest = hashlib.sha256()
            with source.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                raise ValueError(f'local model artifact checksum mismatch: {name}')

    def _load(self):
        """Load the pinned local snapshot only after the first encode request."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback is permitted')
        self.validate_local_artifacts()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True,
                                                     trust_remote_code=False, padding_side='left')
        self.model = AutoModel.from_pretrained(str(self.model_path), local_files_only=True,
                                              trust_remote_code=False, dtype=torch.bfloat16,
                                              attn_implementation='sdpa').to('cuda').eval()
        if self.model.config.hidden_size != self.dimension:
            raise ValueError('local model dimension mismatch')

    def _token_lengths(self, texts: list[str]) -> list[int]:
        """Measure original encoded length without truncating either query or passage."""
        lengths = []
        for start in range(0, len(texts), 256):
            batch = self.tokenizer(texts[start:start + 256], padding=False, truncation=False)
            lengths.extend(len(row) for row in batch['input_ids'])
        return lengths

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Pool last nonpadding token from BF16 SDPA and return float32 vectors."""
        import torch

        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                                return_tensors='pt').to(self.device)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            pooled = last_token_pool(hidden, inputs['attention_mask'])
            return pooled.float().cpu().numpy()

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return unit rows in caller order, sorting batches by length and recording retries."""
        if isinstance(texts, str) or any(not isinstance(text, str) for text in texts):
            raise TypeError('texts must be a sequence of strings')
        if not texts:
            self.stats = {'original_token_counts': [], 'truncated_count': 0, 'max_length': MAX_LENGTH,
                          'elapsed_seconds': 0, 'oom_retries': []}
            return np.empty((0, self.dimension), dtype=np.float32)
        started = time.perf_counter()
        self._load()
        representations = [QUERY_INSTRUCTION + text if is_query else text for text in texts]
        lengths = self._token_lengths(representations)
        order = sorted(range(len(texts)), key=lambda i: (lengths[i], i))
        output = np.empty((len(texts), self.dimension), dtype=np.float32)
        self.stats = {'original_token_counts': lengths, 'truncated_count': sum(n > MAX_LENGTH for n in lengths),
                      'truncated_token_count': sum(max(0, n - MAX_LENGTH) for n in lengths),
                      'max_length': MAX_LENGTH, 'is_query': is_query, 'oom_retries': []}
        offset, batch_size = 0, self.batch_size
        while offset < len(order):
            batch_ids = order[offset:offset + batch_size]
            try:
                output[batch_ids] = normalized(self._encode_batch([representations[i] for i in batch_ids]), self.dimension)
            except RuntimeError as error:
                if 'out of memory' not in str(error).lower():
                    raise
                self.stats['oom_retries'].append({'offset': offset, 'batch_size': batch_size, 'error': str(error)})
                if batch_size == 1:
                    raise
                import torch

                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            offset += len(batch_ids)
        self.batch_size = batch_size
        self.stats['effective_batch_size'] = batch_size
        self.stats['elapsed_seconds'] = time.perf_counter() - started
        return output
