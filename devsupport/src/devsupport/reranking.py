"""Pinned local rerankers with a shared, fixed evidence preparation contract."""

import hashlib
import json
import math
import numbers
import os
import time
from pathlib import Path
from typing import Protocol

from .types import Hit

INSTRUCTION = 'Given a technical support query, retrieve relevant passages that answer the query'
MODELS = {
    'R1': ('Qwen/Qwen3-Reranker-0.6B', 'e61197ed45024b0ed8a2d74b80b4d909f1255473', 'pairwise'),
    'R2': ('jinaai/jina-reranker-v3.5', 'e8a93f33f0b22108f8c2364f8484ce3422552fbc', 'listwise'),
}
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
          'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class Reranker(Protocol):
    """Reorder only the supplied evidence pool."""

    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Return fresh hits preserving all original evidence."""
        ...


def verify_archive(path: Path, variant: str) -> dict:
    """Check pinned revision and every archived file before executing local code."""
    manifest = json.loads((path / 'artifact-manifest.json').read_text(encoding='utf-8'))
    if (manifest['repo'], manifest['revision']) != MODELS[variant][:2]:
        raise ValueError('local archive revision mismatch')
    for name, record in manifest['files'].items():
        source = (path / name).resolve()
        if not source.is_relative_to(path.resolve()) or not source.is_file():
            raise ValueError('unsafe or missing archive file')
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != record['sha256']:
            raise ValueError('archive checksum mismatch: ' + name)
    return manifest


class _LocalReranker:
    """Shared fixed caps, integrity validation, fallback and lazy local loading."""

    variant = ''
    score_kind = ''

    def __init__(self, model_path: str | Path | None = None, tokenizer_path: str | Path | None = None):
        """Configure local paths without allocating a GPU model."""
        cache = Path(__file__).resolve().parents[2] / 'model-cache'
        self.model_path = Path(model_path) if model_path else cache / MODELS[self.variant][2]
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else cache / 'pairwise'
        self.model = self.tokenizer = None
        self.last_trace = {}
        self.archive = None

    def _load(self) -> None:
        """Load verified local BF16 models on CUDA, never silently using CPU."""
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        os.environ['HF_HUB_OFFLINE'] = '1'
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if self.model is None:
            self.archive = verify_archive(self.model_path, self.variant)
            if self.tokenizer_path.resolve() != self.model_path.resolve():
                verify_archive(self.tokenizer_path, 'R1')
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path), local_files_only=True, padding_side='left')
            loader = AutoModel if self.variant == 'R2' else AutoModelForCausalLM
            self.model = loader.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=self.variant == 'R2',
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
            self.model.config.use_cache = False
            if self.variant == 'R2':
                self.model._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path), local_files_only=True, padding_side='left')
        self.last_trace.update(actual_model=MODELS[self.variant][0], device=str(self.model.device),
                               model_files_sha256=self.archive['files'], dtype='bfloat16', attention='sdpa')
        torch.cuda.reset_peak_memory_stats()

    def _prepare(self, query: str, hits: list[Hit]) -> tuple[str, list[str]]:
        """Apply the same Qwen tokenizer and 512-token caps to both rankers."""
        counts = []
        prepared = []
        for text in [query] + [hit.evidence.text for hit in hits]:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            retained = min(len(tokens), 512)
            prepared.append(text if len(tokens) <= 512 else self.tokenizer.decode(tokens[:512], skip_special_tokens=False))
            counts.append({'original': len(tokens), 'retained': retained})
        self.last_trace['lengths'] = {'query': counts[0], 'documents': counts[1:]}
        self.last_trace['prepared_sha256'] = hashlib.sha256(
            json.dumps(prepared, ensure_ascii=False).encode('utf-8')).hexdigest()
        return prepared[0], prepared[1:]

    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Validate the complete pool, score it, and preserve it on runtime failure."""
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a nonempty string')
        if len(hits) > 50 or any(not isinstance(h, Hit) or not isinstance(h.evidence.text, str)
                                 or not isinstance(h.evidence.doc_id, str) or not h.evidence.doc_id for h in hits):
            raise ValueError('invalid candidate pool')
        ids = [h.evidence.doc_id for h in hits]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate candidate IDs')
        self.last_trace = {'requested_model': MODELS[self.variant][0], 'revision': MODELS[self.variant][1],
                           'actual_model': None, 'variant': self.variant, 'input_ids': ids, 'device': None,
                           'fallback': False, 'error': None, 'retries': [], 'forward_count': 0,
                           'prompt_token_counts': [], 'instruction': INSTRUCTION,
                           'query_cap': 512, 'document_cap': 512}
        try:
            if hits:
                self._load()
                prepared_query, docs = self._prepare(query, hits)
                rows = self._infer(prepared_query, docs)
            else:
                rows = []
            if len(rows) != len(hits):
                raise ValueError('model response missing candidates')
            indexes = []
            scores = {}
            for index, score in rows:
                if isinstance(index, bool) or not isinstance(index, numbers.Integral) or not 0 <= index < len(hits):
                    raise ValueError('model response unknown index')
                if index in scores or not isinstance(score, numbers.Real) or not math.isfinite(score):
                    raise ValueError('model response duplicate index or nonfinite score')
                indexes.append(int(index))
                scores[index] = float(score)
            indexes.sort(key=lambda index: (-scores[index], ids[index]))
            result = [Hit(hits[i].evidence, scores[i], rank, self.score_kind)
                      for rank, i in enumerate(indexes, 1)]
        except (RuntimeError, OSError, TimeoutError) as error:
            self.last_trace.update(fallback=True, error=type(error).__name__)
            result = [Hit(h.evidence, h.score, rank, h.score_kind) for rank, h in enumerate(hits, 1)]
        finally:
            self.last_trace['elapsed_seconds'] = time.perf_counter() - started
        self.last_trace['outputs'] = [{'doc_id': h.evidence.doc_id, 'score': h.score, 'rank': h.rank,
                                      'score_kind': h.score_kind} for h in result]
        if self.model is not None:
            import torch
            self.last_trace['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        return result

    def unload(self) -> None:
        """Explicitly release resident model memory between service operations."""
        import gc
        self.model = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenPairwiseReranker(_LocalReranker):
    """Official trained yes/no prompt, independently scoring each query/document pair."""

    variant = 'R1'
    score_kind = 'qwen_yes_probability'

    def _infer(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        """Normalize yes/no logits with bounded batch-halving CUDA OOM retries."""
        import torch

        prompts = [PREFIX + f'<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}' + SUFFIX for doc in docs]
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        if max(lengths) > 4096:
            raise ValueError('prepared prompt exceeds fixed sequence cap')
        self.last_trace['prompt_token_counts'] = lengths
        false_id, true_id = [self.tokenizer.convert_tokens_to_ids(token) for token in ('no', 'yes')]
        result, offset, batch_size = [], 0, 8
        while offset < len(prompts):
            size = min(batch_size, len(prompts) - offset)
            try:
                inputs = self.tokenizer(prompts[offset:offset + size], padding=True, add_special_tokens=False,
                                        return_tensors='pt').to(self.model.device)
                with torch.inference_mode():
                    self.last_trace['forward_count'] += 1
                    logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                    scores = torch.softmax(logits[:, [false_id, true_id]].float(), dim=1)[:, 1].cpu().tolist()
                result.extend((offset + i, score) for i, score in enumerate(scores))
                offset += size
                del inputs, logits
            except torch.cuda.OutOfMemoryError:
                self.last_trace['retries'].append({'offset': offset, 'batch_size': size, 'error': 'OutOfMemoryError'})
                if size == 1:
                    raise
                batch_size = max(1, size // 2)
                torch.cuda.empty_cache()
        return result


class JinaListwiseReranker(_LocalReranker):
    """Official Jina joint contextual rerank; no independent embedding substitute."""

    variant = 'R2'
    score_kind = 'jina_joint_relevance'

    def _infer(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        """Instrument the native rerank path and enforce exactly one all-pool forward."""
        original_batch = self.model._compute_single_batch
        original_forward = self.model.forward

        def joint_batch(q: str, documents: list[str], instruction: str | None = None):
            if len(documents) != len(docs):
                raise RuntimeError('Jina attempted to split the fixed pool')
            return original_batch(q, documents, instruction=INSTRUCTION)

        def measured_forward(*args, **kwargs):
            self.last_trace['forward_count'] += 1
            self.last_trace['prompt_token_counts'].append(int(kwargs['input_ids'].shape[-1]))
            if self.last_trace['forward_count'] > 1:
                raise RuntimeError('Jina attempted multiple forwards')
            return original_forward(*args, **kwargs)

        self.model._compute_single_batch = joint_batch
        self.model.forward = measured_forward
        try:
            rows = self.model.rerank(query, docs, top_n=None)
            if self.last_trace['forward_count'] != 1:
                raise RuntimeError('Jina did not perform a joint forward')
            return [(row['index'], row['relevance_score']) for row in rows]
        finally:
            self.model._compute_single_batch = original_batch
            self.model.forward = original_forward
