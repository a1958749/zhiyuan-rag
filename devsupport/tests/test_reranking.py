import math

import pytest

from devsupport import reranking
from devsupport.types import Evidence, Hit


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, tokens, **kwargs):
        return ''.join(tokens)


def pool(n=50):
    return [Hit(Evidence(f'd{i:02}', 'x' * 600, {'source': i}), i / 100, i + 1, 'RRF') for i in range(n)]


def ranker(cls, result=None, error=None):
    class Injected(cls):
        def _load(self):
            self.tokenizer = Tokenizer()
            self.last_trace['actual_model'] = 'injected-test'

        def _infer(self, query, docs):
            self.seen = (query, docs)
            if error:
                raise error
            return result if result is not None else [(i, 0.5) for i in reversed(range(len(docs)))]
    return Injected()


def test_shared_pool_truncation_and_immutable_evidence():
    hits = pool()
    first = ranker(reranking.QwenPairwiseReranker)
    second = ranker(reranking.JinaListwiseReranker)
    a, b = first.rank('q' * 700, hits), second.rank('q' * 700, hits)
    assert first.seen == second.seen == ('q' * 512, ['x' * 512] * 50)
    assert [h.evidence.doc_id for h in a] == [h.evidence.doc_id for h in b] == [h.evidence.doc_id for h in hits]
    assert all(x.evidence is y.evidence and x is not y for x, y in zip(a, hits))
    assert hits[0].evidence.text == 'x' * 600
    assert first.last_trace['lengths']['documents'][0] == {'original': 600, 'retained': 512}
    assert [h.rank for h in a] == list(range(1, 51))


@pytest.mark.parametrize('result', [[(0, 1), (0, 2)], [(0, 1), (2, 1)], [(0, 1)], [(0, math.nan), (1, 2)], [(False, 1), (1, 2)]])
def test_illegal_model_response_raises(result):
    with pytest.raises(ValueError, match='response'):
        ranker(reranking.JinaListwiseReranker, result).rank('query', pool(2))


@pytest.mark.parametrize('error', [TimeoutError(), RuntimeError('CUDA out of memory'), OSError('load failure')])
def test_fallback_preserves_all_original_scores_order_and_reports_failure(error):
    model = ranker(reranking.JinaListwiseReranker, error=error)
    hits = list(reversed(pool()))
    result = model.rank('query', hits)
    assert [(h.evidence, h.score, h.score_kind) for h in result] == [(h.evidence, h.score, h.score_kind) for h in hits]
    assert model.last_trace['fallback'] is True
    assert model.last_trace['error'] == type(error).__name__


def test_reject_invalid_requests_before_loading():
    model = reranking.JinaListwiseReranker()
    for query, hits in [('', pool()), (None, pool()), ('q', pool(51)), ('q', [pool(1)[0]] * 2)]:
        with pytest.raises((ValueError, TypeError)):
            model.rank(query, hits)


def test_service_rejects_annotations_recursively_and_unknown_models():
    from devsupport.model_service import validate_request
    good = {'model': 'R1', 'query': 'q', 'documents': [{'doc_id': 'd', 'text': 't', 'metadata': {}}]}
    assert validate_request(good) == good
    for bad in [dict(good, model='x'), dict(good, answers=[]), dict(good, documents=[{'doc_id': 'd', 'text': 't', 'metadata': {'nested': {'qrels': []}}}])]:
        with pytest.raises(ValueError):
            validate_request(bad)


def test_native_jina_receives_all_candidates_jointly_and_records_forward():
    from types import SimpleNamespace

    model = reranking.JinaListwiseReranker()
    seen = []

    class Native:
        def forward(self, **kwargs):
            seen.append(kwargs['input_ids'].shape[-1])
            return None

        def _compute_single_batch(self, query, documents, instruction=None):
            assert len(documents) == 50
            assert instruction == reranking.INSTRUCTION
            return self.forward(input_ids=SimpleNamespace(shape=(1, 27000)))

        def rerank(self, query, documents, top_n=None):
            assert top_n is None
            self._compute_single_batch(query, documents)
            return [{'index': i, 'relevance_score': float(i)} for i in range(50)]

    model.model = Native()
    model.last_trace = {'forward_count': 0, 'prompt_token_counts': []}
    assert len(model._infer('q', ['d'] * 50)) == 50
    assert model.last_trace['forward_count'] == 1
    assert model.last_trace['prompt_token_counts'] == seen == [27000]
    assert 'forward' not in model.model.__dict__ or model.model.forward.__self__ is model.model


def test_native_jina_rejects_context_blocking_before_any_forward():
    class Native:
        def forward(self, **kwargs):
            pytest.fail('split inference must not run')

        def _compute_single_batch(self, query, documents, instruction=None):
            pytest.fail('split inference must not run')

        def rerank(self, query, documents, top_n=None):
            self._compute_single_batch(query, documents[:25])

    model = reranking.JinaListwiseReranker()
    model.model = Native()
    model.last_trace = {'forward_count': 0, 'prompt_token_counts': []}
    with pytest.raises(RuntimeError, match='split'):
        model._infer('q', ['d'] * 50)
    assert model.last_trace['forward_count'] == 0


def test_pairwise_uses_yes_no_probability_and_eight_item_batches():
    import torch
    from types import SimpleNamespace

    batches = []

    class Inputs(dict):
        def to(self, device):
            return self

    class PairTokenizer(Tokenizer):
        def convert_tokens_to_ids(self, token):
            return {'no': 0, 'yes': 1}[token]

        def __call__(self, prompts, **kwargs):
            assert kwargs['padding'] is True
            assert all(reranking.PREFIX in prompt and reranking.SUFFIX in prompt for prompt in prompts)
            return Inputs(input_ids=torch.ones(len(prompts), 4, dtype=torch.long))

    class PairModel:
        device = 'cpu'

        def __call__(self, input_ids, **kwargs):
            assert kwargs == {'use_cache': False, 'logits_to_keep': 1}
            batches.append(len(input_ids))
            return SimpleNamespace(logits=torch.tensor([[[0., 1.]]] * len(input_ids)))

    model = reranking.QwenPairwiseReranker()
    model.tokenizer, model.model = PairTokenizer(), PairModel()
    model.last_trace = {'forward_count': 0, 'prompt_token_counts': [], 'retries': []}
    result = model._infer('query', ['document'] * 17)
    assert batches == [8, 8, 1]
    assert model.last_trace['forward_count'] == 3
    assert [index for index, _ in result] == list(range(17))
    assert all(score == pytest.approx(0.7310586) for _, score in result)


def test_service_health_and_request_rejection_without_gpu_loading():
    from fastapi.testclient import TestClient
    from devsupport.model_service import create_app

    with TestClient(create_app()) as client:
        assert client.get('/health').json() == {'status': 'ok', 'resident': []}
        assert client.post('/v1/rerank', json={'model': 'bad', 'query': 'q', 'documents': []}).status_code == 422
        assert client.post('/v1/embeddings', json={'input': ['q'], 'qrels': []}).status_code == 422
        result = client.post('/v1/rerank', json={'model': 'R1', 'query': 'q', 'documents': []}).json()
        assert result['results'] == []
        assert result['trace']['fallback'] is False


def test_jina_loader_uses_registered_automodel_class(monkeypatch):
    import sys
    import torch
    from types import SimpleNamespace

    calls = []

    class Loaded:
        device = 'cuda'
        config = SimpleNamespace(use_cache=True)

        def to(self, device):
            return self

        def eval(self):
            return self

    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(kwargs)
            return Loaded()

    class WrongLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            pytest.fail('Jina config only registers AutoModel')

    tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModel=Loader, AutoModelForCausalLM=WrongLoader, AutoTokenizer=tokenizer))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(reranking, 'verify_archive', lambda *args: {'files': {}})
    model = reranking.JinaListwiseReranker()
    model._load()
    assert calls[0]['trust_remote_code'] is True
    assert calls[0]['local_files_only'] is True
    assert model.model._tokenizer is not None
