"""The shared proxy budget is explicit, offline and includes source formatting."""
import pytest

from devsupport.types import Evidence, Hit


def test_proxy_is_offline_and_treats_special_strings_as_source_text(monkeypatch):
    import urllib.request
    from devsupport.answer import ProxyTokenCounter

    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network forbidden'))
    counter = ProxyTokenCounter()
    assert counter('Hello world') == 2
    assert counter('<|endoftext|>') > 1
    assert counter.exact_generation_tokens is False
    assert counter.name == 'cl100k_base_proxy_v1'


def test_default_budget_uses_proxy_tokens_and_preserves_whole_evidence():
    from devsupport.answer import ProxyTokenCounter, pack_context

    raw = 'configuration ' * 500
    hits = [Hit(Evidence('one', raw, {'source': 'fixture'}), 1, 1, 'test')]
    packed, context, trace = pack_context(hits)
    assert packed == hits
    assert raw in context
    assert len(context.encode('utf-8')) > 4000
    assert trace['used'] == ProxyTokenCounter()(context) <= 4000
    assert trace['counter'] == 'cl100k_base_proxy_v1'


def test_modified_tokenizer_artifact_is_rejected(tmp_path):
    from devsupport.answer import ProxyTokenCounter

    bad = tmp_path / 'ranks.tiktoken'
    bad.write_bytes(b'altered')
    with pytest.raises(ValueError, match='checksum'):
        ProxyTokenCounter(ranks_path=bad)


def test_model_receives_original_source_characters_without_json_reescaping():
    from devsupport.pipeline import Pipeline, PipelineConfig
    from test_adaptive import Provider, Retriever

    raw = '{"python": "a\\nb", "message": "你好"}\nnext line'
    hits = [Hit(Evidence('one', raw, {}), 1, 1, 'test')]
    provider = Provider([{'answer_text': 'Example [1].', 'citation_ids': ['one']}])
    Pipeline(Retriever(hits), provider,
             config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Example?')
    assert raw in provider.calls[0][1]
