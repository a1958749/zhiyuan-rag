import pytest

from test_adaptive import Provider, Retriever, hits


@pytest.mark.parametrize('reply,expected', [
    ({'decision': 'clarify', 'missing_information': ['Python version'], 'reason': 'Version changes API'}, 'clarify'),
    ({'decision': 'clarify', 'missing_information': [], 'reason': 'Need more'}, 'insufficient'),
    ({'decision': 'maybe', 'missing_information': [], 'reason': 'x'}, 'insufficient'),
    ({'decision': 'clarify', 'missing_information': ['favorite color'], 'reason': 'Need more'}, 'insufficient'),
    ({'decision': 'answer', 'missing_information': [], 'reason': 'Supported'}, 'answer'),
])
def test_gate_strict_and_fail_closed(reply, expected):
    from devsupport.pipeline import Pipeline, PipelineConfig

    provider = Provider([reply, {'answer_text': 'Use render [1].', 'citation_ids': ['0']}])
    result = Pipeline(Retriever(hits()), provider, config=PipelineConfig(adaptive='never')).run('Widget.render')
    assert result['decision'] == expected
    assert len(provider.calls) == (2 if expected == 'answer' else 1)
    if expected == 'clarify':
        assert result['missing_information'] == ['Python version']
