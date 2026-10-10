import copy

import pytest

from test_adaptive import Provider, Retriever, hits


@pytest.mark.parametrize('bad', [
    {'answer_text': 'Claim [1].', 'citation_ids': ['foreign']},
    {'answer_text': 'Claim [2].', 'citation_ids': ['0']},
    {'answer_text': 'Claim.', 'citation_ids': []},
    {'answer_text': 'Claim [99].', 'citation_ids': ['0']},
    {'answer_text': 'Claim [1, 99] and [1].', 'citation_ids': ['0']},
])
def test_citation_repair_bounded(bad):
    from devsupport.pipeline import Pipeline, PipelineConfig

    provider = Provider([bad, bad])
    result = Pipeline(Retriever(hits()), provider,
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    assert result['decision'] == 'insufficient'
    assert result['answer_text'] == ''
    assert len(provider.calls) == 2
    assert result['trace']['generation']['attempts'] == 2


def test_valid_repair_and_raw_context_not_mutated():
    from devsupport.pipeline import Pipeline, PipelineConfig

    original = hits()
    before = copy.deepcopy(original)
    provider = Provider([{'answer_text': 'Bad [8]', 'citation_ids': ['unknown']},
                         {'answer_text': 'Use Widget.render [1].', 'citation_ids': ['0']}])
    result = Pipeline(Retriever(original), provider,
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    assert original == before
    assert result['decision'] == 'answer'
    assert result['citation_ids'] == ['0']
    assert result['context'][0]['text'] == original[0].evidence.text


def test_budget_counts_source_labels_and_skips_whole_chunks():
    from devsupport.answer import pack_context, Utf8BudgetCounter

    original = hits()
    packed, text, trace = pack_context(original, 130, Utf8BudgetCounter())
    assert len(text.encode('utf-8')) <= 130
    assert trace['skipped_ids']
    assert len(packed) == 1
    assert original[0].evidence.text in text
    assert '2025' in text
    assert '[1]' in text and '0' in text
    assert trace['exact_generation_tokens'] is False


@pytest.mark.parametrize('code', [
    '```python\na = [1,2]\nitem = a[0]\n```',
    '~~~python\na = [1,2]\nitem = a[0]\n~~~',
    '    a = [1,2]\n    item = a[0]',
    '`a = [1,2]; item = a[0]`',
    '``a = `[1,2]`; item = a[0]``',
    '> ```python\n> a = [1,2]\n> ```',
])
def test_code_numbers_are_not_citations(code):
    import json
    from devsupport.answer import validate_answer
    answer = {'answer_text': 'Use this [1].\n\n' + code, 'citation_ids': ['0']}
    assert validate_answer(json.dumps(answer), hits()) == answer


@pytest.mark.parametrize('prose', ['Claim [99].', 'Claim [0].', 'Claim [1,2].',
                                   'Claim [99', 'Claim [1,', 'Claim [1] and [99',
                                   'Claim [1] plus [2bad].'])
def test_invalid_prose_citations_are_not_hidden_by_code(prose):
    import json
    from devsupport.answer import validate_answer
    with pytest.raises(ValueError):
        validate_answer(json.dumps({'answer_text': prose + '\n\n`x=[1,2]`',
                                    'citation_ids': ['0']}), hits())


@pytest.mark.parametrize('answer', ['Only `a[1]`.', 'Only [1](https://example.org).',
                                   '```text\n[1]\n```', '~~~text\n[1]\n~~~'])
def test_code_or_link_labels_do_not_satisfy_required_plain_citation(answer):
    import json
    from devsupport.answer import validate_answer
    with pytest.raises(ValueError, match='invalid_citations'):
        validate_answer(json.dumps({'answer_text': answer, 'citation_ids': ['0']}), hits())


def test_formatted_prose_keeps_real_citations_and_original_answer():
    import json
    from devsupport.answer import validate_answer
    answer = {'answer_text': '# Usage\n\n- **Use it** [1].\n- See [Python](https://example.org).\n\n'
              '> Supported [2].\n\nThen use `a[99]`.', 'citation_ids': ['0', '1']}
    assert validate_answer(json.dumps(answer), hits()) == answer


def test_validation_diagnostics_preserve_only_bounded_answer_response(monkeypatch):
    import json
    from devsupport.pipeline import Pipeline, PipelineConfig
    from devsupport import pipeline as module
    response = json.dumps({'answer_text': 'Claim [99].' + 'x' * 20000, 'citation_ids': ['0']})
    class TextProvider:
        def complete(self, *args, **kwargs):
            return {'text': response, 'reasoning_content': 'PRIVATE_REASONING',
                    'requested_model': 'test', 'actual_model': 'test'}
    result = Pipeline(Retriever(hits()), TextProvider(),
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    failures = result['trace']['generation']['validation_failures']
    assert failures == [{'error': 'invalid_citations', 'response_text': response[:16384]}] * 2
    assert 'PRIVATE_REASONING' not in json.dumps(result)
    def fail(*args):
        raise ValueError('PRIVATE_EXCEPTION_DETAILS')
    monkeypatch.setattr(module, 'validate_answer', fail)
    result = Pipeline(Retriever(hits()), TextProvider(),
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    assert result['trace']['generation']['validation_failures'][0]['error'] == 'invalid_answer_validation'
    assert 'PRIVATE_EXCEPTION_DETAILS' not in json.dumps(result)


def test_original_fenced_list_reproduction_succeeds_without_repair():
    from devsupport.pipeline import Pipeline, PipelineConfig
    answer = {'answer_text': 'Use this [1].\n```python\na=[1,2]\n```', 'citation_ids': ['0']}
    provider = Provider([answer])
    result = Pipeline(Retriever(hits()), provider,
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    assert result['decision'] == 'answer' and result['answer_text'] == answer['answer_text']
    assert len(provider.calls) == 1
    assert 'validation_failures' not in result['trace']['generation']


def test_unmatched_numeric_prose_bracket_rejects_even_with_valid_citation():
    import json
    from devsupport.answer import validate_answer
    with pytest.raises(ValueError, match='invalid_citation_number_format'):
        validate_answer(json.dumps({'answer_text': 'Supported [1]. Also [99', 'citation_ids': ['0']}), hits())
