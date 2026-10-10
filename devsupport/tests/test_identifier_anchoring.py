"""Offline behavioral checks for deterministic bounded identifier anchoring."""
import json

import pytest

from devsupport.pipeline import Pipeline, PipelineConfig
from test_adaptive import Provider, Retriever, hits


def retrieve(original, rewrites):
    retriever = Retriever(hits())
    provider = Provider([{'subqueries': rewrites}])
    result = Pipeline(retriever, provider, config=PipelineConfig(adaptive='always')).retrieve(original)
    return result, retriever, provider


def test_observed_response_preserves_all_eight_identifiers_and_original_pool():
    original = ('Chroma.from_documents ChromaDb MongoDb OpenAIEmbeddings db.persist '
                'langchain.embeddings.openai langchain.vectorstores persist_directory')
    model = ['langchain Chroma vectorstore get all documents with ids metadata persist_directory',
             'Chroma _collection.get retrieve all documents python langchain embeddings']
    expected = [model[0] + ' ChromaDb MongoDb db.persist langchain.embeddings.openai',
                model[1] + ' Chroma.from_documents OpenAIEmbeddings langchain.vectorstores']
    result, retriever, provider = retrieve(original, model)
    adaptive = result['trace']['adaptive']
    assert retriever.queries == [original, *expected]
    assert adaptive['model_subqueries'] == model
    assert adaptive['subqueries'] == adaptive['effective_subqueries'] == expected
    assert adaptive['anchored_query_count'] == 2
    assert adaptive['appended_identifiers'] == original.split()[:-1]
    assert adaptive['anchoring_policy'] == 'devsupport-identifier-anchoring-v1'
    assert adaptive['additional_rounds'] == 1
    assert result['trace']['final_candidate_ids'] == ['0', '1', '2']
    assert len(provider.calls) == 1


@pytest.mark.parametrize('original,model,expected', [
    ('Widget.render', ['Widget.render usage'], ['Widget.render usage']),
    ('ordinary words', ['some usage'], ['some usage']),
    ('Widget.render', ['Widget.render'], []),
    ('Widget.render', [' Widget.render ', 'Widget.render'], []),
    ('Use `render`', ['prerender'], ['prerender render']),
    ('Widget::render', ['Widget::renderExtra'], ['Widget::renderExtra Widget::render']),
    ('JSONLoader', ['jsonloader MyJSONLoader'], ['jsonloader MyJSONLoader JSONLoader']),
    ('Widget.render', ['Other.Widget.render'], ['Other.Widget.render Widget.render']),
    ('Widget.render', ['Widget.render.extraé'], ['Widget.render.extraé Widget.render']),
    ('Widget.render', ['same', 'same'], ['same Widget.render']),
    ('usage Widget.render', ['usage'], []),
    ('`x`', ['a' * 510, 'b' * 510], ['a' * 510 + ' x', 'b' * 510]),
    ('`x`', ['a' * 511, 'b' * 510], ['a' * 511, 'b' * 510 + ' x']),
])
def test_exact_tokens_noops_and_fitting_bounds(original, model, expected):
    result, retriever, provider = retrieve(original, model)
    assert retriever.queries == [original, *expected]
    adaptive = result['trace']['adaptive']
    assert adaptive['model_subqueries'] == model
    assert adaptive['effective_subqueries'] == expected
    assert adaptive['additional_rounds'] == bool(expected)
    assert len(expected) <= 2 and all(len(q) <= 512 for q in expected)
    assert len(provider.calls) == 1


def test_partial_anchoring_overflow_never_retrieves_partial_queries():
    result, retriever, provider = retrieve('`a` `longname`', ['x' * 510, 'y' * 511])
    adaptive = result['trace']['adaptive']
    assert retriever.queries == ['`a` `longname`']
    assert adaptive['failure_code'] == 'identifier_anchoring_overflow'
    assert adaptive['effective_subqueries'] == adaptive['appended_identifiers'] == []
    assert adaptive['anchored_query_count'] == adaptive['additional_rounds'] == 0
    assert result['trace']['final_candidate_ids'] == ['0', '1', '2']
    assert len(provider.calls) == 1


def test_original_only_anchor_cannot_supply_coverage_for_another_subquery():
    result, retriever, _ = retrieve('usage Widget.render', ['usage', 'configuration'])
    assert retriever.queries == ['usage Widget.render']
    adaptive = result['trace']['adaptive']
    assert adaptive['failure_code'] == 'missing_exact_identifiers'
    assert adaptive['effective_subqueries'] == adaptive['appended_identifiers'] == []
    assert adaptive['anchored_query_count'] == 0


@pytest.mark.parametrize('model', [[], [''], ['a\nb'], ['a\rb'], ['x' * 513], [1], ['a', 'b', 'c']])
def test_structural_failures_cannot_be_repaired(model):
    result, retriever, _ = retrieve('Widget.render', model)
    assert retriever.queries == ['Widget.render']
    assert result['trace']['adaptive']['failure_code'] == 'invalid_subqueries'
    assert 'model_subqueries' not in result['trace']['adaptive']


def test_strict_validator_runs_after_anchoring(monkeypatch):
    from devsupport.adaptive import validate_subqueries

    validated = []

    def validate(text, original):
        validated.append(json.loads(text)['subqueries'])
        return validate_subqueries(text, original)

    monkeypatch.setattr('devsupport.pipeline.validate_subqueries', validate)
    _, retriever, _ = retrieve('Widget.render', ['usage'])
    assert validated == [['usage Widget.render']]
    assert retriever.queries == ['Widget.render', 'usage Widget.render']


def test_anchoring_keeps_gate_and_answer_protocols_and_base_evidence():
    from devsupport.answer import ANSWER_PROMPT, format_question_context, pack_context, Utf8BudgetCounter
    from devsupport.gate import GATE_PROMPT

    class DisjointRetriever(Retriever):
        def search(self, query, k=50):
            self.queries.append(query)
            return self.hits if len(self.queries) == 1 else []

    original = 'Widget.render'
    retriever = DisjointRetriever(hits())
    provider = Provider([{'subqueries': ['usage', 'configuration']},
                         {'decision': 'answer', 'missing_information': [], 'reason': 'Supported.'},
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    counter = Utf8BudgetCounter()
    _, context, _ = pack_context(hits(), 4000, counter)
    result = Pipeline(retriever, provider, token_counter=counter,
                      config=PipelineConfig(adaptive='always')).run(original)
    assert retriever.queries == [original, 'usage Widget.render', 'configuration']
    assert result['trace']['final_candidate_ids'] == ['0', '1', '2']
    assert result['decision'] == 'answer'
    assert result['trace']['generation']['attempts'] == 1
    assert provider.calls[1:] == [(GATE_PROMPT, format_question_context(original, context), 2048),
                                  (ANSWER_PROMPT, format_question_context(original, context), 2048)]
