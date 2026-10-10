"""Query-only capacity preflight and shared-pipeline regressions, without paid calls."""

import pytest

from devsupport.adaptive import should_expand
from devsupport.answer import Utf8BudgetCounter
from devsupport.pipeline import Pipeline, PipelineConfig
from test_adaptive import Provider, Retriever, hits


@pytest.mark.parametrize('query', [
    'compare `' + 'a' * 513 + '`',
    'compare ' + ' '.join(f'field_{i:04d}' for i in range(103)),
])
@pytest.mark.parametrize('mode', ['always', 'selective'])
def test_impossible_identifier_capacity_skips_otherwise_triggered_expansion(query, mode):
    # Two question marks make selective trigger even for an unqualified backtick name.
    assert should_expand(mode, query + '??', hits()) == (False, 'identifier_budget_infeasible')


@pytest.mark.parametrize('query', [
    'compare `' + 'a' * 512 + '` and `' + 'b' * 512 + '`',
    'compare ' + ' '.join(f'field_{i:03d}' for i in range(113)),
    'compare ' + ' '.join(['Widget.render'] * 100),
    'compare ordinary alternatives',
])
def test_capacity_bound_does_not_reject_boundary_duplicates_or_zero_identifiers(query):
    assert should_expand('always', query, hits()) == (True, 'always_policy')


def test_disabled_and_nontriggering_selective_keep_existing_reasons():
    query = '`' + 'a' * 513 + '`'
    assert should_expand('never', query + ' compare??', hits()) == (False, 'disabled')
    assert should_expand('selective', query, hits()) == (False, 'first_round_adequate_heuristic')


@pytest.mark.parametrize('gate_decision', ['answer', 'insufficient'])
def test_impossible_retrieve_keeps_ranked_evidence_and_leaves_gate_independent(gate_decision):
    query = 'compare `' + 'a' * 513 + '`??'
    rows = hits()

    class ReverseRanker:
        def rank(self, query, rows):
            return list(reversed(rows))

    replies = [{'decision': gate_decision, 'missing_information': [], 'reason': 'Evidence checked.'},
               {'answer_text': 'Use it [1].', 'citation_ids': ['2']}]
    provider = Provider(replies)
    retriever = Retriever(rows)
    pipeline = Pipeline(retriever, provider, ReverseRanker(), Utf8BudgetCounter(),
                        PipelineConfig(adaptive='always'))
    baseline = Pipeline(Retriever(rows), Provider([]), ReverseRanker(), Utf8BudgetCounter(),
                        PipelineConfig(adaptive='never')).retrieve(query)
    envelope = pipeline.retrieve(query)
    assert provider.calls == []
    assert retriever.queries == [query]
    assert envelope['context'] == baseline['context']
    assert envelope['trace']['final_candidate_ids'] == ['2', '1', '0']
    adaptive = envelope['trace']['adaptive']
    assert adaptive['triggered'] is False
    assert adaptive['reason'] == 'identifier_budget_infeasible'
    assert adaptive['additional_rounds'] == 0
    assert adaptive['subqueries'] == []
    assert 'fallback' not in adaptive and 'failure_code' not in adaptive
    result = pipeline.answer(query, envelope)
    assert result['decision'] == gate_decision
    assert [call['stage'] for call in result['trace']['provider_calls']] == (
        ['gate', 'generation'] if gate_decision == 'answer' else ['gate'])


def test_preflight_policy_is_traced_and_changes_runtime_identity(monkeypatch):
    import devsupport.pipeline as pipeline_module

    def retrieve():
        return Pipeline(Retriever(hits()), Provider([]), token_counter=Utf8BudgetCounter(),
                        config=PipelineConfig(adaptive='never')).retrieve('Widget.render')

    current = retrieve()
    assert current['trace']['adaptive']['preflight_policy'] == 'devsupport-identifier-capacity-v1'
    monkeypatch.setattr(pipeline_module, 'IDENTIFIER_CAPACITY_POLICY', 'different-policy')
    assert retrieve()['runtime_hash'] != current['runtime_hash']
