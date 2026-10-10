"""Synthetic regressions for recovered failures and provider-only accounting."""
import json

import pytest

from devsupport.pipeline import Pipeline, PipelineConfig
from devsupport.report import markdown, summarize
from test_adaptive import Provider, Retriever, hits


@pytest.mark.parametrize('failure,stage,code', [
    ('json', 'decomposition_validation', 'invalid_json'),
    ('overflow', 'decomposition_validation', 'identifier_anchoring_overflow'),
    ('schema', 'decomposition_validation', 'invalid_subqueries'),
    ('provider', 'decomposition_provider', 'provider_failure'),
    ('retrieval', 'subquery_retrieval', 'retrieval_failed'),
    ('rerank', 'expanded_rerank', 'reranking_failed'),
])
def test_expansion_failure_diagnostics_preserve_exact_first_round(failure, stage, code):
    """Failed expansion keeps order, scores and content and never leaks exception text."""
    class DiagnosticProvider(Provider):
        def complete(self, *args, **kwargs):
            response = super().complete(*args, **kwargs)
            if failure == 'json' and len(self.calls) == 1:
                response['text'] = 'private malformed response'
            return response

    class DiagnosticRetriever(Retriever):
        def search(self, query, k=50):
            rows = super().search(query, k)
            if failure == 'retrieval' and len(self.queries) == 2:
                raise RuntimeError('private exception')
            return rows

    class Ranker:
        calls = 0

        def rank(self, query, rows):
            self.calls += 1
            if failure == 'rerank' and self.calls == 2:
                raise RuntimeError('private exception')
            return list(reversed(rows))

    rewrite = {'subqueries': ['x' * 512 if failure == 'overflow' else 'Widget.render usage']}
    if failure == 'schema':
        rewrite = {'subqueries': []}
    if failure == 'provider':
        rewrite = TimeoutError('private exception')
    provider = DiagnosticProvider([rewrite, {'answer_text': 'Use it [1].', 'citation_ids': ['2']}])
    retriever = DiagnosticRetriever(hits())
    result = Pipeline(retriever, provider, Ranker(),
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    adaptive = result['trace']['adaptive']
    assert adaptive['failure_stage'] == stage
    assert adaptive['failure_code'] == code
    assert adaptive['fallback'] == 'first_round_preserved'
    assert adaptive['additional_rounds'] == int(failure in ('retrieval', 'rerank'))
    assert result['decision'] == 'answer'
    assert result['trace']['final_candidate_ids'] == ['2', '1', '0']
    assert [(r['doc_id'], r['score'], r['text']) for r in result['context']] == [
        (h.evidence.doc_id, h.score, h.evidence.text) for h in reversed(hits())]
    assert len(provider.calls) == 2
    assert len(retriever.queries) == (2 if failure in ('retrieval', 'rerank') else 1)
    assert 'private' not in json.dumps(result)


def row(result, query_id='q'):
    """Wrap a result with fixed metrics to detect denominator changes."""
    return {'query_id': query_id, 'result': result, 'elapsed_seconds': 1,
            'metrics_prepack': {'recall': 0.5}, 'metrics_packed': {'recall': 0.25}}


def test_recovered_failures_count_rows_without_changing_outcomes_or_metrics():
    result = {'decision': 'answer', 'trace': {
        'provider_calls': [{'stage': 'decomposition', 'error': 'network_error'},
                           {'stage': 'generation', 'events': [{'status': 'http_503'}]}],
        'retrieval': [{'error': 'retrieval_failed'}, {'error': 'retrieval_failed'}],
        'reranking': [{'fallback': True, 'error': 'RuntimeError'}],
        'adaptive': {'fallback': 'first_round_preserved', 'failure_stage': 'subquery_retrieval',
                     'failure_code': 'retrieval_failed'}}}
    summary = summarize([row(result)], ['q'])
    assert summary['errors'] == 0 and summary['answers'] == 1 and summary['success'] == 1
    assert summary['operational_failure_rows'] == 1
    assert summary['stage_failure_rows'] == {'provider': 1, 'retrieval': 1, 'reranking': 1,
                                              'validation': 0, 'unknown': 0}
    assert summary['metrics_prepack']['recall'] == 0.5
    assert summary['metrics_packed']['recall'] == 0.25


def test_validation_and_legacy_fallback_do_not_invent_transport_failures():
    results = [{'decision': 'answer', 'trace': {'adaptive': {
        'fallback': 'first_round_preserved', 'failure_stage': 'decomposition_validation',
        'failure_code': 'invalid_json'}}},
        {'decision': 'answer', 'trace': {'adaptive': {'fallback': 'first_round_preserved'}}}]
    summary = summarize([row(result, str(i)) for i, result in enumerate(results)], ['0', '1'])
    assert summary['operational_failure_rows'] == 0
    assert summary['stage_failure_rows']['validation'] == 1
    assert summary['stage_failure_rows']['unknown'] == 1
    assert summary['estimated_cost_configured_currency'] is None


@pytest.mark.parametrize('empty', [True, False])
def test_proven_zero_calls_and_priced_calls_aggregate_without_hiding_unknowns(empty):
    provider = Provider([])
    pipeline = Pipeline(Retriever([] if empty else hits()), provider,
                        config=PipelineConfig(adaptive='never'))
    result = pipeline.retrieve('Widget.render')
    assert result['cost'] == 0 and result['cost_status'] == 'no_provider_calls'
    assert result['cost_scope'] == 'provider_only'

    class PricedProvider(Provider):
        def complete(self, *args, **kwargs):
            return dict(super().complete(*args, **kwargs), cost=0.25)

    priced = Pipeline(Retriever(hits()), PricedProvider([
        {'answer_text': 'Use it [1].', 'citation_ids': ['0']}]),
        config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    summary = summarize([row(result, '0'), row(priced, '1')], ['0', '1'])
    assert summary['estimated_cost_configured_currency'] == 0.25
    assert summary['cost_scope'] == 'provider_only'
    unknown = Pipeline(Retriever(hits()), Provider([
        {'answer_text': 'Use it [1].', 'citation_ids': ['0']}]),
        config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    assert unknown['cost'] is None
    assert summarize([row(result, '0'), row(unknown, '1')], ['0', '1'])[
        'estimated_cost_configured_currency'] is None


def test_unknown_validation_error_is_bounded(monkeypatch):
    """Unexpected validator exceptions cannot become trace data."""
    def invalid(*args):
        raise ValueError('private query-derived detail')

    monkeypatch.setattr('devsupport.pipeline.validate_subqueries', invalid)
    result = Pipeline(Retriever(hits()), Provider([{'subqueries': ['Widget.render usage']}]),
                      config=PipelineConfig(adaptive='always')).retrieve('Widget.render')
    assert result['trace']['adaptive']['failure_code'] == 'unknown_validation_failure'
    assert 'private' not in json.dumps(result)


def test_provider_error_detail_is_retained_and_recovery_is_reported():
    """New bounded fields supplement existing sanitized transport diagnostics."""
    from devsupport.llm import ProviderError

    record = {'attempts': 2, 'events': [{'status': 'http_503'}], 'cost': None}
    result = Pipeline(Retriever(hits()), Provider([
        ProviderError('http_503', record), {'answer_text': 'Use it [1].', 'citation_ids': ['0']}]),
        config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    call = result['trace']['provider_calls'][0]
    assert call['error'] == 'http_503' and call['events'] == record['events'] and call['attempts'] == 2
    summary = summarize([row(result)], ['q'])
    assert summary['errors'] == 0 and summary['operational_failure_rows'] == 1
    assert summary['stage_failure_rows']['provider'] == 1
    assert summary['estimated_cost_configured_currency'] is None


@pytest.mark.parametrize('trace,category', [
    ({'provider_calls': [{'stage': 'generation', 'events': [{'status': 'http_503'}]}]}, 'provider'),
    ({'reranking': [{'retries': [{'error': 'OutOfMemoryError'}]}]}, 'reranking'),
    ({'retrieval': [{'error': 'retrieval_failed'}]}, 'retrieval'),
])
def test_recovered_stage_alone_is_visible(trace, category):
    summary = summarize([row({'decision': 'answer', 'trace': trace})], ['q'])
    assert summary['errors'] == 0 and summary['success'] == 1
    assert summary['operational_failure_rows'] == summary['stage_failure_rows'][category] == 1


@pytest.mark.parametrize('result', [{}, {'trace': {}}, {'trace': {'provider_calls': []}}, {'cost': None}])
def test_legacy_missing_cost_never_implies_zero(result):
    assert summarize([row(result)], ['q'])['estimated_cost_configured_currency'] is None


def test_markdown_exposes_recovered_failures_and_cost_scope():
    summary = summarize([row({'decision': 'answer', 'trace': {
        'retrieval': [{'error': 'retrieval_failed'}]}})], ['q'])
    rendered = markdown({'Full': summary})
    assert 'Operational failure rows' in rendered
    assert 'Validation' in rendered and 'Unknown' in rendered
    assert 'provider-only' in rendered and 'GPU' in rendered


@pytest.mark.parametrize('unknown_call', [1, 2])
def test_any_unknown_call_keeps_pipeline_cost_unavailable(unknown_call):
    """A priced response cannot erase uncertainty in another call in the same result."""
    class MixedProvider(Provider):
        def complete(self, *args, **kwargs):
            response = super().complete(*args, **kwargs)
            response['cost'] = None if len(self.calls) == unknown_call else 0.25
            return response

    result = Pipeline(Retriever(hits()), MixedProvider([
        {'subqueries': ['Widget.render usage']}, {'answer_text': 'Use it [1].', 'citation_ids': ['0']}]),
        config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    assert result['decision'] == 'answer' and result['cost'] is None
    assert result['cost_status'] == 'unverified'
    assert summarize([row(result)], ['q'])['estimated_cost_configured_currency'] is None


def test_failed_repair_preserves_prior_validation_failure_in_report():
    """Actual generation and failed repair retain overlapping failure categories."""
    provider = Provider([{'unexpected': 'invalid answer schema'}, TimeoutError('private transport detail')])
    result = Pipeline(Retriever(hits()), provider,
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    generation = result['trace']['generation']
    assert generation['error'] == 'provider_failure'
    assert generation['validation_failures'][0]['error'] == 'invalid_answer_schema'
    assert len(provider.calls) == 2 and generation['attempts'] == 2
    summary = summarize([row(result)], ['q'])
    assert summary['stage_failure_rows']['validation'] == 1
    assert summary['stage_failure_rows']['provider'] == 1
    assert summary['operational_failure_rows'] == 1
    assert summary['errors'] == 1 and summary['answers'] == 0
