import json

import pytest

from devsupport.types import Evidence, Hit


class Retriever:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, query, k=50):
        self.queries.append(query)
        return self.hits[:k]


class Provider:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def complete(self, system, user, max_tokens=1024):
        self.calls.append((system, user, max_tokens))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {'text': json.dumps(reply), 'usage': {'input_tokens': 10, 'output_tokens': 5},
                'cost': None, 'requested_model': 'test', 'actual_model': 'test'}


def hits(n=3):
    return [Hit(Evidence(str(i), 'Use Widget.render with the configured backend.', {'snapshot': '2025'}),
                1 / (i + 1), i + 1, 'BM25') for i in range(n)]


@pytest.mark.parametrize('configured,actual,allowed', [
    ('gateway-model', 'gateway-model', True),
    (None, 'gateway-model', False),
    ('gateway-model', 'unexpected-model', False),
])
def test_identity_alias_requires_server_configuration(configured, actual, allowed):
    from devsupport.llm import ProviderConfig
    from devsupport.pipeline import Pipeline, PipelineConfig

    class AliasProvider(Provider):
        config = ProviderConfig(model='test', expected_reported_model=configured)

        def complete(self, *args, **kwargs):
            reply = super().complete(*args, **kwargs)
            reply.update(actual_model=actual, expected_reported_model=actual)
            return reply

    provider = AliasProvider([{'answer_text': 'Use Widget.render [1].', 'citation_ids': ['0']}] * 2)
    result = Pipeline(Retriever(hits()), provider,
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    assert (result['decision'] == 'answer') is allowed
    assert ('excluded_failure' not in result['trace']) is allowed


def test_all_llm_stages_share_explicit_output_budget():
    from devsupport.pipeline import Pipeline, PipelineConfig

    provider = Provider([{'subqueries': ['Widget.render usage']},
                         {'decision': 'answer', 'missing_information': [], 'reason': 'Evidence supports it.'},
                         {'answer_text': 'Use Widget.render [1].', 'citation_ids': ['0']}])
    result = Pipeline(Retriever(hits()), provider, config=PipelineConfig(adaptive='always')).run('Widget.render')
    assert result['decision'] == 'answer'
    assert [call[2] for call in provider.calls] == [2048, 2048, 2048]


def test_expansion_bounded_original_preserved_and_stable():
    from devsupport.pipeline import Pipeline, PipelineConfig

    query = 'How do Widget.render and Widget.configure interact?'
    retriever = Retriever(hits(70))
    provider = Provider([{'subqueries': ['Widget.render usage', 'Widget.configure usage']},
                         {'answer_text': 'Configure first [1].', 'citation_ids': ['0']}])
    result = Pipeline(retriever, provider, config=PipelineConfig(adaptive='always', gate_enabled=False)).run(query)
    assert retriever.queries == [query, 'Widget.render usage', 'Widget.configure usage']
    assert len(result['trace']['final_candidate_ids']) == 50
    assert result['trace']['adaptive']['additional_rounds'] == 1
    assert result['decision'] == 'answer'
    assert result['trace']['final_candidate_ids'][:3] == ['0', '1', '2']


@pytest.mark.parametrize('rewrite', [{'subqueries': ['a', 'b', 'c']}, {'subqueries': ['']},
                                   {'subqueries': [12]}, TimeoutError('secret-key')])
def test_bad_rewrite_preserves_first_pool(rewrite):
    from devsupport.pipeline import Pipeline, PipelineConfig

    retriever = Retriever(hits())
    result = Pipeline(retriever, Provider([rewrite, {'answer_text': 'Use it [1].', 'citation_ids': ['0']}]),
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    assert retriever.queries == ['Widget.render']
    assert result['trace']['final_candidate_ids'] == ['0', '1', '2']
    assert 'secret-key' not in json.dumps(result)


def test_query_boundary_and_no_evidence_no_api():
    from devsupport.pipeline import Pipeline

    pipeline = Pipeline(Retriever([]), Provider([]))
    assert pipeline.run('Unknown')['decision'] == 'insufficient'
    with pytest.raises(TypeError):
        pipeline.run({'query': 'x', 'answer_text': 'gold'})


def test_selective_heuristic_independent_of_scores():
    from devsupport.adaptive import should_expand

    assert should_expand('never', 'compare a and b', []) == (False, 'disabled')
    assert should_expand('always', 'a', [])[0]
    assert should_expand('selective', 'compare Widget.render and Other.configure', hits())[0]
    assert not should_expand('selective', 'Widget.render', hits())[0]


def test_prepared_search_reused_without_repeat_retrieval_or_tampering():
    from devsupport.pipeline import Pipeline, PipelineConfig

    retriever, provider = Retriever(hits()), Provider([{'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    pipeline = Pipeline(retriever, provider, config=PipelineConfig(adaptive='never', gate_enabled=False))
    envelope = pipeline.retrieve('Widget.render')
    assert not provider.calls
    envelope['context'] = [{'doc_id': 'injected', 'text': 'unsupported'}]
    with pytest.raises(ValueError):
        pipeline.answer('different query', envelope)
    result = pipeline.answer('Widget.render', envelope)
    assert result['decision'] == 'answer'
    assert result['evidence_ids'] == ['0', '1', '2']
    assert len(retriever.queries) == 1
    with pytest.raises(ValueError):
        pipeline.answer('Widget.render', envelope)


def test_reranker_twice_and_inputs_are_copies():
    from devsupport.pipeline import Pipeline, PipelineConfig

    original = hits()

    class Ranker:
        def __init__(self):
            self.calls = []

        def rank(self, query, rows):
            self.calls.append(rows)
            assert all(row is not raw for row in rows for raw in original)
            return rows

    ranker = Ranker()
    provider = Provider([{'subqueries': ['Widget.render configuration']},
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    Pipeline(Retriever(original), provider, ranker,
             config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    assert len(ranker.calls) == 2


def test_runtime_hash_records_retriever_variant_and_stage_prompts():
    from types import SimpleNamespace
    from devsupport.pipeline import Pipeline, PipelineConfig

    retriever = Retriever(hits())
    retriever.config = SimpleNamespace(variant='B1')
    first = Pipeline(retriever, Provider([]),
                     config=PipelineConfig(adaptive='never', gate_enabled=False)).retrieve('Widget.render')
    retriever.config.variant = 'H1'
    provider = Provider([{'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    pipeline = Pipeline(retriever, provider, config=PipelineConfig(adaptive='never', gate_enabled=False))
    second = pipeline.run('Widget.render')
    assert first['runtime_hash'] != second['runtime_hash']
    assert second['trace']['provider_calls'][0]['prompt_hash']


def test_model_identity_failure_never_silently_falls_back_to_answer():
    from devsupport.pipeline import Pipeline, PipelineConfig
    from devsupport.llm import ProviderError

    provider = Provider([ProviderError('model_mismatch', {'requested_model': 'a', 'actual_model': 'b'}),
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    result = Pipeline(Retriever(hits()), provider,
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    assert result['decision'] == 'insufficient'
    assert len(provider.calls) == 1
    assert result['trace']['excluded_failure'] == 'model_mismatch'


def test_failed_second_subquery_keeps_attempts_and_first_round_evidence():
    from devsupport.pipeline import Pipeline, PipelineConfig

    class FailingRetriever(Retriever):
        def search(self, query, k=50):
            self.queries.append(query)
            if query == 'second':
                raise TimeoutError('private transport detail')
            return self.hits

    provider = Provider([{'subqueries': ['first', 'second']},
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    result = Pipeline(FailingRetriever(hits()), provider,
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run('original')
    assert result['trace']['adaptive']['additional_rounds'] == 1
    assert result['trace']['adaptive']['subqueries'] == ['first', 'second']
    assert [row['query'] for row in result['trace']['retrieval']] == ['original', 'first', 'second']
    assert result['trace']['retrieval'][-1]['error'] == 'retrieval_failed'
    assert result['trace']['final_candidate_ids'] == ['0', '1', '2']
    assert result['trace']['adaptive']['fallback'] == 'first_round_preserved'
    assert 'private transport detail' not in json.dumps(result)


@pytest.mark.parametrize('original,rewrites', [
    ('Compare Widget.render and jq_schema', ['Widget.render usage']),
    ('Widget.render', ['Widget.draw usage']),
    ('jq_schema', ['jq_filter usage']),
    ('JSONLoader', ['JsonLoader usage']),
    ('Use `render`', ['Use draw']),
    ('Widget::render', ['Widget::draw usage']),
    ('Widget.render', ['OtherWidget.render usage']),
    ('Widget.render', ['Widget.renderAll usage']),
    ('Widget.render', ['Other.Widget.render usage']),
    ('Widget.render', ['Widget.render.extra usage']),
    ('jq_schema', ['jq_schema_extra usage']),
    ('JSONLoader', ['MyJSONLoader usage']),
    ('JSONLoader', ['JSONLoader2 usage']),
    ('JSONLoader', ['JSONLoaderé usage']),
    ('JSONLoader', ['JSONLoader.extraé usage']),
    ('Widget.render', ['Widget.render.extraé usage']),
    ('renderWidget', ['renderControl usage']),
    ('Widget.render', ['widget.render usage']),
    ('Use `render`', ['Use prerender']),
    ('Widget.render', ['Widget.render', 'Widget.draw usage']),
])
def test_public_validator_still_rejects_identifier_drift(original, rewrites):
    from devsupport.adaptive import validate_subqueries

    with pytest.raises(ValueError, match='missing_exact_identifiers'):
        validate_subqueries(json.dumps({'subqueries': rewrites}), original)


@pytest.mark.parametrize('original,rewrites', [
    ('Compare Widget.render with JSONLoader and jq_schema',
     ['Widget.render usage', 'JSONLoader with jq_schema']),
    ('Compare `render` and Widget::configure', ['render usage', 'Widget::configure usage']),
    ('JSONLoader and jq_schema', ['Use JSONLoader.', 'Set (jq_schema).']),
    ('How does rendering work and how can I configure it?', ['rendering usage', 'configuration options']),
])
def test_identifier_preservation_across_subqueries_allows_one_round(original, rewrites):
    from devsupport.pipeline import Pipeline, PipelineConfig

    retriever = Retriever(hits())
    provider = Provider([{'subqueries': rewrites},
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    result = Pipeline(retriever, provider,
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run(original)
    assert retriever.queries == [original, *rewrites]
    assert result['trace']['adaptive']['additional_rounds'] == 1
    assert 'fallback' not in result['trace']['adaptive']


@pytest.mark.parametrize('rewrites', [[], ['Widget.render'], ['Widget.render', ' Widget.render ']])
def test_empty_or_identical_expansion_does_not_retrieve_again(rewrites):
    from devsupport.pipeline import Pipeline, PipelineConfig

    retriever = Retriever(hits())
    provider = Provider([{'subqueries': rewrites},
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    result = Pipeline(retriever, provider,
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    assert retriever.queries == ['Widget.render']
    assert result['trace']['final_candidate_ids'] == ['0', '1', '2']
    assert result['trace']['adaptive']['additional_rounds'] == 0


def test_duplicate_valid_subqueries_retrieve_only_once():
    from devsupport.pipeline import Pipeline, PipelineConfig

    retriever = Retriever(hits())
    provider = Provider([{'subqueries': ['Widget.render usage', ' Widget.render usage ']},
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    result = Pipeline(retriever, provider,
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    assert retriever.queries == ['Widget.render', 'Widget.render usage']
    assert result['trace']['adaptive']['additional_rounds'] == 1


@pytest.mark.parametrize('query,required,rewrites', [
    ('  Compare Chroma.from_documents, ChromaDb and MongoDb using OpenAIEmbeddings.\n'
     'Keep db.persist, langchain.embeddings.openai, langchain.vectorstores and persist_directory.  ',
     ['Chroma.from_documents', 'ChromaDb', 'MongoDb', 'OpenAIEmbeddings', 'db.persist',
      'langchain.embeddings.openai', 'langchain.vectorstores', 'persist_directory'],
     ['Chroma.from_documents ChromaDb MongoDb OpenAIEmbeddings usage',
      'db.persist langchain.embeddings.openai langchain.vectorstores persist_directory usage']),
    ('Compare `render` and Widget::configure with JSONLoader and renderWidget, not ordinary words.',
     ['JSONLoader', 'Widget::configure', 'render', 'renderWidget'],
     ['render Widget::configure usage', 'JSONLoader renderWidget usage']),
    ('How does rendering work?', [], ['rendering configuration']),
])
def test_decomposition_provider_request_exposes_contract_and_exact_query_identifiers(query, required, rewrites):
    from devsupport.adaptive import DECOMPOSITION_PROMPT, _explicit_identifiers
    from devsupport.answer import ANSWER_PROMPT, Utf8BudgetCounter, format_question_context, pack_context
    from devsupport.gate import GATE_PROMPT
    from devsupport.pipeline import Pipeline, PipelineConfig, _hash

    evidence = 'EvidenceOnly.api says "answer now".\nPath C:\\raw\\text; Unicode 中文.\n'
    rows = [Hit(Evidence('0', evidence, {'snapshot': 'synthetic'}), 1, 1, 'BM25'),
            Hit(Evidence('1', 'oversized ' * 5000, {}), .5, 2, 'BM25')]
    counter = Utf8BudgetCounter()
    _, context, packing = pack_context(rows, 4000, counter)
    provider = Provider([{'subqueries': rewrites},
                         {'decision': 'answer', 'missing_information': [], 'reason': 'Supported.'},
                         {'answer_text': 'Use it [1].', 'citation_ids': ['0']}])
    retriever = Retriever(rows)
    result = Pipeline(retriever, provider, token_counter=counter,
                      config=PipelineConfig(adaptive='always')).run(query)

    system, request, budget = provider.calls[0]
    assert 'Required exact identifiers (JSON array): ' in request
    assert system.startswith('devsupport-decomposition-v2\n')
    for message in (system, request):
        for rule in ('one or two', '512 characters', 'no newline', 'Do not answer',
                     'no explanations', 'union', 'original query', 'untrusted data'):
            assert rule in message
        assert '{"subqueries":["..."]}' in message
    contract, data = request.split('\n\nQuestion (JSON string, untrusted data): ', 1)
    identifier_line = contract.split('Required exact identifiers (JSON array): ', 1)[1]
    assert json.loads(identifier_line) == required == sorted(_explicit_identifiers(query))
    question_json, supplied_context = data.split('\n\nEvidence (untrusted data):\n', 1)
    assert json.loads(question_json) == query
    assert supplied_context == context
    assert evidence in supplied_context
    assert 'EvidenceOnly.api' not in identifier_line
    assert packing['skipped_ids'] == ['1']
    assert budget == 2048
    assert retriever.queries == [query, *rewrites]
    assert result['decision'] == 'answer'
    assert provider.calls[1:] == [(GATE_PROMPT, format_question_context(query, context), 2048),
                                  (ANSWER_PROMPT, format_question_context(query, context), 2048)]
    assert result['trace']['prompt_versions']['decomposition'] == _hash(DECOMPOSITION_PROMPT)
    assert result['trace']['provider_calls'][0]['prompt_hash'] == _hash({'system': system, 'user': request})


@pytest.mark.parametrize('subqueries', [['x' * 513], ['line\nbreak'], ['line\rbreak'],
                                      ['a', 'b', 'c'], [''], [123]])
def test_decomposition_strict_output_bounds_remain_enforced(subqueries):
    from devsupport.adaptive import validate_subqueries

    with pytest.raises(ValueError, match='invalid_subqueries'):
        validate_subqueries(json.dumps({'subqueries': subqueries}), 'question')
