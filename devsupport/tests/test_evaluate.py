"""Offline contract tests
    no provider, GPU or credential access."""
import copy
import json
from types import SimpleNamespace

import pytest

from devsupport import evaluate as ev
from devsupport import report, review
from devsupport.types import Question


def question(qid='q'):
    return Question(qid, 'title', 'body', 'SECRET GOLD', [{'relevant_corpus_ids': ['d']}], {})


def contract():
    return {'schema': 1, 'source': {'x': 'abc'}, 'config': {'model': 'm'},
            'split_ids': {'dev': ['q'], 'test': [str(i) for i in range(150)]},
            'generation_ids': [str(i) for i in range(30)], 'seeds': [7],
            'profiles': {'H1': {'retrieval': 'H1', 'adaptive': 'never', 'gate_enabled': True}}}


def test_lock_tamper_and_no_overwrite(tmp_path):
    path = tmp_path / 'lock.json'
    ev.freeze(path, contract())
    assert ev.verify_lock(path, contract())
    changed = contract()
    changed['seeds'] = [8]
    with pytest.raises(ValueError):
        ev.verify_lock(path, changed)
    with pytest.raises(FileExistsError):
        ev.freeze(path, contract())
    data = json.loads(path.read_text())
    data['contract']['seeds'] = [9]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ev.verify_lock(path, contract())


@pytest.mark.parametrize('phase', ['pipeline', 'retrieve-only', 'retrieval', 'rerank'])
def test_holdout_selection_preserves_exact_phase_cohort(tmp_path, phase):
    frozen = contract()
    # Nonlexical order ensures selection cannot silently sort the frozen cohort.
    ids = [f't{(i * 37) % 150:03d}' for i in range(150)]
    frozen['split_ids']['test'] = ids
    frozen['generation_ids'] = ids[:30]
    path = tmp_path / 'lock.json'
    ev.freeze(path, frozen)
    assert ev.select_ids(frozen, 'test', phase, path) == (ids[:30] if phase == 'pipeline' else ids)


@pytest.mark.parametrize('phase', ['pipeline', 'retrieve-only', 'retrieval', 'rerank'])
def test_holdout_selection_requires_matching_lock(tmp_path, phase):
    with pytest.raises(ValueError, match='prior frozen evaluation lock'):
        ev.select_ids(contract(), 'test', phase)
    with pytest.raises(FileNotFoundError):
        ev.select_ids(contract(), 'test', phase, tmp_path / 'missing.json')
    path = tmp_path / 'lock.json'
    ev.freeze(path, contract())
    changed = contract()
    changed['seeds'] = [8]
    with pytest.raises(ValueError, match='evaluation lock mismatch'):
        ev.select_ids(changed, 'test', phase, path)


@pytest.mark.parametrize('phase', ['pipeline', 'retrieve-only', 'retrieval', 'rerank'])
@pytest.mark.parametrize('cohort', ['reordered', 'short', 'outside'])
def test_holdout_selection_rejects_malformed_generation_cohort(tmp_path, phase, cohort):
    frozen = contract()
    if cohort == 'reordered':
        frozen['generation_ids'].reverse()
    elif cohort == 'short':
        frozen['generation_ids'].pop()
    else:
        frozen['generation_ids'][-1] = 'outside-holdout'
    path = tmp_path / 'lock.json'
    ev.freeze(path, frozen)
    with pytest.raises(ValueError, match='generation cohort must be first30 holdout IDs'):
        ev.select_ids(frozen, 'test', phase, path)


@pytest.mark.parametrize('phase', ['pipeline', 'retrieve-only', 'retrieval', 'rerank'])
def test_dev_selection_preserves_configured_cohort_without_lock(phase):
    frozen = contract()
    frozen['split_ids']['dev'] = ['dev-z', 'dev-a', 'dev-m']
    assert ev.select_ids(frozen, 'dev', phase) == ['dev-z', 'dev-a', 'dev-m']


def test_profiles_are_isolated_and_share_generation_contract():
    profiles = ev.normalize_profiles({'H1': {'retrieval': 'H1'}, 'Full': {'adaptive': 'selective'}})
    profiles['Full']['gate_enabled'] = True
    assert profiles['H1']['gate_enabled'] is False
    with pytest.raises(ValueError):
        ev.normalize_profiles({'Full': {'generation_max_tokens': 4096}})
    with pytest.raises(ValueError):
        ev.normalize_profiles({'Full': {'prompt': 'different'}})


def test_checkpoint_resume_query_identity_denominators_and_boundary(tmp_path):
    calls = []
    def execute(text):
        assert type(text) is str and 'SECRET' not in text
        calls.append(text)
        raise RuntimeError('private provider detail')
    qs = [question()]
    path = tmp_path / 'rows.jsonl'
    rows = ev.run(qs, execute, path, {'phase': 'pipeline'}, resume=False)
    assert len(rows) == 1 and rows[0]['error'] == 'RuntimeError'
    assert 'private' not in path.read_text()
    assert report.summarize(rows, ['q'])['errors'] == 1
    assert report.summarize(rows, ['q'])['attempted'] == 1
    assert ev.run(qs, execute, path, {'phase': 'pipeline'}, resume=True) == rows
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        ev.run(qs, execute, path, {'phase': 'pipeline'})
    with pytest.raises(ValueError):
        ev.run([Question('q', 'changed', '', '', [], {})], execute, path, {'phase': 'pipeline'}, resume=True)
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError):
        ev.run(qs, execute, path, {'phase': 'pipeline'}, resume=True)


def test_result_hashes_raw_context_and_metrics(tmp_path):
    payload = {'evidence_ids': ['d'], 'context': [{'doc_id': 'd', 'text': 'original', 'metadata': {'url': 'u'}}],
               'decision': 'answer', 'citation_ids': ['d'], 'answer_text': 'x [1]',
               'trace': {'final_candidate_ids': ['d', 'e']}}
    rows = ev.run([question()], lambda text: payload, tmp_path / 'r.jsonl', {'phase': 'pipeline'})
    row = rows[0]
    assert row['prepack_ids'] == ['d', 'e'] and row['packed_ids'] == ['d']
    assert row['context_sha256']['d'] == ev.text_hash('original')
    assert 'metrics_prepack' in row and 'metrics_packed' in row
    assert row['result']['context'][0]['metadata'] == {'url': 'u'}


@pytest.mark.parametrize('separator', ['\u2028', '\u2029', '\x85'])
def test_resume_does_not_split_a_valid_json_string_or_replay_call(tmp_path, separator):
    text = 'original source before' + separator + 'after'
    payload = {'ranked_ids': ['d'], 'evidence_ids': ['d'], 'context': [{'doc_id': 'd', 'text': text}],
               'decision': 'answer', 'citation_ids': ['d'], 'answer_text': 'x [1]'}
    calls = []

    def execute(query):
        calls.append(query)
        return payload

    path = tmp_path / 'rows.jsonl'
    rows = ev.run([question()], execute, path, {'phase': 'pipeline'})
    original = path.read_bytes()
    assert separator.encode() in original
    assert ev.run([question()], execute, path, {'phase': 'pipeline'}, resume=True) == rows
    assert path.read_bytes() == original and len(calls) == 1


def test_budget_whitelist_and_shared_usage_stops_before_call():
    values = iter([{'remain_quota': 900, 'token': 'SECRET'}, {'remain_quota': 850}, {'remain_quota': 700}])
    budget = ev.BudgetGuard({'starting_quota': 1000, 'allowance_quota': 250, 'reserve_quota': 100},
                           lambda: next(values))
    assert budget.check()['remain_quota'] == 900
    assert 'SECRET' not in json.dumps(budget.observations)
    budget.check()
    with pytest.raises(ev.BudgetStopped):
        budget.check()
    for bad in [float('inf'), -1, True]:
        with pytest.raises(ValueError):
            ev.BudgetGuard({'starting_quota': 1000, 'allowance_quota': bad, 'reserve_quota': 100}, lambda: {})
    with pytest.raises(ValueError):
        ev.numeric_balance({'remain_quota': 'secret', 'key': 'secret'})


def test_budget_wrapper_checks_each_call_and_preserves_billed_result():
    calls = []
    class Provider:
        config = SimpleNamespace(model='m')
        def complete(self, *args, **kwargs):
            calls.append(args)
            return {'text': 'billed', 'usage': {'output_tokens': 1}}
    balances = iter([{'remain_quota': 900}, {'remain_quota': 700}])
    guard = ev.BudgetGuard({'starting_quota': 1000, 'allowance_quota': 250, 'reserve_quota': 100},
                          lambda: next(balances))
    wrapped = ev.GuardedProvider(Provider(), guard)
    assert wrapped.complete('s', 'u')['text'] == 'billed'
    with pytest.raises(ev.BudgetStopped):
        wrapped.complete('s', 'u')
    assert len(calls) == 1


def test_rerank_membership_and_annotations_never_online():
    from devsupport.types import Evidence, Hit
    hits = [Hit(Evidence(str(i), 'raw', {}), i, i + 1, 'rrf') for i in range(50)]
    class Ranker:
        last_trace = {}
        def rank(self, query, pool):
            assert query == 'question' and len(pool) == 50
            return pool[:-1]
    with pytest.raises(ValueError):
        ev.rerank_pool('question', hits, Ranker())


def test_summary_zero_and_pair_identity():
    assert report.summarize([], [])['metrics_prepack'] is None
    with pytest.raises(ValueError):
        report.summarize([], ['q'])
    rows = [{'query_id': 'q', 'error': None, 'result': {'decision': 'insufficient'},
             'metrics_prepack': {'mrr': 0}, 'metrics_packed': {'mrr': 0}, 'elapsed_seconds': 1}]
    summary = report.summarize(rows, ['q'])
    assert summary['refusal'] == 1 and summary['success'] == 0
    assert report.paired(rows, rows)['prepack']['mrr']['mean_delta'] == 0
    changed = copy.deepcopy(rows)
    changed[0]['query_id'] = 'other'
    with pytest.raises(ValueError):
        report.paired(rows, changed)


def test_blind_review_has_gold_but_no_method_labels():
    rows = [{'query_id': 'q', 'result': {'answer_text': 'answer', 'context': [
        {'doc_id': 'd', 'text': 'raw', 'metadata': {'url': 'source'}, 'score': .4,
         'score_kind': 'cosine', 'rank': 1}], 'decision': 'answer'}}]
    template, sealed = review.blind_review([question()], {'H1': rows, 'Full': rows}, {}, seed=7)
    encoded = json.dumps(template)
    assert 'H1' not in encoded and 'Full' not in encoded and 'SECRET GOLD' in encoded
    assert len(sealed) == 2
    assert all(item['ratings'] is None for item in template[0]['responses'])
    assert 'score' not in encoded and 'cosine' not in encoded


def test_operational_failure_is_not_knowledge_refusal_and_repair_is_success():
    row = {'result': {'decision': 'insufficient', 'trace': {
        'gate': {'reason': 'gate_validation_or_provider_failure'}}}}
    assert report.outcome(row) == {'error': True, 'success': False, 'refusal': False, 'fallback': False}
    row['result'] = {'decision': 'answer', 'trace': {'generation': {'error': 'citation_or_schema_validation_failed'}}}
    assert report.outcome(row)['success'] and not report.outcome(row)['error']


def test_balance_transport_uses_authorized_endpoint_and_numeric_whitelist():
    from devsupport.llm import ProviderConfig
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, limit):
            return b'{"data":{"remain_quota":123,"token":"SECRET","used_quota":4}}'
    def transport(request, timeout):
        assert request.full_url == 'https://www.xiaoyaoapi.com/api/usage/token/'
        assert request.get_header('Authorization') == 'Bearer memory-only'
        return Response()
    assert ev.check_balance(ProviderConfig(), lambda: 'memory-only', transport) == {'remain_quota': 123, 'used_quota': 4}


def test_inflight_prevents_billed_replay(tmp_path):
    path = tmp_path / 'r.jsonl'
    path.with_suffix('.inflight.json').write_text('{}')
    with pytest.raises(ValueError, match='in-flight'):
        ev.run([question()], lambda text: pytest.fail('must not execute'), path, {})


def test_cached_pool_rejects_other_versions_and_changed_membership():
    from devsupport.types import Evidence
    corpus = {str(i): Evidence(str(i), 'raw', {}) for i in range(50)}
    q = question()
    row = ev.make_record(q, {'ranked_ids': list(corpus), 'scores': list(range(50)),
                            'score_kinds': ['rrf'] * 50}, 1)
    frozen = contract()
    meta = {'variant': 'H1', 'questions_sha256': 'queries', 'index_manifest_sha256': 'index',
            'evaluation_contract': frozen, 'contract_sha256': ev.digest(frozen),
            'settings': frozen['profiles']['H1']}
    pools = ev.cached_pools([row], meta, [q], corpus, 'queries', 'index', frozen)
    assert len(pools[q.query]) == 50
    for field, value in [('contract_sha256', 'stale'), ('settings', {'retrieval': 'B0'})]:
        changed = {**meta, field: value}
        with pytest.raises(ValueError):
            ev.cached_pools([row], changed, [q], corpus, 'queries', 'index', frozen)
    changed_row = {**row, 'record_sha256': 'INVALID'}
    with pytest.raises(ValueError):
        ev.cached_pools([changed_row], meta, [q], corpus, 'queries', 'index', frozen)
    meta['variant'] = 'R1'
    with pytest.raises(ValueError):
        ev.cached_pools([row], meta, [q], corpus, 'queries', 'index', frozen)


def test_budget_ledger_preserves_observed_consumption_across_resume(tmp_path):
    auth = {'starting_quota': 1000, 'allowance_quota': 500, 'reserve_quota': 100}
    path = tmp_path / 'shared-budget.jsonl'
    values = iter([850, 1100, 900])
    first = ev.BudgetGuard(auth, lambda: {'remain_quota': next(values)}, state_path=path)
    for _ in range(3):
        first.check()
    assert first.consumed == 350
    resumed = ev.BudgetGuard(auth, lambda: {'remain_quota': 550}, state_path=path)
    with pytest.raises(ev.BudgetStopped):
        resumed.check()
    assert resumed.consumed == 700
    with pytest.raises(ValueError):
        ev.BudgetGuard({**auth, 'allowance_quota': 600}, lambda: {}, state_path=path)


def test_shared_runtime_profiles_keep_prompt_and_generation_cap_identical():
    from devsupport.answer import ANSWER_PROMPT
    from devsupport.types import Evidence, Hit
    calls = []
    class Provider:
        def complete(self, system, user, max_tokens):
            calls.append((system, user, max_tokens))
            return {'text': json.dumps({'answer_text': 'answer [1]', 'citation_ids': ['d']}),
                    'requested_model': 'm', 'actual_model': 'm'}
    class Retriever:
        def search(self, query, k):
            assert type(query) is str and 'SECRET' not in query
            return [Hit(Evidence('d', 'raw', {}), 1, 1, 'score')]
    from devsupport.pipeline import Pipeline, PipelineConfig
    for _name in ['B0', 'H1']:
        pipeline = Pipeline(Retriever(), Provider(), config=PipelineConfig(adaptive='never', gate_enabled=False))
        result = pipeline.run('question')
        assert result['decision'] == 'answer'
    assert all(c[0] == ANSWER_PROMPT and c[2] == 2048 for c in calls)
    assert len({c[1] for c in calls}) == 1


def test_reranked_cache_cannot_claim_h1_membership():
    with pytest.raises(ValueError, match='provenance'):
        ev.cached_pools([], {'variant': 'H1', 'phase': 'rerank', 'questions_sha256': 'q',
                            'index_manifest_sha256': 'i'}, [], {}, 'q', 'i')


def test_quota_high_watermark_cannot_reset_after_topup():
    balances = iter([{'remain_quota': 850}, {'remain_quota': 1100}, {'remain_quota': 900}])
    guard = ev.BudgetGuard({'starting_quota': 1000, 'allowance_quota': 500, 'reserve_quota': 100},
                          lambda: next(balances))
    guard.check()
    guard.check()
    guard.check()
    assert guard.observations[-1]['shared_usage_quota'] == 350


def test_provider_insufficient_or_failure_stops_paid_loop():
    from devsupport.llm import ProviderError
    class Provider:
        config = None
        def complete(self, *args, **kwargs):
            raise ProviderError('http_402')
    guard = ev.BudgetGuard({'starting_quota': 1000, 'allowance_quota': 500, 'reserve_quota': 100},
                          lambda: {'remain_quota': 900})
    with pytest.raises(ProviderError):
        ev.GuardedProvider(Provider(), guard).complete('s', 'u')
    assert guard.stopped


def test_available_counter_is_quota_source_when_used_counter_is_stale():
    balances = iter([{'total_available': 900, 'total_used': 0, 'unlimited_quota': False},
                     {'total_available': 700, 'total_used': 0}])
    guard = ev.BudgetGuard({'starting_quota': 1000, 'allowance_quota': 250, 'reserve_quota': 100},
                          lambda: next(balances))
    assert guard.check()['remain_quota'] == 900
    with pytest.raises(ev.BudgetStopped):
        guard.check()
    assert guard.observations[-1]['shared_usage_quota'] == 300
    assert 'unlimited_quota' not in guard.observations[0]


def test_contract_hashes_models_source_and_all_data(tmp_path):
    folder = tmp_path / 'devsupport'
    for path in ['src/devsupport', 'scripts', 'configs', 'model-cache/embedding',
                 'model-cache/pairwise', 'model-cache/listwise', 'index']:
        (folder / path).mkdir(parents=True, exist_ok=True)
    (folder / 'src/devsupport/pipeline.py').write_text('source')
    (folder / 'scripts/evaluate_pipeline.py').write_text('runner')
    for name in ['B0', 'B1', 'H0', 'H1', 'R1', 'R2']:
        (folder / 'configs' / (name + '.json')).write_text('{}')
    for name in ['embedding', 'pairwise', 'listwise']:
        (folder / 'model-cache' / name / 'artifact-manifest.json').write_text('{}')
    for name in ['queries', 'corpus']:
        (folder / name).write_text('public fixture')
    (folder / 'index/manifest.json').write_text('{}')
    (folder / 'provider').write_text('{}')
    (folder / 'split').write_text(json.dumps({'dev_ids': ['d' + str(i) for i in range(53)],
                                            'test_ids': ['t' + str(i) for i in range(150)]}))
    config = {name: 'devsupport/' + name for name in ['queries', 'corpus', 'index', 'split', 'provider']}
    config['profiles'] = {'H1': {}}
    path = folder / 'config.json'
    path.write_text(json.dumps(config))
    first = ev.build_contract(path, tmp_path)
    assert first['generation_ids'] == ['t' + str(i) for i in range(30)]
    lock = folder / 'lock'
    ev.freeze(lock, first)
    (folder / 'model-cache/embedding/artifact-manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):
        ev.verify_lock(lock, ev.build_contract(path, tmp_path))


def test_local_retrieval_success_is_counted_without_claiming_generated_answer():
    row = {'query_id': 'q', 'result': {'ranked_ids': ['d']}, 'error': None,
           'elapsed_seconds': 1, 'metrics_prepack': {'mrr': 1}, 'metrics_packed': {'mrr': 0}}
    summary = report.summarize([row], ['q'])
    assert summary['success'] == 1 and summary['answers'] == 0
    assert summary['metrics_packed'] is None


def test_retrieve_only_does_not_count_default_decision_as_refusal():
    row = {'result': {'decision': 'insufficient', 'evaluation_phase': 'retrieve-only',
                      'evidence_ids': ['d'], 'trace': {}}}
    assert report.outcome(row)['success'] and not report.outcome(row)['refusal']


def optional_fixture(tmp_path, variant):
    """Create tiny complete artifact contracts without inference or native dependencies."""
    folder = tmp_path / 'devsupport'
    for path in ['src/devsupport', 'scripts', 'configs', 'model-cache/embedding',
                 'model-cache/pairwise', 'model-cache/listwise', 'index', 'optional']:
        (folder / path).mkdir(parents=True, exist_ok=True)
    for name in ['B0', 'B1', 'H0', 'H1', 'R1', 'R2']:
        (folder / 'configs' / (name + '.json')).write_text('{}')
    for name in ['embedding', 'pairwise', 'listwise']:
        (folder / 'model-cache' / name / 'artifact-manifest.json').write_text('{}')
    for name in ['queries', 'corpus', 'provider']:
        (folder / name).write_text('{}')
    (folder / 'split').write_text(json.dumps({'dev_ids': ['d' + str(i) for i in range(53)],
                                            'test_ids': ['t' + str(i) for i in range(150)]}))
    from devsupport.data import sha256_file
    raw = {'schema': 1, 'count': 1, 'dimension': 1024, 'contextual': False,
           'canonical_corpus_sha256': 'canonical', 'data_sha256': sha256_file(folder / 'corpus'),
           'provenance': {'model': 'shared'}}
    (folder / 'index/manifest.json').write_text(json.dumps(raw))
    names = ['index.faiss', 'index.json'] if variant == 'U' else [
        'dense.faiss', 'vectors.npy', 'ids.json', 'corpus.json', 'audit.json',
        *[f'{leg}/{name}' for leg in ['ordinary', 'identifier'] for name in
          ['data.csc.index.npy', 'indices.csc.index.npy', 'indptr.csc.index.npy',
           'vocab.index.json', 'params.index.json']]]
    for name in names:
        target = folder / 'optional' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('fixture')
    manifest = {**raw, 'contextual': True, 'profile': variant,
                'raw_manifest_sha256': sha256_file(folder / 'index/manifest.json'),
                'files': {name: sha256_file(folder / 'optional' / name) for name in names}}
    filename = 'native-manifest.json' if variant == 'U' else 'manifest.json'
    (folder / 'optional' / filename).write_text(json.dumps(manifest))
    native = tmp_path / 'docsgpt/retriever/classic_rag.py'
    native.parent.mkdir(parents=True)
    native.write_text('native source')
    key = 'native_u_index' if variant == 'U' else 'contextual_index'
    config = {name: 'devsupport/' + name for name in ['queries', 'corpus', 'index', 'split', 'provider']}
    config.update(profiles={variant: {'retrieval': variant}}, **{key: 'devsupport/optional'})
    path = folder / 'config.json'
    path.write_text(json.dumps(config))
    return path, config, folder / 'optional' / filename


@pytest.mark.parametrize('variant', ['U', 'C'])
def test_optional_profiles_bind_complete_artifacts_and_reject_reranking(tmp_path, variant):
    path, config, manifest_path = optional_fixture(tmp_path, variant)
    built = ev.build_contract(path, tmp_path)
    key = 'native_u_index' if variant == 'U' else 'contextual_index'
    assert built['paths'][key] == str(manifest_path.parent.resolve())
    assert key in built['artifacts']
    assert built['generation_ids'] == ['t' + str(i) for i in range(30)]
    with pytest.raises(ValueError, match='identical H1'):
        ev.normalize_profiles({variant: {'retrieval': variant, 'reranker': 'R1'}})
    del config[key]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        ev.build_contract(path, tmp_path)


@pytest.mark.parametrize('variant', ['U', 'C'])
def test_optional_index_rejects_wrong_corpus_model_or_incomplete_files(tmp_path, variant):
    path, _, manifest_path = optional_fixture(tmp_path, variant)
    original = json.loads(manifest_path.read_text())
    for key, value in [('canonical_corpus_sha256', 'other'), ('provenance', {'model': 'other'}), ('files', {})]:
        manifest_path.write_text(json.dumps({**original, key: value}))
        with pytest.raises(ValueError):
            ev.build_contract(path, tmp_path)
    manifest_path.write_text(json.dumps(original))
    (manifest_path.parent / next(iter(original['files']))).write_text('tampered')
    with pytest.raises(ValueError):
        ev.build_contract(path, tmp_path)


def test_native_source_changes_invalidate_optional_lock(tmp_path):
    path, _, _ = optional_fixture(tmp_path, 'U')
    built = ev.build_contract(path, tmp_path)
    assert 'docsgpt/retriever/classic_rag.py' in built['source']
    lock = tmp_path / 'lock.json'
    ev.freeze(lock, built)
    (tmp_path / 'docsgpt/retriever/classic_rag.py').write_text('changed implementation')
    with pytest.raises(ValueError, match='lock mismatch'):
        ev.verify_lock(lock, ev.build_contract(path, tmp_path))


def test_runtime_factory_routes_real_adapters_and_preserves_native_trace(monkeypatch):
    from devsupport import native_baseline, retrieval, index
    from devsupport.types import Evidence, Hit
    seen = []
    class Native:
        def __init__(self, source_id, index_dir, doc_token_limit):
            seen.append((source_id, index_dir, doc_token_limit))
            self.last_trace = {}
        def search(self, query, k):
            assert query == 'exact question' and k == 50
            self.last_trace = {'native_effective_token_budget': 45000, 'ranked_ids': ['d']}
            return [Hit(Evidence('d', 'raw', {}), .25, 1, 'l2_distance')]
    monkeypatch.setattr(native_baseline, 'NativeClassicRetriever', Native)
    runtime = {'paths': {'native_u_index': 'native', 'contextual_index': 'contextual'},
               'config': {'native_u_source_id': 'source'}}
    native = ev.build_retriever(runtime, 'U', object(), object())
    payload = ev.retrieval_result('exact question', native)
    assert seen == [('source', 'native', 50000)]
    assert payload['ranked_ids'] == ['d'] and payload['score_kinds'] == ['l2_distance']
    assert payload['trace']['retrieval'][0]['native']['native_effective_token_budget'] == 45000
    raw = SimpleNamespace(manifest={'provenance': {'model': 'shared'}})
    contextual = object()
    monkeypatch.setattr(index.CorpusIndex, 'load', lambda path, **kw: contextual)
    monkeypatch.setattr(retrieval, 'Retriever', lambda i, e, v: (i, e, v))
    embedder = object()
    assert ev.build_retriever(runtime, 'C', raw, embedder) == (contextual, embedder, 'C')
    assert ev.build_retriever(runtime, 'H1', raw, embedder) == (raw, embedder, 'H1')


def test_native_environment_requires_python312_and_shared_embedding_protocol(monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'version_info', (3, 10, 0))
    with pytest.raises(RuntimeError, match='Python 3.12'):
        ev.prepare_native_environment()
    monkeypatch.setattr(sys, 'version_info', (3, 12, 0))
    for name in ['VECTOR_STORE', 'EMBEDDINGS_NAME', 'EMBEDDINGS_BASE_URL', 'EMBEDDINGS_MAX_QUERY_TOKENS']:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('EMBEDDINGS_MAX_INPUT_TOKENS', '1')
    ev.prepare_native_environment()
    import os
    assert os.environ['EMBEDDINGS_MAX_QUERY_TOKENS'] == '0'
    assert os.environ['EMBEDDINGS_MAX_INPUT_TOKENS'] == '0'
    assert os.environ['EMBEDDINGS_BASE_URL'] == 'http://127.0.0.1:8093'


def test_traced_native_pipeline_preserves_shared_prompt_context_and_every_search():
    from devsupport.answer import ANSWER_PROMPT
    from devsupport.pipeline import Pipeline, PipelineConfig
    from devsupport.types import Evidence, Hit
    calls = []
    class Native:
        last_trace = {}
        def search(self, query, k):
            assert query == 'exact question' and k == 50
            self.last_trace = {'native_effective_token_budget': 45000, 'ranked_ids': ['d']}
            return [Hit(Evidence('d', 'raw', {}), .25, 1, 'l2_distance')]
    class Provider:
        def complete(self, system, user, max_tokens):
            calls.append((system, user, max_tokens))
            return {'text': json.dumps({'answer_text': 'answer [1]', 'citation_ids': ['d']}),
                    'requested_model': 'm', 'actual_model': 'm'}
    traced = ev.TracedRetriever(Native())
    pipeline = Pipeline(traced, Provider(), config=PipelineConfig(adaptive='never', gate_enabled=False))
    result = ev.pipeline_result('exact question', pipeline, 'pipeline')
    assert result['decision'] == 'answer'
    assert result['trace']['retrieval'][0]['native']['native_effective_token_budget'] == 45000
    assert calls[0][0] == ANSWER_PROMPT and calls[0][2] == 2048
    assert result['context'][0]['text'] == 'raw'
    ev.pipeline_result('exact question', pipeline, 'retrieve-only')
    assert len(traced.search_traces) == 1 and len(calls) == 1


def test_native_profile_rejects_different_embedding_endpoint(tmp_path):
    path, config, _ = optional_fixture(tmp_path, 'U')
    config['gpu_url'] = 'http://other-service:8093'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='shared local'):
        ev.build_contract(path, tmp_path)


@pytest.mark.parametrize('script', ['native_u', 'serve_native'])
def test_native_launchers_override_ambient_input_limit(tmp_path, monkeypatch, script):
    """Stop before retrieval/startup and inject credential boundary without reading secrets."""
    import os
    import runpy
    import sys
    from pathlib import Path
    from types import ModuleType
    monkeypatch.setattr(os, 'environ', dict(os.environ, EMBEDDINGS_MAX_INPUT_TOKENS='1'))
    root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(root / 'devsupport/scripts' / (script + '.py')))
    if script == 'serve_native':
        fake = ModuleType('devsupport.src.devsupport.credentials')
        fake.load_credential = lambda path: 'inert-test-placeholder'
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
        namespace['configure']()
    else:
        from devsupport import native_baseline
        query = tmp_path / 'queries.json'
        query.write_text(json.dumps([{'query_id': 'q', 'query': 'exact query'}]))
        monkeypatch.setattr(sys, 'argv', ['native_u.py', 'retrieve', '--queries', str(query),
                                        '--output', str(tmp_path / 'unused.jsonl')])
        class StoppedBeforeSearch(Exception):
            pass
        def stop(*args):
            assert os.environ['EMBEDDINGS_MAX_INPUT_TOKENS'] == '0'
            raise StoppedBeforeSearch()
        monkeypatch.setattr(native_baseline, 'NativeClassicRetriever', stop)
        with pytest.raises(StoppedBeforeSearch):
            namespace['main']()
    assert os.environ['EMBEDDINGS_MAX_INPUT_TOKENS'] == '0'


def test_native_trace_keeps_failed_adaptive_query_without_stale_native_details():
    from devsupport.pipeline import Pipeline, PipelineConfig
    from devsupport.types import Evidence, Hit
    class Native:
        last_trace = {}
        def search(self, query, k):
            if query == 'failing subquery':
                raise RuntimeError('private transport details')
            self.last_trace = {'query': query, 'native_effective_token_budget': 45000}
            return [Hit(Evidence('d', 'raw', {}), .25, 1, 'l2_distance')]
    class Provider:
        def complete(self, system, user, max_tokens):
            return {'text': json.dumps({'subqueries': ['successful subquery', 'failing subquery']}),
                    'requested_model': 'm', 'actual_model': 'm'}
    pipeline = Pipeline(ev.TracedRetriever(Native()), Provider(),
                        config=PipelineConfig(adaptive='always', gate_enabled=False))
    result = ev.pipeline_result('original query', pipeline, 'retrieve-only')
    traces = result['trace']['retrieval']
    assert [item['query'] for item in traces] == ['original query', 'successful subquery', 'failing subquery']
    assert traces[-1] == {'query': 'failing subquery', 'ids': [], 'error': 'retrieval_failed'}
    assert traces[1]['native']['query'] == 'successful subquery'
    assert result['trace']['adaptive']['fallback'] == 'first_round_preserved'
    assert result['evidence_ids'] == ['d']
    assert 'private transport' not in json.dumps(result)
def test_exception_checkpoint_without_optional_httpx(tmp_path, monkeypatch):
    """A minimal core install still durably records ordinary query failures."""
    import builtins
    from devsupport.evaluate import run
    from devsupport.types import Question

    original_import = builtins.__import__

    def without_httpx(name, *args, **kwargs):
        if name == 'httpx':
            raise ModuleNotFoundError('optional httpx absent')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_httpx)
    selected = [Question('offline-q', 'Title', 'Query', '', [], {})]

    def failure(query):
        raise ValueError('private exception detail')

    path = tmp_path / 'rows.jsonl'
    rows = run(selected, failure, path, {})
    assert len(rows) == 1 and rows[0]['error'] == 'ValueError'
    assert 'private exception detail' not in path.read_text(encoding='utf-8')
    assert not path.with_suffix('.inflight.json').exists()
