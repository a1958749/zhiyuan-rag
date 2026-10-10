"""Opt-in contextual service/bridge tests; no model, provider or database calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from devsupport.types import Evidence


def launcher_fixture(tmp_path, monkeypatch):
    from devsupport import embedding, index, llm
    script = Path(__file__).resolve().parents[1] / 'scripts/serve_pipeline.py'
    spec = importlib.util.spec_from_file_location('contextual_launcher_test', script)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    base = tmp_path / 'devsupport'
    (base / 'configs').mkdir(parents=True)
    for name in ['B0', 'H0', 'H1', 'R1', 'R2', 'C', 'Full', 'provider']:
        (base / 'configs' / (name + '.json')).write_text('{}')
    (base / 'reports').mkdir()
    (base / 'reports/budget-current-gateway.json').write_text(json.dumps({
        'starting_quota': 100, 'allowance_quota': 50, 'reserve_quota': 10}))
    provenance = {'model': 'shared', 'dimension': 1024}
    corpus = {'d': Evidence('d', 'original text', {'url': 'original-url'})}
    manifest = {'provenance': provenance, 'canonical_corpus_sha256': 'same-revision',
                'data_sha256': 'same-data', 'count': 1, 'dimension': 1024, 'contextual': False}
    raw = SimpleNamespace(manifest=manifest, corpus=corpus)
    contextual = SimpleNamespace(manifest={**manifest, 'contextual': True}, corpus=dict(corpus))
    loads, providers = [], []
    def load(path, **kwargs):
        loads.append(Path(path).name)
        return raw if Path(path).name == 'raw' else contextual
    monkeypatch.setattr(index.CorpusIndex, 'load', load)
    monkeypatch.setattr(embedding.Embedder, 'provenance', property(lambda self: provenance))
    monkeypatch.setattr(llm, 'AnthropicProvider', lambda config: providers.append(config) or SimpleNamespace(config=config))
    return launcher, raw, contextual, loads, providers


def test_contextual_profile_is_opt_in_and_keeps_original_evidence(tmp_path, monkeypatch):
    launcher, raw, contextual, loads, _ = launcher_fixture(tmp_path, monkeypatch)
    default = launcher.build_service()
    assert set(default.pipelines) == {'B0', 'H0', 'H1', 'R1', 'R2'}
    assert loads == ['raw']
    service = launcher.build_service(enable_contextual=True)
    assert set(service.pipelines) == {'B0', 'H0', 'H1', 'R1', 'R2', 'C'}
    assert service.pipelines['C'].retriever.index is contextual
    assert service.pipelines['C'].retriever.embedder is service.pipelines['H1'].retriever.embedder
    assert service.pipelines['C'].provider is service.pipelines['H1'].provider
    assert service.revision == raw.manifest['canonical_corpus_sha256']
    assert service.evidence(service.revision, doc_id='d') == raw.corpus['d'].to_corpus_dict()


@pytest.mark.parametrize('field,value', [('canonical_corpus_sha256', 'other'), ('data_sha256', 'other'),
                                        ('provenance', {'model': 'other'}), ('contextual', False)])
def test_contextual_mismatch_fails_before_provider(tmp_path, monkeypatch, field, value):
    launcher, _, contextual, _, providers = launcher_fixture(tmp_path, monkeypatch)
    contextual.manifest[field] = value
    with pytest.raises(ValueError, match='contextual'):
        launcher.build_service(enable_contextual=True)
    assert providers == []


def test_contextual_changed_original_bytes_fail_closed(tmp_path, monkeypatch):
    launcher, _, contextual, _, providers = launcher_fixture(tmp_path, monkeypatch)
    contextual.corpus['d'] = Evidence('d', 'prefixed rewritten text', {})
    with pytest.raises(ValueError, match='contextual'):
        launcher.build_service(enable_contextual=True)
    assert providers == []


def test_contextual_incomplete_artifact_uses_verified_loader(tmp_path, monkeypatch):
    from devsupport.index import CorpusIndex
    verified_load = CorpusIndex.load
    launcher, raw, _, _, providers = launcher_fixture(tmp_path, monkeypatch)
    def load(path, **kwargs):
        return raw if Path(path).name == 'raw' else verified_load(path, **kwargs)
    monkeypatch.setattr(CorpusIndex, 'load', load)
    with pytest.raises(ValueError, match='missing index manifest'):
        launcher.build_service(enable_contextual=True)
    assert providers == []


@pytest.mark.parametrize('pool', ['H0', 'H1'])
@pytest.mark.parametrize('adaptive,gate_enabled', [('selective', True), ('never', False)])
def test_full_raw_pool_is_opt_in_and_preserves_pipeline_controls(tmp_path, monkeypatch, pool, adaptive, gate_enabled):
    launcher, raw, _, loads, _ = launcher_fixture(tmp_path, monkeypatch)
    config = {'pool': pool, 'pipeline': {'adaptive': adaptive, 'gate_enabled': gate_enabled}}
    (tmp_path / 'devsupport/configs/Full.json').write_text(json.dumps(config))
    assert 'Full' not in launcher.build_service().pipelines
    service = launcher.build_service(enable_full=True)
    full = service.pipelines['Full']
    assert loads == ['raw', 'raw']
    assert full.retriever.index is raw
    assert full.retriever.config.variant == pool
    assert full.config.adaptive == adaptive
    assert full.config.gate_enabled is gate_enabled
    assert full.reranker is None
    assert all(service.pipelines[name].retriever.config.variant == 'H1' for name in ('R1', 'R2'))


@pytest.mark.parametrize('profile', ['R1', 'R2'])
@pytest.mark.parametrize('pool', ['H0', 'C'])
def test_reranker_profiles_remain_restricted_to_h1(tmp_path, monkeypatch, profile, pool):
    launcher, _, _, _, _ = launcher_fixture(tmp_path, monkeypatch)
    (tmp_path / 'devsupport/configs' / (profile + '.json')).write_text(json.dumps({'pool': pool}))
    with pytest.raises(ValueError, match='reranker profiles require the H1 pool'):
        launcher.build_service()


def test_full_contextual_pool_requires_both_opt_ins(tmp_path, monkeypatch):
    launcher, _, contextual, _, providers = launcher_fixture(tmp_path, monkeypatch)
    (tmp_path / 'devsupport/configs/Full.json').write_text(json.dumps({'pool': 'C'}))
    with pytest.raises(ValueError, match='contextual'):
        launcher.build_service(enable_full=True)
    assert providers == []
    service = launcher.build_service(enable_full=True, enable_contextual=True)
    assert service.pipelines['Full'].retriever.index is contextual
    assert service.pipelines['Full'].retriever.config.variant == 'C'
    assert service.pipelines['R1'].retriever.config.variant == 'H1'
    (tmp_path / 'devsupport/configs/Full.json').write_text(json.dumps({'pool': 'B0'}))
    with pytest.raises(ValueError, match='Full pool'):
        launcher.build_service(enable_full=True, enable_contextual=True)


def test_contextual_native_source_mapping_preserves_revision():
    from devsupport.bridge_client import native_document, source_mapping
    row = {'id': 'c-source', 'metadata': {'devsupport_profile': 'C', 'corpus_revision': 'same-revision',
           'devsupport_corpus': 'freshstack/langchain'}, 'config': {'kind': 'classic', 'retrieval': {
           'retriever': 'devsupport', 'exposure': 'prefetch', 'rephrase_query': False}}}
    mapping = source_mapping(row)
    evidence = {'doc_id': 'd', 'text': 'original', 'metadata': {'url': 'original-url'}}
    native = native_document(evidence, mapping)
    assert mapping['profile'] == 'C' and native['corpus_revision'] == 'same-revision'
    assert native['text'] == evidence['text'] and native['metadata'] == evidence['metadata']


@pytest.mark.parametrize('requested,expected', [(None, ['B0', 'H0', 'H1', 'R1', 'R2']), (['C'], ['C'])])
def test_source_seeding_accepts_optional_c_without_changing_defaults(tmp_path, monkeypatch, requested, expected):
    import runpy
    import sys
    from contextlib import contextmanager
    from types import ModuleType
    created = []
    class Repository:
        def __init__(self, connection):
            pass
        def get_by_id(self, source_id):
            return None
        def create(self, label, **kwargs):
            created.append(kwargs)
    @contextmanager
    def session():
        yield object()
    for name, key, value in [
        ('docsgpt.storage.db.repositories.sources', 'SourcesRepository', Repository),
        ('docsgpt.storage.db.session', 'db_session', session),
    ]:
        module = ModuleType(name)
        setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/seed_sources.py'))
    namespace['main'].__globals__['ROOT'] = tmp_path
    manifest = tmp_path / 'devsupport/indexes/raw/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'canonical_corpus_sha256': 'same-revision'}))
    monkeypatch.setattr(sys, 'argv', ['seed_sources.py'] + (['--profiles', *requested] if requested else []))
    namespace['main']()
    assert [row['metadata']['devsupport_profile'] for row in created] == expected
    assert all(row['metadata']['corpus_revision'] == 'same-revision' for row in created)


@pytest.mark.parametrize('balance', [50, 10, None])
def test_service_paid_profiles_share_fail_closed_budget(tmp_path, monkeypatch, balance):
    from devsupport import evaluate, llm
    launcher, _, _, _, _ = launcher_fixture(tmp_path, monkeypatch)
    calls, checks = [], []
    def checker(config):
        checks.append(config)
        if balance is None:
            raise ValueError('offline')
        return {'remain_quota': balance}
    monkeypatch.setattr(evaluate, 'check_balance', checker)
    monkeypatch.setattr(llm, 'AnthropicProvider', lambda config: SimpleNamespace(
        config=config, complete=lambda *a, **kw: calls.append((a, kw))))
    service = launcher.build_service(enable_full=True, enable_contextual=True)
    provider = service.pipelines['H0'].provider
    assert all(p.provider is provider for p in service.pipelines.values())
    assert checks == calls == []
    for name in ('H0', 'C', 'Full'):
        with pytest.raises(evaluate.BudgetStopped):
            service.pipelines[name].provider.complete('question', max_tokens=10)
    assert calls == [] and len(checks) == 1


@pytest.mark.parametrize('contents', [None, '{', '{}', '[]',
    '{"starting_quota":100,"allowance_quota":Infinity,"reserve_quota":10}'])
def test_service_requires_finite_authorization(tmp_path, monkeypatch, contents):
    launcher, _, _, _, providers = launcher_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'devsupport/reports/budget-current-gateway.json'
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.build_service()
    assert providers == []


def test_service_rejects_corrupt_accounting_before_provider(tmp_path, monkeypatch):
    launcher, _, _, _, providers = launcher_fixture(tmp_path, monkeypatch)
    (tmp_path / 'devsupport/reports/budget-current-gateway.accounting.json').write_text('{}')
    with pytest.raises(ValueError, match='continuity'):
        launcher.build_service()
    assert providers == []


def test_service_observes_external_accounting_and_keeps_config(tmp_path, monkeypatch):
    from devsupport import evaluate, llm
    launcher, _, _, _, _ = launcher_fixture(tmp_path, monkeypatch)
    calls, checks = [], []
    def checker(config):
        checks.append(config)
        return {'remain_quota': 100}
    monkeypatch.setattr(evaluate, 'check_balance', checker)
    monkeypatch.setattr(llm, 'AnthropicProvider', lambda config: SimpleNamespace(
        config=config, complete=lambda *a, **kw: calls.append((a, kw)) or 'answer'))
    provider = launcher.build_service().pipelines['H0'].provider
    assert calls == checks == []
    assert provider.complete('first', max_tokens=10) == 'answer'
    assert provider.config is checks[0]
    budget = tmp_path / 'devsupport/reports/budget-current-gateway.json'
    state = budget.with_suffix('.accounting.json')
    external = evaluate.BudgetGuard(json.loads(budget.read_text()), lambda: {'remain_quota': 50}, state)
    with pytest.raises(evaluate.BudgetStopped):
        external.check()
    # A top-up cannot erase usage recorded by the evaluator's separate guard.
    with pytest.raises(evaluate.BudgetStopped):
        provider.complete('second', max_tokens=10)
    assert len(calls) == 1
    assert evaluate.read_json(state)['consumed'] == 50


def test_service_serializes_check_through_paid_completion(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from devsupport import evaluate, llm
    launcher, _, _, _, _ = launcher_fixture(tmp_path, monkeypatch)
    entered, release, second_started, second_checked = Event(), Event(), Event(), Event()
    calls, checks = [], []
    balance = [100]
    def checker(config):
        checks.append(config)
        if len(checks) > 1:
            second_checked.set()
        return {'remain_quota': balance[0]}
    def complete(*args, **kwargs):
        calls.append(args)
        entered.set()
        assert release.wait(5)
        balance[0] = 50
        return 'answer'
    monkeypatch.setattr(evaluate, 'check_balance', checker)
    monkeypatch.setattr(llm, 'AnthropicProvider', lambda config: SimpleNamespace(config=config, complete=complete))
    service = launcher.build_service(enable_full=True)
    def second():
        second_started.set()
        return service.pipelines['Full'].provider.complete('second')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.pipelines['H0'].provider.complete, 'first')
        assert entered.wait(5)
        other = pool.submit(second)
        try:
            assert second_started.wait(5)
            assert not second_checked.wait(0.2)
        finally:
            release.set()
        assert first.result(timeout=5) == 'answer'
        with pytest.raises(evaluate.BudgetStopped):
            other.result(timeout=5)
    assert len(calls) == 1


def test_service_local_search_survives_unavailable_balance_and_answer_refuses(tmp_path, monkeypatch):
    from devsupport import evaluate, llm
    from devsupport.types import Hit
    launcher, raw, _, _, _ = launcher_fixture(tmp_path, monkeypatch)
    calls, checks = [], []
    def checker(config):
        checks.append(config)
        raise ValueError('offline')
    monkeypatch.setattr(evaluate, 'check_balance', checker)
    monkeypatch.setattr(llm, 'AnthropicProvider', lambda config: SimpleNamespace(
        config=config, complete=lambda *a, **kw: calls.append((a, kw))))
    service = launcher.build_service()
    pipeline = service.pipelines['H0']
    monkeypatch.setattr(pipeline.retriever, 'search',
                        lambda query, k: [Hit(raw.corpus['d'], 1.0, 1, 'test')])
    pipeline.token_counter = len
    envelope = service.search('question', 'H0', service.revision, 'session')
    assert envelope['evidence_ids'] == ['d']
    assert checks == calls == []
    result = service.answer(envelope['run_id'], 'question', 'H0', service.revision, 'session')
    assert result['decision'] == 'insufficient'
    assert result['trace']['generation']['error'] == 'provider_failure'
    assert len(checks) == 1 and calls == []
