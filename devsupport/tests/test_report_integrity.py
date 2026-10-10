"""Offline regressions for phase scope and frozen method identity in reports."""
import copy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from devsupport import data, evaluate as ev


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def invoke(monkeypatch, script, *args):
    """Run a script entry point against only caller-owned fixture paths."""
    monkeypatch.setattr(sys, 'argv', [script, *map(str, args)])
    runpy.run_path(str(SCRIPTS / script), run_name='__main__')


@pytest.fixture
def recorded_run(tmp_path):
    """Write a synthetic partial DEV run with a successful and an operational-error row."""
    queries = tmp_path / 'queries.json'
    queries.write_text(json.dumps([{'query_id': q, 'query_text': q, 'nuggets': []}
                                   for q in ['q1', 'q2', 'q3']]), encoding='utf-8')
    settings = ev.normalize_profiles({'Base': {'retrieval': 'B0'}, 'Full': {'retrieval': 'H1'}})
    contract = {'profiles': settings, 'paths': {'queries': str(queries)},
                'artifacts': {'queries': data.sha256_file(queries)}, 'seeds': [7]}
    run = {'contract_sha256': ev.digest(contract), 'profiles': ['Base', 'Full'],
           'query_ids': ['q1', 'q2', 'q3'], 'phase': 'pipeline', 'split': 'dev', 'controls': False}
    directory = tmp_path / 'run'
    directory.mkdir()
    ev.write_new(directory / 'run.json', run)
    questions = data.load_questions(queries)
    for name in run['profiles']:
        meta = {**run, 'evaluation_contract': contract, 'profile': name,
                'settings': settings[name], 'variant': settings[name]['retrieval']}
        ev.write_new(directory / (name + '.manifest.json'), meta)
        rows = [ev.make_record(questions[0], {'decision': 'answer', 'variant': meta['variant']}, 1),
                ev.make_record(questions[1], {}, 1, 'RuntimeError')]
        (directory / (name + '.jsonl')).write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')
    return directory, tmp_path / 'report', questions


@pytest.mark.parametrize('tamper', ['profile', 'settings', 'variant', 'row_variant', 'swap', 'duplicate'])
def test_report_rejects_method_mislabeling(monkeypatch, recorded_run, tamper):
    """Self-consistent record hashes cannot legitimize a swapped method identity."""
    directory, output, questions = recorded_run
    path = directory / 'Base.manifest.json'
    meta = ev.read_json(path)
    if tamper in ('profile', 'variant'):
        meta[tamper] = 'Full' if tamper == 'profile' else 'H1'
    elif tamper == 'settings':
        meta['settings'] = meta['evaluation_contract']['profiles']['Full']
    elif tamper == 'row_variant':
        rows_path = directory / 'Base.jsonl'
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        rows[0] = ev.make_record(questions[0], {**rows[0]['result'], 'variant': 'H1'}, 1)
        rows_path.write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')
    elif tamper == 'swap':
        meta = ev.read_json(directory / 'Full.manifest.json')
        (directory / 'Base.jsonl').write_bytes((directory / 'Full.jsonl').read_bytes())
    else:
        run = ev.read_json(directory / 'run.json')
        run['profiles'] = ['Base', 'Base']
        (directory / 'run.json').write_text(json.dumps(run), encoding='utf-8')
        meta['profiles'] = run['profiles']
    path.write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='profile|variant'):
        invoke(monkeypatch, 'report_evaluation.py', '--run', directory, '--output', output)
    assert not output.exists()


def test_report_preserves_partial_runs_errors_and_legacy_results(monkeypatch, recorded_run):
    """Historical producer rows lack variant, and an unstarted method need not have JSONL."""
    directory, output, questions = recorded_run
    rows_path = directory / 'Base.jsonl'
    rows = [ev.make_record(questions[0], {'decision': 'answer'}, 1),
            ev.make_record(questions[1], {'balance_observations': []}, 1, 'RuntimeError')]
    rows_path.write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')
    (directory / 'Full.jsonl').unlink()
    invoke(monkeypatch, 'report_evaluation.py', '--run', directory, '--output', output)
    methods = ev.read_json(output / 'summary.json')['methods']
    assert methods['Base']['attempted'] == 2 and methods['Base']['errors'] == 1
    assert methods['Base']['unattempted'] == 1 and not methods['Base']['complete']
    assert methods['Full']['attempted'] == 0


@pytest.mark.parametrize('phase', ['retrieve-only', 'pipeline'])
def test_control_decision_accuracy_requires_pipeline(monkeypatch, tmp_path, phase):
    """Drive runner summaries with synthetic controls and no model/provider access."""
    from devsupport import embedding, index, llm, pipeline, review, service
    from devsupport.types import Question
    controls = tmp_path / 'controls.json'
    controls.write_text(json.dumps([{'query_id': 'c1', 'query': 'ambiguous', 'split': 'dev',
                                    'expected_decision': 'clarify', 'category': 'ambiguous'}]), encoding='utf-8')
    monkeypatch.setattr(review, 'load_controls', lambda *a: json.loads(controls.read_text()))
    frozen = {'profiles': ev.normalize_profiles({'Base': {'retrieval': 'B0'}}),
              'split_ids': {'dev': ['freshstack-dev-id']},
              'paths': {'queries': 'synthetic', 'controls': str(controls), 'index': 'synthetic',
                        'budget_start': 'synthetic-budget'},
              'artifacts': {'queries': 'hash', 'index': 'hash'}, 'config': {}, 'provider': {},
              'generation_max_tokens': 2048, 'context_counter': 'synthetic'}
    monkeypatch.setattr(ev, 'build_contract', lambda *a: copy.deepcopy(frozen))
    monkeypatch.setattr(data, 'load_questions', lambda *a: [Question('c1', '', 'ambiguous', '', [], {})])
    monkeypatch.setattr(index.CorpusIndex, 'load', lambda *a: SimpleNamespace(manifest={'provenance': {}}))
    monkeypatch.setattr(embedding, 'Embedder', lambda: SimpleNamespace(provenance={}))
    monkeypatch.setattr(service, 'ModelEmbedder', lambda *a: object())
    monkeypatch.setattr(ev, 'build_retriever', lambda *a: object())
    monkeypatch.setattr(pipeline, 'Pipeline', lambda *a, **kw: object())
    monkeypatch.setattr(ev, 'pipeline_result', lambda *a: {'decision': 'clarify' if phase == 'pipeline' else 'insufficient'})
    guard = SimpleNamespace(stopped=False, observations=[], check=lambda: None)
    monkeypatch.setattr(ev, 'BudgetGuard', lambda *a, **kw: guard)
    monkeypatch.setattr(ev, 'read_json', lambda *a: {})
    monkeypatch.setattr(ev, 'GuardedProvider', lambda *a: object())
    monkeypatch.setattr(llm, 'AnthropicProvider', lambda *a: object())
    output = tmp_path / 'run'
    invoke(monkeypatch, 'evaluate_pipeline.py', '--config', 'synthetic', '--output', output,
           '--phase', phase, '--controls')
    assert json.loads((output / 'run.json').read_text())['query_ids'] == ['c1']
    summary = json.loads((output / 'Base.summary-n1.json').read_text())
    if phase == 'pipeline':
        assert summary['control_decision_accuracy_attempted'] == 1
    else:
        assert summary['control_decision_accuracy_attempted'] is None
        assert 'pipeline' in summary['control_decision_accuracy_scope']


@pytest.mark.parametrize('ranker', [None, 'R1'])
def test_report_accepts_matching_explicit_variants(monkeypatch, recorded_run, ranker):
    """Explicit labels identify the reranker when present, otherwise the retriever."""
    directory, output, questions = recorded_run
    run = ev.read_json(directory / 'run.json')
    contract = ev.read_json(directory / 'Base.manifest.json')['evaluation_contract']
    contract['profiles']['Full']['reranker'] = ranker
    run['contract_sha256'] = ev.digest(contract)
    (directory / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    for name in run['profiles']:
        meta_path = directory / (name + '.manifest.json')
        meta = ev.read_json(meta_path)
        settings = contract['profiles'][name]
        meta.update(run, evaluation_contract=contract, settings=settings,
                    variant=settings['reranker'] or settings['retrieval'])
        meta_path.write_text(json.dumps(meta), encoding='utf-8')
        row = ev.make_record(questions[0], {'decision': 'answer', 'variant': meta['variant']}, 1)
        (directory / (name + '.jsonl')).write_text(json.dumps(row) + '\n', encoding='utf-8')
    invoke(monkeypatch, 'report_evaluation.py', '--run', directory, '--output', output)
    summary = ev.read_json(output / 'summary.json')
    assert all(method['attempted'] == 1 for method in summary['methods'].values())
    assert 'Full-minus-Base' in summary['paired_bootstrap']
