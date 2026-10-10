"""Offline control decision behavior using only synthetic, complete cohorts."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from devsupport import data, evaluate as ev
from devsupport.types import Question

SCRIPT = Path(__file__).resolve().parents[1] / 'analysis/control_decisions.py'


def load_audit():
    """Load the standalone reporter without executing any runtime service."""
    assert SCRIPT.is_file(), 'control decision audit is not implemented'
    spec = importlib.util.spec_from_file_location('control_decisions', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    """Replace fixture-owned JSON only."""
    path.write_text(json.dumps(value), encoding='utf-8')


def rows_at(directory, name='Mixed'):
    """Read synthetic rows for deliberate corruption tests."""
    return [json.loads(line) for line in (directory / f'{name}.jsonl').read_text().splitlines()]


def save_rows(directory, rows, name='Mixed'):
    """Write fixture-only JSONL."""
    (directory / f'{name}.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')


@pytest.fixture
def recorded_run(tmp_path):
    """Build DEV10 and synthetic TEST30 controls with global20/10/10 label counts."""
    labels = ['answer'] * 5 + ['insufficient'] * 3 + ['clarify'] * 2
    labels += ['answer'] * 15 + ['insufficient'] * 7 + ['clarify'] * 8
    controls = [{'query_id': f'c{i}', 'query': f'synthetic question {i}',
                 'split': 'dev' if i < 10 else 'test', 'expected_decision': label,
                 'nuggets': [{'relevant': ['gold']}]} for i, label in enumerate(labels)]
    controls_path = tmp_path / 'controls.json'
    save(controls_path, controls)
    questions = [Question(r['query_id'], '', r['query'], '', r['nuggets'], {}) for r in controls]
    profiles = {name: {'retrieval': 'H1', 'reranker': None, 'adaptive': 'never', 'gate_enabled': True}
                for name in ['Mixed', 'NoAnswers', 'AllErrors']}
    contract = {'profiles': profiles, 'config': {'profiles': copy.deepcopy(profiles)},
                'paths': {'controls': str(controls_path), 'queries': 'MUST_NOT_READ_REAL_TEST'},
                'artifacts': {'controls': data.sha256_file(controls_path), 'queries': 'historical-queries',
                              'index': 'historical-index'}, 'context_counter': 'proxy',
                'source': {'historical/runtime.py': 'historical-source-hash'}}
    run = {'contract_sha256': ev.digest(contract), 'profiles': list(profiles),
           'query_ids': [q.query_id for q in questions[:10]], 'phase': 'pipeline', 'split': 'dev', 'controls': True}
    directory = tmp_path / 'run'
    directory.mkdir()
    save(directory / 'run.json', run)
    decisions = ['answer', 'insufficient', 'clarify', 'answer', None,
                 'answer', 'insufficient', 'insufficient', 'clarify', 'answer']
    for name in profiles:
        save(directory / f'{name}.manifest.json', {**run, 'evaluation_contract': contract, 'profile': name,
             'settings': profiles[name], 'variant': 'H1', 'questions_sha256': 'historical-queries',
             'index_manifest_sha256': 'historical-index', 'context_counter': 'proxy', 'latency_scope': 'pipeline',
             'query_hashes': [ev.text_hash(q.query) for q in questions[:10]]})
        rows = []
        for i, q in enumerate(questions[:10]):
            result = {'decision': decisions[i] if name == 'Mixed' else 'insufficient',
                      'runtime_hash': name, 'evaluation_phase': 'pipeline', 'trace': {}}
            error = None
            if name == 'AllErrors' or (name == 'Mixed' and i == 4):
                result, error = {}, 'RuntimeError'
            elif name == 'Mixed' and i in (3, 7):
                result['trace'] = {'excluded_failure': 'provider_failure'} if i == 3 else {
                    'gate': {'reason': 'gate_validation_or_provider_failure'}}
            elif name == 'Mixed' and i == 5:
                result['trace'] = {'adaptive': {'fallback': 'first_round_preserved'}}
            rows.append(ev.make_record(q, result, 1, error))
        save_rows(directory, rows, name)
    return directory, questions


def test_complete_confusion_and_all_attempted_denominators(recorded_run):
    """Operational errors override provisional labels while recovered fallbacks count as answers."""
    directory, _ = recorded_run
    report = load_audit().audit(directory)
    method = report['methods']['Mixed']
    assert method['planned'] == method['attempted'] == 10
    assert method['confusion_matrix'] == {
        'answer': {'answer': 1, 'insufficient': 1, 'clarify': 1, 'operational_error': 2},
        'insufficient': {'answer': 1, 'insufficient': 1, 'clarify': 0, 'operational_error': 1},
        'clarify': {'answer': 1, 'insufficient': 0, 'clarify': 1, 'operational_error': 0}}
    expected = {'decision_accuracy': (3, 10), 'answer_coverage': (3, 10),
                'unsupported_control_answer_rate': (1, 3), 'unsupported_control_share_of_answers': (1, 3),
                'false_refusal_insufficient_rate': (1, 5), 'unnecessary_clarification_rate': (1, 5),
                'false_nonanswer_rate': (2, 5), 'correct_clarification_rate': (1, 2),
                'operational_error_rate': (3, 10), 'fallback_event_rate': (1, 10),
                'runtime_identity_unavailable_rate': (1, 10)}
    for key, (numerator, denominator) in expected.items():
        assert method['metrics'][key] == {'numerator': numerator, 'denominator': denominator,
                                          'rate': numerator / denominator}
    per_query = [r for r in report['per_query'] if r['profile'] == 'Mixed']
    assert per_query[3]['observed_outcome'] == 'operational_error'
    assert per_query[4]['runtime_identity_available'] is False
    assert per_query[5]['fallback'] and not per_query[5]['error']
    assert 'answer_text' not in json.dumps(report) and 'synthetic question' not in json.dumps(report)
    assert report['label_verification'] == 'not established'
    assert 'NOT a hallucination rate' in ' '.join(report['notes'])
    assert report['bindings']['historical_source'] == {'historical/runtime.py': 'historical-source-hash'}


def test_no_answers_and_all_failures_remain_complete(recorded_run):
    """Empty answer denominators are null; all failures retain the full expected-label cohort."""
    directory, _ = recorded_run
    methods = load_audit().audit(directory)['methods']
    for name in ['NoAnswers', 'AllErrors']:
        assert methods[name]['metrics']['unsupported_control_share_of_answers'] == {
            'numerator': 0, 'denominator': 0, 'rate': None}
        assert methods[name]['attempted'] == 10
    assert methods['AllErrors']['metrics']['decision_accuracy']['rate'] == 0
    assert methods['AllErrors']['runtime_hash'] is None
    assert methods['AllErrors']['metrics']['runtime_identity_unavailable_rate']['numerator'] == 10
    assert methods['AllErrors']['metrics']['false_nonanswer_rate']['numerator'] == 0


def rebind_contract(directory, contract, run):
    """Change the frozen synthetic contract consistently across all fixture manifests."""
    run['contract_sha256'] = ev.digest(contract)
    save(directory / 'run.json', run)
    for name in run['profiles']:
        path = directory / f'{name}.manifest.json'
        meta = ev.read_json(path)
        meta.update(run, evaluation_contract=contract)
        save(path, meta)


@pytest.mark.parametrize('missing_label', ['answer', 'insufficient', 'clarify'])
def test_zero_expected_label_denominators_are_null(recorded_run, missing_label):
    """A valid global20/10/10 allocation need not put every expected label in DEV10."""
    directory, _ = recorded_run
    controls_path = directory.parent / 'controls.json'
    controls = ev.read_json(controls_path)
    dev_label = 'insufficient' if missing_label == 'answer' else 'answer'
    remaining = {'answer': 20, 'insufficient': 10, 'clarify': 10}
    remaining[dev_label] -= 10
    labels = [dev_label] * 10 + [label for label, count in remaining.items() for _ in range(count)]
    for control, label in zip(controls, labels):
        control['expected_decision'] = label
    save(controls_path, controls)
    contract = ev.read_json(directory / 'Mixed.manifest.json')['evaluation_contract']
    contract['artifacts']['controls'] = data.sha256_file(controls_path)
    rebind_contract(directory, contract, ev.read_json(directory / 'run.json'))
    metrics = load_audit().audit(directory)['methods']['Mixed']['metrics']
    keys = {'answer': ['false_refusal_insufficient_rate', 'unnecessary_clarification_rate', 'false_nonanswer_rate'],
            'insufficient': ['unsupported_control_answer_rate'], 'clarify': ['correct_clarification_rate']}
    for key in keys[missing_label]:
        assert metrics[key] == {'numerator': 0, 'denominator': 0, 'rate': None}


def test_complete_synthetic_test30_uses_control_labels_only(recorded_run):
    """A control TEST30 audit must never open the separate FreshStack query artifact."""
    directory, questions = recorded_run
    run = ev.read_json(directory / 'run.json')
    run.update(split='test', query_ids=[q.query_id for q in questions[10:]])
    contract = ev.read_json(directory / 'Mixed.manifest.json')['evaluation_contract']
    rebind_contract(directory, contract, run)
    for name in run['profiles']:
        meta_path = directory / f'{name}.manifest.json'
        meta = ev.read_json(meta_path)
        meta['query_hashes'] = [ev.text_hash(q.query) for q in questions[10:]]
        save(meta_path, meta)
        save_rows(directory, [ev.make_record(q, {'decision': 'answer', 'runtime_hash': name,
                  'evaluation_phase': 'pipeline'}, 1) for q in questions[10:]], name)
    report = load_audit().audit(directory)
    for method in report['methods'].values():
        assert method['planned'] == method['attempted'] == 30
        assert method['expected_counts'] == {'answer': 15, 'insufficient': 7, 'clarify': 8}
        assert method['metrics']['unsupported_control_answer_rate'] == {
            'numerator': 7, 'denominator': 7, 'rate': 1}
    assert not any('MUST_NOT_READ_REAL_TEST' in path for path in report['bindings']['inputs'])


def test_error_fallback_overlap_and_known_runtime_identity(recorded_run):
    """Known top-level error identities are validated and fallback events overlap errors."""
    directory, questions = recorded_run
    rows = rows_at(directory)
    rows[4] = ev.make_record(questions[4], {'runtime_hash': 'Mixed',
                             'trace': {'adaptive': {'fallback': 'first_round_preserved'}}}, 1, 'RuntimeError')
    save_rows(directory, rows)
    report = load_audit().audit(directory)
    method = report['methods']['Mixed']
    assert method['operational_errors'] == 3 and method['fallback_events'] == 2
    assert method['runtime_identity_unavailable'] == 0
    assert method['runtime_hash'] == 'Mixed'
    row = next(r for r in report['per_query'] if r['profile'] == 'Mixed' and r['query_id'] == 'c4')
    assert row['error'] and row['fallback'] and row['runtime_identity_available']


@pytest.mark.parametrize('tamper', [
    'phase', 'controls_false', 'split', 'partial', 'empty', 'duplicate', 'unknown', 'reorder', 'inflight',
    'missing_rows', 'missing_manifest', 'controls_bytes', 'profile', 'settings', 'variant', 'profile_list',
    'unsafe_name', 'contract_digest', 'contract_config', 'manifest_phase', 'query_hashes', 'query_sha256',
    'metrics', 'runtime_missing', 'runtime_mixed', 'error_runtime_mixed', 'row_phase', 'row_variant', 'decision',
    'manifest_index', 'manifest_context', 'manifest_questions'])
def test_rejects_unbound_or_incomplete_artifacts(recorded_run, tamper):
    """Never publish a summary after integrity, provenance or cohort validation fails."""
    directory, questions = recorded_run
    run = ev.read_json(directory / 'run.json')
    meta_path = directory / 'Mixed.manifest.json'
    meta = ev.read_json(meta_path)
    rows = rows_at(directory)
    if tamper in ['phase', 'controls_false', 'split', 'profile_list', 'unsafe_name']:
        key, value = {'phase': ('phase', 'retrieve-only'), 'controls_false': ('controls', False),
                      'split': ('split', 'other'), 'profile_list': ('profiles', ['Mixed', 'Mixed']),
                      'unsafe_name': ('profiles', ['../Mixed'])}[tamper]
        run[key] = value
        save(directory / 'run.json', run)
    elif tamper in ['partial', 'empty', 'duplicate', 'unknown', 'reorder']:
        if tamper == 'partial':
            rows.pop()
        elif tamper == 'empty':
            rows = []
        elif tamper == 'duplicate':
            rows[-1] = rows[0]
        elif tamper == 'unknown':
            rows[0]['query_id'] = 'alien'
        else:
            rows.reverse()
        save_rows(directory, rows)
    elif tamper == 'inflight':
        save(directory / 'Mixed.inflight.json', {})
    elif tamper in ['missing_rows', 'missing_manifest']:
        (directory / ('Mixed.jsonl' if tamper == 'missing_rows' else 'Mixed.manifest.json')).unlink()
    elif tamper == 'controls_bytes':
        with (directory.parent / 'controls.json').open('a') as handle:
            handle.write(' ')
    elif tamper in ['query_sha256', 'metrics']:
        if tamper == 'metrics':
            rows[0]['metrics_prepack']['coverage@10'] = 123
        else:
            rows[0]['query_sha256'] = 'forged'
        save_rows(directory, rows)
    elif tamper in ['runtime_missing', 'runtime_mixed', 'error_runtime_mixed', 'row_phase', 'row_variant', 'decision']:
        i = 4 if tamper == 'error_runtime_mixed' else 0
        result = copy.deepcopy(rows[i]['result'])
        if tamper == 'runtime_missing':
            result.pop('runtime_hash')
        else:
            key, value = {'runtime_mixed': ('runtime_hash', 'other'),
                          'error_runtime_mixed': ('runtime_hash', 'other'),
                          'row_phase': ('evaluation_phase', 'retrieve-only'),
                          'row_variant': ('variant', 'R1'), 'decision': ('decision', 'unknown')}[tamper]
            result[key] = value
        rows[i] = ev.make_record(questions[i], result, 1, rows[i]['error'])
        save_rows(directory, rows)
    else:
        if tamper == 'contract_digest':
            meta['evaluation_contract']['source']['historical/runtime.py'] = 'different'
        elif tamper == 'contract_config':
            contract = meta['evaluation_contract']
            contract['config']['profiles']['Mixed']['gate_enabled'] = False
            run['contract_sha256'] = ev.digest(contract)
            save(directory / 'run.json', run)
            for name in run['profiles']:
                other_path = directory / f'{name}.manifest.json'
                other = ev.read_json(other_path)
                other.update(run, evaluation_contract=contract)
                save(other_path, other)
            return _assert_rejected(directory)
        elif tamper == 'settings':
            meta['settings'] = {**meta['settings'], 'adaptive': 'always'}
        else:
            key, value = {'profile': ('profile', 'AllErrors'), 'variant': ('variant', 'R1'),
                          'manifest_phase': ('phase', 'retrieve-only'), 'query_hashes': ('query_hashes', []),
                          'manifest_index': ('index_manifest_sha256', 'different'),
                          'manifest_context': ('context_counter', 'different'),
                          'manifest_questions': ('questions_sha256', 'different')}[tamper]
            meta[key] = value
        save(meta_path, meta)
    _assert_rejected(directory)


def _assert_rejected(directory):
    """Check both in-memory validation and no-output publication behavior."""
    module = load_audit()
    with pytest.raises(ValueError):
        module.publish(directory, directory.parent / 'rejected-output')
    assert not (directory.parent / 'rejected-output').exists()


def test_publish_rejects_reserved_error_label_on_non_error_result(recorded_run):
    """A validly reconstructed row cannot declare the reporter's reserved error-column label."""
    directory, questions = recorded_run
    rows = rows_at(directory)
    result = {**rows[0]['result'], 'decision': 'operational_error'}
    rows[0] = ev.make_record(questions[0], result, rows[0]['elapsed_seconds'], None)
    save_rows(directory, rows)
    output = directory.parent / 'reserved-label-output'
    with pytest.raises(ValueError, match='non-error decision must be answer/insufficient/clarify'):
        load_audit().publish(directory, output)
    assert not output.exists()


def test_cli_new_directory_and_bound_sources(monkeypatch, recorded_run):
    """CLI emits compact JSON/Markdown once, with consumed-input and implementation hashes."""
    directory, _ = recorded_run
    output = directory.parent / 'report'
    module = load_audit()
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--run', str(directory), '--output', str(output)])
    module.main()
    report = ev.read_json(output / 'summary.json')
    assert all(data.sha256_file(path) == value for path, value in report['bindings']['inputs'].items())
    assert str(SCRIPT.resolve()) in report['bindings']['implementation']
    assert any(path.endswith('report.py') for path in report['bindings']['implementation'])
    assert 'not human-verified answer faithfulness' in (output / 'summary.md').read_text()
    with pytest.raises(FileExistsError):
        module.publish(directory, output)
    empty = directory.parent / 'existing-empty'
    empty.mkdir()
    with pytest.raises(FileExistsError):
        module.publish(directory, empty)


def test_recheck_consumed_inputs_before_publication(monkeypatch, recorded_run):
    """A post-analysis byte change aborts publication before creating any output."""
    directory, _ = recorded_run
    module = load_audit()
    original = module.audit

    def change_after_analysis(path):
        report = original(path)
        with (directory / 'Mixed.jsonl').open('a') as handle:
            handle.write(' ')
        return report

    monkeypatch.setattr(module, 'audit', change_after_analysis)
    with pytest.raises(ValueError, match='changed'):
        module.publish(directory, directory.parent / 'report')
    assert not (directory.parent / 'report').exists()
