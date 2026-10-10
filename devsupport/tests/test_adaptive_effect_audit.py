"""Synthetic offline adaptive audit contracts; no provider or model execution."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from devsupport import data, evaluate as ev

SCRIPT = Path(__file__).resolve().parents[1] / 'analysis/audit_adaptive_effects.py'


def load_audit():
    """Load the standalone analysis entry point."""
    assert SCRIPT.exists(), 'adaptive effect audit is not implemented'
    spec = importlib.util.spec_from_file_location('adaptive_audit', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    """Replace only fixture-owned JSON."""
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def fixture_run(tmp_path):
    """Construct complete DEV53 with successful, no-op, fallback and unchanged rows."""
    ids = [f'q{i}' for i in range(53)]
    queries = tmp_path / 'queries.json'
    save(queries, [{'query_id': q, 'query_text': q, 'nuggets': [{'relevant': ['gold']}]} for q in ids])
    split = tmp_path / 'split.json'
    save(split, {'dev_ids': ids, 'test_ids': [f't{i}' for i in range(150)]})
    profiles = {n: {'retrieval': 'H0', 'reranker': None, 'gate_enabled': False, 'adaptive': mode}
                for n, mode in [('H0', 'never'), ('Q', 'selective'), ('QAlways', 'always')]}
    contract = {'profiles': profiles, 'paths': {'queries': str(queries), 'split': str(split)},
                'artifacts': {'queries': data.sha256_file(queries), 'split': data.sha256_file(split), 'index': 'index'},
                'split_ids': {'dev': ids, 'test': [f't{i}' for i in range(150)]}, 'seeds': [7],
                'source': {'pipeline.py': 'source'}, 'provider': {'model': 'synthetic'}, 'context_counter': 'proxy'}
    run = {'profiles': list(profiles), 'query_ids': ids, 'contract_sha256': ev.digest(contract),
           'phase': 'retrieve-only', 'split': 'dev', 'controls': False}
    directory = tmp_path / 'run'
    directory.mkdir()
    save(directory / 'run.json', run)
    questions = data.load_questions(queries)
    for name in profiles:
        save(directory / f'{name}.manifest.json', {**run, 'evaluation_contract': contract, 'profile': name,
             'settings': profiles[name], 'variant': 'H0', 'questions_sha256': contract['artifacts']['queries'],
             'index_manifest_sha256': 'index', 'context_counter': 'proxy', 'latency_scope': 'retrieve-only',
             'query_hashes': [ev.text_hash(q.query) for q in questions]})
        rows = []
        for i, q in enumerate(questions):
            triggered = name != 'H0' and (name == 'QAlways' or i < 3)
            adaptive = {'triggered': triggered, 'subqueries': [], 'anchored_query_count': 0,
                        'anchoring_policy': 'test-policy'}
            ranked = ['other']
            retrieval = [{'query': q.query}]
            if triggered and i == 0:
                adaptive.update(subqueries=['sub'], anchored_query_count=1, additional_rounds=1)
                retrieval.append({'query': 'sub'})
                ranked = ['gold']
            elif triggered and i == 2:
                adaptive.update(fallback='first_round_preserved', failure_stage='decomposition_provider',
                                failure_code='provider_failure')
            result = {'evaluation_phase': 'retrieve-only', 'runtime_hash': name, 'evidence_ids': ranked,
                      'trace': {'final_candidate_ids': ranked, 'adaptive': adaptive, 'retrieval': retrieval}}
            rows.append(ev.make_record(q, result, 1))
        (directory / f'{name}.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    return directory, questions


def test_complete_effects_and_failure_denominators(fixture_run):
    """Count triggered no-ops separately and retain all53 quality denominators."""
    run, _ = fixture_run
    report = load_audit().audit(run)
    q = report['methods']['Q']['adaptive_effects']
    assert (q['triggered'], q['untriggered'], q['expanded'], q['fallback'], q['no_op']) == (3, 50, 1, 1, 1)
    assert q['actual_total_additional_queries'] == 1
    assert report['comparisons']['Q-minus-H0']['all53']['prepack_coverage10']['win'] == 1
    assert report['comparisons']['Q-minus-H0']['all53']['prepack_coverage10']['tie'] == 52
    assert not report['preservation_checks']['violations']


@pytest.mark.parametrize('tamper', ['row', 'manifest', 'source', 'reorder', 'incomplete', 'duplicate',
                                   'phase', 'test', 'annotations', 'inflight'])
def test_fail_closed_contracts(fixture_run, tamper):
    """Reject damaged or mixed contracts before writing audit output."""
    directory, _ = fixture_run
    path = directory / 'Q.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if tamper in ('row', 'reorder', 'incomplete', 'duplicate'):
        if tamper == 'row':
            rows[0]['metrics_prepack']['coverage@10'] = 999
        elif tamper == 'reorder':
            rows.reverse()
        elif tamper == 'incomplete':
            rows.pop()
        else:
            rows[-1] = rows[0]
        path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    elif tamper in ('manifest', 'source'):
        path = directory / 'Q.manifest.json'
        meta = ev.read_json(path)
        if tamper == 'manifest':
            meta['settings']['retrieval'] = 'H1'
        else:
            meta['evaluation_contract']['source']['pipeline.py'] = 'different'
        save(path, meta)
    elif tamper == 'annotations':
        (directory.parent / 'queries.json').write_text('[]')
    elif tamper == 'inflight':
        save(directory / 'Q.inflight.json', {})
    else:
        path = directory / 'run.json'
        run = ev.read_json(path)
        run['phase' if tamper == 'phase' else 'split'] = 'pipeline' if tamper == 'phase' else 'test'
        save(path, run)
    with pytest.raises(ValueError):
        load_audit().audit(directory)


@pytest.mark.parametrize('index', [2, 4])
def test_preservation_violation_is_reported_with_ids(fixture_run, index):
    """Validly rehashed fallback/untriggered changes remain visible violations."""
    directory, questions = fixture_run
    path = directory / 'Q.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = copy.deepcopy(rows[index]['result'])
    result['trace']['final_candidate_ids'] = result['evidence_ids'] = ['gold']
    rows[index] = ev.make_record(questions[index], result, 1)
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    violations = load_audit().audit(directory)['preservation_checks']['violations']
    assert violations[0]['query_id'] == f'q{index}'
    assert violations[0]['candidate_prepack_ids'] == ['gold']


def test_operational_error_retained_but_preservation_unavailable(fixture_run):
    """Terminal failures score zero over53 but do not masquerade as changed rankings."""
    directory, questions = fixture_run
    path = directory / 'Q.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[4] = ev.make_record(questions[4], {}, 1, 'RuntimeError')
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    report = load_audit().audit(directory)
    assert report['methods']['Q']['attempted'] == 53
    assert report['methods']['Q']['errors'] == 1
    assert report['preservation_checks']['unavailable'][0]['query_id'] == 'q4'


def test_excluded_failure_takes_precedence_over_fallback(fixture_run):
    """Model mismatch remains terminal even when decomposition preserves round one."""
    directory, questions = fixture_run
    path = directory / 'Q.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = copy.deepcopy(rows[2]['result'])
    result['trace']['excluded_failure'] = 'model_mismatch'
    result['trace']['provider_calls'] = [{'stage': 'decomposition', 'error': 'model_mismatch'}]
    rows[2] = ev.make_record(questions[2], result, 1)
    # Keep a usable fallback beside the terminal failure to distinguish events from statuses.
    fallback_result = copy.deepcopy(rows[2]['result'])
    del fallback_result['trace']['excluded_failure']
    fallback_result['trace']['provider_calls'] = [{'stage': 'decomposition', 'error': 'provider_failure'}]
    rows[3] = ev.make_record(questions[3], fallback_result, 1)
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    report = load_audit().audit(directory)
    method = report['methods']['Q']
    effect = method['adaptive_effects']
    assert effect['operational_error'] == method['errors'] == 1
    assert effect['fallback'] == 1
    assert effect['fallback_event_rows'] == 2
    assert effect['failure_stage_codes'] == [
        {'stage': 'decomposition_provider', 'code': 'provider_failure', 'count': 2}]
    assert method['attempted'] == method['planned'] == 53
    assert len(report['per_query']) == 53
    assert report['per_query'][2]['profiles']['Q']['status'] == 'operational_error'
    assert report['per_query'][3]['profiles']['Q']['status'] == 'fallback'
    comparison = report['comparisons']['Q-minus-H0']
    assert all(metric['n'] == 53 for metric in comparison['all53'].values())
    assert all(metric['n'] == 1 for metric in
               comparison['selection_conditioned_on_candidate']['expanded_only'].values())
    assert report['preservation_checks']['unavailable'][0]['query_id'] == 'q2'


def test_explicit_wrong_phase_on_error_and_cli_output(fixture_run, tmp_path, monkeypatch):
    """The real CLI writes reviewable files and rejects explicit mixed-phase errors."""
    import sys
    directory, questions = fixture_run
    module = load_audit()
    output = tmp_path / 'audit'
    monkeypatch.setattr(sys, 'argv', ['audit', '--run', str(directory), '--output', str(output)])
    module.main()
    assert len(ev.read_json(output / 'audit.json')['per_query']) == 53
    assert 'selection-conditioned' in (output / 'audit.md').read_text()
    path = directory / 'Q.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[4] = ev.make_record(questions[4], {'evaluation_phase': 'pipeline'}, 1, 'RuntimeError')
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    with pytest.raises(ValueError, match='phase'):
        module.audit(directory)
