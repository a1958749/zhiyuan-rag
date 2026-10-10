"""Synthetic contract tests for the explicitly offline normalization hypothesis."""
import json
import shutil
from pathlib import Path
import runpy

import pytest

from devsupport.answer import validate_answer
from devsupport.data import sha256_file
from devsupport.evaluate import digest, text_hash, write_new
from devsupport.types import Evidence, Hit


AUDIT = Path(__file__).resolve().parents[2] / 'analysis/citation_normalization_audit.py'


@pytest.fixture
def normalize():
    return runpy.run_path(str(AUDIT))['normalize_answer']


@pytest.fixture
def packed():
    return [Hit(Evidence(i, 'original context', {}), 1, n, 'synthetic')
            for n, i in enumerate(['a', 'b', 'c'], 1)]


def test_valid_answer_is_byte_unchanged(normalize, packed):
    text = '  ' + json.dumps({'answer_text': 'Use it [2].', 'citation_ids': ['b']}) + '\n'
    assert normalize(text, packed) == (text, [])


def test_only_known_extra_declared_ids_are_pruned(normalize, packed):
    answer = 'Use [2]. `array[3]` and [1](https://example.invalid) are not citations.'
    text = json.dumps({'answer_text': answer, 'citation_ids': ['c', 'b', 'a']})
    normalized, removed = normalize(text, packed)
    assert removed == ['c', 'a']
    assert validate_answer(normalized, packed) == {'answer_text': answer, 'citation_ids': ['b']}


@pytest.mark.parametrize('answer,ids', [
    ('Use [1].', ['a', 'unknown']), ('No inline citation.', ['a', 'b']),
    ('Use [4].', ['a', 'b']), ('Use [2].', ['a', 'c']),
    ('Use [1].', ['a', 'a', 'b']), ('Use [1,2].', ['a', 'b']),
    ('Use [1-2].', ['a', 'b']), ('Only `array[1]`.', ['a', 'b']),
    ('Use [0].', ['a', 'b']),
])
def test_non_superset_failures_remain_rejected(normalize, packed, answer, ids):
    text = json.dumps({'answer_text': answer, 'citation_ids': ids})
    with pytest.raises(ValueError):
        normalize(text, packed)


@pytest.mark.parametrize('text', ['not json', '{', '[]',
                                '{"answer_text":"Use [1].","citation_ids":["a","b"],"extra":true}'])
def test_invalid_json_or_schema_remains_rejected(normalize, packed, text):
    with pytest.raises(ValueError):
        normalize(text, packed)


@pytest.fixture
def artifact_run(tmp_path):
    runtime, run, output = tmp_path / 'runtime', tmp_path / 'run', tmp_path / 'report'
    source = {}
    for name in ('answer', 'llm'):
        relative = f'devsupport/src/devsupport/{name}.py'
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(AUDIT.parents[1] / relative, target)
        source[relative] = sha256_file(target)
    ids = [f'dev-{i}' for i in range(12)]
    profiles = {name: {'retrieval': name, 'reranker': None, 'adaptive': 'never', 'gate_enabled': False}
                for name in ('B0', 'H0', 'H1')}
    contract = {'profiles': profiles, 'source': source, 'split_ids': {'dev': ids}}
    meta = {'split': 'dev', 'phase': 'pipeline', 'controls': False, 'query_ids': ids,
            'profiles': list(profiles), 'contract_sha256': digest(contract)}
    write_new(run / 'run.json', meta)
    bad = json.dumps({'answer_text': 'Use [1].', 'citation_ids': ['a', 'b']})
    for name, settings in profiles.items():
        write_new(run / f'{name}.manifest.json', {**meta, 'evaluation_contract': contract, 'profile': name,
                  'variant': name, 'settings': settings, 'query_hashes': [text_hash(i) for i in ids]})
        rows = []
        for n, qid in enumerate(ids):
            generation = {'attempts': 2 if n < 3 else 1}
            if n < 3:
                generation.update(error='citation_or_schema_validation_failed', validation_failures=[
                    {'error': 'invalid_citations', 'response_text': bad}] * (2 if n < 2 else 1))
            row = {'query_id': qid, 'query_sha256': text_hash(qid), 'error': None, 'packed_ids': ['a', 'b'],
                   'context_sha256': {i: text_hash('context') for i in ['a', 'b']},
                   'result': {'decision': 'insufficient' if n < 2 else 'answer', 'evidence_ids': ['a', 'b'],
                              'context': [{'doc_id': i, 'text': 'context', 'metadata': {}, 'citation_number': k}
                                          for k, i in enumerate(['a', 'b'], 1)],
                              'trace': {'generation': generation, 'provider_calls': [
                                  {'stage': 'generation', **({'error': 'http_403'} if n == 1 else {})}]}}}
            row['record_sha256'] = digest(row)
            rows.append(row)
        (run / f'{name}.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')
    return run, output, runtime


def test_audit_preserves_outcomes_excludes_transport_failures_and_writes_exclusively(artifact_run):
    audit = runpy.run_path(str(AUDIT))['audit']
    report = audit(*artifact_run)
    for method in report['methods'].values():
        assert method['actual_final_contract_failures'] == 1
        assert method['counterfactual_final_recoverable'] == 1
        assert method['potential_earlier_acceptance_attempts'] == 2
        assert method['actual_successes'] == 10
        for query in method['queries']:
            for attempt in query['stored_attempts']:
                if attempt['would_validate_after_pruning']:
                    assert attempt['answer_text_sha256'] == attempt['normalized_answer_text_sha256']
    saved = (artifact_run[1] / 'audit.json').read_text(encoding='utf-8')
    assert 'Use [1].' not in saved and 'response_text' not in saved
    with pytest.raises(FileExistsError):
        audit(*artifact_run)


@pytest.mark.parametrize('tamper', ['record-hash', 'order', 'profile', 'context-order', 'runtime-source'])
def test_audit_rejects_unbound_artifacts_before_output(artifact_run, tamper):
    run, output, runtime = artifact_run
    path = run / 'B0.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if tamper == 'record-hash':
        rows[0]['query_sha256'] = 'changed'
    elif tamper == 'order':
        rows.reverse()
    elif tamper == 'context-order':
        rows[0]['result']['context'].reverse()
        rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    elif tamper == 'profile':
        manifest = run / 'B0.manifest.json'
        value = json.loads(manifest.read_text(encoding='utf-8'))
        value['profile'] = 'H1'
        manifest.write_text(json.dumps(value), encoding='utf-8')
    else:
        (runtime / 'devsupport/src/devsupport/answer.py').write_text('changed', encoding='utf-8')
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    with pytest.raises(ValueError):
        runpy.run_path(str(AUDIT))['audit'](run, output, runtime)
    assert not output.exists()


def test_audit_rejects_archive_boundary_without_proof_of_complete_response(artifact_run):
    run, output, runtime = artifact_run
    prefix = json.dumps({'answer_text': 'Use [1].', 'citation_ids': ['a', 'b']}).ljust(16384)
    original = prefix + ' invalid trailing original response'
    with pytest.raises(ValueError):
        validate_answer(original, [Hit(Evidence(i, 'context', {}), 1, n, 'synthetic')
                                   for n, i in enumerate(['a', 'b'], 1)])
    path = run / 'B0.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows[0]['result']['trace']['generation']['validation_failures'][-1]['response_text'] = original[:16384]
    rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    with pytest.raises(ValueError, match='archived failure bounds'):
        runpy.run_path(str(AUDIT))['audit'](run, output, runtime)
    assert not output.exists()
