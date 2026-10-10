"""Offline synthetic checks for exact consumed-byte evidence bindings."""
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def publisher():
    spec = importlib.util.spec_from_file_location(
        'final_publisher', Path(__file__).with_name('publish-final-evidence-20261010.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consumed_metadata_is_bound_before_later_mutation(publisher, tmp_path):
    path = tmp_path / 'metadata.json'
    original = b'{"method":"H0"}'
    path.write_bytes(original)
    bindings = {}
    # The parsed value and receipt must describe the same bytes, even if a later
    # read observes a changed file. Failure must precede any publication.
    parsed = publisher.read(path, bindings)
    path.write_text('{"method":"Q"}', encoding='utf-8')
    assert parsed == {'method': 'H0'}
    assert list(bindings.values()) == [hashlib.sha256(original).hexdigest()]
    with pytest.raises(ValueError, match='changed'):
        publisher.read(path, bindings)


def test_external_audit_cannot_replace_previously_consumed_hash(publisher, tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text('{}', encoding='utf-8')
    bindings = {}
    publisher.read(path, bindings)
    with pytest.raises(ValueError, match='changed'):
        publisher.bind(path, bindings, expected='0' * 64)


def test_bom_and_unicode_json_values_preserved(publisher, tmp_path):
    path = tmp_path / 'metadata.json'
    value = {'text': 'before\u2028after\u2029tail\u0085end'}
    raw = b'\xef\xbb\xbf' + json.dumps(value, ensure_ascii=False).encode('utf-8')
    path.write_bytes(raw)
    bindings = {}
    assert publisher.read(path, bindings) == value
    assert list(bindings.values()) == [hashlib.sha256(raw).hexdigest()]


def test_expected_hash_must_match_current_bytes(publisher, tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_bytes(b'{}')
    with pytest.raises(ValueError, match='changed'):
        publisher.bind(path, {}, expected='0' * 64)


def test_publication_receipt_must_equal_captured_source_even_if_file_changes_back(publisher, tmp_path, monkeypatch):
    path = tmp_path / 'source.jsonl'
    row = {'query_id': 'q1', 'answer': 'original'}
    row['record_sha256'] = publisher.digest(row)
    path.write_bytes(json.dumps(row).encode() + b'\n')
    original_sha = publisher.checksum(path)
    alternate_sha = '0' * 64
    output = tmp_path / 'public'
    monkeypatch.setattr(publisher, 'OUTPUT', output)
    monkeypatch.setattr(publisher, 'collect', lambda: (
        {'config': {'final_method_aliases': {}}}, 'lock', [('one', path, 'H0', ['q1'])], {}))
    monkeypatch.setattr(publisher, 'load_credential', lambda *_: 'synthetic-credential')
    monkeypatch.setattr(publisher, 'build_contract', lambda *_: {})
    monkeypatch.setattr(publisher, 'verify_lock', lambda *_: 'lock')
    monkeypatch.setattr(publisher.public_module, 'project_record', lambda value, *_: deepcopy(value))
    monkeypatch.setattr(publisher.public_module, '_scan', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publisher.public_module, '_redact_query_keys', lambda _: None)

    def publish(_source, target, *_args, **_kwargs):
        target.mkdir(parents=True)
        (target / 'records.jsonl').write_bytes(b'synthetic-public')
        return {'source_sha256': alternate_sha, 'output_sha256': publisher.checksum(target / 'records.jsonl')}

    monkeypatch.setattr(publisher.public_module, 'publish', publish)
    checksum = publisher.checksum
    original_reads = 0

    def changing_checksum(candidate):
        nonlocal original_reads
        if Path(candidate) == path:
            original_reads += 1
            # The unguarded receipt checks can see alternate bytes, then the
            # final bindings check can see the originals again (ABA).
            return alternate_sha if original_reads == 2 else original_sha
        return checksum(candidate)

    monkeypatch.setattr(publisher, 'checksum', changing_checksum)
    monkeypatch.setattr(publisher.sys, 'argv', ['publisher'])
    with pytest.raises(ValueError, match='captured'):
        publisher.main()
    assert not (output / 'summary.json').exists()


@pytest.fixture
def complete_scope(publisher, tmp_path, monkeypatch):
    root = tmp_path / 'root'
    reports = root / 'reports'
    config = root / 'evaluation.final.v2.json'
    lock = reports / 'final-evaluation-lock-v2.json'
    ids = [f'q{i}' for i in range(150)]
    controls = [{'query_id': f'{split}{i}', 'split': split}
                for split, count in [('dev', 10), ('test', 30)] for i in range(count)]
    methods = ['B0', 'B1', 'H0', 'H1', 'C', 'U', 'R1', 'R2', 'Q', 'QAlways', 'G', 'Full']
    contract = {'split_ids': {'test': ids, 'dev': ['devq']}, 'generation_ids': ids[:30],
                'paths': {'controls': str(root / 'controls.json')}, 'artifacts': {'controls': 'control-hash'},
                'profiles': {m: {'retrieval': m, 'reranker': None} for m in methods}}
    locked = publisher.digest(contract)

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    save(config, {})
    save(lock, {'contract': contract, 'sha256': locked})
    save(root / 'controls.json', controls)
    save(reports / 'final-dev-selection-v1.json', {'selected_method': 'H0',
         'generation_methods': ['B0', 'H1', 'H0'], 'adaptive_holdout_methods': ['Q', 'QAlways'],
         'control_methods': ['H0', 'G']})
    save(reports / 'final-workflow-controller-20261010-v2.json',
         {'stage': 'automated_test_complete_needs_ui_video_public_delivery', 'final_contract_sha256': locked})
    groups = [('test-local-v2', ['B0', 'B1', 'H0', 'H1', 'C'], 'retrieval', False, ids, 'test'),
              ('test-native-u-v2', ['U'], 'retrieval', False, ids, 'test'),
              ('test-rerank-v2', ['R1', 'R2'], 'rerank', False, ids, 'test'),
              ('test-adaptive-v2', ['Q', 'QAlways'], 'retrieve-only', False, ids, 'test'),
              ('test-generation-v2', ['B0', 'H1', 'H0'], 'pipeline', False, ids[:30], 'test'),
              ('test-controls-v2', ['H0', 'G'], 'pipeline', True, [r['query_id'] for r in controls[10:]], 'test'),
              ('dev-decision-controls-v2', ['H0', 'Q', 'G', 'Full'], 'pipeline', True,
               [r['query_id'] for r in controls[:10]], 'dev')]
    for directory, names, phase, is_control, cohort, split in groups:
        meta = {'profiles': names, 'query_ids': cohort, 'phase': phase, 'controls': is_control,
                'split': split, 'contract_sha256': locked}
        save(reports / directory / 'run.json', meta)
        for name in names:
            save(reports / directory / f'{name}.manifest.json',
                 {**meta, 'profile': name, 'evaluation_contract': contract,
                  'settings': contract['profiles'][name], 'variant': name})
    save(reports / 'test-ai-judge-v2/summary-030.json', {'complete': True, 'judged_questions': 30})
    quality = {'generation_contract_sha256': locked, 'input_sha256': {}}
    save(reports / 'test-answer-quality-v2/summary.json', quality)
    monkeypatch.setattr(publisher, 'ROOT', root)
    monkeypatch.setattr(publisher, 'REPORTS', reports)
    monkeypatch.setattr(publisher, 'CONFIG', config)
    monkeypatch.setattr(publisher, 'LOCK', lock)
    monkeypatch.setattr(publisher, 'build_contract', lambda *_: contract)
    monkeypatch.setattr(publisher, 'verify_lock', lambda *_: locked)
    monkeypatch.setattr(publisher, 'load_controls', lambda *_: controls)
    monkeypatch.setattr(publisher.answer_quality, 'audit', lambda *_: quality)
    return reports, contract, save


def test_complete_cohort_requires_all_twenty_files_and_1720_records(publisher, complete_scope):
    _, _, items, bindings = publisher.collect()
    assert len(items) == 20
    assert sum(len(ids) for _, _, _, ids in items) == 1720
    assert len(bindings) > 20


@pytest.fixture
def continued_scope(publisher, complete_scope):
    reports, contract, save = complete_scope
    locked = publisher.digest(contract)
    save(reports / 'final-workflow-controller-20261010-v2.json',
         {'stage': 'stopped_exception', 'final_contract_sha256': locked})
    rows = [{'query_id': qid, 'error': 'retained failed attempt' if i == 2 else None}
            for i, qid in enumerate(contract['generation_ids'])]
    for row in rows:
        row['record_sha256'] = publisher.digest(row)
    lines = [json.dumps(row).encode() + b'\n' for row in rows]
    (reports / 'test-generation-v2/B0.jsonl').write_bytes(b''.join(lines))
    for version, count, stage in [('v3', 19, 'stopped_exception'),
                                  ('v4', 20, 'automated_test_complete_needs_ui_video_public_delivery')]:
        save(reports / f'final-workflow-resume-20261010-{version}.json',
             {'stage': stage, 'contract_sha256': locked, 'preserved_B0_rows': count,
              'preserved_B0_prefix_sha256': hashlib.sha256(b''.join(lines[:count])).hexdigest(),
              'preserved_B0_record_sha256': [r['record_sha256'] for r in rows[:count]],
              'failed_attempts_retained': True})
    return reports, contract, save


def test_completed_same_contract_continuation_accepts_full_scope(publisher, continued_scope):
    _, _, items, bindings = publisher.collect()
    assert len(items) == 20
    assert sum(len(ids) for _, _, _, ids in items) == 1720
    assert sum('final-workflow' in path for path in bindings) == 3
    assert any(path.endswith('B0.jsonl') for path in bindings)


@pytest.mark.parametrize('version,field,value', [
    ('v2', 'stage', 'test_generation30'),
    ('v2', 'final_contract_sha256', 'changed'),
    ('v3', 'stage', 'test_generation30_resume'),
    ('v3', 'contract_sha256', 'changed'),
    ('v4', 'stage', 'test_generation30_resume'),
    ('v4', 'contract_sha256', 'changed'),
    ('v4', 'preserved_B0_rows', 19),
    ('v4', 'failed_attempts_retained', False),
    ('v4', 'preserved_B0_record_sha256', []),
    ('v4', 'preserved_B0_prefix_sha256', 'changed'),
])
def test_continuation_cannot_bypass_completion_or_lineage(publisher, continued_scope, version, field, value):
    reports, _, save = continued_scope
    name = ('final-workflow-controller-20261010-v2.json' if version == 'v2'
            else f'final-workflow-resume-20261010-{version}.json')
    path = reports / name
    data = json.loads(path.read_text())
    data[field] = value
    save(path, data)
    with pytest.raises(ValueError, match='completion|continuation|prefix|contract'):
        publisher.collect()


def test_rewriting_failed_attempt_with_valid_digest_is_rejected(publisher, continued_scope):
    reports, _, _ = continued_scope
    path = reports / 'test-generation-v2/B0.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[2]['error'] = None
    rows[2]['record_sha256'] = publisher.digest({k: v for k, v in rows[2].items() if k != 'record_sha256'})
    path.write_bytes(b''.join(json.dumps(row).encode() + b'\n' for row in rows))
    with pytest.raises(ValueError, match='prefix'):
        publisher.collect()


def test_byte_reformatting_preserved_prefix_is_rejected(publisher, continued_scope):
    reports, _, _ = continued_scope
    path = reports / 'test-generation-v2/B0.jsonl'
    path.write_bytes(path.read_bytes().replace(b'"query_id": ', b'"query_id":  ', 1))
    with pytest.raises(ValueError, match='prefix'):
        publisher.collect()


@pytest.mark.parametrize('field,value', [('settings', {'retrieval': 'other', 'reranker': None}),
                                       ('variant', 'other'), ('profiles', ['B0'])])
def test_retrieval_method_identity_drift_is_rejected(publisher, complete_scope, field, value):
    reports, _, save = complete_scope
    path = reports / 'test-native-u-v2/U.manifest.json'
    data = json.loads(path.read_text())
    data[field] = value
    save(path, data)
    with pytest.raises(ValueError, match='identity'):
        publisher.collect()


@pytest.mark.parametrize('valid_digest', [False, True])
def test_dev_controls_cannot_substitute_another_contract(publisher, complete_scope, valid_digest):
    reports, _, save = complete_scope
    path = reports / 'dev-decision-controls-v2/Q.manifest.json'
    data = json.loads(path.read_text())
    data['evaluation_contract']['artifacts']['controls'] = 'different-controls'
    if valid_digest:
        changed = publisher.digest(data['evaluation_contract'])
        data['contract_sha256'] = changed
        meta_path = path.parent / 'run.json'
        meta = json.loads(meta_path.read_text())
        meta['contract_sha256'] = changed
        save(meta_path, meta)
        # Give the group a consistent alternative digest; the control artifact
        # still must match the actual frozen control collection.
        for other in path.parent.glob('*.manifest.json'):
            item = json.loads(other.read_text())
            item['evaluation_contract']['artifacts']['controls'] = 'different-controls'
            item['contract_sha256'] = changed
            save(other, item)
    save(path, data)
    with pytest.raises(ValueError, match='identity|contract|controls'):
        publisher.collect()
