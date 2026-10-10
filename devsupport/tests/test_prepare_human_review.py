"""Exercise the real frozen-run to shareable human-review handoff offline."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_answer_judge import run_fixture


@pytest.fixture
def preparer():
    folder = Path(__file__).parents[1] / 'analysis'
    path = folder / 'prepare_human_review.py'
    assert path.exists(), 'frozen-run human review handoff must exist'
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location('prepare_human_review', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(folder))


def test_complete_run_produces_unscored_bound_shareable_files(preparer, tmp_path):
    from devsupport.data import sha256_file
    run_fixture(tmp_path)
    output = tmp_path / 'review'
    preparer.prepare(tmp_path, output, 'dev')
    packets_path = output / 'share/packets.json'
    packets = json.loads(packets_path.read_text(encoding='utf-8'))
    manifest = json.loads((output / 'private/manifest.json').read_text())
    mapping = json.loads((output / 'private/sealed-mapping.json').read_text())
    assert manifest['packet_sha256'] == sha256_file(packets_path)
    assert manifest['human_reviews_completed'] == 0 and manifest['split'] == 'dev'
    assert manifest['scope'] == 'DEV rehearsal; not TEST acceptance'
    assert len(mapping) == 4 and len(packets) == 1
    assert set(p.name for p in (output / 'share').iterdir()) == {'packets.json', 'review.html', 'README.txt'}
    text = packets_path.read_text() + (output / 'share/review.html').read_text(encoding='utf-8')
    assert 'LEAK' not in text and 'sealed-mapping' not in text
    assert all(r.get('ratings') is None and r['operational_failure'] is False for r in packets[0]['responses'])
    assert manifest['packet_sha256'] in text


def test_missing_profile_and_wrong_split_do_not_publish(preparer, tmp_path):
    run_fixture(tmp_path)
    output = tmp_path / 'review'
    with pytest.raises(ValueError, match='split'):
        preparer.prepare(tmp_path, output, 'test')
    assert not output.exists()
    (tmp_path / 'C.jsonl').unlink()
    with pytest.raises(ValueError, match='complete'):
        preparer.prepare(tmp_path, output, 'dev')
    assert not output.exists()


def test_handoff_refuses_overwrite(preparer, tmp_path):
    run_fixture(tmp_path)
    output = tmp_path / 'review'
    preparer.prepare(tmp_path, output, 'dev')
    before = (output / 'share/review.html').read_bytes()
    with pytest.raises(FileExistsError):
        preparer.prepare(tmp_path, output, 'dev')
    assert (output / 'share/review.html').read_bytes() == before


def test_real_failure_row_is_not_presented_as_quality_to_score(preparer, tmp_path):
    from devsupport.evaluate import make_record
    from test_answer_judge import fixture
    run_fixture(tmp_path)
    question = fixture()[0][0]
    failure = make_record(question, {'decision': 'error', 'answer_text': '', 'context': []}, 1, 'http_403')
    (tmp_path / 'B0.jsonl').write_text(json.dumps(failure) + '\n')
    output = tmp_path / 'review'
    preparer.prepare(tmp_path, output, 'dev')
    packets = json.loads((output / 'share/packets.json').read_text())
    assert sum(r['operational_failure'] for r in packets[0]['responses']) == 1


@pytest.mark.parametrize('offset', [0, 1])
def test_only_preregistered_first_thirty_test_questions_are_accepted(preparer, tmp_path, offset):
    from dataclasses import replace
    import pyarrow as pa
    import pyarrow.parquet as pq
    from devsupport.data import sha256_file
    from devsupport.evaluate import digest, make_record
    from test_answer_judge import fixture

    contract = run_fixture(tmp_path)
    original, variants, _ = fixture()
    qs = [replace(original[0], query_id=f'q{i:02d}') for i in range(31)]
    pq.write_table(pa.Table.from_pylist([
        {'query_id': q.query_id, 'query_title': q.query_title, 'query_text': q.query_text,
         'answer_text': q.answer_text, 'nuggets': q.nuggets} for q in qs]), tmp_path / 'queries.parquet')
    split = {'dev_ids': ['other'], 'test_ids': [q.query_id for q in qs]}
    (tmp_path / 'split.json').write_text(json.dumps(split))
    contract['split_ids'] = {'dev': split['dev_ids'], 'test': split['test_ids']}
    selected = qs[offset:offset + 30]
    contract['generation_ids'] = [q.query_id for q in selected]
    for key in ('queries', 'split'):
        contract['artifacts'][key] = sha256_file(contract['paths'][key])
    run = {'phase': 'pipeline', 'split': 'test', 'controls': False,
           'query_ids': contract['generation_ids'], 'profiles': list(variants), 'contract_sha256': digest(contract)}
    (tmp_path / 'run.json').write_text(json.dumps(run))
    for name, rows in variants.items():
        path = tmp_path / (name + '.jsonl')
        path.write_text('\n'.join(json.dumps(make_record(q, rows[0]['result'], 1)) for q in selected) + '\n')
        meta_path = path.with_suffix('.manifest.json')
        meta = json.loads(meta_path.read_text())
        meta_path.write_text(json.dumps({**meta, **run, 'evaluation_contract': contract}))
    if offset:
        with pytest.raises(ValueError, match='first 30'):
            preparer.prepare(tmp_path, tmp_path / 'review', 'test')
        assert not (tmp_path / 'review').exists()
    else:
        manifest = preparer.prepare(tmp_path, tmp_path / 'review', 'test')
        assert manifest['scope'] == 'TEST acceptance cohort'
        assert manifest['query_ids'] == split['test_ids'][:30]
