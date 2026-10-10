"""Synthetic complete evaluation artifacts; no real TEST annotations or inference."""
import importlib.util
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from devsupport.answer import pack_context, ProxyTokenCounter
from devsupport.data import load_corpus, sha256_file
from devsupport.evaluate import digest, freeze, make_record, text_hash, write_new
from devsupport.types import Hit, Question

BASE = Path(__file__).resolve().parents[1]


def load():
    path = BASE / 'analysis/evaluation_context.py'
    assert path.is_file(), 'complete-cohort context auditor is not implemented'
    spec = importlib.util.spec_from_file_location('evaluation_context', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def save_rows(path, rows):
    path.write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')


def fixture(tmp_path, split='dev', phase='retrieval', missing_gold=False):
    run = tmp_path / 'run'
    run.mkdir()
    dev, test = [f'd{i}' for i in range(53)], [f't{i}' for i in range(150)]
    selected = dev[:2] if split == 'dev' else (test[:30] if phase == 'pipeline' else test)
    questions = [Question(qid, 'Title', 'Body ' + qid, '',
                          [] if missing_gold and qid == selected[0] else
                          [{'text': 'Fact', 'relevant_corpus_ids': ['positive']}], {}) for qid in dev + test]
    query_path, corpus_path, split_path = (tmp_path / name for name in ('queries.parquet', 'corpus.json', 'split.json'))
    pq.write_table(pa.Table.from_pylist([{'query_id': q.query_id, 'query_title': q.query_title,
                  'query_text': q.query_text, 'nuggets': q.nuggets} for q in questions]), query_path)
    write_new(corpus_path, [{'doc_id': 'positive', 'text': 'Exact source fact'},
                           {'doc_id': 'negative', 'text': 'Different content'}])
    write_new(split_path, {'dev_ids': dev, 'test_ids': test})
    profiles = {name: {'retrieval': name, 'reranker': None, 'adaptive': 'never', 'gate_enabled': False}
                for name in ('B0', 'H0')}
    paths = {'queries': query_path, 'corpus': corpus_path, 'split': split_path}
    contract = {'paths': {k: str(p) for k, p in paths.items()},
                'artifacts': {k: sha256_file(p) for k, p in paths.items()}, 'profiles': profiles,
                'split_ids': {'dev': dev[:2], 'test': test}, 'generation_ids': test[:30], 'seeds': [7],
                'source': {'devsupport/src/devsupport/' + name: sha256_file(BASE / 'src/devsupport' / name)
                           for name in ('answer.py', 'data.py', 'metrics.py', 'types.py')}}
    contract['artifacts']['index'] = 'fixture-index-hash'
    contract['context_counter'] = 'cl100k_base_proxy_v1; 4000 proxy tokens, not exact gateway generation tokens'
    meta = {'contract_sha256': digest(contract), 'split': split, 'phase': phase,
            'profiles': list(profiles), 'query_ids': selected, 'controls': False}
    write_new(run / 'run.json', meta)
    corpus = load_corpus(corpus_path)
    qs = {q.query_id: q for q in questions}
    for name in profiles:
        rows = []
        for i, qid in enumerate(selected):
            ids = ['positive'] if name == 'H0' else ['negative']
            result = {'ranked_ids': ids, 'scores': [1.0]}
            if phase == 'pipeline':
                hits = [Hit(corpus[d], 1.0, j + 1, 'fixture') for j, d in enumerate(ids)]
                packed, _, packing = pack_context(hits, counter=ProxyTokenCounter())
                result = {'evidence_ids': ids,
                          'context': [{**h.evidence.to_corpus_dict(), 'score': h.score, 'citation_number': j + 1}
                                      for j, h in enumerate(packed)],
                          'decision': 'insufficient' if i == 0 else 'answer',
                          'trace': {'final_candidate_ids': ids, 'packing': packing,
                                    'generation': {'error': 'provider_failure'} if i == 0 else {}}}
            rows.append(make_record(qs[qid], result, 1.0))
        save_rows(run / (name + '.jsonl'), rows)
        write_new(run / (name + '.manifest.json'), {**meta, 'evaluation_contract': contract,
                  'profile': name, 'settings': profiles[name], 'variant': name,
                  'query_hashes': [text_hash(qs[qid].query) for qid in selected],
                  'questions_sha256': contract['artifacts']['queries'],
                  'index_manifest_sha256': contract['artifacts']['index'], 'latency_scope': phase,
                  'context_counter': contract['context_counter']})
    lock = tmp_path / 'lock.json'
    freeze(lock, contract)
    return run, lock


@pytest.mark.parametrize('separator', ['\u2028', '\u2029', '\x85'])
def test_audit_preserves_unicode_inside_a_physical_jsonl_record(tmp_path, separator):
    run, _ = fixture(tmp_path)
    path = run / 'B0.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[0]['result']['diagnostic_text'] = 'document before' + separator + 'after'
    rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
    original = path.read_bytes()
    assert separator.encode() in original
    result = load().audit([run])
    assert result['complete'] and result['methods']['B0']['n'] == 2
    assert path.read_bytes() == original


@pytest.fixture
def recorded(tmp_path):
    return fixture(tmp_path)


def test_real_record_replay_and_all_pair_quality(recorded):
    report = load().audit([recorded[0]])
    assert report['complete'] and report['questions'] == 2
    assert report['methods']['H0']['metrics']['context_nugget_coverage']['mean'] == 1
    assert report['methods']['B0']['metrics']['context_nugget_coverage']['mean'] == 0
    assert report['paired']['H0-minus-B0']['context_nugget_coverage']['mean_delta'] == 1
    assert {r['packing_scope'] for r in report['rows']} == {'replayed ranking-only context'}


@pytest.mark.parametrize('field', ['metrics_prepack', 'metrics_packed'])
def test_cross_python_metric_roundoff_preserves_original_checksum_and_discloses_delta(tmp_path, field):
    run, _ = fixture(tmp_path, phase='pipeline')
    path = run / 'H0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    original = rows[0][field]['ndcg@10']
    recorded_value = math.nextafter(original, math.inf)
    rows[0][field]['ndcg@10'] = recorded_value
    rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    save_rows(path, rows)
    before = path.read_bytes()
    report = load().audit([run])
    assert report['metric_recompute_tolerance'] == {'rel_tol': 1e-12, 'abs_tol': 1e-12}
    item = next(r for r in report['rows'] if r['method'] == 'H0' and r['query_id'] == rows[0]['query_id'])
    assert item['record_metric_roundoff'] == {field: {'ndcg@10': {
        'recorded': recorded_value, 'recomputed': original}}}
    assert path.read_bytes() == before


@pytest.mark.parametrize('mutation', ['large_delta', 'boolean', 'metric_key', 'query_hash', 'unsigned_roundoff'])
def test_metric_tolerance_cannot_bypass_integrity_or_other_record_fields(recorded, mutation):
    run, _ = recorded
    path = run / 'H0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == 'large_delta':
        rows[0]['metrics_prepack']['ndcg@10'] += 4e-12
    elif mutation == 'boolean':
        rows[0]['metrics_prepack']['ndcg@10'] = True
    elif mutation == 'metric_key':
        rows[0]['metrics_prepack']['unknown_metric'] = 1.0
    elif mutation == 'query_hash':
        rows[0]['query_sha256'] = 'wrong'
    else:
        rows[0]['metrics_prepack']['ndcg@10'] = math.nextafter(1.0, math.inf)
    if mutation != 'unsigned_roundoff':
        rows[0]['record_sha256'] = digest({k: v for k, v in rows[0].items() if k != 'record_sha256'})
    save_rows(path, rows)
    with pytest.raises(ValueError, match='checksum'):
        load().audit([run])


def test_generation_failure_retains_observed_context_quality(tmp_path):
    run, _ = fixture(tmp_path, phase='pipeline')
    report = load().audit([run])
    assert report['methods']['H0']['operational_errors'] == 1
    assert report['methods']['H0']['metrics']['context_nugget_coverage']['mean'] == 1
    assert {r['packing_scope'] for r in report['rows']} == {'validated observed pipeline context'}


def test_missing_gold_not_silently_removed_from_paired_cohort(tmp_path):
    run, _ = fixture(tmp_path, missing_gold=True)
    report = load().audit([run])
    assert report['methods']['H0']['metrics']['context_nugget_coverage']['n'] == 1
    assert report['paired']['H0-minus-B0']['context_nugget_coverage']['unavailable']


@pytest.mark.parametrize('tamper', ['partial', 'reorder', 'checksum', 'input', 'source', 'profile', 'inflight', 'controls'])
def test_incomplete_or_changed_artifact_rejected(recorded, tamper):
    run, _ = recorded
    path = run / 'H0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if tamper in ('partial', 'reorder', 'checksum'):
        if tamper == 'checksum':
            rows[0]['record_sha256'] = 'wrong'
        save_rows(path, rows[:1] if tamper == 'partial' else rows[::-1] if tamper == 'reorder' else rows)
    elif tamper == 'inflight':
        write_new(run / 'H0.inflight.json', {})
    elif tamper == 'input':
        (run.parent / 'corpus.json').write_text('[]')
    elif tamper == 'controls':
        meta = json.loads((run / 'run.json').read_text())
        meta['controls'] = True
        save(run / 'run.json', meta)
    else:
        meta = json.loads((run / 'H0.manifest.json').read_text())
        if tamper == 'source':
            meta['evaluation_contract']['source']['devsupport/src/devsupport/answer.py'] = 'wrong'
        else:
            meta['settings']['retrieval'] = 'B1'
        save(run / 'H0.manifest.json', meta)
    with pytest.raises(ValueError):
        load().audit([run])


@pytest.mark.parametrize('phase,expected', [('retrieval', 150), ('pipeline', 30)])
def test_synthetic_test_cohort_requires_exact_lock(tmp_path, phase, expected):
    run, lock = fixture(tmp_path, split='test', phase=phase)
    tool = load()
    with pytest.raises(ValueError, match='lock'):
        tool.audit([run], split='test')
    report = tool.audit([run], split='test', lock=lock)
    assert report['questions'] == expected
    value = json.loads(lock.read_text())
    value['sha256'] = 'wrong'
    save(lock, value)
    with pytest.raises(ValueError):
        tool.audit([run], split='test', lock=lock)


def test_duplicate_input_profiles_and_existing_output_rejected(recorded, tmp_path):
    tool = load()
    with pytest.raises(ValueError):
        tool.audit([recorded[0], recorded[0]])
    output = tmp_path / 'out'
    output.mkdir()
    with pytest.raises(FileExistsError):
        tool.publish([recorded[0]], output)


def rewrite_result(run, change):
    """Resign a changed result using the real evaluator, isolating semantic validation."""
    path = run / 'H0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[0]
    change(row['result'])
    q = Question(row['query_id'], 'Title', 'Body ' + row['query_id'], '',
                 [{'text': 'Fact', 'relevant_corpus_ids': ['positive']}], {})
    rows[0] = make_record(q, row['result'], row['elapsed_seconds'], row['error'])
    save_rows(path, rows)


def test_actual_pipeline_rank_and_score_kind_fields_are_supported(tmp_path):
    run, _ = fixture(tmp_path, phase='pipeline')
    rewrite_result(run, lambda result: result['context'][0].update(rank=1, score_kind='dense'))
    assert load().audit([run])['methods']['H0']['metrics']['context_nugget_coverage']['mean'] == 1


@pytest.mark.parametrize('tamper', ['text', 'metadata', 'citation', 'packing', 'unknown'])
def test_resigned_semantic_context_drift_rejected(tmp_path, tamper):
    run, _ = fixture(tmp_path, phase='pipeline')

    def change(result):
        if tamper == 'text':
            result['context'][0]['text'] = 'Fabricated source'
        elif tamper == 'metadata':
            result['context'][0]['metadata']['url'] = 'https://fabricated.invalid'
        elif tamper == 'citation':
            result['context'][0]['citation_number'] = 2
        elif tamper == 'packing':
            result['trace']['packing']['budget'] = 9999
        else:
            result['trace']['final_candidate_ids'].append('unknown')

    rewrite_result(run, change)
    with pytest.raises(ValueError):
        load().audit([run])


def test_json_parse_snapshot_mutation_rejected(recorded, monkeypatch):
    run, _ = recorded
    tool = load()
    original = tool.json.loads
    changed = False

    def mutate(raw, *args, **kwargs):
        nonlocal changed
        value = original(raw, *args, **kwargs)
        if isinstance(value, dict) and 'controls' in value and not changed:
            changed = True
            save(run / 'run.json', {**value, 'split': 'test'})
        return value

    monkeypatch.setattr(tool.json, 'loads', mutate)
    with pytest.raises(ValueError, match='changed'):
        tool.audit([run])


def test_publication_rechecks_changed_input(recorded, tmp_path, monkeypatch):
    tool = load()
    original = tool.audit

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (recorded[0] / 'H0.jsonl').write_text('')
        return result

    monkeypatch.setattr(tool, 'audit', mutate)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='changed'):
        tool.publish([recorded[0]], output)
    assert not output.exists()


def test_short_test_cohort_rejected_even_with_consistent_lock(tmp_path):
    run, lock = fixture(tmp_path, split='test')
    split_path = tmp_path / 'split.json'
    split = json.loads(split_path.read_text())
    split['test_ids'] = split['test_ids'][:-1]
    save(split_path, split)
    meta = json.loads((run / 'run.json').read_text())
    meta['query_ids'] = split['test_ids']
    for name in meta['profiles']:
        path = run / (name + '.manifest.json')
        manifest = json.loads(path.read_text())
        contract = manifest['evaluation_contract']
        contract['split_ids']['test'] = split['test_ids']
        contract['artifacts']['split'] = sha256_file(split_path)
        meta['contract_sha256'] = digest(contract)
        manifest.update(meta)
        save(path, manifest)
        rows = [json.loads(line) for line in (run / (name + '.jsonl')).read_text().splitlines()]
        save_rows(run / (name + '.jsonl'), rows[:-1])
    save(run / 'run.json', meta)
    save(lock, {'contract': contract, 'sha256': digest(contract)})
    with pytest.raises(ValueError, match='150'):
        load().audit([run], split='test', lock=lock)


def test_source_manifest_parse_mutation_rejected(recorded, monkeypatch):
    run, _ = recorded
    tool = load()
    original = tool.json.loads
    changed = False

    def mutate(raw, *args, **kwargs):
        nonlocal changed
        value = original(raw, *args, **kwargs)
        if isinstance(value, dict) and value.get('profile') == 'B0' and not changed:
            changed = True
            save(run / 'B0.manifest.json', {**value, 'settings': {'retrieval': 'H0'}})
        return value

    monkeypatch.setattr(tool.json, 'loads', mutate)
    with pytest.raises(ValueError, match='changed'):
        tool.audit([run])


def test_bound_packing_protocol_must_match_replayed_policy(recorded):
    run, _ = recorded
    meta = json.loads((run / 'run.json').read_text())
    for name in meta['profiles']:
        path = run / (name + '.manifest.json')
        manifest = json.loads(path.read_text())
        manifest['evaluation_contract']['context_counter'] = 'utf8_bytes; budget9000'
        manifest['context_counter'] = 'utf8_bytes; budget9000'
        meta['contract_sha256'] = digest(manifest['evaluation_contract'])
        manifest.update(meta)
        save(path, manifest)
    save(run / 'run.json', meta)
    with pytest.raises(ValueError, match='packing'):
        load().audit([run])


def test_test30_pipeline_cannot_mask_short_retrieval_run(tmp_path):
    run, lock = fixture(tmp_path, split='test', phase='pipeline')
    first_meta = json.loads((run / 'run.json').read_text())
    first_meta['profiles'] = ['H0']
    save(run / 'run.json', first_meta)
    first_manifest = json.loads((run / 'H0.manifest.json').read_text())
    first_manifest.update(first_meta)
    save(run / 'H0.manifest.json', first_manifest)
    other = tmp_path / 'retrieval'
    other.mkdir()
    second_meta = {**first_meta, 'phase': 'retrieval', 'profiles': ['B0']}
    save(other / 'run.json', second_meta)
    second_manifest = json.loads((run / 'B0.manifest.json').read_text())
    second_manifest.update(second_meta)
    second_manifest['latency_scope'] = 'retrieval'
    save(other / 'B0.manifest.json', second_manifest)
    records = []
    for qid in second_meta['query_ids']:
        q = Question(qid, 'Title', 'Body ' + qid, '',
                     [{'text': 'Fact', 'relevant_corpus_ids': ['positive']}], {})
        records.append(make_record(q, {'ranked_ids': ['negative'], 'scores': [1.0]}, 1.0))
    save_rows(other / 'B0.jsonl', records)
    with pytest.raises(ValueError, match='cohort'):
        load().audit([run, other], split='test', lock=lock)


@pytest.mark.parametrize('field', ['query_hashes', 'questions_sha256', 'index_manifest_sha256', 'latency_scope'])
@pytest.mark.parametrize('missing', [False, True])
def test_manifest_query_index_and_phase_bindings_are_required(recorded, field, missing):
    run, _ = recorded
    path = run / 'H0.manifest.json'
    manifest = json.loads(path.read_text())
    if missing:
        manifest.pop(field)
    else:
        manifest[field] = ['wrong'] if field == 'query_hashes' else 'wrong'
    save(path, manifest)
    with pytest.raises(ValueError, match='manifest'):
        load().audit([run])
