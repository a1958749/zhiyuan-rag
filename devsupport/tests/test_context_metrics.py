"""Tiny offline metric and fail-closed cached-run contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from devsupport.metrics import evaluate_query
from devsupport.types import Question


@pytest.fixture
def audit():
    path = Path(__file__).parents[1] / 'analysis/context_metrics.py'
    assert path.exists(), 'context quality audit must exist'
    spec = importlib.util.spec_from_file_location('context_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question(nuggets=None):
    return Question('q', 'title', 'body', '', nuggets if nuggets is not None else [
        {'relevant': {'a': 1, 'b': 2, 'n': 0}}, {'relevant': ['c']}, {'relevant': []}], {})


def test_actual_count_precision_ap_recall_and_freshstack_coverage(audit):
    result = audit.context_quality(['n', 'a', 'b'], question(), {'a', 'b', 'c', 'n'})
    assert result['packed_context_doc_precision'] == pytest.approx(2 / 3)
    assert result['context_average_precision_qrels'] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert result['context_doc_recall'] == pytest.approx(2 / 3)
    assert result['context_nugget_coverage'] == pytest.approx(1 / 3)
    assert result['unjudged_nugget_count'] == 1
    assert result['positive_packed_doc_count'] == 2


@pytest.mark.parametrize('nuggets', [[], [{}], [{'relevant': []}]])
def test_missing_annotations_are_null_not_zero(audit, nuggets):
    result = audit.context_quality(['n'], question(nuggets), {'n'})
    assert all(result[key] is None for key in audit.QUALITY_KEYS)


def test_empty_or_no_positive_packed_with_known_gold_is_zero(audit):
    for ids in ([], ['n']):
        result = audit.context_quality(ids, question(), {'a', 'b', 'c', 'n'})
        assert all(result[key] == 0 for key in audit.QUALITY_KEYS)


@pytest.mark.parametrize('ids', [['a', 'a'], ['missing']])
def test_duplicate_unknown_packed_ids_rejected(audit, ids):
    with pytest.raises(ValueError, match='duplicate|unknown'):
        audit.context_quality(ids, question(), {'a', 'b', 'c', 'n'})


def test_unknown_qrel_rejected(audit):
    with pytest.raises(ValueError, match='unknown'):
        audit.context_quality([], question(), {'a'})


def cached_row():
    q = question()
    return {'query_id': q.query_id, 'query_sha256': hashlib.sha256(q.query.encode()).hexdigest(),
            'ranked_ids': ['a', 'n'], 'scores': [1, 0], 'error': None,
            'metrics': evaluate_query(['a', 'n'], q)}


def test_valid_retrieval_rows_and_metric_drift(audit):
    row = cached_row()
    audit.validate_rows('B0', [row], ['q'], {'q': question()}, {'a', 'b', 'c', 'n'}, {})
    row['metrics']['precision@10'] = 0.9
    with pytest.raises(ValueError, match='metric'):
        audit.validate_rows('B0', [row], ['q'], {'q': question()}, {'a', 'b', 'c', 'n'}, {})


def test_recomputed_metric_roundoff_tolerance_and_missing_keys(audit):
    row = cached_row()
    row['metrics']['ndcg@50'] += 1e-15
    audit.validate_rows('B0', [row], ['q'], {'q': question()}, {'a', 'b', 'c', 'n'}, {})
    del row['metrics']['ndcg@50']
    with pytest.raises(ValueError, match='metric'):
        audit.validate_rows('B0', [row], ['q'], {'q': question()}, {'a', 'b', 'c', 'n'}, {})


@pytest.mark.parametrize('mutation', ['missing_error', 'error', 'duplicate', 'unknown', 'query', 'order'])
def test_retrieval_scope_and_positive_status(audit, mutation):
    row = cached_row()
    rows = [row]
    if mutation == 'missing_error':
        del row['error']
    elif mutation == 'error':
        row['error'] = 'TimeoutError'
    elif mutation == 'duplicate':
        row['ranked_ids'] = ['a', 'a']
    elif mutation == 'unknown':
        row['ranked_ids'] = ['unknown']
    elif mutation == 'query':
        row['query_sha256'] = 'changed'
    else:
        rows *= 2
    with pytest.raises(ValueError):
        audit.validate_rows('B0', rows, ['q'], {'q': question()}, {'a', 'b', 'c', 'n'}, {})


def test_rerank_positive_status_and_candidate_membership(audit):
    row = cached_row()
    row['seed'] = None
    config = {'model': 'model', 'revision': 'revision'}
    row['trace'] = {'variant': 'R1', 'fallback': False, 'error': None,
                    'actual_model': 'model', 'requested_model': 'model', 'revision': 'revision',
                    'forward_count': 1, 'outputs': [{'doc_id': i, 'score': s, 'rank': r}
                        for r, (i, s) in enumerate(zip(row['ranked_ids'], row['scores']), 1)]}
    args = ('R1', [row], ['q'], {'q': question()}, {'a', 'b', 'c', 'n'}, config)
    audit.validate_rows(*args, candidates={'q': ['n', 'a']})
    with pytest.raises(ValueError, match='candidate'):
        audit.validate_rows(*args, candidates={'q': ['a', 'b']})
    row['trace']['fallback'] = True
    with pytest.raises(ValueError, match='successful'):
        audit.validate_rows(*args, candidates={'q': ['n', 'a']})


def test_manifest_requires_order_profile_and_hashes(audit):
    manifest = {'variant': 'R1', 'split': 'dev', 'query_ids': ['q'], 'seeds': [None],
                'config_sha256': 'config', 'candidate_file_sha256': 'file',
                'candidate_manifest_sha256': 'manifest'}
    expected = {k: v for k, v in manifest.items() if k.endswith('sha256')}
    audit.validate_manifest(manifest, 'R1', ['q'], expected)
    for key, changed in [('split', 'test'), ('variant', 'R2'), ('query_ids', []),
                         ('seeds', [0]), ('candidate_file_sha256', 'changed')]:
        with pytest.raises(ValueError):
            audit.validate_manifest({**manifest, key: changed}, 'R1', ['q'], expected)


def test_macro_uses_only_annotated_denominator(audit):
    rows = [{'quality': audit.context_quality([], question(), {'a', 'b', 'c', 'n'})},
            {'quality': audit.context_quality([], question([]), set())}]
    summary = audit.summarize(rows)
    assert summary['n'] == 2
    assert summary['metrics']['context_doc_recall'] == {'mean': 0, 'n': 1, 'missing': 1}


def test_dev_projection_excludes_holdout_and_detects_duplicates(audit, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / 'queries.parquet'
    rows = [{'query_id': i, 'query_title': 'title', 'query_text': 'body',
             'nuggets': [{'relevant_corpus_ids': ['a' if i == 'q' else 'holdout-only']}]} for i in ('q', 'test')]
    pq.write_table(pa.Table.from_pylist(rows), path)
    selected = audit.load_dev_questions(path, ['q'])
    assert list(selected) == ['q']
    assert selected['q'].nuggets[0]['relevant_corpus_ids'] == ['a']
    pq.write_table(pa.Table.from_pylist(rows + [rows[0]]), path)
    with pytest.raises(ValueError, match='duplicate'):
        audit.load_dev_questions(path, ['q'])


def test_source_binding_accepts_only_checkout_newline_difference(audit, tmp_path):
    archived = tmp_path / 'archived.py'
    executing = tmp_path / 'executing.py'
    archived.write_bytes(b'x = 1\n')
    executing.write_bytes(b'x = 1\r\n')
    digest = audit.sha256_file(archived)
    audit.validate_semantics(digest, archived, executing)
    executing.write_bytes(b'x = 2\r\n')
    with pytest.raises(ValueError, match='implementation drift'):
        audit.validate_semantics(digest, archived, executing)
    executing.write_bytes(b'x = 1\r\n')
    with pytest.raises(ValueError, match='implementation drift'):
        audit.validate_semantics('changed', archived, executing)


def test_full_seven_profile_audit_and_index_binding(audit, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    base = tmp_path / 'devsupport'

    def save(name, value):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    dev_ids = [f'q{i:02}' for i in range(53)]
    split = save('reports/split.json', {'dev_ids': dev_ids, 'test_ids': ['test']})
    (base / 'data').mkdir()
    docs = [f'd{i}' for i in range(50)]
    corpus = base / 'data/corpus.parquet'
    pq.write_table(pa.Table.from_pylist([{'doc_id': i, 'text': 'small evidence'} for i in docs]), corpus)
    queries = base / 'data/queries.parquet'
    pq.write_table(pa.Table.from_pylist([{'query_id': i, 'query_title': 'title', 'query_text': 'body',
        'nuggets': [{'relevant_corpus_ids': ['d0']}]} for i in dev_ids]), queries)
    historical = {name: audit.sha256_file(audit.ROOT / f'devsupport/src/devsupport/{name}')
                  for name in ('data.py', 'metrics.py', 'types.py')}
    (base / 'src/devsupport').mkdir(parents=True)
    for name in historical:
        (base / f'src/devsupport/{name}').write_bytes((audit.ROOT / f'devsupport/src/devsupport/{name}').read_bytes())
    for variant, run in audit.RUNS.items():
        config = {'variant': variant, 'model': 'model', 'revision': 'revision'}
        cp = save(f'configs/{variant}.json', config)
        manifest = {'variant': variant, 'split': 'dev', 'query_ids': dev_ids,
                    'config_sha256': audit.sha256_file(cp)}
        if variant in ('R1', 'R2'):
            h1 = base / 'reports/retrieval-v1/dev/H1.jsonl'
            manifest.update(seeds=[None], candidate_file_sha256=audit.sha256_file(h1),
                            candidate_manifest_sha256=audit.sha256_file(h1.with_suffix('.manifest.json')),
                            implementation_sha256=historical)
        else:
            index = save(f'indexes/{"contextual" if variant == "C" else "raw"}/manifest.json',
                         {'data_sha256': audit.sha256_file(corpus), 'contextual': variant == 'C',
                          'provenance': {'model': 'embedding'}})
            manifest.update(split_sha256=audit.sha256_file(split), questions_sha256=audit.sha256_file(queries),
                            index_manifest_sha256=audit.sha256_file(index), code_sha256=historical,
                            query_provenance={'model': 'embedding'})
            manifest['run_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        save(f'reports/{run}/dev/{variant}.manifest.json', manifest)
        rows = []
        for qid in dev_ids:
            q = Question(qid, 'title', 'body', '', [{'relevant_corpus_ids': ['d0']}], {})
            row = {'query_id': qid, 'ranked_ids': docs, 'scores': [1] * 50,
                   'metrics': evaluate_query(docs, q), 'error': None,
                   'query_sha256': hashlib.sha256(q.query.encode()).hexdigest()}
            if variant in ('R1', 'R2'):
                row.update(seed=None, trace={'variant': variant, 'actual_model': 'model',
                    'requested_model': 'model', 'revision': 'revision', 'fallback': False, 'error': None,
                    'forward_count': 1, 'outputs': [{'doc_id': i, 'score': 1, 'rank': r}
                                                  for r, i in enumerate(docs, 1)]})
            rows.append(row)
        (base / f'reports/{run}/dev/{variant}.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    rows, summary = audit.build_audit(tmp_path)
    assert len(rows) == 371
    assert summary['methods']['C']['metrics']['context_doc_recall'] == {'mean': 1, 'n': 53, 'missing': 0}
    assert summary['paired_C_minus_baseline']['B0']['context_doc_recall']['n'] == 53
    assert 'reports/rerank-v1/dev/R2.manifest.json' in summary['input_sha256']
    save('indexes/raw/manifest.json', {'data_sha256': 'changed'})
    with pytest.raises(ValueError, match='index corpus'):
        audit.build_audit(tmp_path)
