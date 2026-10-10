"""Summarize the existing five-query DEV order experiment without new inference."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))

from devsupport.data import load_questions  # noqa: E402
from devsupport.metrics import evaluate_query  # noqa: E402


def main() -> None:
    """Validate all permutations and recompute gold metrics before summarizing."""
    base = ROOT / 'devsupport'
    source = base / 'reports/rerank-order-v1/dev/R2.jsonl'
    meta_path = source.with_suffix('.manifest.json')
    pool_path = base / 'reports/retrieval-v1/dev/H1.jsonl'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    files = [source, meta_path, pool_path, pool_path.with_suffix('.manifest.json'),
             base / 'data/queries.parquet', base / 'reports/split.json']
    hashes = {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if hashes[pool_path.relative_to(base).as_posix()] != meta['candidate_file_sha256']:
        raise ValueError('candidate artifact changed')
    if hashes[pool_path.with_suffix('.manifest.json').relative_to(base).as_posix()] != meta['candidate_manifest_sha256']:
        raise ValueError('candidate manifest changed')
    pool_meta = json.loads(pool_path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if hashes['data/queries.parquet'] != pool_meta['questions_sha256']:
        raise ValueError('question annotations changed')
    split = json.loads((base / 'reports/split.json').read_text(encoding='utf-8'))
    if meta['split'] != 'dev' or meta['query_ids'] != split['dev_ids'][:5] or meta['seeds'] != [0, 1, 2]:
        raise ValueError('expected the original five DEV queries and three permutations')
    questions = {q.query_id: q for q in load_questions(base / 'data/queries.parquet')
                 if q.query_id in meta['query_ids']}
    pools = {r['query_id']: r for r in map(json.loads, pool_path.read_bytes().splitlines())}
    rows = list(map(json.loads, source.read_bytes().splitlines()))
    expected = {(qid, seed) for qid in meta['query_ids'] for seed in meta['seeds']}
    if len(rows) != len(expected) or {(r['query_id'], r['seed']) for r in rows} != expected:
        raise ValueError('incomplete or duplicated query-permutation cohort')
    for row in rows:
        original = pools[row['query_id']]['ranked_ids']
        shuffled = original.copy()
        random.Random(row['seed']).shuffle(shuffled)
        trace = row['trace']
        if trace['fallback'] or trace.get('error') or trace['input_ids'] != shuffled:
            raise ValueError('failed inference or changed input permutation')
        if len(row['ranked_ids']) != len(original) or set(row['ranked_ids']) != set(original):
            raise ValueError('reranking changed candidate membership')
        if row['metrics'] != evaluate_query(row['ranked_ids'], questions[row['query_id']]):
            raise ValueError('stored metrics disagree with original annotations')
    metrics = ('coverage@10', 'alpha_ndcg@10', 'recall@10')
    per_query = []
    for qid in meta['query_ids']:
        group = [r for r in rows if r['query_id'] == qid]
        overlaps = [len(set(a['ranked_ids'][:10]) & set(b['ranked_ids'][:10])) / 10
                    for a, b in itertools.combinations(group, 2)]
        per_query.append({'query_id': qid, 'pairwise_top10_overlap_mean': statistics.fmean(overlaps),
                          'metrics': {k: {'min': min(r['metrics'][k] for r in group),
                                          'max': max(r['metrics'][k] for r in group),
                                          'range': max(r['metrics'][k] for r in group)
                                          - min(r['metrics'][k] for r in group)} for k in metrics}})
    report = {'independent_queries': 5, 'permutations_per_query': 3, 'inference_records': 15,
              'artifact_sha256': hashes,
              'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'per_seed_means': {str(seed): {k: statistics.fmean(r['metrics'][k] for r in rows
                                                                if r['seed'] == seed) for k in metrics}
                                 for seed in meta['seeds']}, 'per_query': per_query,
              'limitations': ['Five first DEV queries, not 15 independent questions or the full DEV cohort.',
                              'Three seeded permutations only; no best-order selection or tuning.',
                              'No repeat of each fixed order, so nondeterminism is not separately estimated.',
                              'Ranking stability is not evidence of answer correctness.']}
    output = source.parent / 'order-audit.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(output), 'per_seed_means': report['per_seed_means'],
                      'per_query': per_query}))


if __name__ == '__main__':
    main()
