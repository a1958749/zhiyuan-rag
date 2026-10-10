"""Compare real local rankers on the identical frozen H1 development pools."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport' / 'src'))


def implementation_hashes(base: Path) -> dict[str, str]:
    """Bind ranking, scoring, annotation parsing and this orchestration entrypoint."""
    from devsupport.data import sha256_file

    hashes = {name: sha256_file(base / 'src/devsupport' / name)
              for name in ('reranking.py', 'metrics.py', 'data.py', 'types.py')}
    hashes['runner'] = sha256_file(Path(__file__))
    return hashes


def main():
    """Save every true inference trace; fallbacks never count as successful reranking."""
    from devsupport.data import load_questions, sha256_file
    from devsupport.index import CorpusIndex
    from devsupport.metrics import evaluate_query
    from devsupport.reranking import JinaListwiseReranker, QwenPairwiseReranker
    from devsupport.types import Hit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=['R1', 'R2'], default=['R1', 'R2'])
    parser.add_argument('--pools', type=Path, default=ROOT / 'devsupport/reports/retrieval-v1/dev/H1.jsonl')
    parser.add_argument('--order-sensitivity', action='store_true')
    args = parser.parse_args()
    base = ROOT / 'devsupport'
    split = json.loads((base / 'reports/split.json').read_text(encoding='utf-8'))
    dev_ids = split['dev_ids']
    questions = {q.query_id: q for q in load_questions(base / 'data/queries.parquet')}
    pool_rows = [json.loads(line) for line in args.pools.read_bytes().splitlines()]
    pools = {row['query_id']: row for row in pool_rows}
    if len(pools) != len(pool_rows) or set(pools) != set(dev_ids):
        raise ValueError('candidate file must contain all and only frozen development IDs')
    pool_manifest = json.loads(args.pools.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if pool_manifest.get('variant') != 'H1' or pool_manifest.get('split') != 'dev':
        raise ValueError('expected H1 development candidate provenance')
    index = CorpusIndex.load(base / 'indexes/raw')
    if pool_manifest['index_manifest_sha256'] != sha256_file(base / 'indexes/raw/manifest.json'):
        raise ValueError('candidate/index provenance mismatch')
    if pool_manifest['questions_sha256'] != sha256_file(base / 'data/queries.parquet'):
        raise ValueError('candidate/query provenance mismatch')
    for qid, pool in pools.items():
        if pool['query_sha256'] != hashlib.sha256(questions[qid].query.encode('utf-8')).hexdigest():
            raise ValueError('query text changed')
        ids = pool['ranked_ids']
        if pool['error'] or len(ids) != 50 or len(set(ids)) != 50 or not set(ids) <= index.corpus.keys():
            raise ValueError('invalid candidate pool')
    output = base / 'reports' / ('rerank-order-v1' if args.order_sensitivity else 'rerank-v1') / 'dev'
    output.mkdir(parents=True, exist_ok=True)
    selected = dev_ids[:5] if args.order_sensitivity else dev_ids
    seeds = [0, 1, 2] if args.order_sensitivity else [None]
    for variant in args.variants:
        config_path = base / 'configs' / (variant + '.json')
        contract = {
            'variant': variant, 'split': 'dev', 'query_ids': selected, 'seeds': seeds,
            'candidate_file_sha256': sha256_file(args.pools),
            'candidate_manifest_sha256': sha256_file(args.pools.with_suffix('.manifest.json')),
            'config_sha256': sha256_file(config_path),
            'implementation_sha256': implementation_hashes(base),
        }
        meta_path = output / (variant + '.manifest.json')
        result_path = output / (variant + '.jsonl')
        if meta_path.exists():
            if json.loads(meta_path.read_text(encoding='utf-8')) != contract:
                raise ValueError('reranking contract changed; preserve old experiment and use a new run')
        elif result_path.exists():
            raise ValueError('missing reranking contract')
        else:
            meta_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')
        rows = [json.loads(line) for line in result_path.read_bytes().splitlines()] \
            if result_path.exists() else []
        completed = {(row['query_id'], row['seed']) for row in rows}
        allowed = {(qid, seed) for qid in selected for seed in seeds}
        if len(completed) != len(rows) or not completed <= allowed:
            raise ValueError('invalid reranking checkpoint')
        ranker = QwenPairwiseReranker() if variant == 'R1' else JinaListwiseReranker()
        for qid in selected:
            pool = pools[qid]
            original_ids = pool['ranked_ids']
            for seed in seeds:
                if (qid, seed) in completed:
                    continue
                positions = list(range(len(original_ids)))
                if seed is not None:
                    random.Random(seed).shuffle(positions)
                hits = [Hit(index.corpus[original_ids[i]], pool['scores'][i], rank, pool['score_kinds'][i])
                        for rank, i in enumerate(positions, 1)]
                ranked = ranker.rank(questions[qid].query, hits)
                ids = [hit.evidence.doc_id for hit in ranked]
                if len(ids) != 50 or set(ids) != set(original_ids):
                    raise ValueError('reranker changed candidate membership')
                row = {'query_id': qid, 'seed': seed, 'ranked_ids': ids,
                       'scores': [hit.score for hit in ranked], 'trace': ranker.last_trace,
                       'metrics': evaluate_query(ids, questions[qid])}
                with result_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                print(json.dumps({'variant': variant, 'query_id': qid, 'seed': seed,
                                  'fallback': row['trace']['fallback'],
                                  'seconds': row['trace']['elapsed_seconds'],
                                  'alpha_ndcg@10': row['metrics']['alpha_ndcg@10']}), flush=True)
        successful = [row for row in rows if not row['trace']['fallback']]
        keys = rows[0]['metrics'].keys()
        summary = {
            'variant': variant, 'n': len(rows), 'successful': len(successful),
            'fallbacks': len(rows) - len(successful),
            'metrics_with_fallback': {k: statistics.fmean(r['metrics'][k] for r in rows) for k in keys},
            'metrics_success_only': {k: statistics.fmean(r['metrics'][k] for r in successful) for k in keys}
            if successful else None,
            'timing_note': 'real local GPU ranker; first inference includes load and artifact verification',
        }
        (output / (variant + '.summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
        ranker.unload()
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
