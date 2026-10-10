"""Compare historical byte packing with proxy-token packing on cached DEV rankings only."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))


def main():
    """Measure the packing boundary without inference, query rewriting or holdout access."""
    from devsupport.answer import ProxyTokenCounter, Utf8BudgetCounter, pack_context
    from devsupport.data import load_corpus, load_questions, sha256_file
    from devsupport.evaluate import write_new
    from devsupport.metrics import evaluate_query
    from devsupport.types import Hit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankings', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / 'devsupport'
    split = json.loads((base / 'reports/split.json').read_text(encoding='utf-8'))
    questions = {q.query_id: q for q in load_questions(base / 'data/queries.parquet')}
    corpus = load_corpus(base / 'data/corpus.parquet', duplicate_policy='official_last')
    rows, summaries = [], {}
    for path in args.rankings:
        original = [json.loads(line) for line in path.read_bytes().splitlines()]
        if [row['query_id'] for row in original] != split['dev_ids']:
            raise ValueError('requires complete ordered DEV53 cached rankings')
        if path.stem in summaries:
            raise ValueError('ranking labels must be unique')
        summaries[path.stem] = {'ranking_sha256': sha256_file(path), 'policies': {}}
        for counter in (Utf8BudgetCounter(), ProxyTokenCounter()):
            selected = []
            for row in original:
                if len(row['ranked_ids']) != len(set(row['ranked_ids'])):
                    raise ValueError('duplicate ranked evidence')
                hits = [Hit(corpus[doc_id], 0.0, rank, 'cached_diagnostic')
                        for rank, doc_id in enumerate(row['ranked_ids'], 1)]
                packed, _, trace = pack_context(hits, counter=counter)
                ids = [hit.evidence.doc_id for hit in packed]
                selected.append({'variant': path.stem, 'query_id': row['query_id'], 'evidence_ids': ids,
                                 'packing': trace, 'metrics': evaluate_query(ids, questions[row['query_id']])})
            summaries[path.stem]['policies'][counter.name] = {
                'n': len(selected), 'mean_chunks': statistics.fmean(len(r['evidence_ids']) for r in selected),
                'mean_budget_used': statistics.fmean(r['packing']['used'] for r in selected),
                'packed_nugget_coverage': statistics.fmean(r['metrics']['coverage@50'] for r in selected)}
            rows.extend(selected)
    write_new(args.output / 'rows.json', rows)
    write_new(args.output / 'summary.json', {
        'scope': 'DEV-only context-budget protocol diagnostic; not method gain or answer accuracy',
        'split_sha256': sha256_file(base / 'reports/split.json'),
        'corpus_sha256': sha256_file(base / 'data/corpus.parquet'),
        'script_sha256': sha256_file(Path(__file__)), 'methods': summaries})
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
