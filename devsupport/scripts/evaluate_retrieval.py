"""Run baseline retrieval against a fixed local split, using the production Retriever."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport' / 'src'))

def main():
    """Only development is accessible until a separately frozen holdout lock exists."""
    from devsupport.data import load_questions, sha256_file
    from devsupport.embedding import Embedder
    from devsupport.experiments import run_retrieval, summarize
    from devsupport.index import CorpusIndex
    from devsupport.retrieval import Retriever

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=['B0', 'B1', 'H0', 'H1', 'C'],
                        default=['B0', 'B1', 'H0', 'H1'])
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--run-name', default='retrieval-v1')
    args = parser.parse_args()
    if not args.run_name.replace('-', '').replace('_', '').isalnum():
        parser.error('run-name must contain letters, digits, dash or underscore')
    base = ROOT / 'devsupport'
    split_path = base / 'reports' / 'split.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    if args.split == 'test':
        parser.error('holdout evaluation is reserved for the frozen final experiment runner')
    questions = {q.query_id: q for q in load_questions(base / 'data' / 'queries.parquet')}
    selected = [questions[qid] for qid in split[args.split + '_ids']]
    embedder = Embedder(base / 'model-cache' / 'embedding')
    code_files = ['retrieval.py', 'index.py', 'lexical.py', 'embedding.py', 'config.py',
                  'metrics.py', 'data.py', 'types.py', 'experiments.py']
    code_hashes = {f: sha256_file(base / 'src' / 'devsupport' / f) for f in code_files}
    output = base / 'reports' / args.run_name / args.split
    summaries = {}
    index = None
    index_path = None
    for variant in args.variants:
        target = base / 'indexes' / ('contextual' if variant == 'C' else 'raw')
        if target != index_path:
            index = CorpusIndex.load(target, expected_provenance=embedder.provenance)
            index_path = target
        config = base / 'configs' / (variant + '.json')
        manifest = {
            'schema': 1, 'variant': variant, 'split': args.split,
            'query_ids': [q.query_id for q in selected], 'split_sha256': sha256_file(split_path),
            'questions_sha256': sha256_file(base / 'data' / 'queries.parquet'),
            'index_manifest_sha256': sha256_file(target / 'manifest.json'),
            'config_sha256': sha256_file(config), 'code_sha256': code_hashes,
            'runner_sha256': sha256_file(Path(__file__)),
            'query_provenance': embedder.provenance,
        }
        manifest['run_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        engine = Retriever(index, embedder, variant)
        rows = run_retrieval(selected, engine, output / (variant + '.jsonl'), manifest)
        summaries[variant] = summarize(rows)
        (output / (variant + '.summary.json')).write_text(
            json.dumps(summaries[variant], indent=2), encoding='utf-8')
        print(json.dumps({'variant': variant, **summaries[variant]}), flush=True)


if __name__ == '__main__':
    main()
