"""Export frozen embeddings or run genuine upstream U retrieval (no generation).

Run with the native app environment: .venv/Scripts/python.exe -X utf8.
Queries are JSON [{"query_id": "...", "query": "..."}]; no gold fields accepted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'devsupport' / 'src'))


def main() -> None:
    """Convert without model calls, or retrieve supplied query strings explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--raw', type=Path, default=ROOT / 'devsupport/indexes/raw')
    export.add_argument('--destination', type=Path, default=ROOT / 'indexes/devsupport-native-u')
    retrieve = commands.add_parser('retrieve')
    retrieve.add_argument('--source-id', default='devsupport-native-u')
    retrieve.add_argument('--index-dir', type=Path, default=ROOT / 'indexes/devsupport-native-u')
    retrieve.add_argument('--queries', type=Path, required=True)
    retrieve.add_argument('--output', type=Path, required=True)
    retrieve.add_argument('--doc-token-limit', type=int, default=50000)
    args = parser.parse_args()
    if args.command == 'retrieve':
        # This standalone benchmark process explicitly selects U's documented
        # shared embedding protocol. The native API/worker need the same env.
        os.environ['EMBEDDINGS_BASE_URL'] = 'http://127.0.0.1:8093'
        os.environ['EMBEDDINGS_NAME'] = 'Qwen/Qwen3-Embedding-0.6B'
        os.environ['EMBEDDINGS_MAX_QUERY_TOKENS'] = '0'
        os.environ['EMBEDDINGS_MAX_INPUT_TOKENS'] = '0'
        os.environ['VECTOR_STORE'] = 'faiss'
    from devsupport.native_baseline import NativeClassicRetriever, export_native_index

    if args.command == 'export':
        print(json.dumps(export_native_index(args.raw, args.destination), ensure_ascii=False))
        return
    queries = json.loads(args.queries.read_text(encoding='utf-8-sig'))
    if not isinstance(queries, list) or not queries:
        raise ValueError('nonempty query-only list required')
    for row in queries:
        if (not isinstance(row, dict) or set(row) != {'query_id', 'query'}
                or not isinstance(row['query_id'], str) or not isinstance(row['query'], str)):
            raise ValueError('each row must contain only query_id and query strings')
    if len({row['query_id'] for row in queries}) != len(queries):
        raise ValueError('duplicate query IDs')
    if args.output.exists():
        raise FileExistsError(args.output)
    adapter = NativeClassicRetriever(args.source_id, args.index_dir, args.doc_token_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for row in queries:
            hits = adapter.search(row['query'], k=50)
            record = {'query_id': row['query_id'], 'profile': 'U', 'trace': adapter.last_trace,
                      'hits': [{'doc_id': h.evidence.doc_id, 'score': h.score,
                                'score_kind': h.score_kind, 'rank': h.rank} for h in hits]}
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'query_id': row['query_id'], 'count': len(hits)}), flush=True)


if __name__ == '__main__':
    main()
