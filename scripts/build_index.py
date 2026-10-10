"""Build the full audited raw corpus index with real offline CUDA inference.

Run from any directory: python F:/DevSupportCopilot/scripts/build_index.py.
An interrupted invocation resumes matching safe embedding checkpoints automatically.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'devsupport/src'))

from devsupport.data import load_corpus_audited  # noqa: E402
from devsupport.embedding import Embedder  # noqa: E402
from devsupport.index import CorpusIndex  # noqa: E402


def parser() -> argparse.ArgumentParser:
    """Return the explicit pinned full-corpus CLI options."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--corpus', type=Path, default=ROOT / 'devsupport/data/corpus.parquet')
    result.add_argument('--output', type=Path, default=ROOT / 'devsupport/indexes/raw')
    result.add_argument('--model', type=Path, default=ROOT / 'devsupport/model-cache/embedding')
    result.add_argument('--batch-size', type=int, default=16)
    result.add_argument('--chunk-size', type=int, default=1024)
    result.add_argument('--expected-raw-count', type=int, default=49514,
                        help='expected official raw rows; explicit override only for artificial fixtures')
    result.add_argument('--expected-count', type=int, default=49505,
                        help='expected official unique IDs; explicit override only for artificial fixtures')
    result.add_argument('--contextual', action='store_true',
                        help='optional factual contextual index; choose a separate --output directory')
    result.add_argument('--rebuild-lexical', action='store_true',
                        help='offline maintenance: atomically publish new BM25 artifacts in verified --output; '
                             'interrupted attempts can safely retry without re-embedding')
    return result


def main(argv: list[str] | None = None) -> int:
    """Build dense/ordinary BM25/identifier BM25 and report real manifest statistics."""
    args = parser().parse_args(argv)
    started = time.perf_counter()
    if args.rebuild_lexical:
        index = CorpusIndex.rebuild_lexical(args.output)
        print(json.dumps({'stage': 'lexical-rebuilt', 'count': len(index.ids), 'output': str(args.output),
                          'wall_seconds': time.perf_counter() - started,
                          'lexical_seconds': index.manifest['lexical_rebuild_seconds']}), flush=True)
        return 0
    corpus, audit = load_corpus_audited(args.corpus, duplicate_policy='official_last')
    if audit['raw_row_count'] != args.expected_raw_count or len(corpus) != args.expected_count:
        raise ValueError(f'full corpus count mismatch: raw={audit["raw_row_count"]}, unique={len(corpus)}')
    embedder = Embedder(args.model, batch_size=args.batch_size)
    print(json.dumps({'stage': 'corpus-audited', 'raw_rows': audit['raw_row_count'],
                      'unique_ids': len(corpus), 'data_sha256': audit['source_sha256'],
                      'model': embedder.provenance}), flush=True)
    index = CorpusIndex.build(corpus, args.output, embedder, data_sha256=audit['source_sha256'],
                              audit=audit, contextual=args.contextual, chunk_size=args.chunk_size)
    # Perform the complete safe round trip before declaring the persisted index ready.
    CorpusIndex.load(args.output, expected_provenance=embedder.provenance,
                     expected_data_sha256=audit['source_sha256'])
    print(json.dumps({'stage': 'complete', 'count': len(index.ids), 'dimension': index.vectors.shape[1],
                      'output': str(args.output), 'wall_seconds': time.perf_counter() - started,
                      'embedding_seconds': index.manifest['embedding_seconds'],
                      'truncated_count': index.manifest['truncated_count'],
                      'truncated_token_count': index.manifest['truncated_token_count'],
                      'oom_retries': index.manifest['oom_retries']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
