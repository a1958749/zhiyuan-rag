"""Build or resume a verified full FreshStack index without loading annotations."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport' / 'src'))

def main():
    """Keep data, model and representation contracts explicit on every resume."""
    from devsupport.data import load_corpus_audited
    from devsupport.embedding import Embedder
    from devsupport.index import CorpusIndex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contextual', action='store_true')
    parser.add_argument('--rebuild-lexical', action='store_true')
    args = parser.parse_args()
    base = ROOT / 'devsupport'
    target = base / 'indexes' / ('contextual' if args.contextual else 'raw')
    if args.rebuild_lexical:
        index = CorpusIndex.rebuild_lexical(target)
    else:
        corpus, audit = load_corpus_audited(base / 'data' / 'corpus.parquet')
        embedder = Embedder(base / 'model-cache' / 'embedding')
        index = CorpusIndex.build(corpus, target, embedder,
                                 data_sha256=audit['source_sha256'], audit=audit,
                                 contextual=args.contextual)
    checked = CorpusIndex.load(target)
    print(json.dumps({'status': 'complete_verified', 'count': len(checked.ids),
                      'dimension': checked.manifest['dimension'],
                      'contextual': checked.manifest['contextual'],
                      'truncated_count': index.manifest.get('truncated_count'),
                      'embedding_seconds': index.manifest.get('embedding_seconds')}, indent=2))


if __name__ == '__main__':
    main()
