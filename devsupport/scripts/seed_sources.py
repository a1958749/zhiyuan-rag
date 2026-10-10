"""Seed stable owned native source aliases using the existing DocsGPT app venv."""
import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    """Use native repository transactions without changing embeddings or chunking."""
    from docsgpt.storage.db.repositories.sources import SourcesRepository
    from docsgpt.storage.db.session import db_session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', default='local')
    parser.add_argument('--profiles', nargs='+', choices=['B0', 'H0', 'H1', 'R1', 'R2', 'C', 'Full'],
                        default=['B0', 'H0', 'H1', 'R1', 'R2'])
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'devsupport/indexes/raw/manifest.json').read_text())
    revision = manifest['canonical_corpus_sha256']
    with db_session() as conn:
        repo = SourcesRepository(conn)
        for profile in args.profiles:
            source_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f'devsupport:{args.owner}:{revision}:{profile}'))
            existing = repo.get_by_id(source_id)
            if existing:
                if existing['user_id'] != args.owner:
                    raise ValueError('source identity belongs to another owner')
            else:
                repo.create(f'DevSupport · {profile} · LangChain 2024-10', source_id=source_id,
                            user_id=args.owner, type='file', retriever='devsupport',
                            metadata={'devsupport_corpus': 'freshstack/langchain',
                                      'devsupport_profile': profile, 'corpus_revision': revision,
                                      'knowledge_date': '2024-10'},
                            config={'kind': 'classic', 'retrieval': {'retriever': 'devsupport',
                                    'exposure': 'prefetch', 'chunks': 50, 'rephrase_query': False,
                                    'prescreen': None, 'reranker': None}})
            print(json.dumps({'profile': profile, 'source_id': source_id, 'corpus_revision': revision}))


if __name__ == '__main__':
    main()
