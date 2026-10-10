"""Inspect private service health, optionally run an explicitly requested paid query."""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))


def main():
    """Health alone is read-only; --query explicitly exercises shared search and answer."""
    from devsupport.bridge_client import BridgeClient
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query')
    parser.add_argument('--profile', default='H1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    client = BridgeClient()
    with urllib.request.urlopen(client.url + '/health', timeout=10) as response:
        result = json.load(response)
    if args.query:
        mapping = {'profile': args.profile, 'corpus_revision': result['corpus_revision']}
        import uuid
        session = uuid.uuid4().hex
        prepared = client.search(args.query, mapping, session)
        result = client.answer(args.query, mapping, session, prepared['run_id'])
        result['text_hashes'] = {row['doc_id']: hashlib.sha256(row['text'].encode()).hexdigest()
                                 for row in result['context']}
    text = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
