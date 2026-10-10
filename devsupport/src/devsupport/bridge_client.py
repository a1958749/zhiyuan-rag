"""Private loopback transport and trusted native source mapping (standard library only)."""
import hashlib
import json
import os
import urllib.error
import urllib.request

PROFILES = frozenset(('B0', 'H0', 'H1', 'R1', 'R2', 'C', 'Full'))


def source_mapping(row: dict | None) -> dict | None:
    """Resolve only persisted source metadata; malformed marked sources fail closed."""
    row = row or {}
    metadata = row.get('metadata') or {}
    if not metadata.get('devsupport_profile'):
        return None
    config = row.get('config') or {}
    retrieval = config.get('retrieval') or {}
    if (metadata['devsupport_profile'] not in PROFILES or not metadata.get('corpus_revision')
            or metadata.get('devsupport_corpus') != 'freshstack/langchain'
            or config.get('kind', 'classic') != 'classic'
            or retrieval.get('retriever') != 'devsupport' or retrieval.get('exposure') != 'prefetch'
            or retrieval.get('rephrase_query') is not False or retrieval.get('prescreen')
            or retrieval.get('reranker')):
        raise ValueError('Invalid persisted DevSupport source configuration')
    return {'source_id': str(row['id']), 'profile': metadata['devsupport_profile'],
            'corpus_revision': metadata['corpus_revision'], 'knowledge_date': '2024-10'}


def select_mapping(authorized_sources: list) -> dict | None:
    """Select exactly one authorized alias; reject mixed sources and profiles."""
    mappings = [entry.get('devsupport') for entry in authorized_sources if entry.get('devsupport')]
    if not mappings:
        return None
    if len(authorized_sources) != 1 or len(mappings) != 1:
        raise ValueError('Choose one DevSupport profile; mixed profiles and uploaded sources are unsupported')
    return mappings[0]


def native_document(raw: dict, mapping: dict) -> dict:
    """Overlay native source identity without modifying original evidence metadata/text."""
    metadata = raw.get('metadata') or {}
    title = str(metadata.get('title') or metadata.get('path') or raw['doc_id'])
    return {**raw, 'title': title, 'filename': title,
            'source': metadata.get('url') or metadata.get('source') or '',
            'source_id': mapping['source_id'], 'chunk_key': hashlib.md5(raw['text'].encode('utf-8')).hexdigest(),
            'corpus_revision': mapping['corpus_revision'], 'knowledge_date': mapping['knowledge_date']}


class BridgeClient:
    """Backend-only bounded HTTP client; never retries a consuming answer request."""

    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get('DEVSUPPORT_SERVICE_URL', 'http://127.0.0.1:8094')
        if not self.url.startswith('http://127.0.0.1:'):
            raise ValueError('DevSupport service must use IPv4 loopback')

    def call(self, operation: str, body: dict) -> dict:
        """Call the private service and sanitize transport errors."""
        headers = {'Content-Type': 'application/json'}
        token = os.environ.get('DEVSUPPORT_SERVICE_TOKEN')
        if token:
            headers['X-Devsupport-Token'] = token
        request = urllib.request.Request(self.url + '/v1/' + operation,
                                         json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if operation == 'evidence' and exc.code == 404:
                return None
            raise RuntimeError('DevSupport service rejected the request') from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError('DevSupport service unavailable or rejected the request') from exc

    def search(self, query: str, mapping: dict, session: str) -> dict:
        """Prepare a server-held run after native source authorization."""
        return self.call('search', {'query': query, 'profile': mapping['profile'],
                                   'revision': mapping['corpus_revision'], 'session': session})

    def answer(self, query: str, mapping: dict, session: str, run_id: str) -> dict:
        """Consume exactly that scoped prepared run."""
        return self.call('answer', {'query': query, 'profile': mapping['profile'],
                                   'revision': mapping['corpus_revision'], 'session': session, 'run_id': run_id})

    def evidence(self, mapping: dict, **lookup) -> dict:
        """Read immutable snapshot content only after native source-use authorization."""
        return self.call('evidence', {'revision': mapping['corpus_revision'], **lookup})
