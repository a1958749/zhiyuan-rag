"""Native adapter accepting only an explicitly authorized server context."""
from devsupport.src.devsupport.bridge_client import BridgeClient, native_document, source_mapping
from docsgpt.retriever.base import BaseRetriever


def trusted_source_mapping(row: dict | None) -> dict | None:
    """Strictly validate marked rows before the upstream lenient config parser."""
    mapping = source_mapping(row)
    if mapping:
        from docsgpt.storage.db.source_config import SourceConfig
        SourceConfig.model_validate(row['config'])
    return mapping


class DevSupportRetriever(BaseRetriever):
    """Prefetch the shared pipeline without invoking native answer generation."""

    def __init__(self, source=None, chunks=6, doc_token_limit=50000,
                 devsupport_context=None, **kwargs):
        super().__init__()
        self.source = source or {}
        self.chunks, self.doc_token_limit = chunks, doc_token_limit
        self.context = devsupport_context
        self.envelope = None

    def search(self, query='') -> list[dict]:
        """Preserve full run metadata even when no evidence was retrieved."""
        if not self.context or not self.context.get('session') or self.chunks <= 0:
            raise ValueError('DevSupport requires an authorized source and enabled retrieval')
        mapping = self.context['mapping']
        query = query or self.source.get('question', '')
        self.envelope = BridgeClient().search(query, mapping, self.context['session'])
        self.context['run_id'] = self.envelope['run_id']
        self.context['query'] = query
        return [native_document(row, mapping) for row in self.envelope['context']]
