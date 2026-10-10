"""Launch the private shared pipeline with existing core Python environment."""
import argparse
import json
import sys
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))


def build_service(enable_full=False, enable_contextual=False):
    """Create fixed profiles; startup loads no weights and makes no provider calls."""
    from devsupport.embedding import Embedder
    from devsupport.evaluate import BudgetGuard, GuardedProvider, check_balance, read_json
    from devsupport.index import CorpusIndex
    from devsupport.llm import AnthropicProvider, ProviderConfig
    from devsupport.pipeline import Pipeline, PipelineConfig
    from devsupport.retrieval import Retriever
    from devsupport.service import ModelEmbedder, ModelReranker, PipelineService
    base = ROOT / 'devsupport'
    index = CorpusIndex.load(base / 'indexes/raw')
    provenance = Embedder().provenance
    if provenance != index.manifest['provenance']:
        raise ValueError('local embedding semantic contract differs from index')
    embedder = ModelEmbedder(provenance)
    names = ('B0', 'H0', 'H1', 'R1', 'R2') + (('C',) if enable_contextual else ())
    names += ('Full',) if enable_full else ()
    configs = {name: json.loads((base / 'configs' / (name + '.json')).read_text()) for name in names}
    if enable_full:
        pool = configs['Full'].get('pool', 'H1')
        if pool not in ('H0', 'H1', 'C'):
            raise ValueError('Full pool must be H0, H1 or C')
        if pool == 'C' and not enable_contextual:
            raise ValueError('Full C pool requires explicit contextual enabling')
    contextual = None
    if enable_contextual:
        contextual = CorpusIndex.load(base / 'indexes/contextual', expected_provenance=provenance,
                                       expected_data_sha256=index.manifest['data_sha256'])
        if (contextual.manifest.get('contextual') is not True
                or any(contextual.manifest.get(key) != index.manifest.get(key) for key in
                       ('provenance', 'canonical_corpus_sha256', 'data_sha256', 'count', 'dimension'))
                or contextual.corpus != index.corpus):
            raise ValueError('contextual index must preserve the shared original corpus and embedding contract')
    provider_config = ProviderConfig(**json.loads((base / 'configs/provider.json').read_text()))
    budget_path = base / 'reports/budget-current-gateway.json'
    authorization = read_json(budget_path)
    if not isinstance(authorization, dict):
        raise ValueError('finite numeric budget authorization required')
    guard = BudgetGuard(authorization, lambda: check_balance(provider_config),
                        state_path=budget_path.with_suffix('.accounting.json'))
    provider_lock = Lock()

    class SerializedProvider(GuardedProvider):
        """Keep UI threads from checking the same balance while a paid call is in flight."""

        def complete(self, *args, **kwargs):
            """Serialize the existing guard and complete call across all UI profiles."""
            with provider_lock:
                return super().complete(*args, **kwargs)

    provider = SerializedProvider(AnthropicProvider(provider_config), guard)
    pipelines = {}
    for name in names:
        config = configs[name]
        variant = name if name in ('B0', 'H0', 'H1', 'C') else config.get('pool', 'H1')
        if name in ('R1', 'R2') and variant != 'H1':
            raise ValueError('reranker profiles require the H1 pool')
        ranker = name if name in ('R1', 'R2') else config.get('reranker')
        pipeline_config = config.get('pipeline', {'adaptive': 'never', 'gate_enabled': False})
        pipelines[name] = Pipeline(Retriever(contextual if variant == 'C' else index, embedder, variant), provider,
                                   ModelReranker(ranker) if ranker else None,
                                   config=PipelineConfig(**pipeline_config))
    return PipelineService(pipelines, index.manifest['canonical_corpus_sha256'], index.corpus)


def main():
    """Bind loopback only; Full needs explicit enabling after dev selection."""
    import uvicorn
    from devsupport.service import create_app
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8094)
    parser.add_argument('--enable-full', action='store_true')
    parser.add_argument('--enable-contextual', action='store_true')
    args = parser.parse_args()
    uvicorn.run(create_app(build_service(args.enable_full, args.enable_contextual)), host='127.0.0.1', port=args.port,
                workers=1, access_log=False)


if __name__ == '__main__':
    main()
