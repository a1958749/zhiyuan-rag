"""Run frozen local retrieval/reranking or shared query-only pipeline experiments.

Optional U calls the genuine upstream adapter in native Python 3.12; C requires a
complete contextual index. Both use the shared query-only generation pipeline.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'devsupport/src'))


class NoPaidProvider:
    """Fail closed if a local-only phase accidentally reaches the generator."""

    def complete(self, *args, **kwargs):
        raise RuntimeError('paid generation disabled in local phase')


def main():
    """Build shared adapters only after freeze/checkpoint authorization has been checked."""
    from devsupport.data import load_questions, sha256_file
    from devsupport.evaluate import (BudgetGuard, GuardedProvider, build_contract, cached_pools, check_balance,
                                     digest, read_json, rerank_pool, run, select_ids, write_new,
                                     build_retriever, pipeline_result, prepare_native_environment,
                                     retrieval_result, TracedRetriever, dependency_unavailable)
    from devsupport.embedding import Embedder
    from devsupport.index import CorpusIndex
    from devsupport.llm import AnthropicProvider, ProviderConfig
    from devsupport.pipeline import Pipeline, PipelineConfig
    from devsupport.service import ModelEmbedder, ModelReranker
    from devsupport.report import summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--phase', choices=['retrieval', 'rerank', 'retrieve-only', 'pipeline'], required=True)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--pools', type=Path, help='Frozen complete H1 JSONL for rerank-only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--controls', action='store_true', help='Use frozen config controls path instead of FreshStack')
    args = parser.parse_args()
    contract = build_contract(args.config, ROOT)
    ids = select_ids(contract, args.split, args.phase, args.lock)
    profiles = args.profiles or list(contract['profiles'])
    if len(set(profiles)) != len(profiles) or not set(profiles) <= set(contract['profiles']):
        parser.error('profiles must be a unique subset of frozen names')
    for name in profiles:
        settings = contract['profiles'][name]
        if args.phase == 'retrieval' and (settings['reranker'] or settings['adaptive'] != 'never'):
            parser.error('retrieval phase accepts only base local profiles without adaptive or reranker')
        if args.phase == 'rerank' and (not settings['reranker'] or settings['adaptive'] != 'never'):
            parser.error('rerank phase requires ranker plus adaptive never')
    if any(contract['profiles'][name]['retrieval'] == 'U' for name in profiles):
        prepare_native_environment()
    questions = {q.query_id: q for q in load_questions(contract['paths']['queries'])}
    if args.controls:
        from devsupport.review import load_controls
        from devsupport.types import Question
        if args.phase not in ('pipeline', 'retrieve-only') or 'controls' not in contract['paths']:
            parser.error('controls require pipeline/retrieve-only and a frozen controls JSON path')
        controls = [r for r in load_controls(contract['paths']['controls']) if r['split'] == args.split]
        ids = [r['query_id'] for r in controls]
        questions = {r['query_id']: Question(r['query_id'], '', r['query'], '', r.get('nuggets', []),
                                           {'expected_decision': r['expected_decision']}) for r in controls}
    selected = [questions[qid] for qid in ids]
    # All runs in a directory must share the frozen contract, split and phase.
    directory_contract = {'contract_sha256': digest(contract), 'split': args.split, 'phase': args.phase,
                          'profiles': profiles, 'query_ids': ids, 'controls': args.controls}
    directory_meta = args.output / 'run.json'
    if directory_meta.exists():
        if not args.resume or read_json(directory_meta) != directory_contract:
            raise ValueError('output already exists or changed run contract')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('output directory contains unbound files')
        write_new(directory_meta, directory_contract)
    index = CorpusIndex.load(contract['paths']['index'])
    provenance = Embedder().provenance
    if provenance != index.manifest['provenance']:
        raise ValueError('embedding semantic contract differs from index')
    embedder = ModelEmbedder(provenance, contract['config'].get('gpu_url', 'http://127.0.0.1:8093'))
    retrievers = {variant: build_retriever(contract, variant, index, embedder)
                  for variant in dict.fromkeys(contract['profiles'][name]['retrieval'] for name in profiles)}
    paid = args.phase == 'pipeline' or (args.phase == 'retrieve-only' and
                                      any(contract['profiles'][n]['adaptive'] != 'never' for n in profiles))
    guard, provider = None, NoPaidProvider()
    if paid:
        if 'budget_start' not in contract['paths']:
            parser.error('paid operations require root-authorized finite budget_start JSON in frozen config')
        config = ProviderConfig(**contract['provider'])
        guard = BudgetGuard(read_json(contract['paths']['budget_start']), lambda: check_balance(config),
                            state_path=Path(contract['paths']['budget_start']).with_suffix('.accounting.json'))
        provider = GuardedProvider(AnthropicProvider(config), guard)
    pools = None
    cache_hashes = {}
    if args.phase == 'rerank':
        if not args.pools:
            parser.error('rerank-only requires --pools with identical frozen H1 candidates')
        cache_manifest = args.pools.with_suffix('.manifest.json')
        rows = [json.loads(line) for line in args.pools.read_bytes().splitlines()]
        meta = read_json(cache_manifest)
        if meta.get('split') != args.split or meta.get('query_ids') != ids:
            raise ValueError('candidate split/order mismatch')
        pools = cached_pools(rows, meta, selected, index.corpus, contract['artifacts']['queries'],
                             contract['artifacts']['index'], contract)
        cache_hashes = {'pools_sha256': sha256_file(args.pools), 'pool_manifest_sha256': sha256_file(cache_manifest)}
    for name in profiles:
        if guard is not None and guard.stopped:
            break
        settings = contract['profiles'][name]
        retriever = retrievers[settings['retrieval']]
        if settings['retrieval'] == 'U' and args.phase in ('pipeline', 'retrieve-only'):
            retriever = TracedRetriever(retriever)
        ranker = ModelReranker(settings['reranker'], embedder.url) if settings['reranker'] else None
        pipeline = Pipeline(retriever, provider, ranker,
                            config=PipelineConfig(adaptive=settings['adaptive'], gate_enabled=settings['gate_enabled'],
                                                  generation_max_tokens=contract['generation_max_tokens']))

        def execute(query):
            """Only query text crosses into production; gold annotations remain in run()."""
            if args.phase in ('pipeline', 'retrieve-only'):
                return pipeline_result(query, pipeline, args.phase)
            if pools is None:
                return retrieval_result(query, retriever)
            start = time.perf_counter()
            hits = rerank_pool(query, deepcopy(pools[query]), ranker)
            trace = {'reranking': [deepcopy(ranker.last_trace)]}
            return {'ranked_ids': [h.evidence.doc_id for h in hits], 'scores': [h.score for h in hits],
                    'score_kinds': [h.score_kind for h in hits], 'trace': trace,
                    'retrieval_elapsed_seconds': time.perf_counter() - start,
                    'source_rows': [h.evidence.to_corpus_dict() for h in hits]}

        artifact_key = {'U': 'native_u_index', 'C': 'contextual_index'}.get(settings['retrieval'], 'index')
        manifest = {**directory_contract, 'profile': name, 'variant': settings['reranker'] or settings['retrieval'],
                    'settings': settings, 'questions_sha256': contract['artifacts']['queries'],
                    'index_manifest_sha256': contract['artifacts'][artifact_key], **cache_hashes,
                    'context_counter': contract['context_counter'], 'latency_scope': args.phase,
                    'evaluation_contract': contract}
        local_dependency_failed = False

        def execute_with_dependency_stop(query):
            """Remember an outage even when the failed query is the final row of a profile."""
            nonlocal local_dependency_failed
            try:
                return execute(query)
            except Exception as exc:
                local_dependency_failed = dependency_unavailable(exc)
                raise

        result = run(selected, execute_with_dependency_stop, args.output / (name + '.jsonl'),
                     manifest, args.resume, guard)
        summary = summarize(result, ids, allow_partial=True)
        summary.update({'phase': args.phase, 'split': args.split, 'profile': name})
        if args.controls:
            summary['control_decision_accuracy_scope'] = 'pipeline only; retrieve-only does not run the decision gate'
            summary['control_decision_accuracy_attempted'] = None
            if args.phase == 'pipeline':
                from devsupport.report import outcome
                valid = [r for r in result if not outcome(r)['error']]
                correct = sum(r['result'].get('decision') == questions[r['query_id']].metadata['expected_decision']
                              for r in valid)
                summary['control_decision_accuracy_attempted'] = correct / len(result) if result else None
        # Summary is versioned by durable row count; resuming cannot overwrite earlier observations.
        target = args.output / f'{name}.summary-n{len(result)}.json'
        if not target.exists():
            write_new(target, summary)
        print(json.dumps({'profile': name, 'attempted': len(result), 'planned': len(ids), 'errors': summary['errors']}))
        if local_dependency_failed or len(result) < len(ids):
            break
    if guard is not None:
        # Accounting snapshots saved separately include only numeric whitelisted balances.
        target = args.output / f'balance-{time.time_ns()}.json'
        try:
            guard.check()
        except RuntimeError:
            pass
        write_new(target, {'observations': guard.observations, 'stopped': guard.stopped})


if __name__ == '__main__':
    main()
