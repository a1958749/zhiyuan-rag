"""Offline DEV53 packed-context qrel diagnostics; no generation or model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))

from devsupport.answer import ProxyTokenCounter, TOKENIZER_PATH, pack_context  # noqa: E402
from devsupport.data import load_corpus, relevant_doc_ids, sha256_file  # noqa: E402
from devsupport.metrics import bootstrap_paired, evaluate_query  # noqa: E402
from devsupport.types import Hit, Question  # noqa: E402

QUALITY_KEYS = ('packed_context_doc_precision', 'context_average_precision_qrels',
                'context_doc_recall', 'context_nugget_coverage')
RUNS = {'B0': 'retrieval-v1', 'B1': 'retrieval-v1', 'H0': 'retrieval-v1',
        'H1': 'retrieval-v1', 'C': 'contextual-v1', 'R1': 'rerank-v1', 'R2': 'rerank-v1'}


def unique_known(ids: list[str], corpus_ids: set[str]) -> None:
    """Reject malformed, repeated or unknown evidence instead of silently repairing it."""
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('invalid evidence IDs')
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate evidence IDs')
    if not set(ids) <= corpus_ids:
        raise ValueError('unknown evidence IDs')


def context_quality(ids: list[str], question: Question, corpus_ids: set[str]) -> dict:
    """Score actual unique packed documents against positive qrels in packed order.

    AP divides by positives in this context, not all gold positives. Coverage retains
    every nugget in its denominator, including nuggets without positive judgments.
    Missing/all-unjudged gold yields nulls; known-positive gold with no hits yields zeros.
    """
    unique_known(ids, corpus_ids)
    groups = [relevant_doc_ids(nugget) for nugget in question.nuggets]
    for nugget in question.nuggets:
        for key in ('relevant_corpus_ids', 'non_relevant_corpus_ids', 'relevant'):
            if not {str(i) for i in (nugget.get(key) or [])} <= corpus_ids:
                raise ValueError('unknown qrel IDs')
    gold = set().union(*groups) if groups else set()
    positives = set(ids) & gold
    result = {'packed_doc_count': len(ids), 'positive_packed_doc_count': len(positives),
              'known_positive_doc_count': len(gold), 'nugget_count': len(groups),
              'unjudged_nugget_count': sum(not group for group in groups),
              'annotation_status': 'known_positive' if gold else 'missing_or_all_unjudged'}
    result.update(dict.fromkeys(QUALITY_KEYS))
    if gold:
        seen_positive = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ids, 1):
            if doc_id in gold:
                seen_positive += 1
                precision_sum += seen_positive / rank
        result.update({
            'packed_context_doc_precision': len(positives) / len(ids) if ids else 0.0,
            'context_average_precision_qrels': precision_sum / len(positives) if positives else 0.0,
            'context_doc_recall': len(positives) / len(gold),
            'context_nugget_coverage': sum(bool(set(ids) & group) for group in groups) / len(groups),
        })
    return result


def validate_manifest(manifest: dict, variant: str, dev_ids: list[str], expected: dict) -> None:
    """Require exact profile, ordered DEV cohort, and supplied provenance hash bindings."""
    required = {'variant': variant, 'split': 'dev', 'query_ids': dev_ids, **expected}
    if variant in ('R1', 'R2'):
        required['seeds'] = [None]
    else:
        unsigned = {k: v for k, v in manifest.items() if k != 'run_sha256'}
        required['run_sha256'] = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    if any(key not in manifest or manifest[key] != value for key, value in required.items()):
        raise ValueError(f'{variant}: manifest profile/cohort/provenance mismatch')


def validate_rows(variant: str, rows: list[dict], dev_ids: list[str], questions: dict[str, Question],
                  corpus_ids: set[str], config: dict, candidates: dict | None = None) -> None:
    """Fail closed on incomplete cohorts, unsuccessful runs, metric drift and altered pools."""
    if [row.get('query_id') for row in rows] != dev_ids or len(set(dev_ids)) != len(dev_ids):
        raise ValueError('requires complete ordered DEV cohort')
    for row in rows:
        qid = row['query_id']
        ids = row['ranked_ids']
        unique_known(ids, corpus_ids)
        if len(ids) > 50 or len(row['scores']) != len(ids) or any(
                not isinstance(s, (int, float)) or not math.isfinite(s) for s in row['scores']):
            raise ValueError('invalid ranked scores or depth')
        if variant in ('R1', 'R2'):
            trace = row.get('trace', {})
            if ('seed' not in row or row['seed'] is not None or trace.get('fallback') is not False
                    or 'error' not in trace or trace['error'] is not None
                    or trace.get('variant') != variant or trace.get('actual_model') != config['model']
                    or trace.get('requested_model') != config['model']
                    or trace.get('revision') != config['revision'] or trace.get('forward_count', 0) <= 0):
                raise ValueError('reranking must be explicitly successful with matching profile')
            if candidates is None or set(ids) != set(candidates[qid]):
                raise ValueError('reranking candidate membership changed')
            outputs = trace.get('outputs', [])
            if ([(o.get('doc_id'), o.get('score'), o.get('rank')) for o in outputs]
                    != [(i, s, rank) for rank, (i, s) in enumerate(zip(ids, row['scores']), 1)]):
                raise ValueError('reranking output trace mismatch')
        elif ('error' not in row or row['error'] is not None
              or row.get('query_sha256') != hashlib.sha256(questions[qid].query.encode()).hexdigest()):
            raise ValueError('retrieval must be explicitly successful with unchanged query')
        recomputed = evaluate_query(ids, questions[qid])
        stored = row.get('metrics', {})
        if (stored.keys() != recomputed.keys() or any(
                not isinstance(stored[k], (int, float)) or not math.isfinite(stored[k])
                or not math.isclose(stored[k], v, rel_tol=1e-12, abs_tol=1e-12)
                for k, v in recomputed.items())):
            raise ValueError('stored retrieval metric drift')


def summarize(rows: list[dict]) -> dict:
    """Macro-average only defined quality values, publishing each effective denominator."""
    metrics = {}
    for key in QUALITY_KEYS:
        values = [row['quality'][key] for row in rows if row['quality'][key] is not None]
        metrics[key] = {'mean': statistics.fmean(values) if values else None,
                        'n': len(values), 'missing': len(rows) - len(values)}
    return {'n': len(rows), 'metrics': metrics,
            'mean_packed_docs': statistics.fmean(row['quality']['packed_doc_count'] for row in rows),
            'unjudged_nugget_count': sum(row['quality']['unjudged_nugget_count'] for row in rows)}


def read_json(path: Path) -> dict:
    """Read a JSON manifest without importing model services."""
    return json.loads(path.read_text(encoding='utf-8'))


def validate_semantics(historical_hash: str, archived: Path, executing: Path) -> None:
    """Bind archived source bytes and allow only CRLF/LF checkout differences in execution."""
    if (sha256_file(archived) != historical_hash
            or archived.read_bytes().replace(b'\r\n', b'\n') != executing.read_bytes().replace(b'\r\n', b'\n')):
        raise ValueError('annotation/metric implementation drift')


def load_dev_questions(path: Path, dev_ids: list[str]) -> dict[str, Question]:
    """Project only DEV rows and required columns; never materialize holdout annotations."""
    import pyarrow.parquet as pq
    rows = pq.read_table(path, columns=['query_id', 'query_title', 'query_text', 'nuggets'],
                         filters=[('query_id', 'in', dev_ids)]).to_pylist()
    questions = {r['query_id']: Question(r['query_id'], r['query_title'], r['query_text'], '',
                                       r['nuggets'] or [], {}) for r in rows}
    if len(rows) != len(questions) or set(questions) != set(dev_ids):
        raise ValueError('duplicate or missing DEV query IDs')
    return questions


def build_audit(project_root: Path) -> tuple[list[dict], dict]:
    """Validate all seven frozen caches and repack original evidence with the shared policy."""
    base = project_root / 'devsupport'
    bound: dict[str, str] = {}

    def bind(path: Path) -> str:
        """Hash each input once and retain a relative provenance ledger."""
        name = path.relative_to(base).as_posix()
        if name not in bound:
            bound[name] = sha256_file(path)
        return bound[name]

    split_path = base / 'reports/split.json'
    dev_ids = read_json(split_path)['dev_ids']
    if len(dev_ids) != 53 or len(set(dev_ids)) != 53:
        raise ValueError('requires complete ordered DEV53 split')
    questions_path = base / 'data/queries.parquet'
    corpus_path = base / 'data/corpus.parquet'
    split_hash, query_hash, corpus_hash = bind(split_path), bind(questions_path), bind(corpus_path)
    questions = load_dev_questions(questions_path, dev_ids)
    corpus = load_corpus(corpus_path, duplicate_policy='official_last')
    corpus_ids = set(corpus)
    counter = ProxyTokenCounter()
    all_rows, summaries, rankings = [], {}, {}
    for variant, run in RUNS.items():
        path = base / f'reports/{run}/dev/{variant}.jsonl'
        manifest_path = path.with_suffix('.manifest.json')
        manifest = read_json(manifest_path)
        config_path = base / f'configs/{variant}.json'
        config = read_json(config_path)
        if config.get('variant') != variant:
            raise ValueError('config profile mismatch')
        expected = {'config_sha256': bind(config_path)}
        bind(path)
        bind(manifest_path)
        if variant in ('R1', 'R2'):
            h1 = base / 'reports/retrieval-v1/dev/H1.jsonl'
            expected.update(candidate_file_sha256=bind(h1),
                            candidate_manifest_sha256=bind(h1.with_suffix('.manifest.json')))
        else:
            index_path = base / ('indexes/contextual/manifest.json' if variant == 'C'
                                 else 'indexes/raw/manifest.json')
            index = read_json(index_path)
            if (index.get('data_sha256') != corpus_hash or index.get('contextual') is not (variant == 'C')
                    or manifest.get('query_provenance') != index.get('provenance')
                    or not manifest.get('query_provenance')):
                raise ValueError('index corpus/profile/query provenance mismatch')
            expected.update(split_sha256=split_hash, questions_sha256=query_hash,
                            index_manifest_sha256=bind(index_path))
        validate_manifest(manifest, variant, dev_ids, expected)
        # Historical retrieval implementations remain bound by their archived manifest.
        # Require current annotation/metric semantics to match before recomputing values.
        historical = manifest['implementation_sha256' if variant in ('R1', 'R2') else 'code_sha256']
        for name in ('data.py', 'metrics.py', 'types.py'):
            archived = base / f'src/devsupport/{name}'
            bind(archived)
            validate_semantics(historical.get(name), archived, ROOT / f'devsupport/src/devsupport/{name}')
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        validate_rows(variant, rows, dev_ids, questions, corpus_ids, config,
                      candidates=rankings.get('H1'))
        rankings[variant] = {r['query_id']: r['ranked_ids'] for r in rows}
        if variant == 'H1' and any(len(r['ranked_ids']) != 50 for r in rows):
            raise ValueError('requires complete H1 top-50 candidate pools')
        selected = []
        for row in rows:
            hits = [Hit(corpus[i], score, rank, 'cached_diagnostic')
                    for rank, (i, score) in enumerate(zip(row['ranked_ids'], row['scores']), 1)]
            packed, context, trace = pack_context(hits, budget=4000, counter=counter)
            ids = [hit.evidence.doc_id for hit in packed]
            selected.append({'variant': variant, 'query_id': row['query_id'], 'packed_ids': ids,
                             'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
                             'packing': trace, 'quality': context_quality(ids, questions[row['query_id']], corpus_ids)})
        summaries[variant] = summarize(selected)
        all_rows.extend(selected)
    paired = {}
    for baseline in ('B0', 'H0', 'H1'):
        paired[baseline] = {}
        for key in QUALITY_KEYS:
            a = [r['quality'][key] for r in all_rows if r['variant'] == baseline]
            b = [r['quality'][key] for r in all_rows if r['variant'] == 'C']
            paired[baseline][key] = (bootstrap_paired(a, b) if None not in a + b else
                                     {'status': 'unavailable: complete DEV53 has null quality values'})
    sources = {f'devsupport/src/devsupport/{name}': sha256_file(ROOT / f'devsupport/src/devsupport/{name}')
               for name in ('answer.py', 'llm.py', 'data.py', 'metrics.py', 'types.py')}
    sources['analysis/context_metrics.py'] = sha256_file(Path(__file__))
    sources['tokenizer'] = sha256_file(TOKENIZER_PATH)
    # Detect mutations during the read/compute window before publishing a successful artifact.
    if any(sha256_file(base / name) != digest for name, digest in bound.items()):
        raise ValueError('input changed during audit')
    return all_rows, {'schema': 1, 'scope': 'offline DEV53 packed-context qrels; retrieval order only; '
                     'not Ragas, LLM judgment, faithfulness or answer accuracy; negative controls excluded',
                     'project_root': str(project_root.resolve()), 'query_ids': dev_ids,
                     'corpus_policy': 'official_last', 'budget': 4000, 'counter': counter.name,
                     'retrieval_metric_comparison': {'rel_tol': 1e-12, 'abs_tol': 1e-12},
                     'exact_generation_tokens': False, 'input_sha256': bound, 'source_sha256': sources,
                     'methods': summaries, 'paired_C_minus_baseline': paired}


def main() -> None:
    """Publish exclusively into a new report directory after all validation succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.project_root / 'devsupport/reports/context-quality-dev-v1'
    if output.exists():
        raise FileExistsError(f'preserve existing audit: {output}')
    rows, summary = build_audit(args.project_root)
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (('rows.json', rows), ('summary.json', summary)):
        with (output / name).open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write('\n')
    print(json.dumps({'output': str(output), 'rows': len(rows), 'methods': summary['methods']}))


if __name__ == '__main__':
    main()
