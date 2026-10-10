"""Audit complete frozen evaluation contexts offline, without any model calls."""
import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_metrics import QUALITY_KEYS, context_quality, load_dev_questions, summarize, unique_known  # noqa: E402
from devsupport.answer import ProxyTokenCounter, TOKENIZER_PATH, pack_context  # noqa: E402
from devsupport.data import load_corpus, sha256_file  # noqa: E402
from devsupport.evaluate import digest, make_record, text_hash, write_new  # noqa: E402
from devsupport.metrics import bootstrap_paired  # noqa: E402
from devsupport.report import outcome  # noqa: E402
from devsupport.types import Hit  # noqa: E402

METRIC_TOLERANCE = {'rel_tol': 1e-12, 'abs_tol': 1e-12}


def _verify_record(record: dict, expected: dict) -> dict:
    """Verify original integrity and exact nonmetric fields, disclosing bounded metric roundoff."""
    if record.get('record_sha256') != digest({k: v for k, v in record.items() if k != 'record_sha256'}):
        raise ValueError('record/query/metric checksum mismatch')
    excluded = {'record_sha256', 'metrics_prepack', 'metrics_packed'}
    if digest({k: v for k, v in record.items() if k not in excluded}) != digest(
            {k: v for k, v in expected.items() if k not in excluded}):
        raise ValueError('record/query/metric checksum mismatch')
    differences = {}
    for field in ('metrics_prepack', 'metrics_packed'):
        actual, target = record.get(field), expected[field]
        if not isinstance(actual, dict) or set(actual) != set(target):
            raise ValueError('record/query/metric checksum mismatch')
        for key, value in actual.items():
            wanted = target[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not math.isclose(value, wanted, **METRIC_TOLERANCE)):
                raise ValueError('record/query/metric checksum mismatch')
            if value != wanted:
                differences.setdefault(field, {})[key] = {'recorded': value, 'recomputed': wanted}
    return differences


def _bytes(path: Path, inputs: dict) -> bytes:
    """Consume and bind one exact byte snapshot, rejecting inconsistent rereads."""
    raw = Path(path).read_bytes()
    name, hashed = str(Path(path).resolve()), hashlib.sha256(raw).hexdigest()
    if name in inputs and inputs[name] != hashed:
        raise ValueError('input changed during audit')
    inputs[name] = hashed
    return raw


def _json(path: Path, inputs: dict):
    """Parse the exact JSON snapshot entered in the provenance ledger."""
    return json.loads(_bytes(path, inputs).decode('utf-8-sig'))


def _recheck(inputs: dict) -> None:
    """Fail closed if any consumed input changed before acceptance/publication."""
    if any(sha256_file(Path(name)) != expected for name, expected in inputs.items()):
        raise ValueError('consumed input changed before publication')


def _cohort(contract: dict, split_data: dict, meta: dict, split: str) -> list[str]:
    """Require complete DEV selection or exact TEST150/fixed-first30 cohorts."""
    full_dev, test = split_data['dev_ids'], split_data['test_ids']
    for values in (full_dev, test):
        if (not isinstance(values, list) or not values or any(type(i) is not str or not i for i in values)
                or len(values) != len(set(values))):
            raise ValueError('invalid split IDs')
    if set(full_dev) & set(test):
        raise ValueError('DEV/TEST overlap')
    dev = contract['split_ids']['dev']
    if (not dev or len(dev) != len(set(dev)) or not set(dev) <= set(full_dev)
            or contract['split_ids']['test'] != test):
        raise ValueError('contract split mismatch')
    if split == 'test' and (len(test) != 150 or contract['generation_ids'] != test[:30]):
        raise ValueError('TEST requires frozen150 and first30')
    ids = dev if split == 'dev' else contract['generation_ids'] if meta['phase'] == 'pipeline' else test
    if meta['query_ids'] != ids:
        raise ValueError('complete ordered cohort required')
    return ids


def _packed(row: dict, phase: str, corpus: dict, counter) -> tuple[list, dict, str]:
    """Replay raw whole chunks and verify observed contexts, regardless of later generation errors."""
    ids, scores = row['prepack_ids'], row['prepack_scores']
    unique_known(ids, set(corpus))
    hits = [Hit(corpus[i], score if score is not None else 0.0, n, 'audit')
            for n, (i, score) in enumerate(zip(ids, scores), 1)]
    packed, _, packing = pack_context(hits, budget=4000, counter=counter)
    if phase in ('retrieval', 'rerank'):
        if row['packed_ids'] or row['result'].get('context'):
            raise ValueError('ranking-only run contains unexpected observed context')
        return packed, packing, 'replayed ranking-only context'
    result = row['result']
    observed = result.get('context', [])
    if row['packed_ids'] != [h.evidence.doc_id for h in packed] or len(observed) != len(packed):
        raise ValueError('observed packing membership differs from shared policy')
    for n, (item, hit) in enumerate(zip(observed, packed), 1):
        expected = {**hit.evidence.to_corpus_dict(), 'score': hit.score, 'citation_number': n}
        if 'rank' in item:
            if type(item['rank']) is not int or item['rank'] != hit.rank:
                raise ValueError('observed context rank mismatch')
            expected['rank'] = item['rank']
        if 'score_kind' in item:
            if type(item['score_kind']) is not str or not item['score_kind']:
                raise ValueError('invalid observed score kind')
            expected['score_kind'] = item['score_kind']
        if digest(item) != digest(expected):
            raise ValueError('observed context differs from raw source/score/citation')
    if ids or observed or 'packing' in result.get('trace', {}):
        if digest(result.get('trace', {}).get('packing')) != digest(packing):
            raise ValueError('observed packing trace differs from shared policy')
    return packed, packing, 'validated observed pipeline context'


def audit(runs: list[Path], split: str = 'dev', lock: Path | None = None) -> dict:
    """Validate complete modern runs and compute qrel context quality for every attempted query.

    Args:
        runs: Complete evaluate_pipeline output directories with disjoint profile names.
        split: Explicit dev or test selection; no automatic holdout materialization.
        lock: Required immutable exact contract for TEST.

    Returns:
        Per-query contexts, method summaries, all paired metric intervals and byte provenance.

    Raises:
        ValueError: Incomplete, changed, mixed, unlocked or malformed artifacts.
    """
    if not runs or split not in ('dev', 'test'):
        raise ValueError('runs and explicit split required')
    if split == 'test' and lock is None:
        raise ValueError('TEST requires exact lock')
    inputs, methods, rows_by_method, all_rows = {}, {}, {}, []
    sources = [Path(__file__), Path(__file__).with_name('context_metrics.py'), TOKENIZER_PATH]
    sources += [BASE / 'src/devsupport' / name for name in
                ('answer.py', 'data.py', 'metrics.py', 'types.py', 'evaluate.py', 'report.py')]
    for path in sources:
        _bytes(path, inputs)
    contract, ids, corpus, questions = None, None, None, None
    phases = {}
    try:
        counter = ProxyTokenCounter()
        for run in map(Path, runs):
            if list(run.glob('*.inflight.json')):
                raise ValueError('unresolved inflight record')
            meta = _json(run / 'run.json', inputs)
            names = meta['profiles']
            if (meta['split'] != split or meta.get('controls') is not False
                    or meta['phase'] not in ('retrieval', 'rerank', 'retrieve-only', 'pipeline')
                    or not names or len(names) != len(set(names))):
                raise ValueError('invalid run split/phase/profile/control scope')
            if any(type(n) is not str or not n.replace('-', '').replace('_', '').isalnum() for n in names):
                raise ValueError('invalid profile name')
            for name in names:
                if name in methods:
                    raise ValueError('duplicate profile across runs')
                manifest = _json(run / f'{name}.manifest.json', inputs)
                current = manifest['evaluation_contract']
                if (meta['contract_sha256'] != digest(current)
                        or any(manifest.get(k) != v for k, v in meta.items())
                        or manifest['profile'] != name or manifest['settings'] != current['profiles'][name]
                        or manifest['variant'] != (manifest['settings']['reranker'] or
                                                   manifest['settings']['retrieval'])):
                    raise ValueError('manifest/profile/contract mismatch')
                if contract is None:
                    contract = current
                    if ('context_counter' in contract and contract['context_counter'] !=
                            'cl100k_base_proxy_v1; 4000 proxy tokens, not exact gateway generation tokens'):
                        raise ValueError('unsupported frozen packing protocol')
                    for filename in ('answer.py', 'data.py', 'metrics.py', 'types.py'):
                        rel = 'devsupport/src/devsupport/' + filename
                        if contract['source'].get(rel) != inputs[str((ROOT / rel).resolve())]:
                            raise ValueError('annotation/metric/packing source drift')
                    for key in ('queries', 'corpus', 'split'):
                        path = Path(contract['paths'][key])
                        inputs[str(path.resolve())] = sha256_file(path)
                        if inputs[str(path.resolve())] != contract['artifacts'][key]:
                            raise ValueError('dataset artifact checksum mismatch')
                    split_data = _json(Path(contract['paths']['split']), inputs)
                    ids = _cohort(contract, split_data, meta, split)
                    if split == 'test' and _json(Path(lock), inputs) != {'contract': contract,
                                                                       'sha256': digest(contract)}:
                        raise ValueError('evaluation lock mismatch')
                    questions = load_dev_questions(Path(contract['paths']['queries']), ids)
                    corpus = load_corpus(contract['paths']['corpus'], duplicate_policy='official_last')
                    _recheck(inputs)
                elif current != contract or meta['query_ids'] != ids:
                    raise ValueError('mixed frozen contract/cohort')
                if _cohort(contract, split_data, meta, split) != ids:
                    raise ValueError('mixed phase cohort')
                index_key = {'U': 'native_u_index', 'C': 'contextual_index'}.get(
                    manifest['settings']['retrieval'], 'index')
                required = {'query_hashes': [text_hash(questions[qid].query) for qid in ids],
                            'questions_sha256': contract['artifacts']['queries'],
                            'index_manifest_sha256': contract['artifacts'][index_key],
                            'latency_scope': meta['phase'], 'context_counter': contract['context_counter']}
                if any(manifest.get(k) != v for k, v in required.items()):
                    raise ValueError('manifest query/index/phase/packing identity mismatch')
                records = [json.loads(line) for line in _bytes(run / f'{name}.jsonl', inputs).splitlines()]
                if [r['query_id'] for r in records] != ids:
                    raise ValueError('complete ordered record cohort required')
                scored = []
                for record in records:
                    elapsed = record['elapsed_seconds']
                    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError('invalid elapsed time')
                    q = questions[record['query_id']]
                    expected = make_record(q, record['result'], elapsed, record['error'])
                    roundoff = _verify_record(record, expected)
                    packed, packing, scope = _packed(record, meta['phase'], corpus, counter)
                    packed_ids = [h.evidence.doc_id for h in packed]
                    item = {'method': name, 'query_id': q.query_id, 'packing_scope': scope,
                            'packed_ids': packed_ids, 'packing': packing,
                            'context_sha256': {h.evidence.doc_id: hashlib.sha256(h.evidence.text.encode()).hexdigest()
                                               for h in packed},
                            'quality': context_quality(packed_ids, q, set(corpus)), 'outcome': outcome(record),
                            'record_metric_roundoff': roundoff}
                    scored.append(item)
                methods[name] = {**summarize(scored), 'operational_errors': sum(r['outcome']['error'] for r in scored),
                                 'fallbacks': sum(r['outcome']['fallback'] for r in scored)}
                rows_by_method[name] = scored
                all_rows.extend(scored)
                phases[name] = meta['phase']
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError('missing or invalid evaluation artifact') from exc
    seed = contract['seeds'][0]
    if type(seed) is not int:
        raise ValueError('invalid bootstrap seed')
    pairs = {}
    for baseline, candidate in combinations(methods, 2):
        metrics = {}
        for key in QUALITY_KEYS:
            a = [r['quality'][key] for r in rows_by_method[baseline]]
            b = [r['quality'][key] for r in rows_by_method[candidate]]
            metrics[key] = ({'unavailable': 'missing gold; complete cohort retained', 'n': len(ids)}
                            if None in a or None in b else bootstrap_paired(a, b, seed))
        pairs[f'{candidate}-minus-{baseline}'] = metrics
    _recheck(inputs)
    return {'schema': 'devsupport.evaluation-context.v1', 'complete': True, 'split': split,
            'questions': len(ids), 'query_ids': ids, 'methods': methods, 'paired': pairs, 'rows': all_rows,
            'phases': phases, 'contract_sha256': digest(contract), 'input_sha256': inputs,
            'metric_recompute_tolerance': dict(METRIC_TOLERANCE),
            'historical_runtime_sources': contract['source'], 'model_calls': 0, 'human_ratings': 0,
            'notes': ['Positive-qrel document diagnostics, not answer accuracy or faithfulness.',
                      'Ranking-only contexts are replayed; pipeline contexts are verified observed inputs.',
                      'Later generation failures retain context quality and have separate error counts.',
                      'All attempted queries retained; missing gold never silently shrinks paired cohorts.',
                      'Single runs; exploratory paired percentile bootstrap, 2000 samples, no multiplicity correction.',
                      'Current semantic sources verified; historical runtime execution is not asserted reproduced.']}


def publish(runs: list[Path], output: Path, split: str = 'dev', lock: Path | None = None) -> dict:
    """Recheck consumed bytes before exclusively publishing a new report."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('report output exists; never overwrite')
    report = audit(runs, split, lock)
    _recheck(report['input_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'summary.json', report)
    return report


def main() -> None:
    """Audit recorded contexts without loading model services or credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    report = publish(args.runs, args.output, args.split, args.lock)
    print(json.dumps({'complete': report['complete'], 'questions': report['questions'],
                      'methods': list(report['methods']), 'model_calls': 0, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
