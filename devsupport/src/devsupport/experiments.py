"""Resumable offline experiment execution, separate from online retrieval."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .metrics import evaluate_query
from .types import Question


def _save_json(path: Path, value: object) -> None:
    """Publish metadata atomically; a crash never makes half a manifest readable."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding='utf-8')
    temporary.replace(path)


def _query_hash(question: Question) -> str:
    """Bind each result to original query text without copying gold answer data."""
    return hashlib.sha256(question.query.encode('utf-8')).hexdigest()


def run_retrieval(questions: list[Question], engine: object, output: str | Path, manifest: dict) -> list[dict]:
    """Run frozen searches; online engine receives only query text and a fixed cap.

    Runtime failures contribute zero metrics and remain visible in the denominator.
    Malformed rankings and corrupted checkpoints abort rather than become valid scores.
    Caller manifest must bind source code, configuration, corpus, query file and split.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.with_suffix('.manifest.json')
    questions_by_id = {q.query_id: q for q in questions}
    if len(questions_by_id) != len(questions):
        raise ValueError('duplicate requested query IDs')
    if metadata.exists():
        if json.loads(metadata.read_text(encoding='utf-8')) != manifest:
            raise ValueError('experiment manifest mismatch; use a new run directory')
    elif output.exists():
        raise ValueError('missing experiment manifest')
    else:
        _save_json(metadata, manifest)
    rows = []
    seen = set()
    if output.exists():
        for line in output.read_bytes().splitlines():
            try:
                row = json.loads(line)
                qid = row['query_id']
                if qid in seen or qid not in questions_by_id:
                    raise ValueError('unknown or duplicate ID')
                if row['query_sha256'] != _query_hash(questions_by_id[qid]):
                    raise ValueError('query changed')
                if row['metrics'] != evaluate_query(row['ranked_ids'], questions_by_id[qid]):
                    raise ValueError('metrics changed')
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('invalid experiment checkpoint') from error
            seen.add(qid)
            rows.append(row)
    for q in questions:
        if q.query_id in seen:
            continue
        started = time.perf_counter()
        failure = None
        try:
            hits = engine.search(q.query, k=50)
        except (TimeoutError, ConnectionError, RuntimeError, OSError) as error:
            hits, failure = [], type(error).__name__
        ids = [hit.evidence.doc_id for hit in hits]
        if (len(ids) > 50 or len(set(ids)) != len(ids) or
                any(hit.rank != rank or not math.isfinite(hit.score) for rank, hit in enumerate(hits, 1))):
            raise ValueError('invalid retrieval ranking')
        row = {
            'query_id': q.query_id, 'query_sha256': _query_hash(q), 'ranked_ids': ids,
            'scores': [hit.score for hit in hits], 'score_kinds': [hit.score_kind for hit in hits],
            'metrics': evaluate_query(ids, q), 'elapsed_seconds': time.perf_counter() - started,
            'stage_seconds': {} if failure else dict(getattr(engine, 'timings', {})),
            'leg_ids': {} if failure else dict(getattr(engine, 'leg_ids', {})), 'error': failure,
        }
        # Append one complete record and flush before beginning the next expensive query.
        with output.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
        rows.append(row)
    by_id = {row['query_id']: row for row in rows}
    return [by_id[q.query_id] for q in questions]


def summarize(rows: list[dict]) -> dict:
    """Aggregate all requested queries; retrieval timing is explicitly not end-to-end latency."""
    if not rows:
        raise ValueError('cannot summarize an empty experiment')
    keys = rows[0]['metrics'].keys()
    elapsed = sorted(row['elapsed_seconds'] for row in rows)
    return {
        'n': len(rows), 'errors': sum(row['error'] is not None for row in rows),
        'metrics': {key: statistics.fmean(row['metrics'][key] for row in rows) for key in keys},
        'retrieval_seconds_p50': statistics.median(elapsed),
        'retrieval_seconds_p95': elapsed[max(0, math.ceil(len(elapsed) * 0.95) - 1)],
        'latency_scope': 'in-process retrieval only; first query may include lazy model loading',
    }
