"""Offline accounting evidence; missing transport/usage/cost is never fabricated."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from devsupport.evaluate import digest, write_new  # noqa: E402
from devsupport.report import summarize  # noqa: E402


def _number(value) -> bool:
    """Accept finite nonnegative numeric accounting, excluding booleans."""
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def summarize_calls(rows: list[dict]) -> dict:
    """Separate observed stage invocations, known transports and incomplete response accounting."""
    calls, missing_trace = [], 0
    for row in rows:
        trace = row.get('result', {}).get('trace', {})
        observed = trace.get('provider_calls')
        if observed is None:
            missing_trace += 1
        elif not isinstance(observed, list) or any(not isinstance(c, dict) for c in observed):
            raise ValueError('invalid recorded provider trace')
        else:
            calls.extend(observed)
    attempts, unknown_attempts, retries, usage = 0, 0, 0, {}
    usage_complete = not missing_trace
    costs, known_costs = [], []
    for call in calls:
        count = call.get('attempts')
        if count is None:
            unknown_attempts += 1
            usage_complete = False
        elif type(count) is not int or count < 0:
            raise ValueError('invalid transport attempts')
        else:
            attempts += count
            retries += max(0, count - 1)
        observed_usage = call.get('usage')
        if observed_usage is not None:
            if not isinstance(observed_usage, dict):
                raise ValueError('invalid response usage')
            for key, value in observed_usage.items():
                if type(value) is not int or value < 0:
                    raise ValueError('invalid response usage counter')
                usage[key] = usage.get(key, 0) + value
        dimensions_complete = isinstance(observed_usage, dict) and {'input_tokens', 'output_tokens'} <= observed_usage.keys()
        if (count != 0 and not dimensions_complete) or (count is not None and count > 1):
            # Call-level usage describes a response, not a complete reconstructed retry ledger.
            usage_complete = False
        cost = call.get('cost')
        known = call.get('known_response_cost', cost)
        for value in (cost, known):
            if value is not None and not _number(value):
                raise ValueError('invalid recorded cost')
        costs.append(cost)
        if known is not None:
            known_costs.append(known)
    return {'rows': len(rows), 'rows_missing_provider_trace': missing_trace,
            'stage_invocations': len(calls), 'invocations_missing_attempts': unknown_attempts,
            'known_transport_attempts': attempts,
            'transport_attempts_total': None if missing_trace or unknown_attempts else attempts,
            'known_retry_extra_attempts': retries, 'reported_response_usage': usage,
            'usage_complete_for_recorded_attempts': usage_complete,
            'known_response_cost_estimate_sum': sum(known_costs) if known_costs else None,
            'estimated_provider_cost_total': (sum(costs) if not missing_trace and all(c is not None for c in costs)
                                               else None),
            'costs_missing': sum(c is None for c in costs)}


def _recheck(inputs: dict) -> None:
    """Recheck every consumed byte binding before accepting or publishing."""
    for name, expected in inputs.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('accounting input changed')


def audit(directory: Path, split: str = 'dev', lock: Path | None = None) -> dict:
    """Reuse complete-cohort validators, then summarize their exact bound JSONL snapshots."""
    import control_decisions
    import evaluation_context
    directory = Path(directory).resolve()
    raw = (directory / 'run.json').read_bytes()
    meta = json.loads(raw)
    names = meta.get('profiles')
    if (split not in ('dev', 'test') or meta.get('split') != split or not isinstance(names, list) or not names
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n.replace('-', '').replace('_', '').isalnum() for n in names)):
        raise ValueError('invalid accounting split or profile scope')
    inputs = {str((directory / 'run.json').resolve()): hashlib.sha256(raw).hexdigest(),
              str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if split == 'test':
        if lock is None:
            raise ValueError('TEST accounting requires exact frozen lock')
        lock = Path(lock).resolve()
        lock_raw = lock.read_bytes()
        frozen = json.loads((directory / f'{names[0]}.manifest.json').read_bytes())['evaluation_contract']
        if json.loads(lock_raw) != {'contract': frozen, 'sha256': digest(frozen)}:
            raise ValueError('accounting TEST lock mismatch')
        inputs[str(lock)] = hashlib.sha256(lock_raw).hexdigest()
    if meta.get('controls'):
        validation = control_decisions.audit(directory)
        verified = {**validation['bindings']['inputs'], **validation['bindings']['implementation']}
    else:
        validation = evaluation_context.audit([directory], split, lock)
        verified = validation['input_sha256']
    for name, expected in inputs.items():
        if name in verified and verified[name] != expected:
            raise ValueError('accounting scope changed during validation')
    inputs.update(verified)
    methods, per_query = {}, []
    for name in names:
        path = directory / f'{name}.jsonl'
        bound = str(path.resolve())
        row_bytes = path.read_bytes()
        if hashlib.sha256(row_bytes).hexdigest() != inputs[bound]:
            raise ValueError('validated rows changed before accounting')
        rows = [json.loads(line) for line in row_bytes.splitlines()]
        methods[name] = {'accounting': summarize_calls(rows), 'operational': summarize(rows, meta['query_ids'])}
        per_query.extend({'method': name, 'query_id': row['query_id'], **summarize_calls([row])} for row in rows)
    _recheck(inputs)
    return {'schema': 'devsupport.provider-accounting.v1', 'complete': True, 'split': split,
            'phase': meta['phase'], 'controls': meta.get('controls'), 'query_ids': meta['query_ids'],
            'contract_sha256': meta['contract_sha256'], 'methods': methods, 'per_query': per_query,
            'input_sha256': inputs, 'model_calls': 0,
            'notes': ['Stage invocations can stop before transport; they are not billed model requests.',
                      'Missing transport counts stay unknown; known attempts are only recorded evidence.',
                      'Response usage is not total billed tokens; retries are not reconstructed as complete usage.',
                      'Currency costs are configured-rate estimates, not verified payment or all-in GPU costs.',
                      'No balances, credentials or model requests; raw provider text is excluded from this output.']}


def main() -> None:
    """Publish a new complete offline accounting report without provider interaction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('accounting output exists; never overwrite')
    report = audit(args.run, args.split, args.lock)
    _recheck(report['input_sha256'])
    write_new(args.output / 'summary.json', report)
    print(json.dumps({'complete': True, 'questions': len(report['query_ids']), 'model_calls': 0,
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
