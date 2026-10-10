"""Offline decision-proxy tradeoffs from complete frozen pipeline control artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from devsupport.data import sha256_file
from devsupport.evaluate import digest, make_record, text_hash, write_new
from devsupport.report import outcome
from devsupport.review import load_controls
from devsupport.types import Question

LABELS = ('answer', 'insufficient', 'clarify')
OBSERVED = (*LABELS, 'operational_error')
NOTES = [
    'Label verification is not established by this tool. Frozen expected_decision labels are decision proxies, '
    'not human-verified answer faithfulness.',
    'unsupported_control_answer_rate is NOT a hallucination rate or evidence-level unsupported-claim metric.',
    'Every attempted control, including operational errors, stays in its applicable denominator. Errors take '
    'precedence over provisional decisions and are not counted as refusal. Fallback events count rows and '
    'can overlap terminal errors; they do not automatically mean terminal failure.',
    'All rates include raw numerator and denominator; zero denominators are null. No human verification, '
    'bootstrap, composite score or automatic relabeling is inferred.',
    'Hashes bind consumed bytes and analysis implementation, not external authenticity. Historical runtime '
    'source hashes need not match current source. Different configured profiles may have different runtime hashes.',
]


def metric(numerator: int, denominator: int) -> dict:
    """Expose raw counts and make an empty denominator explicitly unavailable."""
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


def safe_name(value: object) -> bool:
    """Use the existing reporter's safe profile filename convention."""
    return isinstance(value, str) and value.replace('-', '').replace('_', '').isalnum()


def read_bound(path: Path, inputs: dict, jsonl: bool = False) -> object:
    """Parse exactly the input bytes whose hash is recorded, failing closed on missing data."""
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f'missing input artifact: {path.name}')
    raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode('utf-8-sig')
        return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(text)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f'invalid input artifact: {path.name}') from exc


def validate_profiles(contract: dict, names: list[str]) -> None:
    """Check frozen method settings against recorded configuration without loading runtime models."""
    frozen = contract['profiles']
    configured = contract['config']['profiles']
    if not isinstance(frozen, dict) or not isinstance(configured, dict) or set(frozen) != set(configured):
        raise ValueError('profile configuration mismatch')
    if not set(names) <= set(frozen):
        raise ValueError('profile list not in frozen contract')
    allowed = {'retrieval', 'reranker', 'adaptive', 'gate_enabled'}
    for name, settings in configured.items():
        if not safe_name(name) or not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError('invalid profile configuration')
        normalized = {'retrieval': 'H1', 'reranker': None, 'adaptive': 'never', 'gate_enabled': False, **settings}
        if (frozen[name] != normalized or normalized['retrieval'] not in ('B0', 'B1', 'H0', 'H1', 'U', 'C')
                or normalized['reranker'] not in (None, 'R1', 'R2')
                or (normalized['reranker'] and normalized['retrieval'] != 'H1')
                or normalized['adaptive'] not in ('never', 'always', 'selective')
                or type(normalized['gate_enabled']) is not bool):
            raise ValueError('profile configuration mismatch')


def recheck(bindings: dict, directory: Path) -> None:
    """Verify consumed bytes and absence of any in-flight marker before publication."""
    for group in ('inputs', 'implementation'):
        for filename, expected in bindings[group].items():
            path = Path(filename)
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f'consumed {group} changed: {path.name}')
    if any(directory.glob('*.inflight.json')):
        raise ValueError('in-flight run cannot be reported')


def summarize_profile(name: str, rows: list[dict], controls: list[dict]) -> tuple[dict, list[dict]]:
    """Classify operational errors first and retain all expected-label denominators."""
    confusion = {label: dict.fromkeys(OBSERVED, 0) for label in LABELS}
    per_query = []
    known_hashes = set()
    for row, control in zip(rows, controls):
        flags = outcome(row)
        if not flags['error'] and row['result'].get('decision') not in LABELS:
            raise ValueError('non-error decision must be answer/insufficient/clarify')
        observed = 'operational_error' if flags['error'] else row['result']['decision']
        runtime_hash = row['result'].get('runtime_hash')
        if runtime_hash is not None and (not isinstance(runtime_hash, str) or not runtime_hash.strip()):
            raise ValueError('invalid runtime hash')
        if not runtime_hash and not row['error']:
            raise ValueError('runtime hash missing without top-level runner error')
        if runtime_hash:
            known_hashes.add(runtime_hash)
        expected = control['expected_decision']
        confusion[expected][observed] += 1
        per_query.append({'query_id': row['query_id'], 'profile': name, 'expected_label': expected,
                          'observed_outcome': observed, 'error': flags['error'], 'fallback': flags['fallback'],
                          'top_level_error': bool(row['error']), 'runtime_identity_available': bool(runtime_hash)})
    if len(known_hashes) > 1:
        raise ValueError('inconsistent known runtime identity within profile')
    attempted = len(rows)
    expected_counts = {label: sum(confusion[label].values()) for label in LABELS}
    answers = sum(confusion[label]['answer'] for label in LABELS)
    unsupported = confusion['insufficient']['answer']
    false_refusal = confusion['answer']['insufficient']
    unnecessary = confusion['answer']['clarify']
    counts = {'operational_error': sum(r['error'] for r in per_query),
              'fallback_event': sum(r['fallback'] for r in per_query),
              'runtime_identity_unavailable': sum(not r['runtime_identity_available'] for r in per_query)}
    metrics = {
        'decision_accuracy': metric(sum(confusion[label][label] for label in LABELS), attempted),
        'answer_coverage': metric(answers, attempted),
        'unsupported_control_answer_rate': metric(unsupported, expected_counts['insufficient']),
        'unsupported_control_share_of_answers': metric(unsupported, answers),
        'false_refusal_insufficient_rate': metric(false_refusal, expected_counts['answer']),
        'unnecessary_clarification_rate': metric(unnecessary, expected_counts['answer']),
        'false_nonanswer_rate': metric(false_refusal + unnecessary, expected_counts['answer']),
        'correct_clarification_rate': metric(confusion['clarify']['clarify'], expected_counts['clarify']),
        **{key + '_rate': metric(value, attempted) for key, value in counts.items()},
    }
    return {'planned': len(controls), 'attempted': attempted, 'expected_counts': expected_counts,
            'operational_errors': counts['operational_error'], 'fallback_events': counts['fallback_event'],
            'runtime_identity_unavailable': counts['runtime_identity_unavailable'],
            'runtime_hash': next(iter(known_hashes), None), 'confusion_matrix': confusion, 'metrics': metrics}, per_query


def audit(directory: Path) -> dict:
    """Validate complete recorded control cohorts; never open FreshStack queries or provider credentials."""
    directory = Path(directory).resolve()
    implementation = [Path(__file__).resolve(), *[
        Path(__file__).resolve().parents[1] / 'src/devsupport' / (name + '.py')
        for name in ('data', 'evaluate', 'metrics', 'report', 'review', 'types')]]
    bindings = {'inputs': {}, 'implementation': {str(path): sha256_file(path) for path in implementation}}
    inputs = bindings['inputs']
    try:
        run = read_bound(directory / 'run.json', inputs)
        names = run['profiles']
        if (not isinstance(names, list) or not names or any(not safe_name(name) for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('profiles must be unique valid names')
        if run.get('controls') is not True or run.get('phase') != 'pipeline' or run.get('split') not in ('dev', 'test'):
            raise ValueError('requires controls=true, pipeline phase and dev/test split')
        contract, selected, questions = None, None, None
        methods, per_query = {}, []
        for name in names:
            path = directory / f'{name}.jsonl'
            meta = read_bound(path.with_suffix('.manifest.json'), inputs)
            frozen = meta['evaluation_contract']
            if (digest(frozen) != run['contract_sha256'] or any(meta.get(k) != v for k, v in run.items())
                    or (contract is not None and frozen != contract)):
                raise ValueError('mixed or tampered evaluation contract')
            if contract is None:
                contract = frozen
                validate_profiles(contract, names)
                controls_path = Path(contract['paths']['controls']).resolve()
                read_bound(controls_path, inputs)
                if inputs[str(controls_path)] != contract['artifacts']['controls']:
                    raise ValueError('control annotations changed')
                controls = load_controls(controls_path)
                selected = [r for r in controls if r['split'] == run['split']]
                ids = [r['query_id'] for r in selected]
                if (len(ids) != (10 if run['split'] == 'dev' else 30) or len(set(ids)) != len(ids)
                        or any(not isinstance(i, str) or not i for i in ids) or ids != run['query_ids']):
                    raise ValueError('incomplete or reordered frozen control cohort')
                questions = [Question(r['query_id'], '', r['query'], '', r.get('nuggets', []), {}) for r in selected]
                bindings['contract_sha256'] = run['contract_sha256']
                bindings['historical_source'] = contract.get('source', {})
            settings = contract['profiles'][name]
            variant = settings.get('reranker') or settings['retrieval']
            artifact_key = {'U': 'native_u_index', 'C': 'contextual_index'}.get(settings['retrieval'], 'index')
            if (meta.get('profile') != name or meta.get('settings') != settings or meta.get('variant') != variant
                    or meta.get('questions_sha256') != contract['artifacts']['queries']
                    or meta.get('index_manifest_sha256') != contract['artifacts'][artifact_key]
                    or meta.get('context_counter') != contract['context_counter']
                    or meta.get('latency_scope') != 'pipeline'
                    or meta.get('query_hashes') != [text_hash(q.query) for q in questions]):
                raise ValueError('profile/manifest provenance mismatch')
            rows = read_bound(path, inputs, jsonl=True)
            if [row['query_id'] for row in rows] != run['query_ids']:
                raise ValueError('incomplete, duplicate, unknown or reordered rows')
            for row, question in zip(rows, questions):
                result = row['result']
                if (('variant' in result and result['variant'] != variant)
                        or ('evaluation_phase' in result and result['evaluation_phase'] != 'pipeline')
                        or 'ranked_ids' in result):
                    raise ValueError('result phase/variant mismatch')
                if row != make_record(question, result, row['elapsed_seconds'], row['error']):
                    raise ValueError('result integrity mismatch')
            methods[name], query_rows = summarize_profile(name, rows, selected)
            per_query.extend(query_rows)
        recheck(bindings, directory)
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError('invalid or missing control artifact structure') from exc
    return {'schema': 1, 'run': run, 'label_verification': 'not established', 'notes': NOTES,
            'methods': methods, 'per_query': per_query, 'bindings': bindings}


def markdown(report: dict) -> str:
    """Render compact counts and metric denominators alongside the full JSON confusion tables."""
    lines = ['# Control decision proxies', '', *report['notes'], '',
             '| Profile | Planned | Attempted | Operational errors | Fallback events | Runtime unavailable |',
             '|---|---:|---:|---:|---:|---:|']
    for name, method in report['methods'].items():
        values = [method[key] for key in ('planned', 'attempted', 'operational_errors', 'fallback_events',
                                         'runtime_identity_unavailable')]
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '| Profile | Metric | Numerator | Denominator | Rate |', '|---|---|---:|---:|---:|']
    for name, method in report['methods'].items():
        for key, value in method['metrics'].items():
            rate = 'null' if value['rate'] is None else f"{value['rate']:.6f}"
            lines.append(f"| {name} | {key} | {value['numerator']} | {value['denominator']} | {rate} |")
    lines += ['', 'Full expected-by-observed confusion matrices, per-query IDs/labels/outcomes/flags and '
              'consumed input and implementation hashes are in summary.json.']
    return '\n'.join(lines) + '\n'


def publish(directory: Path, output: Path) -> None:
    """Validate and recheck inputs before creating an exclusively new output directory."""
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError('output must be a new directory')
    report = audit(directory)
    rendered = markdown(report)
    recheck(report['bindings'], directory)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'summary.json', report)
    with (output / 'summary.md').open('x', encoding='utf-8') as handle:
        handle.write(rendered)


def main() -> None:
    """Accept only recorded run artifacts and a new report destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    publish(args.run, args.output)


if __name__ == '__main__':
    main()
