"""Offline DEV counterfactual only; never imported by the production pipeline."""
# ruff: noqa: E402 -- standalone CLI bootstraps the isolated package before imports.
import argparse
import ast
import inspect
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'devsupport/src'))

from devsupport import answer, llm
from devsupport.answer import citation_prose, validate_answer
from devsupport.data import sha256_file
from devsupport.evaluate import digest, read_json, text_hash, write_new
from devsupport.report import outcome
from devsupport.types import Evidence, Hit

PROFILES = ('B0', 'H0', 'H1')
SCOPE = ('Post-hoc DEV counterfactual on stored rejected responses only. Original outcomes are unchanged. '
         'Potential earlier acceptance is not measured repair savings, cost, faithfulness, human quality or TEST gain.')


def normalize_answer(text: str, packed: list) -> tuple[str, list[str]]:
    """Return a strict-valid candidate without changing factual answer text."""
    try:
        validate_answer(text, packed)
        return text, []
    except ValueError:
        pass
    body = llm.parse_json(text)
    if set(body) != {'answer_text', 'citation_ids'}:
        raise ValueError('not_normalizable')
    prose, ids = body['answer_text'], body['citation_ids']
    if (not isinstance(prose, str) or not prose.strip() or not isinstance(ids, list) or not ids
            or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids))):
        raise ValueError('not_normalizable')
    context_ids = {hit.evidence.doc_id for hit in packed}
    if not set(ids) <= context_ids:
        raise ValueError('not_normalizable')
    candidates = re.findall(r'\[(\d[^\]]*)(\]|$)', citation_prose(prose))
    if not candidates or any(not closing or not re.fullmatch(r'\d+', value) for value, closing in candidates):
        raise ValueError('not_normalizable')
    numbers = [int(value) for value, _ in candidates]
    if any(n < 1 or n > len(packed) for n in numbers):
        raise ValueError('not_normalizable')
    cited = {packed[n - 1].evidence.doc_id for n in numbers}
    if not cited < set(ids):
        raise ValueError('not_normalizable')
    removed = [i for i in ids if i not in cited]
    candidate = json.dumps({**body, 'citation_ids': [i for i in ids if i in cited]}, ensure_ascii=False)
    validated = validate_answer(candidate, packed)
    if validated['answer_text'] != prose:
        raise ValueError('answer text changed')
    return candidate, removed


def audit(run_path: Path, output: Path, runtime_root: Path) -> dict:
    """Inspect only completed bound DEV artifacts and write an exclusive report."""
    run_path, output, runtime_root = Path(run_path), Path(output), Path(runtime_root)
    if output.exists():
        raise FileExistsError(output)
    inputs = {}

    def read(path: Path) -> dict:
        inputs[str(path.resolve())] = sha256_file(path)
        return read_json(path)

    run = read(run_path / 'run.json')
    ids = run['query_ids']
    if (run.get('split') != 'dev' or run.get('phase') != 'pipeline' or run.get('controls') is not False
            or len(ids) != 12 or len(set(ids)) != 12 or run['profiles'][:3] != list(PROFILES)):
        raise ValueError('expected fixed ordered DEV12 pipeline cohort and B0/H0/H1')
    methods = {}
    for name in PROFILES:
        meta = read(run_path / f'{name}.manifest.json')
        contract = meta['evaluation_contract']
        settings = contract['profiles'][name]
        if (digest(contract) != run['contract_sha256'] or any(meta.get(k) != v for k, v in run.items())
                or contract['split_ids']['dev'] != ids or meta.get('profile') != name
                or meta.get('settings') != settings or settings.get('retrieval') != name
                or settings.get('reranker') is not None or settings.get('adaptive') != 'never'
                or meta.get('variant') != name or len(meta.get('query_hashes', [])) != len(ids)):
            raise ValueError('cohort or profile contract mismatch')
        for module in ('answer', 'llm'):
            relative = f'devsupport/src/devsupport/{module}.py'
            path = runtime_root / relative
            inputs[str(path.resolve())] = sha256_file(path)
            if inputs[str(path.resolve())] != contract['source'][relative]:
                raise ValueError('archived runtime source mismatch')
            source = path.read_text(encoding='utf-8')
            if module == 'answer':
                if source != Path(answer.__file__).read_text(encoding='utf-8'):
                    raise ValueError('executing strict answer validator differs')
            else:
                original = next(node for node in ast.parse(source).body
                                if isinstance(node, ast.FunctionDef) and node.name == 'parse_json')
                executing = ast.parse(inspect.getsource(llm.parse_json)).body[0]
                if ast.dump(original) != ast.dump(executing):
                    raise ValueError('executing JSON parser differs')
        path = run_path / f'{name}.jsonl'
        inputs[str(path.resolve())] = sha256_file(path)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [row['query_id'] for row in rows] != ids:
            raise ValueError('incomplete or reordered DEV cohort')
        for row, query_hash in zip(rows, meta['query_hashes']):
            if (row.get('record_sha256') != digest({k: v for k, v in row.items() if k != 'record_sha256'})
                    or row['query_sha256'] != query_hash or row['result'].get('variant', name) != name):
                raise ValueError('record integrity or variant mismatch')
        methods[name] = analyze_rows(rows)
    for path, expected in inputs.items():
        if sha256_file(path) != expected:
            raise ValueError('input changed during audit')
    report = {'scope': SCOPE, 'query_ids': ids, 'profiles': list(PROFILES), 'input_sha256': inputs,
              'analysis_sha256': sha256_file(__file__),
              'executing_source_sha256': {module: sha256_file(path) for module, path in
                                         [('answer', answer.__file__), ('llm', llm.__file__)]}, 'methods': methods}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'audit.json', report)
    return report


def analyze_rows(rows: list[dict]) -> dict:
    """Keep actual outcomes separate from final and earlier-attempt counterfactuals."""
    result = {'actual_successes': 0, 'actual_operational_errors': 0, 'actual_final_contract_failures': 0,
              'counterfactual_final_recoverable': 0, 'potential_earlier_acceptance_attempts': 0, 'queries': []}
    for row in rows:
        payload = row['result']
        context = payload.get('context', [])
        if ([c['doc_id'] for c in context] != row['packed_ids']
                or payload.get('evidence_ids', []) != row['packed_ids']
                or len(set(row['packed_ids'])) != len(context)
                or [c.get('citation_number') for c in context] != list(range(1, len(context) + 1))
                or row['context_sha256'] != {c['doc_id']: text_hash(c['text']) for c in context}):
            raise ValueError('context identity/order mismatch')
        packed = [Hit(Evidence(c['doc_id'], c['text'], c['metadata']), 0, n, 'audit')
                  for n, c in enumerate(context, 1)]
        trace = payload.get('trace', {})
        generation = trace.get('generation', {})
        flags = outcome(row)
        excluded = bool(row.get('error') or trace.get('excluded_failure')
                        or any(call.get('error') for call in trace.get('provider_calls', []))
                        or generation.get('error') == 'provider_failure'
                        or trace.get('gate', {}).get('reason') == 'gate_validation_or_provider_failure')
        failures = generation.get('validation_failures', [])
        attempts = generation.get('attempts', 0)
        if type(attempts) is not int or not 0 <= attempts <= 2 or not isinstance(failures, list):
            raise ValueError('invalid archived generation attempts')
        final_failure = (not excluded and flags['error'] and payload.get('decision') != 'answer'
                         and generation.get('error') == 'citation_or_schema_validation_failed'
                         and len(failures) == attempts and attempts > 0)
        result['actual_successes'] += int(flags['success'])
        result['actual_operational_errors'] += int(flags['error'])
        result['actual_final_contract_failures'] += int(final_failure)
        inspected = []
        for index, failure in enumerate(failures, 1):
            text = failure['response_text']
            # The producer stores text[:16384] without an independent completeness flag.
            # A full buffer could hide invalid trailing content, even if its JSON parses.
            if not isinstance(text, str) or len(text) >= 16384 or index > attempts:
                raise ValueError('invalid archived failure bounds')
            item = {'attempt': index, 'stage': 'generation' if index == 1 else 'citation_repair',
                    'text_sha256': text_hash(text), 'would_validate_after_pruning': False, 'removed_ids': []}
            try:
                body = llm.parse_json(text)
                if isinstance(body.get('answer_text'), str):
                    item['answer_text_sha256'] = text_hash(body['answer_text'])
            except ValueError:
                pass
            try:
                normalized, removed = normalize_answer(text, packed)
                if not removed:
                    raise ValueError('archived rejected text already validates')
            except ValueError:
                pass
            else:
                item.update(would_validate_after_pruning=True, removed_ids=removed,
                            normalized_text_sha256=text_hash(normalized),
                            normalized_answer_text_sha256=text_hash(llm.parse_json(normalized)['answer_text']))
            item['potential_earlier_acceptance'] = bool(
                not excluded and index < attempts and item['would_validate_after_pruning'])
            result['potential_earlier_acceptance_attempts'] += int(item['potential_earlier_acceptance'])
            inspected.append(item)
        recoverable = bool(final_failure and inspected and inspected[-1]['would_validate_after_pruning'])
        result['counterfactual_final_recoverable'] += int(recoverable)
        result['queries'].append({'query_id': row['query_id'], 'record_sha256': row['record_sha256'],
                                  'actual_outcome': flags, 'excluded_operational_failure': excluded,
                                  'actual_final_contract_failure': bool(final_failure),
                                  'counterfactual_final_recoverable': recoverable, 'stored_attempts': inspected})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.output, args.runtime_root)
    print(json.dumps({name: {k: v for k, v in method.items() if k != 'queries'}
                      for name, method in report['methods'].items()}, indent=2))
