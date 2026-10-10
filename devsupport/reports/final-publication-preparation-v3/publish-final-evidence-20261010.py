"""Publish complete frozen TEST and DEV-control evidence, offline and exclusively."""
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('F:/DevSupportCopilot')
REPORTS = ROOT / 'devsupport/reports'
OUTPUT = REPORTS / 'public-final-completed-v2'
CONFIG = ROOT / 'devsupport/configs/evaluation.final.v2.json'
LOCK = REPORTS / 'final-evaluation-lock-v2.json'
sys.path.insert(0, str(ROOT / 'devsupport/src'))
sys.path.insert(0, str(ROOT / 'devsupport/analysis'))
import answer_quality  # noqa: E402
from devsupport.credentials import load_credential  # noqa: E402
from devsupport.evaluate import build_contract, digest, verify_lock, write_new  # noqa: E402
from devsupport.review import load_controls  # noqa: E402

spec = importlib.util.spec_from_file_location('public_records', ROOT / 'devsupport/analysis/public_records.py')
public_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_module)


def bind(path, bindings, raw=None, expected=None):
    """Bind consumed bytes and reject drift or conflicting external receipts."""
    path = Path(path).resolve()
    raw = path.read_bytes() if raw is None else raw
    actual = hashlib.sha256(raw).hexdigest()
    key = str(path)
    if (expected is not None and expected != actual) or (key in bindings and bindings[key] != actual):
        raise ValueError('consumed source changed or receipt conflicts')
    bindings[key] = actual
    return raw


def read(path, bindings):
    """Parse precisely the bytes captured by the evidence receipt."""
    return json.loads(bind(path, bindings).decode('utf-8-sig'))


def checksum(path):
    """Bind the exact local bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_automated_completion(bindings: dict, locked: str, generation_ids: list[str]) -> None:
    """Require terminal completion and preserve paid rows across the actual continuation chain."""
    complete_stage = 'automated_test_complete_needs_ui_video_public_delivery'
    original = read(REPORTS / 'final-workflow-controller-20261010-v2.json', bindings)
    if original.get('final_contract_sha256') != locked:
        raise ValueError('controller contract differs from the frozen evaluation')
    if original.get('stage') == complete_stage:
        return
    if original.get('stage') != 'stopped_exception':
        raise ValueError('automated completion is not established')
    continued = [read(REPORTS / f'final-workflow-resume-20261010-{version}.json', bindings)
                 for version in ('v3', 'v4')]
    for state, stage, count in zip(continued, ('stopped_exception', complete_stage), (19, 20)):
        if (state.get('stage') != stage or state.get('contract_sha256') != locked
                or state.get('preserved_B0_rows') != count
                or state.get('failed_attempts_retained') is not True):
            raise ValueError('continuation completion, contract or retained scope is invalid')
    raw = bind(REPORTS / 'test-generation-v2/B0.jsonl', bindings)
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if [row.get('query_id') for row in rows] != generation_ids:
        raise ValueError('continuation requires the complete ordered generation cohort')
    for state, count in zip(continued, (19, 20)):
        prefix = rows[:count]
        if (hashlib.sha256(b''.join(lines[:count])).hexdigest() != state.get('preserved_B0_prefix_sha256')
                or [row.get('record_sha256') for row in prefix] != state.get('preserved_B0_record_sha256')
                or any(row.get('record_sha256') != digest({k: v for k, v in row.items() if k != 'record_sha256'})
                       for row in prefix)):
            raise ValueError('preserved billed prefix changed during continuation')


def collect():
    """Require complete frozen scope before any record is projected or published."""
    bindings = {}
    bind(CONFIG, bindings)
    contract = build_contract(CONFIG, ROOT)
    locked = verify_lock(LOCK, contract)
    if read(LOCK, bindings) != {'contract': contract, 'sha256': locked}:
        raise ValueError('lock differs from the exact consumed contract')
    selected = read(REPORTS / 'final-dev-selection-v1.json', bindings)
    if (selected['selected_method'] != 'H0' or selected['generation_methods'] != ['B0', 'H1', 'H0']
            or selected['adaptive_holdout_methods'] != ['Q', 'QAlways']
            or selected['control_methods'] != ['H0', 'G']):
        raise ValueError('pre-TEST selection differs from the approved frozen methods')
    verify_automated_completion(bindings, locked, contract['generation_ids'])
    bind(Path(contract['paths']['controls']), bindings)
    controls = load_controls(Path(contract['paths']['controls']))
    control_ids = {split: [r['query_id'] for r in controls if r['split'] == split] for split in ('dev', 'test')}
    retrieval_ids = contract['split_ids']['test']
    generation_ids = contract['generation_ids']
    if (len(retrieval_ids) != 150 or generation_ids != retrieval_ids[:30]
            or len(control_ids['test']) != 30 or len(control_ids['dev']) != 10):
        raise ValueError('approved TEST scope differs from frozen contract')
    groups = [
        ('test-local', 'test-local-v2', 'retrieval', False, ['B0', 'B1', 'H0', 'H1', 'C'], retrieval_ids),
        ('test-native', 'test-native-u-v2', 'retrieval', False, ['U'], retrieval_ids),
        ('test-rerank', 'test-rerank-v2', 'rerank', False, ['R1', 'R2'], retrieval_ids),
        ('test-adaptive', 'test-adaptive-v2', 'retrieve-only', False, selected['adaptive_holdout_methods'], retrieval_ids),
        ('test-generation', 'test-generation-v2', 'pipeline', False, selected['generation_methods'], generation_ids),
        ('test-controls', 'test-controls-v2', 'pipeline', True, selected['control_methods'], control_ids['test']),
        ('dev-controls', 'dev-decision-controls-v2', 'pipeline', True, ['H0', 'Q', 'G', 'Full'], control_ids['dev']),
    ]
    items = []
    for label, directory, phase, is_controls, methods, ids in groups:
        folder = REPORTS / directory
        metadata = read(folder / 'run.json', bindings)
        split = 'dev' if label == 'dev-controls' else 'test'
        if (metadata['query_ids'] != ids or metadata['profiles'] != methods or metadata['phase'] != phase
                or metadata['controls'] is not is_controls or metadata['split'] != split
                or list(folder.glob('*.inflight.json'))):
            raise ValueError('incomplete, mixed, reordered or unresolved original scope')
        for method in methods:
            manifest_path = folder / f'{method}.manifest.json'
            manifest = read(manifest_path, bindings)
            source_contract = manifest['evaluation_contract']
            settings = source_contract['profiles'][method]
            if (manifest['query_ids'] != ids or manifest['profile'] != method
                    or manifest['profiles'] != methods or manifest['settings'] != settings
                    or manifest['variant'] != (settings['reranker'] or settings['retrieval'])
                    or manifest['contract_sha256'] != digest(source_contract)
                    or manifest['contract_sha256'] != metadata['contract_sha256']
                    or manifest['phase'] != phase or manifest['split'] != split
                    or manifest['controls'] is not is_controls):
                raise ValueError('original method manifest identity mismatch')
            if split == 'test' and (metadata['contract_sha256'] != locked
                                   or manifest['evaluation_contract'] != contract):
                raise ValueError('TEST source is outside the final frozen contract')
            if split == 'dev' and (
                    source_contract['split_ids']['dev'] != contract['split_ids']['dev']
                    or source_contract['artifacts']['controls'] != contract['artifacts']['controls']
                    or source_contract['paths']['controls'] != contract['paths']['controls']):
                raise ValueError('DEV split or controls identity changed')
            items.append((f'{label}-{method}', folder / f'{method}.jsonl', method, ids))
    assessment = REPORTS / 'test-ai-judge-v2'
    summary_path = assessment / 'summary-030.json'
    summary = read(summary_path, bindings)
    if (summary.get('complete') is not True or summary.get('judged_questions') != 30
            or (assessment / 'inflight.json').exists()):
        raise ValueError('complete same-cohort TEST assessment required')
    quality = answer_quality.audit(REPORTS / 'test-generation-v2', assessment, 'test')
    saved_quality_path = REPORTS / 'test-answer-quality-v2/summary.json'
    if quality != read(saved_quality_path, bindings) or quality['generation_contract_sha256'] != locked:
        raise ValueError('assessment must match the exact frozen generation and saved complete quality audit')
    for path, expected in quality['input_sha256'].items():
        bind(path, bindings, expected=expected)
    items.append(('test-assessment', assessment / 'results.jsonl', 'AI assessment', generation_ids))
    return contract, locked, items, bindings


def main():
    """Preflight the entire set before creating one exclusive public directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError('final public output exists; never overwrite')
    contract, locked, items, bindings = collect()
    credentials = tuple(load_credential(ROOT / '.secrets' / name) for name in ('deepseek.dpapi', 'internal.dpapi'))
    for label, path, method, ids in items:
        raw = bind(path, bindings)
        rows = [json.loads(line) for line in raw.splitlines()]
        if [r['query_id'] for r in rows] != ids:
            raise ValueError('all ordered attempted records required; never publish a successful subset')
        for row in rows:
            field = 'sha256' if method == 'AI assessment' else 'record_sha256'
            if row.get(field) != digest({k: v for k, v in row.items() if k != field}):
                raise ValueError('original record digest mismatch')
        projected = [public_module.project_record(row, bindings[str(path.resolve())],
                     hashlib.sha256(line).hexdigest(), method) for row, line in zip(rows, raw.splitlines())]
        public_module._scan(json.dumps(projected, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                            credentials, projected, check_key_pattern=False)
        for row in projected:
            public_module._redact_query_keys(row)
        public_module._scan(json.dumps(projected, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                            credentials, projected)
    for path, expected in bindings.items():
        if checksum(path) != expected:
            raise ValueError('consumed source changed before publication')
    verify_lock(LOCK, build_contract(CONFIG, ROOT))
    if args.preflight:
        print(json.dumps({'preflight': True, 'cohorts': len(items),
                          'records': sum(len(ids) for _, _, _, ids in items), 'paid_calls': 0}))
        return
    receipts = {label: public_module.publish(path, OUTPUT / label, method, credentials, redact_query_keys=True)
                for label, path, method, _ in items}
    for label, path, _, _ in items:
        if receipts[label]['source_sha256'] != bindings[str(path.resolve())]:
            raise ValueError('publication receipt differs from captured source bytes')
        if (checksum(path) != receipts[label]['source_sha256']
                or checksum(OUTPUT / label / 'records.jsonl') != receipts[label]['output_sha256']):
            raise ValueError('original/public byte verification failed')
    for path, expected in bindings.items():
        if checksum(path) != expected:
            raise ValueError('consumed source changed during publication')
    verify_lock(LOCK, build_contract(CONFIG, ROOT))
    report = {'schema': 'devsupport.final-publication.v2', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'final_contract_sha256': locked, 'cohorts': len(items),
              'records': sum(len(ids) for _, _, _, ids in items), 'publications': receipts,
              'input_sha256': bindings, 'script_sha256': checksum(Path(__file__)),
              'helper_sha256': checksum(Path(public_module.__file__)), 'credential_literal_count': len(credentials),
              'credential_matches': 0, 'paid_calls': 0, 'human_ratings': 0,
              'all_attempts_retained': True, 'private_sources_unchanged': True,
              'final_method_aliases': contract['config']['final_method_aliases'],
              'acceptance_scope': 'Frozen ordered cohorts and projection integrity; not UI/video/archive acceptance',
              'reused_dev_public_views': ['public-dev-completed-v1', 'public-native-modern-dev-preflight-v1']}
    write_new(OUTPUT / 'summary.json', report)
    print(json.dumps({k: report[k] for k in ('cohorts', 'records', 'credential_matches', 'paid_calls')}))


if __name__ == '__main__':
    main()
