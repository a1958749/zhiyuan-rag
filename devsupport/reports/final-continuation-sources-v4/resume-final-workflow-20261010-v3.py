"""Continue only unattempted frozen v2 work after a verified terminal provider stop."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('F:/DevSupportCopilot')
REPORTS = ROOT / 'devsupport/reports'
CORE = Path('F:/DevSupportCopilotWork/.venv-core-repro/Scripts/python.exe')
CONFIG = ROOT / 'devsupport/configs/evaluation.final.v2.json'
LOCK = REPORTS / 'final-evaluation-lock-v2.json'
STATE = REPORTS / 'final-workflow-resume-20261010-v3.json'
sys.path.insert(0, str(ROOT / 'devsupport/src'))
from devsupport.evaluate import build_contract, digest, verify_lock, write_new

spec = importlib.util.spec_from_file_location('previous_controller',
    Path(__file__).with_name('final-workflow-controller-20261010-v2.py'))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def preflight():
    old = read(REPORTS / 'final-workflow-controller-20261010-v2.json')
    if old['stage'] != 'stopped_exception' or previous.process_live(old['pid']):
        raise ValueError('previous controller must be verifiably terminal')
    contract = build_contract(CONFIG, ROOT)
    verify_lock(LOCK, contract)
    if digest(contract) != old['final_contract_sha256']:
        raise ValueError('original v2 contract changed')
    previous.health()
    ids = contract['split_ids']['test']
    previous.complete_run(REPORTS / 'test-local-v2', ids, ['B0', 'B1', 'H0', 'H1', 'C'])
    previous.complete_run(REPORTS / 'test-native-u-v2', ids, ['U'])
    previous.complete_run(REPORTS / 'test-rerank-v2', ids, ['R1', 'R2'])
    generation = REPORTS / 'test-generation-v2'
    if list(generation.glob('*.inflight.json')):
        raise ValueError('unreconciled paid attempt')
    if read(generation / 'run.json') != {'contract_sha256': digest(contract), 'split': 'test',
        'phase': 'pipeline', 'profiles': ['B0', 'H1', 'H0'],
        'query_ids': contract['generation_ids'], 'controls': False}:
        raise ValueError('original generation scope changed')
    prefix = generation / 'B0.jsonl'
    rows = [json.loads(line) for line in prefix.read_bytes().splitlines()]
    if len(rows) != 19 or [r['query_id'] for r in rows] != contract['generation_ids'][:19]:
        raise ValueError('unexpected prior progress; inspect before resuming')
    return contract, prefix.read_bytes(), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    contract, prefix_bytes, prefix_rows = preflight()
    if not args.execute:
        print(json.dumps({'preflight': True, 'preserved_B0_rows': 19,
            'unattempted_generation_rows': 71, 'same_frozen_v2_contract': True,
            'local_retrieval_not_repeated': True, 'paid_calls': 0}))
        return
    state = {'pid': os.getpid(), 'stage': 'resuming_unattempted_v2_work',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'project_complete': False,
        'contract_sha256': digest(contract), 'preserved_B0_rows': 19,
        'preserved_B0_prefix_sha256': hashlib.sha256(prefix_bytes).hexdigest(),
        'preserved_B0_record_sha256': [r['record_sha256'] for r in prefix_rows],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reason': 'Prior controller terminal with incomplete generation after provider failure; '
            'balance is available, above existing reserve, no pacing lock or inflight call.',
        'failed_attempts_retained': True, 'no_paid_ui_calls': True}
    write_new(STATE, state)

    def save(stage):
        state.update(stage=stage, observed_at_utc=datetime.now(timezone.utc).isoformat())
        temp = STATE.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        os.replace(temp, STATE)
        print(json.dumps({'stage': stage}), flush=True)

    def command(stage, arguments):
        verify_lock(LOCK, build_contract(CONFIG, ROOT))
        save(stage)
        result = subprocess.run([str(CORE), '-X', 'utf8', '-u', *arguments], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT / 'devsupport/src'),
                'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}, check=False)
        state['last_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError('stage returned nonzero')

    def evaluation(stage, phase, profiles, folder, controls=False, resume=False):
        previous.health()
        argv = ['devsupport/scripts/evaluate_pipeline.py', '--config', str(CONFIG),
            '--split', 'test', '--lock', str(LOCK), '--phase', phase, '--profiles',
            *profiles, '--output', str(folder)]
        if controls:
            argv.append('--controls')
        if resume:
            argv.append('--resume')
        command(stage, argv)
        ids = contract['generation_ids'] if phase == 'pipeline' else contract['split_ids']['test']
        if controls:
            from devsupport.review import load_controls
            ids = [r['query_id'] for r in load_controls(Path(contract['paths']['controls']))
                if r['split'] == 'test']
        previous.complete_run(folder, ids, profiles)

    try:
        generation = REPORTS / 'test-generation-v2'
        evaluation('test_generation30_resume', 'pipeline', ['B0', 'H1', 'H0'], generation, resume=True)
        if not (generation / 'B0.jsonl').read_bytes().startswith(prefix_bytes):
            raise ValueError('previous billed rows changed')
        command('test_generation_context', ['devsupport/analysis/evaluation_context.py', '--split', 'test',
            '--lock', str(LOCK), '--runs', str(generation), '--output', str(REPORTS / 'test-generation-context-v2')])
        command('test_generation_accounting', ['devsupport/analysis/provider_accounting.py', '--split', 'test',
            '--lock', str(LOCK), '--run', str(generation), '--output', str(REPORTS / 'test-generation-accounting-v2')])
        command('test_generation_report', ['devsupport/scripts/report_evaluation.py', '--run', str(generation),
            '--output', str(REPORTS / 'test-generation-report-v2')])
        judge = REPORTS / 'test-ai-judge-v2'
        flags = ['--split', 'test', '--output-tokens', '8192', '--decision-label-protocol', 'explicit_coverage']
        command('test_judge_preflight', ['devsupport/analysis/answer_judge.py', '--run', str(generation),
            '--output', str(REPORTS / 'test-ai-judge-preflight-v2'), *flags])
        previous.health()
        command('test_judge_execute', ['devsupport/analysis/answer_judge.py', '--run', str(generation),
            '--output', str(judge), *flags, '--execute'])
        summary = read(judge / 'summary-030.json')
        if summary.get('complete') is not True or summary.get('judged_questions') != 30:
            raise ValueError('incomplete TEST AI assessment')
        command('test_answer_quality', ['devsupport/analysis/answer_quality.py', '--split', 'test',
            '--run', str(generation), '--assessment', str(judge), '--output', str(REPORTS / 'test-answer-quality-v2')])
        controls = REPORTS / 'test-controls-v2'
        evaluation('test_controls30', 'pipeline', ['H0', 'G'], controls, controls=True)
        command('test_control_report', ['devsupport/analysis/control_decisions.py', '--run', str(controls),
            '--output', str(REPORTS / 'test-controls-report-v2')])
        command('test_control_accounting', ['devsupport/analysis/provider_accounting.py', '--split', 'test',
            '--lock', str(LOCK), '--run', str(controls), '--output', str(REPORTS / 'test-controls-accounting-v2')])
        adaptive = REPORTS / 'test-adaptive-v2'
        evaluation('test_adaptive150', 'retrieve-only', ['Q', 'QAlways'], adaptive)
        command('test_all_context', ['devsupport/analysis/evaluation_context.py', '--split', 'test',
            '--lock', str(LOCK), '--runs', str(REPORTS / 'test-local-v2'), str(REPORTS / 'test-native-u-v2'),
            str(REPORTS / 'test-rerank-v2'), str(adaptive), '--output', str(REPORTS / 'test-context-v2')])
        command('test_adaptive_accounting', ['devsupport/analysis/provider_accounting.py', '--split', 'test',
            '--lock', str(LOCK), '--run', str(adaptive), '--output', str(REPORTS / 'test-adaptive-accounting-v2')])
        save('automated_test_complete_needs_ui_video_public_delivery')
    except Exception as exc:
        state['error_type'] = type(exc).__name__
        save('stopped_exception')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
