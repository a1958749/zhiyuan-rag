"""Verify complete offline AI assessments and compare paired all-attempt utility."""
import argparse
from dataclasses import asdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer_judge as judge  # noqa: E402
from devsupport.data import sha256_file  # noqa: E402
from devsupport.evaluate import digest, write_new  # noqa: E402
from devsupport.metrics import bootstrap_paired  # noqa: E402

ASSESSMENT = 'offline AI judge; not human review or independent model validation'
UTILITY_NOTE = ('Unweighted mean of five 0..3 scores. Operational failures contribute zero. '
                'Missing AI judgments keep full attempted utility unavailable.')


def _read_bound(path: Path, inputs: dict) -> bytes:
    """Hash the same byte snapshot consumed by this audit."""
    raw = path.read_bytes()
    inputs[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return raw


def _recheck(inputs: dict) -> None:
    """Reject a changed input before accepting or publishing the result."""
    for name, expected in inputs.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError('consumed input changed before publication')


def _summary(packets: list, mapping: dict, rows: list) -> dict:
    """Recompute the existing judge's complete summary, including every failure."""
    return {'assessment': ASSESSMENT, 'complete': True, 'recording_complete': True,
            'judged_questions': len(packets), 'recorded_questions': len(packets),
            'planned_questions': len(packets), 'methods': judge.summarize_judgments(packets, mapping, rows),
            'utility_note': UTILITY_NOTE}


def audit(run: Path, assessment: Path, split: str = 'dev') -> dict:
    """Validate recorded inputs and calculate complete-cohort paired AI statistics.

    Args:
        run: Complete generation directory with at least two comparison methods.
        assessment: Complete AI judge directory created from that exact generation.
        split: Explicit dev or test assessment; test requires the frozen first 30 IDs.

    Returns:
        Hash-bound descriptive statistics. AI score utility is not answer accuracy.

    Raises:
        ValueError: A consumed artifact is incomplete, changed or inconsistent.
    """
    run, assessment = Path(run), Path(assessment)
    if (assessment / 'inflight.json').exists():
        raise ValueError('unresolved judge inflight')
    try:
        analysis_sources = {**judge.source_contract(), 'analysis/answer_quality.py': sha256_file(Path(__file__))}
        inputs = {str((BASE / name).resolve()): expected for name, expected in analysis_sources.items()}
        meta = json.loads(_read_bound(assessment / 'manifest.json', inputs).decode('utf-8-sig'))
        recorded_inputs = meta['inputs']
        if not isinstance(recorded_inputs, dict) or not recorded_inputs:
            raise ValueError('invalid recorded generation inputs')
        _recheck(recorded_inputs)
        questions, variants, corpus, contract, generation_inputs = judge.load_run(run, split)
        if generation_inputs != recorded_inputs:
            raise ValueError('generation input changed during audit')
        inputs.update(recorded_inputs)
        ids = [q.query_id for q in questions]
        if split == 'test' and len(ids) != 30:
            raise ValueError('TEST requires all frozen first 30 questions')
        seed, tokens = meta['seed'], judge.validate_output_tokens(meta['output_tokens'])
        protocol = judge.validate_decision_label_protocol(meta.get('decision_label_protocol', 'legacy'))
        if type(seed) is not int:
            raise ValueError('invalid recorded seed')
        packets, mapping = judge.build_packets(questions, variants, corpus, seed)
        original = asdict(judge.ProviderConfig(**contract['provider']))
        expected_provider = asdict(judge.JudgeProviderConfig(**{**original, 'retries': 0,
                                                              'max_output_tokens': tokens}))
        if (meta.get('schema') != 1 or meta.get('assessment') != ASSESSMENT
                or meta.get('split') != split or meta.get('query_ids') != ids
                or meta.get('max_paid_calls') != len(ids)
                or meta.get('generation_contract_sha256') != digest(contract)
                or meta.get('inputs') != generation_inputs
                or meta.get('provider') != expected_provider
                or meta.get('provider_sha256') != digest(expected_provider)
                or meta.get('packets_sha256') != digest(packets)
                or meta.get('mapping_sha256') != digest(mapping)
                or meta.get('rubric_sha256') != digest(judge.RUBRIC)
                or meta.get('prompt_sha256') != digest(judge.system_prompt(tokens, protocol))
                or not isinstance(meta.get('sources'), dict) or not meta['sources']):
            raise ValueError('judge manifest provenance mismatch')
        if (json.loads(_read_bound(assessment / 'packets.json', inputs).decode('utf-8-sig')) != packets
                or json.loads(_read_bound(assessment / 'sealed-mapping.json', inputs).decode('utf-8-sig')) != mapping):
            raise ValueError('blinded packets or mapping changed')
        rows = [json.loads(line) for line in
                _read_bound(assessment / 'results.jsonl', inputs).splitlines()]
        if [row['query_id'] for row in rows] != ids:
            raise ValueError('complete ordered judge cohort required')
        for row, packet in zip(rows, packets):
            if (row.get('sha256') != digest({k: v for k, v in row.items() if k != 'sha256'})
                    or row.get('manifest_sha256') != digest(meta)
                    or row.get('packet_sha256') != digest(packet) or row.get('status') != 'ok'
                    or row.get('error') is not None):
                raise ValueError('invalid or failed judge record')
            judge.validate_review(row['reviews'], packet)
        summary = _summary(packets, mapping, rows)
        if json.loads(_read_bound(assessment / f'summary-{len(ids):03d}.json', inputs).decode('utf-8-sig')) != summary:
            raise ValueError('saved judge summary differs from recorded scores')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('missing or invalid assessment artifact') from exc

    methods = list(variants)
    utilities = {name: [] for name in methods}
    answered = {name: {key: [] for key in judge.DIMENSIONS} for name in methods}
    per_question = []
    for packet, row in zip(packets, rows):
        reviews = {r['review_id']: r for r in row['reviews']['reviews']}
        item = {'query_id': packet['query_id'], 'utilities': {}}
        for response in packet['responses']:
            name = mapping[response['review_id']]['method']
            utility = 0.0
            if not response['operational_failure']:
                scores = reviews[response['review_id']]['scores']
                utility = statistics.fmean(scores.values())
                if response['decision'] == 'answer':
                    for key in judge.DIMENSIONS:
                        answered[name][key].append(scores[key])
            utilities[name].append(utility)
            item['utilities'][name] = utility
        per_question.append(item)
    for name, target in summary['methods'].items():
        n = len(answered[name]['faithfulness'])
        target.update(answers=n, answer_coverage=n / len(ids), answered_quality_n=n,
                      answered_dimensions={k: statistics.fmean(v) if v else None for k, v in answered[name].items()})
    pairs = {f'{candidate}-minus-{baseline}': bootstrap_paired(utilities[baseline], utilities[candidate], seed)
             for baseline, candidate in combinations(methods, 2)}
    _recheck(inputs)
    return {'schema': 'devsupport.answer-quality.v1', 'assessment': ASSESSMENT, 'complete': True,
            'decision_label_protocol': protocol,
            'split': split, 'questions': len(ids), 'query_ids': ids, 'human_ratings': 0,
            'methods': summary['methods'], 'per_question': per_question, 'paired': pairs,
            'generation_contract_sha256': digest(contract), 'judge_manifest_sha256': digest(meta),
            'historical_judge_sources': meta['sources'], 'analysis_sources': analysis_sources,
            'input_sha256': inputs,
            'notes': ['All attempted queries paired; generation failures contribute zero utility.',
                      'AI-rated legitimate abstentions remain scored; answer coverage is reported separately.',
                      'Conditional answer-only means have their own denominator and are not paired causal effects.',
                      'Single generations; exploratory percentile-bootstrap intervals, no multiplicity correction.',
                      'AI assessment only; underlying gateway model unverified; not accuracy or human validation.',
                      'Historical judge source hashes are retained; current code regenerates the exact packets '
                      'and validates saved ratings. Historical source execution is not asserted reproduced.']}


def publish(run: Path, assessment: Path, output: Path, split: str = 'dev') -> dict:
    """Recheck all consumed bytes before exclusively writing a new report directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('report output exists; never overwrite')
    report = audit(run, assessment, split)
    _recheck(report['input_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'summary.json', report)
    return report


def main() -> None:
    """Compute an existing complete AI assessment; this command never invokes a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    args = parser.parse_args()
    result = publish(args.run, args.assessment, args.output, args.split)
    print(json.dumps({'complete': result['complete'], 'questions': result['questions'],
                      'pairs': len(result['paired']), 'output': str(args.output), 'model_calls': 0}))


if __name__ == '__main__':
    main()
