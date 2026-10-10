"""Complete synthetic generation/judge artifacts; no network or credential use."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from devsupport.data import sha256_file
from devsupport.evaluate import digest, make_record, read_json, write_new
from devsupport.llm import ProviderConfig
from devsupport.types import Question

BASE = Path(__file__).resolve().parents[1]


def load(name):
    path = BASE / 'analysis' / (name + '.py')
    assert path.is_file(), f'{name} utility is not implemented'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def save_rows(path, rows):
    path.write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')


def fixture_at(tmp_path, split='dev', count=2, names=('H0', 'Q', 'G', 'Full'), decision_label_protocol=None):
    """Use the real offline judge loader, blinder, validator and checkpoint writer."""
    judge = load('answer_judge')
    generation, assessed = tmp_path / 'generation', tmp_path / 'judge'
    generation.mkdir()
    questions = [Question(f'q{i}', 'Synthetic title', f'Synthetic body {i}', 'Reference',
                          [{'_id': 'n', 'text': 'Nugget', 'relevant_corpus_ids': ['d']}], {})
                 for i in range(count)]
    query_path = tmp_path / 'queries.parquet'
    pq.write_table(pa.Table.from_pylist([{'query_id': q.query_id, 'query_title': q.query_title,
                   'query_text': q.query_text, 'answer_text': q.answer_text, 'nuggets': q.nuggets}
                  for q in questions]), query_path)
    corpus, split_path, budget = tmp_path / 'corpus.json', tmp_path / 'split.json', tmp_path / 'budget.json'
    write_new(corpus, [{'doc_id': 'd', 'text': 'Exact evidence'}])
    ids = [q.query_id for q in questions]
    write_new(split_path, {'dev_ids': ids if split == 'dev' else ['other'],
                           'test_ids': ids if split == 'test' else ['not-read']})
    write_new(budget, {'starting_quota': 100, 'allowance_quota': 50, 'reserve_quota': 10})
    paths = {'queries': query_path, 'corpus': corpus, 'split': split_path, 'budget_start': budget}
    config = ProviderConfig(credential_path='NEVER_READ', min_request_interval_seconds=65,
                            rate_state_path='NEVER_READ')
    profiles = {name: {'retrieval': 'H0', 'reranker': None, 'adaptive': 'never',
                      'gate_enabled': name in ('G', 'Full')} for name in names}
    contract = {'paths': {k: str(v) for k, v in paths.items()},
                'artifacts': {k: sha256_file(v) for k, v in paths.items()},
                'profiles': profiles, 'provider': asdict(config), 'seeds': [7],
                'split_ids': {'dev': ids if split == 'dev' else ['other'],
                              'test': ids if split == 'test' else ['not-read']},
                'generation_ids': ids if split == 'test' else ['not-read']}
    run = {'phase': 'pipeline', 'split': split, 'controls': False, 'query_ids': ids,
           'profiles': list(profiles), 'contract_sha256': digest(contract)}
    write_new(generation / 'run.json', run)
    for name in profiles:
        rows = []
        for i, q in enumerate(questions):
            result = {'answer_text': 'Synthetic answer', 'decision': 'answer', 'citation_ids': ['d'],
                      'context': [{'doc_id': 'd', 'text': 'Exact evidence', 'citation_number': 1}], 'trace': {}}
            error = None
            if name == 'G' and i == 0:
                result, error = {}, 'RuntimeError'
            elif name == 'Full' and i == 1:
                result['decision'] = 'insufficient'
            rows.append(make_record(q, result, 1, error))
        save_rows(generation / (name + '.jsonl'), rows)
        write_new(generation / (name + '.manifest.json'),
                  {**run, 'evaluation_contract': contract, 'profile': name,
                   'settings': profiles[name], 'variant': 'H0'})
    qs, variants, evidence, _, inputs = judge.load_run(generation, split)
    packets, mapping = judge.build_packets(qs, variants, evidence, seed=7)
    provider_config = {**asdict(config), 'retries': 0}
    manifest = {'schema': 1, 'assessment': 'offline AI judge; not human review or independent model validation',
                'model_identity': 'gateway-reported alias; underlying identity unverified',
                'inputs': inputs, 'sources': judge.source_contract(), 'provider': provider_config,
                'provider_sha256': digest(provider_config), 'seed': 7, 'split': split,
                'generation_contract_sha256': digest(contract), 'query_ids': ids,
                'output_tokens': 2048, 'max_paid_calls': count}
    if decision_label_protocol is not None:
        manifest['decision_label_protocol'] = decision_label_protocol

    class Provider:
        config = SimpleNamespace(max_output_tokens=2048, max_input_bytes=200000)
        guard = SimpleNamespace(stopped=False, observations=[])

        def complete(self, system, user, max_tokens):
            packet = json.loads(user.split('\nReturn only')[0])['packet']
            reviews = []
            for response in packet['responses']:
                name = mapping[response['review_id']]['method']
                score = {'H0': 1, 'Q': 3, 'G': 2, 'Full': 1}.get(name, 2)
                reviews.append({'review_id': response['review_id'],
                                'scores': {key: score for key in judge.DIMENSIONS},
                                'reason': 'Synthetic supported rationale.',
                                'refusal': 'none' if response['decision'] == 'answer' else 'false_refusal',
                                'nugget_coverage': {'N001': 'covered'}})
            return {'text': json.dumps({'reviews': reviews})}

    rows = judge.run_judge(packets, mapping, manifest, assessed, Provider())
    summary = {'assessment': manifest['assessment'], 'complete': True, 'recording_complete': True,
               'judged_questions': count, 'recorded_questions': count, 'planned_questions': count,
               'methods': judge.summarize_judgments(packets, mapping, rows),
               'utility_note': 'Unweighted mean of five 0..3 scores. Operational failures contribute zero. '
                               'Missing AI judgments keep full attempted utility unavailable.'}
    write_new(assessed / f'summary-{count:03d}.json', summary)
    return generation, assessed


@pytest.fixture
def recorded(tmp_path):
    return fixture_at(tmp_path)


@pytest.mark.parametrize('protocol', ['explicit', 'explicit_coverage'])
def test_quality_audit_uses_recorded_explicit_decision_protocol(tmp_path, protocol):
    run, assessed = fixture_at(tmp_path, decision_label_protocol=protocol)
    report = load('answer_quality').audit(run, assessed)
    assert report['complete'] and report['methods']['Full']['refusals']['false_refusal'] == 1
    path = assessed / 'manifest.json'
    meta = read_json(path)
    meta['decision_label_protocol'] = 'unknown'
    save(path, meta)
    with pytest.raises(ValueError, match='decision label protocol'):
        load('answer_quality').audit(run, assessed)


def test_all_attempted_utility_and_answer_coverage(recorded):
    report = load('answer_quality').audit(*recorded)
    assert report['complete'] is True and report['human_ratings'] == 0
    assert report['methods']['G']['attempted'] == 2
    assert report['methods']['G']['operational_failures'] == 1
    assert report['methods']['G']['attempted_utility_mean'] == 1
    assert report['methods']['G']['conditional_quality_mean'] == 2
    assert report['methods']['Full']['answer_coverage'] == .5
    assert report['methods']['Full']['answered_dimensions']['faithfulness'] == 1
    assert report['methods']['Full']['refusals']['false_refusal'] == 1
    assert len(report['paired']) == 6
    assert report['paired']['Q-minus-H0']['mean_delta'] == 2
    assert report['paired']['Q-minus-H0']['ci_low'] == 2
    assert report['paired']['G-minus-H0']['mean_delta'] == 0
    assert report['paired']['G-minus-H0']['ci_low'] == -1
    assert report['paired']['G-minus-H0']['ci_high'] == 1
    assert report['paired']['Q-minus-H0']['samples'] == 2000


def test_five_method_ablation_audit_keeps_all_pairs_and_failures(tmp_path):
    names = ('H0', 'Q', 'G', 'Full', 'FullMinusBM25')
    report = load('answer_quality').audit(*fixture_at(tmp_path, names=names))
    assert set(report['methods']) == set(names)
    assert len(report['paired']) == 10
    assert all(pair['n'] == 2 and pair['samples'] == 2000 for pair in report['paired'].values())
    assert report['paired']['FullMinusBM25-minus-H0']['mean_delta'] == 1
    assert report['methods']['G']['operational_failures'] == 1
    assert report['methods']['G']['attempted_utility_mean'] == 1
    assert report['methods']['Full']['refusals']['false_refusal'] == 1
    assert report['human_ratings'] == 0 and report['complete'] is True


@pytest.mark.parametrize('tamper', ['inflight', 'partial', 'reorder', 'judge_error', 'checksum',
                                  'row_packet', 'mapping', 'summary', 'input', 'prompt', 'rubric', 'provider'])
def test_changed_or_incomplete_artifacts_rejected(recorded, tamper):
    generation, assessed = recorded
    path = assessed / 'results.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if tamper == 'inflight':
        write_new(assessed / 'inflight.json', {})
    elif tamper == 'partial':
        save_rows(path, rows[:1])
    elif tamper == 'reorder':
        save_rows(path, rows[::-1])
    elif tamper in ('judge_error', 'checksum', 'row_packet'):
        rows[0][{'judge_error': 'status', 'checksum': 'sha256', 'row_packet': 'packet_sha256'}[tamper]] = 'invalid'
        if tamper != 'checksum':
            rows[0]['sha256'] = digest({k: v for k, v in rows[0].items() if k != 'sha256'})
        save_rows(path, rows)
    elif tamper == 'mapping':
        value = read_json(assessed / 'sealed-mapping.json')
        value[next(iter(value))]['method'] = 'Q'
        save(assessed / 'sealed-mapping.json', value)
    elif tamper == 'summary':
        value = read_json(assessed / 'summary-002.json')
        value['methods']['G']['attempted_utility_mean'] = 3
        save(assessed / 'summary-002.json', value)
    elif tamper == 'input':
        with (generation / 'H0.jsonl').open('a') as f:
            f.write('\n')
    else:
        value = read_json(assessed / 'manifest.json')
        if tamper == 'provider':
            value['provider']['model'] = 'another-model'
            value['provider_sha256'] = digest(value['provider'])
        else:
            value[tamper + '_sha256'] = 'different'
        save(assessed / 'manifest.json', value)
    with pytest.raises(ValueError):
        load('answer_quality').audit(generation, assessed)


def test_invalid_review_not_accepted_after_rehash(recorded):
    path = recorded[1] / 'results.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['reviews']['reviews'][0]['scores']['faithfulness'] = True
    rows[0]['sha256'] = digest({k: v for k, v in rows[0].items() if k != 'sha256'})
    save_rows(path, rows)
    with pytest.raises(ValueError):
        load('answer_quality').audit(*recorded)


def test_manifest_mutation_after_parsing_prevents_publication(recorded, tmp_path, monkeypatch):
    quality = load('answer_quality')
    original_loads = json.loads
    manifest_path = recorded[1] / 'manifest.json'
    mutated = []

    def mutate_after_parse(*args, **kwargs):
        value = original_loads(*args, **kwargs)
        if isinstance(value, dict) and value.get('assessment') == quality.ASSESSMENT and not mutated:
            changed = {**value, 'split': 'test'}
            save(manifest_path, changed)
            mutated.append(True)
        return value

    monkeypatch.setattr(json, 'loads', mutate_after_parse)
    output = tmp_path / 'race-publication'
    with pytest.raises(ValueError, match='changed'):
        quality.publish(*recorded, output)
    assert mutated == [True]
    assert not output.exists()


def test_generation_inflight_and_wrong_split_rejected(recorded):
    quality = load('answer_quality')
    with pytest.raises(ValueError):
        quality.audit(*recorded, split='test')
    write_new(recorded[0] / 'H0.inflight.json', {})
    with pytest.raises(ValueError):
        quality.audit(*recorded)


def test_test30_frozen_first_cohort_only(tmp_path):
    generation, assessed = fixture_at(tmp_path, split='test', count=30)
    report = load('answer_quality').audit(generation, assessed, split='test')
    assert report['split'] == 'test' and report['questions'] == 30
    assert report['paired']['Q-minus-H0']['n'] == 30


def test_short_test_cohort_rejected(tmp_path):
    recorded = fixture_at(tmp_path, split='test', count=2)
    with pytest.raises(ValueError, match='30'):
        load('answer_quality').audit(*recorded, split='test')


def test_publication_rechecks_input_and_preserves_existing(recorded, tmp_path, monkeypatch):
    quality = load('answer_quality')
    original = quality.audit
    output = tmp_path / 'publication'

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        with (recorded[1] / 'results.jsonl').open('a') as f:
            f.write('\n')
        return report

    monkeypatch.setattr(quality, 'audit', changed)
    with pytest.raises(ValueError, match='changed'):
        quality.publish(*recorded, output)
    assert not output.exists()
    monkeypatch.setattr(quality, 'audit', original)
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'summary.json').write_text('preserve')
    with pytest.raises((FileExistsError, ValueError)):
        quality.publish(*recorded, existing)
    assert (existing / 'summary.json').read_text() == 'preserve'
