"""Offline AI-judge contracts; no credentials or live transport."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from devsupport.evaluate import digest, make_record
from devsupport.types import Evidence, Question


@pytest.fixture
def judge():
    path = Path(__file__).parents[1] / 'analysis/answer_judge.py'
    assert path.exists(), 'offline judge utility must exist'
    spec = importlib.util.spec_from_file_location('answer_judge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    q = Question('q', 'Title', 'Exact body', 'Exact reference',
                 [{'_id': 'n', 'text': 'Exact nugget', 'relevant_corpus_ids': ['d']}], {})
    corpus = {'d': Evidence('d', 'Exact evidence', {'score': 900, 'method': 'LEAK'})}
    variants = {}
    for name in ('B0', 'H0', 'H1', 'C'):
        result = {'answer_text': 'answer', 'decision': 'answer', 'citation_ids': ['d'],
                  'context': [{'doc_id': 'd', 'text': 'Exact evidence', 'score': 900,
                               'metadata': {'method': 'LEAK'}, 'citation_number': 1}],
                  'trace': {'generation': {'error': 'citation_or_schema_validation_failed'}}}
        variants[name] = [make_record(q, result, 1)]
    return [q], variants, corpus


def prepared(judge):
    return judge.build_packets(*fixture(), seed=7)


def test_explicit_decision_labels_are_eligible_only_and_do_not_mutate_packet(judge):
    packet = prepared(judge)[0][0]
    packet['responses'][0].update(decision='insufficient', answer='')
    packet['responses'][1].update(decision='clarify', answer='')
    packet['responses'][2]['operational_failure'] = True
    original = json.dumps(packet)
    body = json.loads(judge.prompt(packet, 200000, 8192, 'explicit').split('\n', 1)[0])
    assert body['allowed_refusal_labels'] == {
        a['review_id']: (['none'] if a['decision'] == 'answer' else ['supported_abstention', 'false_refusal'])
        for a in packet['responses'] if not a['operational_failure']}
    assert body['packet']['responses'] == [a for a in packet['responses'] if not a['operational_failure']]
    assert json.dumps(packet) == original
    assert 'allowed_refusal_labels' not in json.loads(judge.prompt(packet, 200000).split('\n', 1)[0])
    assert 'allowed_refusal_labels' in judge.system_prompt(8192, 'explicit')


def test_explicit_protocol_is_bound_and_invalid_protocol_stops_before_output(judge, tmp_path):
    packets, mapping = prepared(judge)
    provider = FakeProvider(response(judge, packets[0]))
    seen = []
    original_complete = provider.complete

    def capture(system, user, max_tokens):
        seen.append((system, user))
        return original_complete(system, user, max_tokens)

    provider.complete = capture
    destination = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='decision label protocol'):
        judge.run_judge(packets, mapping, {'decision_label_protocol': 'unknown'}, destination, provider)
    assert not destination.exists() and provider.calls == 0
    out = tmp_path / 'explicit'
    rows = judge.run_judge(packets, mapping, {'decision_label_protocol': 'explicit'}, out, provider)
    assert rows[0]['status'] == 'ok'
    meta = json.loads((out / 'manifest.json').read_text())
    assert meta['decision_label_protocol'] == 'explicit'
    assert meta['prompt_sha256'] == digest(judge.system_prompt(2048, 'explicit'))
    assert seen[0][0] == judge.system_prompt(2048, 'explicit')
    assert 'allowed_refusal_labels' in json.loads(seen[0][1].split('\n', 1)[0])
    with pytest.raises(ValueError, match='contract'):
        judge.run_judge(packets, mapping, {}, out, provider, resume=True)


def response(judge, packet):
    return {'reviews': [{'review_id': a['review_id'],
                         'scores': {key: 3 for key in judge.DIMENSIONS},
                         'reason': 'Supported by supplied evidence.', 'refusal': 'none',
                         'nugget_coverage': {n['nugget_id']: 'covered' for n in packet['nuggets']}}
                        for a in packet['responses'] if not a['operational_failure']]}


def test_explicit_coverage_requires_all_nuggets_for_empty_abstention(judge):
    packet = prepared(judge)[0][0]
    packet['responses'][0].update(decision='insufficient', answer='')
    packet['responses'][1]['operational_failure'] = True
    packet['nuggets'].append({'nugget_id': 'N002', 'text': 'Second exact nugget'})
    original = json.dumps(packet)
    body = json.loads(judge.prompt(packet, 200000, 8192, 'explicit_coverage').split('\n', 1)[0])
    assert body['required_nugget_ids'] == ['N001', 'N002']
    assert body['allowed_refusal_labels'][packet['responses'][0]['review_id']] == [
        'supported_abstention', 'false_refusal']
    assert body['packet']['responses'] == [a for a in packet['responses'] if not a['operational_failure']]
    assert json.dumps(packet) == original
    assert 'including empty abstentions' in judge.system_prompt(8192, 'explicit_coverage')
    assert 'required_nugget_ids' not in json.loads(judge.prompt(packet, 200000, 8192, 'explicit').split('\n', 1)[0])
    value = response(judge, packet)
    value['reviews'][0]['refusal'] = 'supported_abstention'
    value['reviews'][0]['nugget_coverage'] = {}
    with pytest.raises(ValueError, match='invalid nugget coverage'):
        judge.validate_review(value, packet)


def test_explicit_coverage_is_delivered_and_bound(judge, tmp_path):
    packets, mapping = prepared(judge)
    provider = FakeProvider(response(judge, packets[0]))
    seen = []
    original_complete = provider.complete

    def capture(system, user, max_tokens):
        seen.append((system, user))
        return original_complete(system, user, max_tokens)

    provider.complete = capture
    out = tmp_path / 'coverage'
    rows = judge.run_judge(packets, mapping, {'decision_label_protocol': 'explicit_coverage'}, out, provider)
    assert rows[0]['status'] == 'ok'
    meta = json.loads((out / 'manifest.json').read_text())
    assert meta['decision_label_protocol'] == 'explicit_coverage'
    assert meta['prompt_sha256'] == digest(judge.system_prompt(2048, 'explicit_coverage'))
    assert seen[0][0] == judge.system_prompt(2048, 'explicit_coverage')
    assert json.loads(seen[0][1].split('\n', 1)[0])['required_nugget_ids'] == ['N001']
    with pytest.raises(ValueError, match='contract'):
        judge.run_judge(packets, mapping, {'decision_label_protocol': 'explicit'}, out, provider, resume=True)


def test_blinding_exact_text_and_no_scores(judge):
    packets, mapping = prepared(judge)
    assert prepared(judge) == (packets, mapping)
    encoded = json.dumps(packets)
    assert all(name not in encoded for name in ('B0', 'H0', 'H1', 'LEAK', '900', 'score'))
    assert packets[0]['question'] == 'Title Exact body'
    assert packets[0]['reference_answer'] == 'Exact reference'
    assert packets[0]['nuggets'][0]['text'] == 'Exact nugget'
    assert packets[0]['gold_context'][0]['text'] == 'Exact evidence'
    assert all(not a['operational_failure'] for a in packets[0]['responses'])
    assert {m['method'] for m in mapping.values()} == {'B0', 'H0', 'H1', 'C'}


def test_incomplete_missing_gold_and_oversize_fail_closed(judge):
    questions, variants, corpus = fixture()
    variants['C'] = []
    with pytest.raises(ValueError, match='complete'):
        judge.build_packets(questions, variants, corpus)
    with pytest.raises(ValueError, match='gold'):
        judge.build_packets(*fixture()[:2], {})
    packets, _ = prepared(judge)
    with pytest.raises(ValueError, match='byte cap'):
        judge.prompt(packets[0], 100)


@pytest.mark.parametrize('mutation', ['bool', 'float', 'nan', 'range', 'missing', 'duplicate', 'nugget', 'extra'])
def test_strict_schema(judge, mutation):
    packet = prepared(judge)[0][0]
    value = response(judge, packet)
    first = value['reviews'][0]
    if mutation in ('bool', 'float', 'nan', 'range'):
        first['scores']['correctness'] = {'bool': True, 'float': 1.0, 'nan': float('nan'), 'range': 4}[mutation]
    elif mutation == 'missing':
        value['reviews'].pop()
    elif mutation == 'duplicate':
        value['reviews'][1]['review_id'] = first['review_id']
    elif mutation == 'nugget':
        first['nugget_coverage'] = {'other': 'covered'}
    else:
        first['extra'] = 'unsupported'
    with pytest.raises(ValueError):
        judge.validate_review(value, packet)


class FakeProvider:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0
        self.config = SimpleNamespace(max_input_bytes=200000, max_output_tokens=2048)
        self.guard = SimpleNamespace(stopped=False, observations=[])

    def complete(self, system, user, max_tokens):
        assert max_tokens == 2048
        self.calls += 1
        if isinstance(self.payload, BaseException):
            raise self.payload
        return {'text': json.dumps(self.payload), 'usage': {'output_tokens': 50}}


def test_checkpoint_resume_binding_and_inflight(judge, tmp_path):
    packets, mapping = prepared(judge)
    provider = FakeProvider(response(judge, packets[0]))
    manifest = {'input_sha256': 'one', 'source_sha256': 'two'}
    rows = judge.run_judge(packets, mapping, manifest, tmp_path, provider)
    assert rows[0]['status'] == 'ok' and provider.calls == 1
    assert judge.run_judge(packets, mapping, manifest, tmp_path, provider, resume=True) == rows
    assert provider.calls == 1
    with pytest.raises(ValueError, match='contract'):
        judge.run_judge(packets, mapping, {**manifest, 'source_sha256': 'changed'}, tmp_path, provider, resume=True)
    (tmp_path / 'inflight.json').write_text('{}')
    with pytest.raises(ValueError, match='in-flight'):
        judge.run_judge(packets, mapping, manifest, tmp_path, provider, resume=True)


@pytest.mark.parametrize('bad', [RuntimeError('SECRET'), {'reviews': []}])
def test_failure_checkpoint_never_replayed(judge, tmp_path, bad):
    packets, mapping = prepared(judge)
    provider = FakeProvider(bad)
    rows = judge.run_judge(packets, mapping, {}, tmp_path, provider)
    assert rows[0]['status'] == 'judge_error'
    assert 'SECRET' not in json.dumps(rows)
    judge.run_judge(packets, mapping, {}, tmp_path, provider, resume=True)
    assert provider.calls == 1


def test_tampered_checkpoint_and_no_overwrite(judge, tmp_path):
    packets, mapping = prepared(judge)
    provider = FakeProvider(response(judge, packets[0]))
    judge.run_judge(packets, mapping, {}, tmp_path, provider)
    with pytest.raises(FileExistsError):
        judge.run_judge(packets, mapping, {}, tmp_path, provider)
    path = tmp_path / 'results.jsonl'
    row = json.loads(path.read_text())
    row['reviews']['reviews'][0]['scores']['correctness'] = 0
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='integrity'):
        judge.run_judge(packets, mapping, {}, tmp_path, provider, resume=True)


def test_operational_errors_zero_utility_refusals_explicit(judge):
    questions, variants, corpus = fixture()
    variants['B0'][0]['error'] = 'ProviderError'
    variants['C'][0]['result']['decision'] = 'insufficient'
    variants['C'][0]['result']['trace'] = {}
    packets, mapping = judge.build_packets(questions, variants, corpus)
    value = response(judge, packets[0])
    refusal_id = next(k for k, v in mapping.items() if v['method'] == 'C')
    next(r for r in value['reviews'] if r['review_id'] == refusal_id)['refusal'] = 'supported_abstention'
    judge.validate_review(value, packets[0])
    summary = judge.summarize_judgments(packets, mapping, [{'status': 'ok', 'reviews': value}])
    assert summary['B0']['conditional_quality_n'] == 0
    assert summary['B0']['attempted_utility_mean'] == 0
    assert summary['H1']['conditional_quality_mean'] == 3
    assert summary['C']['refusals']['supported_abstention'] == 1


def test_crash_leaves_inflight(judge, tmp_path):
    packets, mapping = prepared(judge)
    with pytest.raises(KeyboardInterrupt):
        judge.run_judge(packets, mapping, {}, tmp_path, FakeProvider(KeyboardInterrupt()))
    assert (tmp_path / 'inflight.json').exists()


def test_manifest_binds_own_source_and_prompt(judge):
    value = judge.source_contract()
    assert 'analysis/answer_judge.py' in value
    assert 'src/devsupport/pacing.py' in value
    assert all(len(v) == 64 for v in value.values())
    assert digest(value) != digest({**value, 'analysis/answer_judge.py': 'changed'})


def test_prompt_ends_with_schema_reminder(judge):
    user = judge.prompt(prepared(judge)[0][0], 200000)
    assert user.endswith('Return only the JSON object with reviews; no Markdown or surrounding prose.')


def run_fixture(tmp_path, names=('B0', 'H0', 'H1', 'C')):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from devsupport.data import sha256_file
    from devsupport.evaluate import write_new

    questions, variants, _ = fixture()
    variants = {name: variants['H0'] for name in names}
    q = questions[0]
    query_path = tmp_path / 'queries.parquet'
    pq.write_table(pa.Table.from_pylist([{'query_id': q.query_id, 'query_title': q.query_title,
                    'query_text': q.query_text, 'answer_text': q.answer_text, 'nuggets': q.nuggets}]), query_path)
    corpus_path = tmp_path / 'corpus.json'
    write_new(corpus_path, [{'doc_id': 'd', 'text': 'Exact evidence'}])
    split_path, budget_path = tmp_path / 'split.json', tmp_path / 'budget.json'
    write_new(split_path, {'dev_ids': ['q'], 'test_ids': ['not-read']})
    write_new(budget_path, {'starting_quota': 100, 'allowance_quota': 50, 'reserve_quota': 10})
    paths = dict(queries=query_path, corpus=corpus_path, split=split_path, budget_start=budget_path)
    contract = {'paths': {k: str(v) for k, v in paths.items()},
                'artifacts': {k: sha256_file(v) for k, v in paths.items()},
                'profiles': {name: {'retrieval': name, 'reranker': None,
                                    'adaptive': 'never', 'gate_enabled': False} for name in variants},
                'split_ids': {'dev': ['q'], 'test': ['not-read']}, 'generation_ids': ['not-read']}
    run = {'phase': 'pipeline', 'split': 'dev', 'controls': False, 'query_ids': ['q'],
           'profiles': list(variants), 'contract_sha256': digest(contract)}
    write_new(tmp_path / 'run.json', run)
    for method, rows in variants.items():
        (tmp_path / (method + '.jsonl')).write_text(json.dumps(rows[0]) + '\n')
        write_new(tmp_path / (method + '.manifest.json'),
                  {**run, 'evaluation_contract': contract, 'profile': method,
                   'settings': contract['profiles'][method], 'variant': method})
    return contract


@pytest.mark.parametrize('count', [2, 5, 8])
def test_ablation_cohort_packets_preserve_every_method(judge, count):
    questions, original, corpus = fixture()
    variants = {f'Method{i}': original['H0'] for i in range(count)}
    packets, mapping = judge.build_packets(questions, variants, corpus, seed=7)
    assert len(packets[0]['responses']) == count
    assert {origin['method'] for origin in mapping.values()} == set(variants)
    assert all(answer['answer'] == 'answer' for answer in packets[0]['responses'])
    assert all(answer['context'] == [{'doc_id': 'd', 'text': 'Exact evidence', 'citation_number': 1}]
               for answer in packets[0]['responses'])
    assert 'Method' not in judge.prompt(packets[0], 200000)
    assert judge.validate_review(response(judge, packets[0]), packets[0])
    assert judge.build_packets(questions, variants, corpus, seed=7) == (packets, mapping)


@pytest.mark.parametrize('count', [0, 1])
def test_comparative_packets_require_two_methods(judge, count):
    questions, variants, corpus = fixture()
    with pytest.raises(ValueError, match='at least two'):
        judge.build_packets(questions, dict(list(variants.items())[:count]), corpus)


@pytest.mark.parametrize('count', [2, 5, 8])
def test_loader_preserves_complete_ablation_cohort(judge, tmp_path, count):
    names = tuple(f'Method{i}' for i in range(count))
    run_fixture(tmp_path, names)
    questions, variants, corpus, _, inputs = judge.load_run(tmp_path)
    assert list(variants) == list(names) and questions[0].query_id == 'q'
    assert set(corpus) == {'d'} and len(inputs) == 5 + 2 * count


def test_ablation_loader_rejects_missing_fifth_output(judge, tmp_path):
    run_fixture(tmp_path, ('B0', 'H0', 'H1', 'C', 'FullMinusBM25'))
    (tmp_path / 'FullMinusBM25.jsonl').write_text('')
    with pytest.raises(ValueError, match='complete identically ordered'):
        judge.load_run(tmp_path)


@pytest.mark.parametrize('profiles', [['B0'], ['B0', 'H0', 'H1', 'C', 'B0']])
def test_loader_rejects_insufficient_or_duplicate_methods(judge, tmp_path, profiles):
    run_fixture(tmp_path)
    path = tmp_path / 'run.json'
    run = json.loads(path.read_text())
    run['profiles'] = profiles
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='unique.*at least two'):
        judge.load_run(tmp_path)


@pytest.mark.parametrize('operation', ['swap', 'duplicate'])
def test_run_rejects_reassigned_generation_files(judge, tmp_path, operation):
    run_fixture(tmp_path)
    for suffix in ('.jsonl', '.manifest.json'):
        first, second = tmp_path / ('B0' + suffix), tmp_path / ('H0' + suffix)
        original = first.read_bytes()
        first.write_bytes(second.read_bytes())
        if operation == 'swap':
            second.write_bytes(original)
    with pytest.raises(ValueError, match='profile'):
        judge.load_run(tmp_path)


@pytest.mark.parametrize('field', ['profile', 'settings', 'variant'])
@pytest.mark.parametrize('mutation', ['missing', 'mismatched'])
def test_run_rejects_incomplete_or_mismatched_profile_metadata(judge, tmp_path, field, mutation):
    run_fixture(tmp_path)
    path = tmp_path / 'B0.manifest.json'
    meta = json.loads(path.read_text())
    if mutation == 'missing':
        meta.pop(field)
    else:
        meta[field] = meta['evaluation_contract']['profiles']['H0'] if field == 'settings' else 'H0'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='profile'):
        judge.load_run(tmp_path)


def test_run_validates_split_complete_integrity_and_annotations(judge, tmp_path):
    run_fixture(tmp_path)
    questions, variants, corpus, _, inputs = judge.load_run(tmp_path, 'dev')
    assert questions[0].query_id == 'q' and set(corpus) == {'d'} and len(variants) == 4
    assert len(inputs) == 13
    with pytest.raises(ValueError, match='split'):
        judge.load_run(tmp_path, 'test')
    path = tmp_path / 'B0.jsonl'
    row = json.loads(path.read_text())
    row['result']['answer_text'] = 'tampered'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='integrity'):
        judge.load_run(tmp_path, 'dev')
    path.write_text('')
    with pytest.raises(ValueError, match='complete'):
        judge.load_run(tmp_path, 'dev')


def test_input_annotation_change_rejected(judge, tmp_path):
    run_fixture(tmp_path)
    (tmp_path / 'corpus.json').write_text('[]')
    with pytest.raises(ValueError, match='artifact changed'):
        judge.load_run(tmp_path, 'dev')


def test_budget_stop_does_not_claim_paid_failure(judge, tmp_path):
    from devsupport.evaluate import BudgetStopped
    packets, mapping = prepared(judge)
    rows = judge.run_judge(packets, mapping, {}, tmp_path, FakeProvider(BudgetStopped()))
    assert rows == [] and not (tmp_path / 'inflight.json').exists()
    summary = judge.summarize_judgments(packets, mapping, rows)
    assert summary['B0']['attempted_utility_mean'] is None
    assert summary['B0']['judge_missing'] == 1


def test_explicit_test_split_only_accepts_frozen_generation_ids(judge, tmp_path):
    from devsupport.data import sha256_file
    run_fixture(tmp_path)
    split = {'dev_ids': ['other'], 'test_ids': ['q']}
    (tmp_path / 'split.json').write_text(json.dumps(split))
    run = json.loads((tmp_path / 'run.json').read_text())
    run['split'] = 'test'
    for method in run['profiles']:
        path = tmp_path / (method + '.manifest.json')
        meta = json.loads(path.read_text())
        contract = meta['evaluation_contract']
        contract['artifacts']['split'] = sha256_file(tmp_path / 'split.json')
        contract['generation_ids'] = ['q']
        contract['split_ids'] = {'dev': ['other'], 'test': ['q']}
        run['contract_sha256'] = digest(contract)
        path.write_text(json.dumps({**meta, **run}))
    (tmp_path / 'run.json').write_text(json.dumps(run))
    assert judge.load_run(tmp_path, 'test')[0][0].query_id == 'q'
    with pytest.raises(ValueError, match='split'):
        judge.load_run(tmp_path)


def test_test_loader_rejects_selected_ids_outside_first_thirty(judge, tmp_path):
    from devsupport.data import sha256_file
    run_fixture(tmp_path)
    split = {'dev_ids': ['other'], 'test_ids': [f'not_selected_{i}' for i in range(30)] + ['q']}
    (tmp_path / 'split.json').write_text(json.dumps(split))
    run = json.loads((tmp_path / 'run.json').read_text())
    run['split'] = 'test'
    for method in run['profiles']:
        path = tmp_path / (method + '.manifest.json')
        meta = json.loads(path.read_text())
        contract = meta['evaluation_contract']
        contract['artifacts']['split'] = sha256_file(tmp_path / 'split.json')
        contract['generation_ids'] = ['q']
        contract['split_ids'] = {'dev': split['dev_ids'], 'test': split['test_ids']}
        run['contract_sha256'] = digest(contract)
        path.write_text(json.dumps({**meta, **run}))
    (tmp_path / 'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='first 30'):
        judge.load_run(tmp_path, 'test')


def test_final_balance_is_saved_after_durable_result(judge, tmp_path):
    from devsupport.evaluate import BudgetGuard, read_json
    authorization = {'starting_quota': 100, 'allowance_quota': 50, 'reserve_quota': 10}
    guard = BudgetGuard(authorization, lambda: {'remain_quota': 90}, tmp_path / 'budget.accounting.json')
    judge.finalize_accounting(guard, tmp_path)
    snapshots = list(tmp_path.glob('balance-*.json'))
    assert len(snapshots) == 1
    assert read_json(snapshots[0])['observations'][0]['shared_usage_quota'] == 10
    assert read_json(snapshots[0])['final_balance_reconciled'] is True
    assert read_json(tmp_path / 'budget.accounting.json')['consumed'] == 10


def test_failed_judge_final_balance_marks_unreconciled_call(judge, tmp_path):
    from devsupport.evaluate import BudgetGuard, GuardedProvider, read_json

    checks = []

    def balance():
        checks.append(True)
        return {'remain_quota': 90}

    authorization = {'starting_quota': 100, 'allowance_quota': 50, 'reserve_quota': 10}
    guard = BudgetGuard(authorization, balance, tmp_path / 'budget.accounting.json')
    output = tmp_path / 'judge'
    packets, mapping = prepared(judge)
    provider = GuardedProvider(FakeProvider({'reviews': []}), guard)
    rows = judge.run_judge(packets, mapping, {}, output, provider)
    assert rows[0]['status'] == 'judge_error' and guard.stopped
    judge.finalize_accounting(guard, output)
    snapshot = read_json(next(output.glob('balance-*.json')))
    assert snapshot['final_balance_reconciled'] is False
    assert snapshot['accounting_note'] == (
        'Final balance check incomplete; last potentially billed call cost may be pending.')
    assert snapshot['stopped'] is True and checks == [True]
    assert snapshot['observations'] == rows[0]['balance_observations']

@pytest.mark.parametrize('limit', [2048, 4096, 8192])
def test_judge_budget_reaches_transport_and_checkpoint(judge, tmp_path, limit):
    from devsupport.llm import AnthropicProvider
    packets, mapping = prepared(judge)
    payloads = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return json.dumps({'model': 'claude-opus-5-5', 'stop_reason': 'end_turn',
                'content': [{'type': 'text', 'text': json.dumps(response(judge, packets[0]))}],
                'usage': {'input_tokens': 10, 'output_tokens': 20}}).encode()

    def transport(request, timeout):
        payloads.append(json.loads(request.data))
        return Reply()

    config = judge.JudgeProviderConfig(max_output_tokens=limit, retries=0)
    provider = AnthropicProvider(config, credential_loader=lambda: 'offline-placeholder', transport=transport)
    provider.guard = SimpleNamespace(stopped=False, observations=[])
    rows = judge.run_judge(packets, mapping, {'output_tokens': limit}, tmp_path, provider)
    assert rows[0]['status'] == 'ok'
    assert len(payloads) == 1 and payloads[0]['max_tokens'] == limit
    assert f'entire response fits {limit} output tokens.' in payloads[0]['system']
    assert 'Exact reference' in payloads[0]['messages'][0]['content']
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['output_tokens'] == limit
    assert manifest['prompt_sha256'] == digest(payloads[0]['system'])


@pytest.mark.parametrize('limit', [True, False, 2048.0, '4096', None, 0, 2049, 16384])
def test_invalid_judge_budget_rejected_before_calls(judge, tmp_path, limit):
    with pytest.raises(ValueError, match='output token'):
        judge.JudgeProviderConfig(max_output_tokens=limit)
    packets, mapping = prepared(judge)
    provider = FakeProvider(response(judge, packets[0]))
    provider.config.max_output_tokens = limit
    with pytest.raises(ValueError, match='output token'):
        judge.run_judge(packets, mapping, {}, tmp_path, provider)
    assert provider.calls == 0


def test_changed_budget_resume_and_manifest_mismatch_rejected(judge, tmp_path):
    packets, mapping = prepared(judge)
    provider = FakeProvider(response(judge, packets[0]))
    provider.config.max_output_tokens = 2048
    judge.run_judge(packets, mapping, {}, tmp_path, provider)
    provider.config.max_output_tokens = 8192
    with pytest.raises(ValueError, match='contract'):
        judge.run_judge(packets, mapping, {}, tmp_path, provider, resume=True)
    with pytest.raises(ValueError, match='output token'):
        judge.run_judge(packets, mapping, {'output_tokens': 2048}, tmp_path / 'new', provider)
    assert provider.calls == 1


@pytest.mark.parametrize('code', ['output_truncated', 'http_429', 'SECRET credential'])
def test_safe_provider_error_diagnostics_terminal(judge, tmp_path, code):
    from devsupport.llm import ProviderError
    packets, mapping = prepared(judge)
    provider = FakeProvider(ProviderError(code, {
        'truncated': True, 'stop_reason': 'max_tokens', 'usage': {'output_tokens': 2048},
        'http_error': {'retry_after_seconds': 65, 'provider_error_code': 429,
                       'provider_error_type': 'rate_limit_error', 'body': 'SECRET'},
        'headers': {'Authorization': 'SECRET'}, 'raw_body': 'SECRET', 'reasoning': 'SECRET',
        'events': [{'error': 'SECRET'}]}))
    rows = judge.run_judge(packets, mapping, {}, tmp_path, provider)
    row = rows[0]
    assert row['status'] == 'judge_error' and row['error'] == 'ProviderError'
    assert row['provider_error_code'] == (code if 'SECRET' not in code else 'unknown_provider_error')
    assert row['provider']['http_error'] == {'retry_after_seconds': 65, 'provider_error_code': 429,
                                            'provider_error_type': 'rate_limit_error'}
    assert 'SECRET' not in json.dumps(rows)
    provider.guard.stopped = False
    assert judge.run_judge(packets, mapping, {}, tmp_path, provider, resume=True) == rows
    assert provider.calls == 1


@pytest.mark.parametrize('limit', [None, 4096, 8192])
@pytest.mark.parametrize('protocol', ['legacy', 'explicit', 'explicit_coverage'])
def test_cli_preflight_binds_selected_budget_without_network(judge, tmp_path, monkeypatch, limit, protocol):
    import sys
    from dataclasses import asdict
    questions, variants, corpus = fixture()
    config = judge.ProviderConfig(min_request_interval_seconds=65, rate_state_path='offline-rate.json')
    contract = {'provider': asdict(config)}
    monkeypatch.setattr(judge, 'load_run', lambda *args: (questions, variants, corpus, contract, {}))
    monkeypatch.setattr(judge, 'AnthropicProvider', lambda *args: pytest.fail('preflight created provider'))
    args = ['answer_judge', '--run', str(tmp_path), '--output', str(tmp_path / 'preflight')]
    if limit is not None:
        args += ['--output-tokens', str(limit)]
    if protocol != 'legacy':
        args += ['--decision-label-protocol', protocol]
    monkeypatch.setattr(sys, 'argv', args)
    judge.main()
    value = json.loads((tmp_path / 'preflight/preflight.json').read_text())
    expected = limit or 2048
    assert value['output_tokens'] == value['provider']['max_output_tokens'] == expected
    assert value['provider']['retries'] == 0
    assert value['provider']['min_request_interval_seconds'] == 65
    assert value['provider']['rate_state_path'] == 'offline-rate.json'
    assert value.get('decision_label_protocol', 'legacy') == protocol
    assert value['prompt_sha256'] == digest(judge.system_prompt(expected, protocol))
    packet = judge.build_packets(questions, variants, corpus)[0][0]
    assert value['input_bytes'] == [len((judge.system_prompt(expected, protocol) +
        judge.prompt(packet, config.max_input_bytes, expected, protocol)).encode('utf-8'))]


@pytest.mark.parametrize('settings', [
    {'base_url': 'https://unapproved.example/v1'}, {'model': 'bad model'}, {'thinking': 'enabled'},
    {'max_input_bytes': 200001}, {'timeout_seconds': 91}, {'retries': 2},
    {'min_request_interval_seconds': 121}, {'rate_state_path': ''}, {'protocol': 'other'},
    {'expected_reported_model': 'bad model'}, {'input_per_million': -1},
])
def test_judge_budget_preserves_base_validation(judge, settings):
    with pytest.raises(ValueError):
        judge.JudgeProviderConfig(max_output_tokens=8192, **settings)


def test_shared_generation_config_keeps_original_cap(judge):
    with pytest.raises(ValueError):
        judge.ProviderConfig(max_output_tokens=4096)


@pytest.mark.parametrize('value', ['True', '2048.0', '0', '2049', '16384'])
def test_invalid_cli_budget_fails_before_loading_or_network(judge, tmp_path, monkeypatch, value):
    import sys
    monkeypatch.setattr(judge, 'load_run', lambda *args: pytest.fail('invalid budget loaded inputs'))
    monkeypatch.setattr(sys, 'argv', ['judge', '--run', str(tmp_path), '--output', str(tmp_path / 'out'),
                                    '--output-tokens', value, '--execute'])
    with pytest.raises(SystemExit) as error:
        judge.main()
    assert error.value.code == 2
    assert not (tmp_path / 'out').exists()

@pytest.mark.parametrize('length', [251, 256, 1000])
def test_reason_hard_bound_accepts_exact_text_and_scores(judge, length):
    packet = prepared(judge)[0][0]
    value = response(judge, packet)
    reason = ' Evidence supports this. ' + 'x' * (length - 26) + ' '
    assert len(reason) == length
    value['reviews'][0]['reason'] = reason
    before = json.dumps(value, ensure_ascii=False)
    validated = judge.validate_review(value, packet)
    assert json.dumps(validated, ensure_ascii=False) == before
    assert validated['reviews'][0]['scores'] == {key: 3 for key in judge.DIMENSIONS}


@pytest.mark.parametrize('reason', ['x' * 1001, '', ' \t\n', 'x' + ' ' * 1000])
def test_reason_hard_bound_rejects_overlong_or_blank(judge, reason):
    packet = prepared(judge)[0][0]
    value = response(judge, packet)
    value['reviews'][0]['reason'] = reason
    with pytest.raises(ValueError, match='reason'):
        judge.validate_review(value, packet)


def test_reason_prompt_has_concise_target_and_hard_maximum(judge):
    for limit in (2048, 4096, 8192):
        system = judge.system_prompt(limit)
        assert 'aim for at most 240 characters' in system
        assert 'hard maximum 1000 characters' in system


def test_prompt_only_eligible_exact_inputs_and_canonical_archive(judge, tmp_path):
    packets, mapping = prepared(judge)
    packet = packets[0]
    failed = packet['responses'][2]
    failed['operational_failure'] = True
    failed['answer_text'] = 'FAILED_ANSWER_SENTINEL'
    failed['context'] = [{'doc_id': 'failed', 'text': 'FAILED_CONTEXT_SENTINEL'}]
    packet['responses'][0]['context'][0]['text'] = ' Exact evidence\nwith Unicode: \u4e2d\u6587 '
    before = json.dumps(packets, ensure_ascii=False)
    user = judge.prompt(packet, 200000)
    payload = json.loads(user.split('\n', 1)[0])
    eligible = [a for a in packet['responses'] if not a['operational_failure']]
    assert payload['allowed_review_ids'] == [a['review_id'] for a in eligible]
    assert payload['packet'] == {**packet, 'responses': eligible}
    assert payload['packet']['reference_answer'] == 'Exact reference'
    assert payload['packet']['gold_context'][0]['text'] == 'Exact evidence'
    assert payload['packet']['nuggets'][0]['text'] == 'Exact nugget'
    assert 'FAILED_ANSWER_SENTINEL' not in user and 'FAILED_CONTEXT_SENTINEL' not in user
    assert failed['review_id'] not in user
    assert all(name not in user for name in ('B0', 'H0', 'H1', 'LEAK'))
    assert 'exactly one entry for each ID in allowed_review_ids and no other IDs' in judge.SYSTEM
    assert json.dumps(packets, ensure_ascii=False) == before

    class CaptureProvider(FakeProvider):
        def complete(self, system, sent_user, max_tokens):
            assert sent_user == user
            return super().complete(system, sent_user, max_tokens)

    provider = CaptureProvider(response(judge, packet))
    rows = judge.run_judge(packets, mapping, {}, tmp_path, provider)
    assert rows[0]['status'] == 'ok' and provider.calls == 1
    assert json.dumps(packets, ensure_ascii=False) == before
    assert json.loads((tmp_path / 'packets.json').read_text(encoding='utf-8')) == packets
    assert json.loads((tmp_path / 'sealed-mapping.json').read_text()) == mapping
    failed_method = mapping[failed['review_id']]['method']
    summary = judge.summarize_judgments(packets, mapping, rows)[failed_method]
    assert summary['attempted'] == summary['operational_failures'] == 1
    assert summary['conditional_quality_n'] == 0 and summary['attempted_utility_mean'] == 0


def test_filtered_prompt_byte_cap_uses_exact_serialized_input(judge):
    packet = prepared(judge)[0][0]
    packet['responses'][0]['operational_failure'] = True
    packet['responses'][0]['answer_text'] = 'FAILED_OVERSIZE' * 20000
    user = judge.prompt(packet, 200000)
    size = len((judge.SYSTEM + user).encode('utf-8'))
    assert size < 200000
    assert judge.prompt(packet, size) == user
    with pytest.raises(ValueError, match='byte cap'):
        judge.prompt(packet, size - 1)


def test_all_failed_preserves_archive_and_denominator_without_provider(judge, tmp_path):
    packets, mapping = prepared(judge)
    for answer in packets[0]['responses']:
        answer['operational_failure'] = True
    payload = json.loads(judge.prompt(packets[0], 200000).split('\n', 1)[0])
    assert payload['allowed_review_ids'] == [] and payload['packet']['responses'] == []
    provider = FakeProvider(RuntimeError('must never call'))
    rows = judge.run_judge(packets, mapping, {}, tmp_path, provider)
    assert provider.calls == 0 and rows[0]['status'] == 'ok'
    assert rows[0]['reviews'] == {'reviews': []}
    assert json.loads((tmp_path / 'packets.json').read_text()) == packets
    for result in judge.summarize_judgments(packets, mapping, rows).values():
        assert result['attempted'] == result['operational_failures'] == 1
        assert result['conditional_quality_n'] == 0 and result['attempted_utility_mean'] == 0


def test_failed_review_id_remains_strict_terminal_failure(judge, tmp_path):
    packets, mapping = prepared(judge)
    invalid = response(judge, packets[0])
    packets[0]['responses'][2]['operational_failure'] = True
    with pytest.raises(ValueError, match='review IDs'):
        judge.validate_review(invalid, packets[0])
    provider = FakeProvider(invalid)
    rows = judge.run_judge(packets, mapping, {}, tmp_path, provider)
    assert rows[0]['status'] == 'judge_error' and provider.calls == 1
    assert judge.run_judge(packets, mapping, {}, tmp_path, provider, resume=True) == rows
    assert provider.calls == 1
