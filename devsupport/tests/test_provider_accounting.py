"""Observed accounting must not turn missing transport/usage/cost into zero."""
import importlib.util
import json
from pathlib import Path

import pytest


def helper():
    path = Path(__file__).parents[1] / 'analysis/provider_accounting.py'
    assert path.is_file(), 'missing provider accounting implementation'
    spec = importlib.util.spec_from_file_location('provider_accounting_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(calls):
    return {'result': {'trace': {'provider_calls': calls}}}


def test_missing_attempts_are_unknown_not_zero():
    result = helper().summarize_calls([row([{'stage': 'gate', 'error': 'provider_failure', 'cost': None}])])
    assert result['stage_invocations'] == 1
    assert result['known_transport_attempts'] == 0
    assert result['transport_attempts_total'] is None
    assert result['invocations_missing_attempts'] == 1
    assert result['estimated_provider_cost_total'] is None


def test_explicit_zero_attempts_remains_known_without_invented_usage():
    result = helper().summarize_calls([row([{'stage': 'gate', 'attempts': 0, 'cost': None}])])
    assert result['transport_attempts_total'] == 0
    assert result['invocations_missing_attempts'] == 0
    assert result['reported_response_usage'] == {}
    assert result['estimated_provider_cost_total'] is None


def test_retries_with_unknown_usage_are_not_full_billed_tokens_or_cost():
    call = {'stage': 'generation', 'attempts': 2, 'usage': {'input_tokens': 20, 'output_tokens': 10},
            'events': [{'status': 'network_error', 'usage': None},
                       {'status': 'ok', 'usage': {'input_tokens': 20, 'output_tokens': 10}}],
            'cost': None, 'known_response_cost': 0.02}
    result = helper().summarize_calls([row([call])])
    assert result['transport_attempts_total'] == 2 and result['known_retry_extra_attempts'] == 1
    assert result['reported_response_usage'] == {'input_tokens': 20, 'output_tokens': 10}
    assert result['usage_complete_for_recorded_attempts'] is False
    assert result['estimated_provider_cost_total'] is None
    assert result['known_response_cost_estimate_sum'] == 0.02


def test_missing_trace_cannot_claim_no_model_activity():
    result = helper().summarize_calls([{'result': {}}, row([])])
    assert result['rows_missing_provider_trace'] == 1
    assert result['transport_attempts_total'] is None
    assert result['estimated_provider_cost_total'] is None


def test_single_response_accounting_sums_only_recorded_dimensions():
    calls = [{'stage': 'generation', 'attempts': 1,
              'usage': {'input_tokens': 20, 'output_tokens': 10, 'cache_read_input_tokens': 5},
              'cost': 0.02, 'known_response_cost': 0.02}]
    result = helper().summarize_calls([row(calls), row([])])
    assert result['transport_attempts_total'] == 1
    assert result['reported_response_usage']['input_tokens'] == 20
    assert result['reported_response_usage']['cache_read_input_tokens'] == 5
    assert result['usage_complete_for_recorded_attempts'] is True
    assert result['estimated_provider_cost_total'] == 0.02


@pytest.mark.parametrize('usage', [{'input_tokens': 20}, {'output_tokens': 10}, {'cache_read_input_tokens': 5}])
def test_partial_response_dimensions_are_not_complete(usage):
    result = helper().summarize_calls([row([{'attempts': 1, 'usage': usage, 'cost': None}])])
    assert result['reported_response_usage'] == usage
    assert result['usage_complete_for_recorded_attempts'] is False


def test_explicit_zero_response_dimensions_are_complete():
    result = helper().summarize_calls([row([{'attempts': 1,
                                          'usage': {'input_tokens': 0, 'output_tokens': 0}, 'cost': None}])])
    assert result['usage_complete_for_recorded_attempts'] is True


@pytest.mark.parametrize('field,value', [('attempts', -1), ('attempts', True), ('cost', -1), ('cost', float('nan'))])
def test_invalid_numeric_accounting_rejected(field, value):
    call = {'stage': 'generation', 'attempts': 1, 'cost': 0.02}
    call[field] = value
    with pytest.raises(ValueError):
        helper().summarize_calls([row([call])])


def context_fixture(tmp_path, split='dev'):
    path = Path(__file__).with_name('test_evaluation_context.py')
    spec = importlib.util.spec_from_file_location('accounting_context_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture(tmp_path, split=split, phase='pipeline')


def test_complete_pipeline_preserves_all_failures_and_missing_trace(tmp_path):
    run, _ = context_fixture(tmp_path)
    result = helper().audit(run)
    assert result['complete'] and len(result['query_ids']) == 2
    assert result['methods']['B0']['operational']['errors'] == 1
    assert result['methods']['B0']['accounting']['rows_missing_provider_trace'] == 2
    assert result['methods']['B0']['accounting']['transport_attempts_total'] is None
    assert len(result['per_query']) == 4 and result['model_calls'] == 0


@pytest.mark.parametrize('kind', ['partial', 'inflight'])
def test_partial_or_inflight_pipeline_rejected(tmp_path, kind):
    run, _ = context_fixture(tmp_path)
    if kind == 'partial':
        p = run / 'B0.jsonl'
        p.write_text(p.read_text().splitlines()[0]+'\n', encoding='utf-8')
    else:
        (run / 'B0.inflight.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        helper().audit(run)


def test_test_requires_exact_lock_and_first30_cohort(tmp_path):
    run, lock = context_fixture(tmp_path, 'test')
    with pytest.raises(ValueError, match='requires exact frozen lock'):
        helper().audit(run, 'test')
    result = helper().audit(run, 'test', lock)
    assert len(result['query_ids']) == 30 and len(result['per_query']) == 60
    altered = json.loads(lock.read_text())
    altered['sha256'] = 'wrong'
    lock.write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError, match='lock mismatch'):
        helper().audit(run, 'test', lock)


def test_mutation_after_shared_validation_rejected(tmp_path, monkeypatch):
    run, _ = context_fixture(tmp_path)
    module = helper()
    import evaluation_context
    original = evaluation_context.audit

    def change_after_validation(*args, **kwargs):
        result = original(*args, **kwargs)
        p = run / 'B0.jsonl'
        p.write_bytes(p.read_bytes()+b'\n')
        return result

    monkeypatch.setattr(evaluation_context, 'audit', change_after_validation)
    with pytest.raises(ValueError, match='validated rows changed'):
        module.audit(run)


def test_controls_use_only_control_validator(tmp_path, monkeypatch):
    path = Path(__file__).with_name('test_control_decisions.py')
    spec = importlib.util.spec_from_file_location('accounting_control_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    run, _ = fixture.recorded_run.__wrapped__(tmp_path)
    module = helper()
    import evaluation_context

    def disallowed(*args, **kwargs):
        raise AssertionError('controls must not load FreshStack query annotations')

    monkeypatch.setattr(evaluation_context, 'audit', disallowed)
    result = module.audit(run)
    assert result['controls'] and len(result['query_ids']) == 10
    assert len(result['per_query']) == 30
    assert result['methods']['AllErrors']['accounting']['transport_attempts_total'] is None
