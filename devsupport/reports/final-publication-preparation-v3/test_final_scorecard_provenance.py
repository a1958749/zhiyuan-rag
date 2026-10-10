"""Synthetic scorecard checks for source coverage and queue substitution."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def renderer():
    spec = importlib.util.spec_from_file_location(
        'scorecard', Path(__file__).with_name('render-final-scorecard-20261010.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('field', [None, {}, []])
def test_summary_without_nonempty_byte_bindings_rejected(renderer, tmp_path, monkeypatch, field):
    path = tmp_path / 'observations/summary.json'
    path.parent.mkdir()
    value = {} if field is None else {'input_sha256': field}
    path.write_text(json.dumps(value))
    monkeypatch.setattr(renderer, 'REPORTS', tmp_path)
    with pytest.raises(ValueError, match='bindings'):
        renderer.read_report('observations', {})


def test_summary_must_bind_required_original_record(renderer, tmp_path, monkeypatch):
    unrelated = tmp_path / 'unrelated.json'
    unrelated.write_bytes(b'{}')
    path = tmp_path / 'observations/summary.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'input_sha256': {str(unrelated): renderer.publication.checksum(unrelated)}}))
    monkeypatch.setattr(renderer, 'REPORTS', tmp_path)
    with pytest.raises(ValueError, match='coverage'):
        renderer.read_report('observations', {}, required_inputs=[tmp_path / 'original.jsonl'])


@pytest.mark.parametrize('expected', [None, '', False])
def test_summary_cannot_replace_recorded_hash_with_missing_value(renderer, tmp_path, monkeypatch, expected):
    original = tmp_path / 'original.jsonl'
    original.write_bytes(b'{}\n')
    path = tmp_path / 'observations/summary.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'input_sha256': {str(original): expected}}))
    monkeypatch.setattr(renderer, 'REPORTS', tmp_path)
    with pytest.raises(ValueError, match='bindings'):
        renderer.read_report('observations', {}, required_inputs=[original])


@pytest.mark.parametrize('field,value', [('split', 'dev'), ('phase', 'pipeline'), ('controls', True),
                                       ('query_ids', ['other']), ('methods', {'H0': {}})])
def test_provider_queue_substitution_rejected(renderer, field, value):
    report = {'complete': True, 'contract_sha256': 'frozen', 'split': 'test', 'phase': 'retrieve-only',
              'controls': False, 'query_ids': ['q1'], 'methods': {'Q': {}, 'QAlways': {}}}
    report[field] = value
    with pytest.raises(ValueError, match='queue'):
        renderer.validate_observations(report, 'adaptive', 'frozen', ['q1'], ['Q', 'QAlways'])


def test_exact_provider_queue_accepted(renderer):
    report = {'complete': True, 'contract_sha256': 'frozen', 'split': 'test', 'phase': 'pipeline',
              'controls': True, 'query_ids': ['c1'], 'methods': {'G': {}, 'H0': {}}}
    renderer.validate_observations(report, 'controls', 'frozen', ['c1'], ['H0', 'G'])
