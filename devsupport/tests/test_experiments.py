"""Regression contracts for honest, resumable offline experiments."""

import json

import pytest

from devsupport.experiments import run_retrieval, summarize
from devsupport.types import Evidence, Hit, Question


def question(qid):
    return Question(qid, 'find API', 'query body', 'SECRET GOLD ANSWER',
                    [{'text': 'SECRET NUGGET', 'relevant_corpus_ids': ['doc']}], {})


class Search:
    timings = {'total_seconds': 0.02}
    leg_ids = {'dense': ['doc']}

    def search(self, query, k):
        assert query == 'find API query body'
        assert k == 50
        return [Hit(Evidence('doc', 'raw evidence', {}), 0.5, 1, 'cosine')]


def test_resume_calls_only_missing_queries_and_never_passes_annotations(tmp_path):
    output = tmp_path / 'run.jsonl'
    manifest = {'variant': 'B0', 'split': 'dev', 'index_sha256': 'a'}
    run_retrieval([question('one')], Search(), output, manifest)

    class Count(Search):
        calls = 0

        def search(self, query, k):
            self.calls += 1
            return super().search(query, k)

    engine = Count()
    rows = run_retrieval([question('one'), question('two')], engine, output, manifest)
    assert engine.calls == 1
    assert len(rows) == 2
    assert 'SECRET' not in output.read_text()
    assert rows[0]['metrics']['coverage@10'] == 1


def test_changed_manifest_cannot_mix_results(tmp_path):
    output = tmp_path / 'run.jsonl'
    run_retrieval([question('one')], Search(), output, {'index_sha256': 'a'})
    with pytest.raises(ValueError, match='manifest'):
        run_retrieval([question('two')], Search(), output, {'index_sha256': 'b'})


def test_failed_question_stays_in_denominator_with_explicit_failure(tmp_path):
    class Fails(Search):
        def search(self, query, k):
            raise TimeoutError('provider detail MUST NOT be logged')

    rows = run_retrieval([question('one')], Fails(), tmp_path / 'run.jsonl', {})
    summary = summarize(rows)
    assert summary['n'] == 1
    assert summary['errors'] == 1
    assert summary['metrics']['coverage@10'] == 0
    assert rows[0]['error'] == 'TimeoutError'
    assert 'MUST NOT' not in json.dumps(rows)


def test_duplicate_or_out_of_order_ranking_is_integrity_failure(tmp_path):
    class Duplicates(Search):
        def search(self, query, k):
            one = super().search(query, k)[0]
            return [one, one]

    with pytest.raises(ValueError, match='ranking'):
        run_retrieval([question('one')], Duplicates(), tmp_path / 'run.jsonl', {})


def test_partial_checkpoint_is_rejected_instead_of_silently_skipped(tmp_path):
    output = tmp_path / 'run.jsonl'
    run_retrieval([question('one')], Search(), output, {})
    with output.open('a') as f:
        f.write('{"query_id":')
    with pytest.raises(ValueError, match='checkpoint'):
        run_retrieval([question('one'), question('two')], Search(), output, {})
