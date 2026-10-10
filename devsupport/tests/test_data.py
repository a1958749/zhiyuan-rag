"""Behavioral tests using artificial data only."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from devsupport import data
from devsupport.data import build_split, load_corpus, load_questions, serialize_corpus, sha256_file, validate_dataset
from devsupport.types import Evidence, Question


FIXTURE = Path(__file__).parent / "fixtures" / "mini_freshstack.json"


def test_official_parquet_preserves_raw_utf8_text_and_metadata(tmp_path: Path) -> None:
    raw = "  人工 fixture\nreturn x\r\n"
    metadata = {"url": "https://example.invalid/a", "start_byte": 7, "end_byte": 33}
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist([{"_id": "repo/a.py_7_33", "text": raw, "metadata": metadata}]), path)
    corpus = load_corpus(path)
    assert corpus["repo/a.py_7_33"].text == raw
    assert corpus["repo/a.py_7_33"].metadata == metadata
    with pytest.raises(FrozenInstanceError):
        corpus["repo/a.py_7_33"].text = "changed"


def test_jsonl_rejects_duplicate_corpus_ids(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    row = {"doc_id": "a", "text": "artificial"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*a"):
        load_corpus(path)


def test_fixture_and_parquet_preserve_nested_question_annotations(tmp_path: Path) -> None:
    questions = load_questions(FIXTURE)
    assert questions[0].query == questions[0].query_title + " " + questions[0].query_text
    assert questions[0].nuggets[0]["relevant_corpus_ids"] == ["a", "b"]
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["questions"]
    path = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert load_questions(path) == questions


def test_corpus_serialization_excludes_nested_offline_labels() -> None:
    evidence = Evidence("a", "artificial", {
        "url": "https://example.invalid/a", "start_byte": 0, "end_byte": 10,
        "answer_text": "hidden", "nested": {"nuggets": ["hidden"], "source": "fixture"},
        "qrels": {"a": 1}, "relevant_corpus_ids": ["hidden"],
    })
    result = serialize_corpus({"a": evidence})
    assert result == [{"doc_id": "a", "text": "artificial", "metadata": {
        "url": "https://example.invalid/a", "start_byte": 0, "end_byte": 10,
        "nested": {"source": "fixture"},
    }}]
    result[0]["metadata"]["nested"]["source"] = "edited"
    assert evidence.metadata["nested"]["source"] == "fixture"


def test_validation_reports_issues_without_deleting_questions() -> None:
    corpus = load_corpus(FIXTURE)
    questions = load_questions(FIXTURE)
    questions.append(replace(questions[0], nuggets=[{
        "relevant_corpus_ids": ["missing"], "non_relevant_corpus_ids": ["also-missing"],
    }], query_text=""))
    corpus["empty"] = Evidence("empty", "", {})
    report = validate_dataset(corpus, questions)
    assert report["corpus_count"] == 7
    assert report["question_count"] == 5
    assert report["empty_corpus_ids"] == ["empty"]
    assert report["duplicate_query_ids"] == [questions[0].query_id]
    assert report["empty_query_ids"] == [questions[0].query_id]
    assert report["unknown_qrel_corpus_ids"] == ["also-missing", "missing"]
    assert len(questions) == 5


def test_split_is_input_order_independent_and_keeps_near_duplicates_together() -> None:
    questions = load_questions(FIXTURE)
    split = build_split(questions, dev_count=2)
    assert split == build_split(list(reversed(questions)), dev_count=2)
    assert set(split["dev_ids"]).isdisjoint(split["test_ids"])
    assert set(split["dev_ids"]) | set(split["test_ids"]) == {q.query_id for q in questions}
    assert ("q1", "q2") in split["near_duplicate_groups"]
    for group in split["near_duplicate_groups"]:
        assert set(group) <= set(split["dev_ids"]) or set(group) <= set(split["test_ids"])
    assert split["actual_counts"] == {"dev": 2, "test": 2}
    changed_answers = [replace(q, answer_text="different", nuggets=[]) for q in questions]
    assert split == build_split(changed_answers, dev_count=2)


def test_split_discloses_group_forced_deviation_and_rejects_duplicate_ids() -> None:
    queries = [Question(str(i), "", "Same artificial query?", "", [], {}) for i in range(3)]
    split = build_split(queries, dev_count=1)
    assert split["actual_counts"] == {"dev": 0, "test": 3}
    assert split["count_deviation"] == -1
    assert isinstance(split["dev_ids"], tuple)
    with pytest.raises(ValueError, match="duplicate"):
        build_split(queries + [queries[0]])


def test_hash_tracks_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_official_last_duplicate_policy_selects_last_and_audits_every_duplicate_row(tmp_path: Path) -> None:
    original = {"_id": "a", "text": "ARTIFICIAL original", "metadata": {"url": "https://example.invalid/a"}}
    same = {"_id": "b", "text": "ARTIFICIAL same", "metadata": {"start_byte": 0, "end_byte": 15}}
    last = {"_id": "a", "text": "ARTIFICIAL last", "metadata": {"url": "https://example.invalid/last"}}
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([original, same, original, same, last]), encoding="utf-8")
    source_bytes = path.read_bytes()
    corpus, audit = data.load_corpus_audited(path, duplicate_policy="official_last")
    assert corpus["a"].text == last["text"]
    assert corpus["a"].metadata["url"] == last["metadata"]["url"]
    assert load_corpus(path, duplicate_policy="official_last") == corpus
    assert path.read_bytes() == source_bytes
    assert audit["raw_row_count"] == 5
    assert audit["unique_id_count"] == 2
    assert audit["extra_rows"] == 3
    assert audit["conflicting_groups"] == 1
    assert audit["duplicate_policy"] == "official_last"
    assert audit["row_index_base"] == 0
    groups = {group["doc_id"]: group for group in audit["duplicate_groups"]}
    assert groups["a"]["selected_row_index"] == 4
    assert groups["b"]["selected_row_index"] == 3
    assert [row["row_index"] for row in groups["a"]["rows"]] == [0, 2, 4]
    assert [row["row_index"] for row in groups["b"]["rows"]] == [1, 3]
    assert groups["a"]["conflicting"] is True
    assert groups["b"]["conflicting"] is False
    assert groups["a"]["rows"][0]["text_sha256"] == groups["a"]["rows"][1]["text_sha256"]
    assert groups["a"]["rows"][0]["text_sha256"] != groups["a"]["rows"][2]["text_sha256"]
    assert groups["a"]["rows"][0]["metadata_sha256"] != groups["a"]["rows"][2]["metadata_sha256"]
    assert audit["source_sha256"] == sha256_file(path)
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(path)


def test_unsupported_duplicate_policy_raises_even_without_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate_policy"):
        load_corpus(FIXTURE, duplicate_policy="guess")
