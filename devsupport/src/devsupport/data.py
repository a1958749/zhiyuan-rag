"""Read official rows unchanged and prepare reproducible offline evaluation splits."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path

from .types import Evidence, Question


def _rows(path: str | Path, section: str) -> list[dict]:
    """Read Parquet, JSONL or a JSON array/object fixture without altering rows."""
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(source).to_pylist()
    with source.open(encoding="utf-8") as handle:
        if source.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        elif source.suffix.lower() == ".json":
            rows = json.load(handle)
            if isinstance(rows, dict):
                rows = rows[section]
        else:
            raise ValueError(f"unsupported dataset format: {source.suffix}")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("dataset must contain a list of row objects")
    return rows


def _identifier(row: dict, keys: tuple[str, ...]) -> str:
    """Read a required nonempty ID without changing a supplied string."""
    value = next((row[key] for key in keys if key in row), None)
    if value is None or str(value) == "":
        raise ValueError(f"missing ID: expected one of {keys}")
    return str(value)


def _text(row: dict, key: str) -> str:
    """Read exact text, treating a missing or null text as empty for validation."""
    value = row.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def load_corpus(path: str | Path, duplicate_policy: str = "error") -> dict[str, Evidence]:
    """Load exact corpus chunks, rejecting duplicate IDs unless explicitly requested otherwise.

    Args:
        path: Official Parquet or a JSON/JSONL corpus fixture.
        duplicate_policy: "error" by default; "official_last" explicitly matches FreshStack's
            last-row dictionary semantics. Use load_corpus_audited to retain the selection audit.

    Returns:
        Evidence indexed by the exact source ID.

    Raises:
        ValueError: A corpus ID is duplicate or a required field is malformed.
    """
    corpus, _ = load_corpus_audited(path, duplicate_policy=duplicate_policy)
    return corpus


def load_corpus_audited(
    path: str | Path, duplicate_policy: str = "official_last",
) -> tuple[dict[str, Evidence], dict]:
    """Load a corpus with a full duplicate-row audit, leaving the source file untouched.

    Args:
        path: The archived official corpus or an artificial JSON/JSONL fixture.
        duplicate_policy: "official_last" selects the last raw row for each exact ID, as the
            official evaluator does. "error" rejects duplicate IDs instead of choosing a row.

    Returns:
        Evidence map and an audit with raw/unique counts, extra rows, conflicting groups,
        zero-based row indices, exact-text and canonical-JSON metadata hashes, selected row
        indices, and source-byte SHA-256. The original rows, IDs and qrels are never rewritten.

    Raises:
        ValueError: The policy is unknown, data is malformed, or strict policy finds duplicates.
    """
    if duplicate_policy not in ("error", "official_last"):
        raise ValueError("duplicate_policy must be 'error' or 'official_last'")
    rows = _rows(path, "corpus")
    corpus: dict[str, Evidence] = {}
    positions: dict[str, list[int]] = {}
    for row_index, row in enumerate(rows):
        doc_id = _identifier(row, ("_id", "doc_id", "id"))
        if doc_id in corpus and duplicate_policy == "error":
            raise ValueError(f"duplicate corpus ID: {doc_id}")
        positions.setdefault(doc_id, []).append(row_index)
        metadata = deepcopy(row.get("metadata") or {})
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be an object for {doc_id}")
        for key in ("url", "start_byte", "end_byte"):
            metadata.setdefault(key, row.get(key))
        corpus[doc_id] = Evidence(doc_id, _text(row, "text"), metadata)
    duplicate_groups = []
    for doc_id, indices in positions.items():
        if len(indices) < 2:
            continue
        audited_rows = []
        for index in indices:
            text_hash = hashlib.sha256(_text(rows[index], "text").encode("utf-8")).hexdigest()
            raw_metadata = json.dumps(rows[index].get("metadata"), sort_keys=True,
                                      ensure_ascii=False, separators=(",", ":"))
            metadata_hash = hashlib.sha256(raw_metadata.encode("utf-8")).hexdigest()
            audited_rows.append({"row_index": index, "text_sha256": text_hash, "metadata_sha256": metadata_hash})
        signatures = {(row["text_sha256"], row["metadata_sha256"]) for row in audited_rows}
        duplicate_groups.append({"doc_id": doc_id, "rows": audited_rows,
                                 "selected_row_index": indices[-1], "conflicting": len(signatures) > 1})
    audit = {
        "duplicate_policy": duplicate_policy, "row_index_base": 0, "raw_row_count": len(rows),
        "unique_id_count": len(corpus), "extra_rows": len(rows) - len(corpus),
        "conflicting_groups": sum(group["conflicting"] for group in duplicate_groups),
        "duplicate_groups": duplicate_groups, "source_sha256": sha256_file(path),
    }
    return corpus, audit


def load_questions(path: str | Path) -> list[Question]:
    """Load offline questions, retaining nested nuggets, answers and duplicate rows.

    Args:
        path: Official Parquet or a JSON/JSONL questions fixture.

    Returns:
        Questions in source order. Validation reports duplicate IDs separately.
    """
    questions = []
    for row in _rows(path, "questions"):
        nuggets = deepcopy(row.get("nuggets") or [])
        if not isinstance(nuggets, list) or any(not isinstance(nugget, dict) for nugget in nuggets):
            raise ValueError("nuggets must be a list of annotation objects")
        metadata = deepcopy(row.get("metadata") or {})
        if not isinstance(metadata, dict):
            raise ValueError("question metadata must be an object")
        if "answer_id" in row:
            metadata["answer_id"] = row["answer_id"]
        questions.append(Question(
            _identifier(row, ("query_id", "_id", "id")), _text(row, "query_title"),
            _text(row, "query_text"), _text(row, "answer_text"), nuggets, metadata,
        ))
    return questions


def relevant_doc_ids(nugget: dict) -> set[str]:
    """Extract binary-positive judgments from an official nugget or graded fixture."""
    labels = nugget.get("relevant_corpus_ids", nugget.get("relevant", [])) or []
    if isinstance(labels, dict):
        return {str(doc_id) for doc_id, grade in labels.items() if grade > 0}
    return {str(doc_id) for doc_id in labels}


def serialize_corpus(corpus: dict[str, Evidence]) -> list[dict]:
    """Produce detached corpus-only rows safe for indexing or live retrieval.

    Args:
        corpus: The corpus evidence map, containing no Question objects.

    Returns:
        Rows with annotation fields recursively excluded from metadata.
    """
    if any(not isinstance(item, Evidence) for item in corpus.values()):
        raise TypeError("corpus serialization accepts Evidence objects only")
    return [item.to_corpus_dict() for item in corpus.values()]


def validate_dataset(corpus: dict[str, Evidence], questions: list[Question]) -> dict:
    """Report dataset integrity issues without mutating data or dropping annotations.

    Args:
        corpus: Loaded evidence keyed by corpus ID.
        questions: Offline questions with all source annotations.

    Returns:
        Counts, duplicate/empty IDs, unknown positive and negative qrel IDs, and validity.
    """
    corpus_counts = Counter(item.doc_id for item in corpus.values())
    query_counts = Counter(query.query_id for query in questions)
    qrel_ids: set[str] = set()
    nugget_count = 0
    unjudged_count = 0
    for query in questions:
        for nugget in query.nuggets:
            nugget_count += 1
            if not relevant_doc_ids(nugget):
                unjudged_count += 1
            for key in ("relevant_corpus_ids", "non_relevant_corpus_ids", "relevant"):
                qrel_ids.update(str(doc_id) for doc_id in (nugget.get(key) or []))
    report = {
        "corpus_count": len(corpus), "question_count": len(questions), "nugget_count": nugget_count,
        "unjudged_nugget_count": unjudged_count,
        "unjudged_question_count": sum(not any(relevant_doc_ids(n) for n in q.nuggets) for q in questions),
        "empty_corpus_ids": sorted(item.doc_id for item in corpus.values() if not item.text.strip()),
        "empty_query_ids": sorted({q.query_id for q in questions if not q.query_text.strip()}),
        "duplicate_corpus_ids": sorted(key for key, count in corpus_counts.items() if count > 1),
        "duplicate_query_ids": sorted(key for key, count in query_counts.items() if count > 1),
        "unknown_qrel_corpus_ids": sorted(qrel_ids - set(corpus)),
    }
    issue_keys = ("empty_corpus_ids", "empty_query_ids", "duplicate_corpus_ids", "duplicate_query_ids",
                  "unknown_qrel_corpus_ids")
    report["valid"] = not any(report[key] for key in issue_keys)
    return report


def _normalized_query(query: Question) -> str:
    """Normalize only title/body for deterministic text-based duplicate grouping."""
    text = unicodedata.normalize("NFKC", query.query).casefold()
    return " ".join(re.findall(r"\w+", text))


def build_split(questions: list[Question], dev_count: int = 53, seed: int = 20261008) -> dict:
    """Freeze a seeded group-safe split using only original query title/body similarity.

    Args:
        questions: Offline queries; duplicate IDs are rejected.
        dev_count: Desired number of development queries, before indivisible grouping.
        seed: Local RNG seed, independent of input row order.

    Returns:
        Immutable ID/group tuples, actual counts, requested count and signed deviation.
        Connected groups use normalized SequenceMatcher ratio >=0.9. A subset-sum selection
        minimizes count deviation; ties prefer a smaller dev set. Answers and qrels never enter it.
    """
    if not isinstance(dev_count, int) or isinstance(dev_count, bool) or dev_count < 0:
        raise ValueError("dev_count must be a nonnegative integer")
    by_id = {q.query_id: q for q in questions}
    if len(by_id) != len(questions):
        raise ValueError("duplicate query IDs cannot be split")
    ids = sorted(by_id)
    normalized = {key: _normalized_query(by_id[key]) for key in ids}
    parent = {key: key for key in ids}

    def root(key: str) -> str:
        """Find a group representative with path compression."""
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            a, b = normalized[left], normalized[right]
            matcher = SequenceMatcher(None, a, b, autojunk=False)
            if (a and b and matcher.real_quick_ratio() >= 0.9 and matcher.quick_ratio() >= 0.9
                    and matcher.ratio() >= 0.9):
                parent[root(right)] = root(left)
    components: dict[str, list[str]] = {}
    for key in ids:
        components.setdefault(root(key), []).append(key)
    groups = sorted(tuple(group) for group in components.values())
    shuffled = groups.copy()
    random.Random(seed).shuffle(shuffled)
    choices: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(shuffled):
        for count, selected in list(choices.items()):
            choices.setdefault(count + len(group), selected + (index,))
    count = min(choices, key=lambda number: (abs(number - dev_count), number))
    dev_ids = tuple(sorted(key for index in choices[count] for key in shuffled[index]))
    dev_set = set(dev_ids)
    test_ids = tuple(key for key in ids if key not in dev_set)
    return {
        "dev_ids": dev_ids, "test_ids": test_ids,
        "near_duplicate_groups": tuple(group for group in groups if len(group) > 1),
        "actual_counts": {"dev": len(dev_ids), "test": len(test_ids)},
        "requested_dev_count": dev_count, "count_deviation": len(dev_ids) - dev_count, "seed": seed,
        "grouping_policy": "NFKC/casefold/word normalization; SequenceMatcher>=0.9; connected components",
    }


def sha256_file(path: str | Path) -> str:
    """Hash original source bytes for an immutable source manifest.

    Args:
        path: Source file to hash without decoding or normalizing its bytes.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
