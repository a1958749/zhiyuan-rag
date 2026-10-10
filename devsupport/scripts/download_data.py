"""Verify or download the immutable FreshStack artifacts and write their audit/split."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from devsupport.data import build_split, load_corpus_audited, load_questions, sha256_file, validate_dataset

ARTIFACTS = {
    "corpus": {
        "repo": "freshstack/corpus-oct-2024", "revision": "069f66dc323e163b48b10d08408d282733d4393b",
        "file": "langchain/train-00000-of-00001.parquet", "local_file": "corpus.parquet",
        "sha256": "f1e63bf897f30062704cd77b22a4c4961ea74824c0511c17e20098f4f203fa52",
    },
    "queries": {
        "repo": "freshstack/queries-oct-2024", "revision": "023ac3a14caf9d6ebb13d01adcfb8fa05e5a9630",
        "file": "langchain/test-00000-of-00001.parquet", "local_file": "queries.parquet",
        "sha256": "319effb837de2f3cd8c6bda5d07a406db44ae2486f5239a0f145ca3eef71ac45",
    },
}


def prepare(data_dir: Path, reports_dir: Path) -> dict:
    """Fail on altered local data; audit every row and freeze a text-only query split."""
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for role, specification in ARTIFACTS.items():
        target = data_dir / specification["local_file"]
        if not target.exists():
            from huggingface_hub import hf_hub_download

            source = hf_hub_download(
                repo_id=specification["repo"], repo_type="dataset", revision=specification["revision"],
                filename=specification["file"], cache_dir=data_dir / ".download-cache", token=False,
            )
            temporary = target.with_suffix(".download")
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != specification["sha256"]:
                raise ValueError(f"download SHA-256 mismatch for {role}")
            os.replace(temporary, target)
        actual_hash = sha256_file(target)
        if actual_hash != specification["sha256"]:
            raise ValueError(f"existing immutable artifact SHA-256 mismatch for {role}; file left untouched")
        artifacts[role] = {**specification, "bytes": target.stat().st_size}

    corpus, duplicate_audit = load_corpus_audited(data_dir / "corpus.parquet", duplicate_policy="official_last")
    questions = load_questions(data_dir / "queries.parquet")
    validation = validate_dataset(corpus, questions)
    if not validation["valid"]:
        raise ValueError("full dataset validation failed; no rows or annotations were removed")
    split = build_split(questions, dev_count=53, seed=20261008)
    if split["actual_counts"]["dev"] != 53 or split["actual_counts"]["test"] != 150:
        raise ValueError("frozen source did not reproduce the approved 53/150 group-disjoint split")
    manifest = {
        "dataset": "FreshStack oct-2024 langchain", "license": "CC-BY-SA-4.0",
        "snapshot_date": "2024-10", "artifacts": artifacts, "duplicate_audit": duplicate_audit,
        "validation": validation, "split": split,
        "canonical_policy": "official_last; original 49,514 raw rows archived, 49,505 unique IDs indexed",
        "online_boundary": "only corpus records enter retrieval; queries, answers, nuggets and qrels stay offline",
    }
    destination = reports_dir / "data-manifest.json"
    encoded = json.dumps(manifest, indent=2, ensure_ascii=False)
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise ValueError("existing frozen manifest differs; refusing silent split/provenance rewrite")
    destination.write_text(encoded, encoding="utf-8")
    (reports_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Prepare artifacts and print a small safe verification summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.data, args.reports)
    print(json.dumps({"artifacts": manifest["artifacts"], "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
