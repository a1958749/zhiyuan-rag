"""Download pinned public model artifacts without executing their remote code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODELS = {
    "embedding": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "pairwise": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
    "listwise": ("jinaai/jina-reranker-v3.5", "e8a93f33f0b22108f8c2364f8484ce3422552fbc"),
}


def file_hash(path: Path) -> str:
    """Return SHA-256 without loading large weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(cache: Path, names: list[str]) -> dict:
    """Archive model revisions and hashes; leave model execution to the reviewed runtime."""
    from huggingface_hub import snapshot_download

    manifest = {}
    for role in names:
        repo, revision = MODELS[role]
        target = cache / role
        snapshot_download(
            repo_id=repo, revision=revision, local_dir=target, token=False, max_workers=4,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja", "README.md", "LICENSE*"],
        )
        files = {
            str(path.relative_to(target)).replace("\\", "/"): {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in sorted(target.rglob("*"))
            if path.is_file() and ".cache" not in path.parts and path.name != "artifact-manifest.json"
        }
        manifest[role] = {"repo": repo, "revision": revision, "files": files}
        print(json.dumps({"role": role, "revision": revision, "file_count": len(files)}), flush=True)
        (target / "artifact-manifest.json").write_text(json.dumps(manifest[role], indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Run a role-selectable public artifact download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    download(args.cache, args.roles)


if __name__ == "__main__":
    main()
