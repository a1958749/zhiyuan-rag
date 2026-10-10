"""Export an explicit sanitized file list and verify packaging integrity offline."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

KEY_PATTERN = re.compile(rb'sk-[A-Za-z0-9_-]{24,}')
DUMMY_FIXTURE = 'tests/guardrails/test_redaction_boundaries.py'
DUMMY_HASHES = {'8ae48cf4dc50c10781a19c456c17116dc23c0ad57aeb93253278e5ed5b43bb02'}
ACCOUNT_NAME = re.compile(r'(^|[-_.])(balance|budget|accounting|billing|quota)([-_.]|$)')
BLOCKED_PARTS = {'.git', '.secrets', '__pycache__', '.venv', 'venv', 'runtime', 'indexes', 'inputs'}
BLOCKED_SUFFIXES = {'.dpapi', '.safetensors', '.faiss', '.npy', '.pt', '.pth', '.db', '.sqlite', '.sqlite3'}


def allowed_path(root: Path, name: str) -> Path:
    """Require a canonical root-relative public file with no sensitive storage path."""
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name
            or any(part in ('', '.', '..') or part.endswith((' ', '.')) for part in name.split('/'))):
        raise ValueError('invalid relative archive path')
    relative = PurePosixPath(name)
    parts = [p.casefold() for p in relative.parts]
    basename = parts[-1]
    if (relative.is_absolute() or any(p in BLOCKED_PARTS for p in parts)
            or relative.suffix.casefold() in BLOCKED_SUFFIXES
            or basename == '.env' or basename.startswith('.env.')
            or parts[:2] == ['devsupport', 'data']
            or ('model-cache' in parts and basename not in ('artifact-manifest.json', 'readme.md', 'license'))
            or ('reports' in parts and (ACCOUNT_NAME.search(basename) or 'account-screenshot' in basename))):
        raise ValueError('sensitive or bulky artifact excluded from handoff')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path != root / name or not path.is_file():
        raise ValueError('missing file, filesystem alias or path escapes archive root')
    return path


def export(root: Path, files: list[str], output: Path, forbidden_values: tuple[str, ...] = ()) -> dict:
    """Scan exact snapshots, publish exclusively and verify every archived file hash.

    Args:
        root: Source root for the explicit allowlist.
        files: Canonical relative file names, without directory/glob expansion.
        output: New ZIP filename, outside selected inputs.
        forbidden_values: Additional secret strings supplied in memory, never serialized.

    Returns:
        Packaging-only verification receipt. Semantic project acceptance is separate.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('handoff exists; never overwrite')
    if (not isinstance(files, list) or not files or any(not isinstance(n, str) for n in files)
            or len({n.casefold() for n in files}) != len(files)):
        raise ValueError('explicit nonempty unique file list required')
    if any(not isinstance(v, str) or not v for v in forbidden_values):
        raise ValueError('literal credential scan values must be nonempty strings')
    snapshots, inputs = {}, {}
    for name in files:
        path = allowed_path(root, name)
        if path == output:
            raise ValueError('output cannot be an input')
        raw = path.read_bytes()
        hashed = hashlib.sha256(raw).hexdigest()
        if any(v.encode('utf-8') in raw for v in forbidden_values):
            raise ValueError('credential literal in selected file')
        if KEY_PATTERN.search(raw) and not (name == DUMMY_FIXTURE and hashed in DUMMY_HASHES):
            raise ValueError('credential pattern in selected file')
        snapshots[name] = raw
        inputs[name] = {'sha256': hashed, 'bytes': len(raw)}
    for name, expected in inputs.items():
        if hashlib.sha256(allowed_path(root, name).read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('selected input changed before export')
    manifest = {'schema': 'devsupport.handoff.v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'acceptance_scope': 'packaging integrity only; not project or semantic acceptance',
                'files': inputs, 'scan': 'long sk-key pattern and caller-supplied in-memory literals',
                'media_limit': 'Byte scan does not inspect rendered pixels; selected media requires visual review.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr('DevSupportCopilot/' + name, raw)
        archive.writestr('handoff-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    zip_bytes = output.read_bytes()
    zip_hash = hashlib.sha256(zip_bytes).hexdigest()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()
        expected_names = {'DevSupportCopilot/' + name for name in files} | {'handoff-manifest.json'}
        if len(names) != len(set(names)) or set(names) != expected_names or archive.testzip() is not None:
            raise ValueError('archive integrity verification failed')
        for name, expected in inputs.items():
            if hashlib.sha256(archive.read('DevSupportCopilot/' + name)).hexdigest() != expected['sha256']:
                raise ValueError('archived bytes differ from selected snapshot')
        if json.loads(archive.read('handoff-manifest.json')) != manifest:
            raise ValueError('archived manifest differs')
    if hashlib.sha256(output.read_bytes()).hexdigest() != zip_hash:
        raise ValueError('archive changed after verification')
    return {'path': str(output), 'sha256': zip_hash,
            'bytes': len(zip_bytes), 'files': len(files), 'zip_integrity_verified': True,
            'acceptance_scope': manifest['acceptance_scope'], 'manifest': manifest}


def main() -> None:
    """Package a public JSON file-list; do not accept secrets via command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--files', type=Path, required=True, help='Public JSON array of relative file names')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = json.loads(args.files.read_text(encoding='utf-8-sig'))
    receipt = export(args.root, files, args.output)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'manifest'}))


if __name__ == '__main__':
    main()
