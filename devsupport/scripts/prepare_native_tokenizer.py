"""Warm the native parser's Hugging Face cache from the frozen local tokenizer."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Copy verified tokenizer bytes into an offline native cache, without network calls."""
    manifest = json.loads((ROOT / 'devsupport/indexes/raw/manifest.json').read_text())
    provenance = manifest['provenance']
    source = ROOT / 'devsupport/model-cache/embedding/tokenizer.json'
    expected = provenance['model_files_sha256']['tokenizer.json']['sha256']
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('frozen tokenizer checksum mismatch')
    cache = ROOT / 'devsupport/runtime/native-tokenizer-cache'
    model = cache / ('models--' + provenance['model'].replace('/', '--'))
    snapshot = model / 'snapshots' / provenance['revision']
    snapshot.mkdir(parents=True, exist_ok=True)
    destination = snapshot / 'tokenizer.json'
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise ValueError('existing native tokenizer differs')
    if not destination.exists():
        shutil.copyfile(source, destination)
    (model / 'refs').mkdir(exist_ok=True)
    (model / 'refs/main').write_text(provenance['revision'], encoding='utf-8')
    from huggingface_hub import hf_hub_download
    resolved = hf_hub_download(provenance['model'], 'tokenizer.json', cache_dir=cache, local_files_only=True)
    print(json.dumps({'cache': str(cache), 'resolved': resolved, 'sha256': expected}))


if __name__ == '__main__':
    main()
