"""Archive boundaries exercised with real files; no credentials or network."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / 'analysis/export_handoff.py'
    assert path.is_file(), 'handoff exporter is not implemented'
    spec = importlib.util.spec_from_file_location('export_handoff', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def package(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'README.md').write_text('Public source', encoding='utf-8')
    (root / 'report.json').write_text('{"human_ratings":0}', encoding='utf-8')
    return root, tmp_path / 'handoff.zip'


def test_exact_files_and_manifest_are_verified(package):
    root, output = package
    result = load().export(root, ['README.md', 'report.json'], output)
    assert result['zip_integrity_verified'] is True
    with zipfile.ZipFile(output) as archive:
        assert archive.read('DevSupportCopilot/README.md') == (root / 'README.md').read_bytes()
        manifest = json.loads(archive.read('handoff-manifest.json'))
        assert len(manifest['files']) == 2
        assert manifest['acceptance_scope'] == 'packaging integrity only; not project or semantic acceptance'


@pytest.mark.parametrize('name', ['../outside', 'C:/secret', 'a\\b', '.secrets/x.dpapi', '.env',
                                'devsupport/data/corpus.parquet', 'devsupport/indexes/raw/ids.json',
                                'devsupport/model-cache/embedding/model.safetensors',
                                'devsupport/reports/budget-current-gateway.json', '.git/config'])
def test_forbidden_path_never_publishes(package, name):
    root, output = package
    with pytest.raises(ValueError):
        load().export(root, [name], output)
    assert not output.exists()


@pytest.mark.parametrize('text,known', [('sk-' + 'x' * 32, ()), ('private-value-42', ('private-value-42',))])
def test_secret_content_rejected_before_publication(package, text, known):
    root, output = package
    (root / 'README.md').write_text(text)
    with pytest.raises(ValueError, match='credential'):
        load().export(root, ['README.md'], output, forbidden_values=known)
    assert not output.exists()


def test_dummy_fixture_exception_requires_exact_bytes(package):
    root, output = package
    path = root / 'tests/guardrails/test_redaction_boundaries.py'
    path.parent.mkdir(parents=True)
    path.write_text('sk-' + 'y' * 32)
    with pytest.raises(ValueError, match='credential'):
        load().export(root, [path.relative_to(root).as_posix()], output)
    assert not output.exists()


def test_changed_input_rejected_before_zip_creation(package, monkeypatch):
    root, output = package
    original = Path.read_bytes
    changed = []

    def changing(path):
        value = original(path)
        if path == root / 'README.md' and not changed:
            path.write_text('Modified while read')
            changed.append(True)
        return value

    monkeypatch.setattr(Path, 'read_bytes', changing)
    with pytest.raises(ValueError, match='changed'):
        load().export(root, ['README.md'], output)
    assert not output.exists()


def test_existing_output_and_duplicate_entries_rejected(package):
    root, output = package
    quality = load()
    with pytest.raises(ValueError, match='unique'):
        quality.export(root, ['README.md', 'README.md'], output)
    output.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        quality.export(root, ['README.md'], output)
    assert output.read_bytes() == b'keep'


def test_public_symlink_cannot_select_excluded_private_storage(package):
    root, output = package
    secret = root / '.secrets/value.dpapi'
    secret.parent.mkdir()
    secret.write_bytes(b'Encrypted private payload')
    link = root / 'public.txt'
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip('This Windows host cannot create a symlink')
    with pytest.raises(ValueError):
        load().export(root, ['public.txt'], output)
    assert not output.exists()


def test_windows_trailing_dot_alias_cannot_duplicate_a_file(package):
    root, output = package
    if not (root / 'README.md.').exists():
        pytest.skip('Host does not resolve trailing-dot Windows aliases')
    with pytest.raises(ValueError):
        load().export(root, ['README.md.', 'README.md'], output)
    assert not output.exists()


@pytest.mark.parametrize('name', ['gate-protocol-dev-v1/balance.json', 'gateway-stop-v2-balance.json',
                                'quota-accounting.json'])
def test_account_credit_artifact_names_rejected(package, name):
    root, output = package
    relative = 'devsupport/reports/' + name
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"remain_quota":42}')
    with pytest.raises(ValueError):
        load().export(root, [relative], output)
    assert not output.exists()


def test_approved_upstream_dummy_fixture_can_be_included(package):
    root, output = package
    source = Path(__file__).resolve().parents[2] / 'tests/guardrails/test_redaction_boundaries.py'
    target = root / 'tests/guardrails/test_redaction_boundaries.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    result = load().export(root, [target.relative_to(root).as_posix()], output)
    assert result['zip_integrity_verified'] is True


def test_zip_rewrite_after_validation_cannot_receive_verified_receipt(package, monkeypatch):
    root, output = package
    original = zipfile.ZipFile.__exit__
    changed = []

    def mutate_after_verify(archive, *args):
        result = original(archive, *args)
        if archive.mode == 'r' and not changed:
            output.write_bytes(b'corrupt replacement')
            changed.append(True)
        return result

    monkeypatch.setattr(zipfile.ZipFile, '__exit__', mutate_after_verify)
    with pytest.raises(ValueError, match='changed'):
        load().export(root, ['README.md'], output)
    assert changed == [True]
