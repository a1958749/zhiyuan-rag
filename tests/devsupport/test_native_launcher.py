"""Operator settings must not be shadowed by an old local mock endpoint."""
import importlib.util
from pathlib import Path
from unittest.mock import patch


def test_gateway_launcher_disables_legacy_endpoint(monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    path = Path(__file__).resolve().parents[2] / 'devsupport/scripts/serve_native.py'
    spec = importlib.util.spec_from_file_location('native_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://127.0.0.1:8092/v1')
    with patch('devsupport.src.devsupport.credentials.load_credential', return_value='test-only-key'):
        module.configure()
    assert os.environ['OPENAI_BASE_URL'] == ''
    assert os.environ['DEVSUPPORT_GATEWAY_API_KEY'] == 'test-only-key'
    assert os.environ['INTERNAL_KEY'] == 'test-only-key'
    assert os.environ['EMBEDDINGS_NAME'] == 'Qwen/Qwen3-Embedding-0.6B'
    assert os.environ['EMBEDDINGS_BASE_URL'] == 'http://127.0.0.1:8093'
    assert os.environ['EMBEDDINGS_MAX_QUERY_TOKENS'] == '0'
