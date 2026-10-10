"""Changing metric logic must invalidate a resumed reranking experiment."""

import importlib.util
from pathlib import Path


def test_metric_code_change_changes_reranking_contract(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'scripts/evaluate_reranking.py'
    spec = importlib.util.spec_from_file_location('rerank_runner', source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert hasattr(runner, 'implementation_hashes'), 'runner must bind all scoring implementation code'
    folder = tmp_path / 'src/devsupport'
    folder.mkdir(parents=True)
    for name in ('reranking.py', 'metrics.py', 'data.py', 'types.py'):
        (folder / name).write_text('version one', encoding='utf-8')
    before = runner.implementation_hashes(tmp_path)
    (folder / 'metrics.py').write_text('version two', encoding='utf-8')
    after = runner.implementation_hashes(tmp_path)
    assert before != after, 'old and new scoring code must not share an experiment contract'
    assert 'runner' in before
