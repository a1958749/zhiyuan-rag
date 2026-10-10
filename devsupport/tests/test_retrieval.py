"""Retrieval contracts on explicitly artificial tiny corpora (no annotations)."""

import importlib
import json

import numpy as np
import pytest

from devsupport.types import Evidence


def api(name):
    """Fail explicitly on the missing production feature during RED."""
    assert importlib.util.find_spec('devsupport.' + name), f'missing production {name} module'
    return importlib.import_module('devsupport.' + name)


class TinyEmbedder:
    """Deterministic artificial 3D representation, never a real-model claim."""

    dimension = 3
    provenance = {'model': 'artificial-fixture', 'revision': 'test', 'dimension': 3,
                  'query_instruction': 'fixture', 'max_length': 2048}
    stats = {}

    def encode(self, texts, is_query=False):
        return np.array([[1, 0, 0] if 'JSONLoader' in text else [0, 1, 0] for text in texts], dtype=np.float32)


@pytest.fixture
def corpus():
    return {'z': Evidence('z', 'JSONLoader jq_schema TS-999 version 1.2.3', {'path': '/a.py'}),
            'a': Evidence('a', 'ordinary document', {'heading': 'Actual heading', 'answer_text': 'FORBIDDEN'})}


def test_identifier_tokens_preserve_complete_code_and_controlled_case():
    lexical = api('lexical')
    doc = lexical.identifier_tokens('JSONLoader jq_schema TS-999 pkg@1.2.3')
    assert {'JSONLoader', 'jsonloader', 'json', 'loader', 'jq_schema', 'jq', 'schema',
            'TS-999', 'ts-999', 'pkg@1.2.3', '1.2.3'} <= set(doc)
    assert lexical.identifier_tokens('JSONLoader') == lexical.identifier_tokens('JSONLoader', is_query=True)
    assert 'JSONLoader' not in lexical.identifier_tokens('jsonloader')
    assert lexical.ordinary_tokens('JSONLoader jq_schema TS-999') == ['jsonloader', 'jq_schema', 'ts', '999']


def test_rrf_exact_dedup_stable_ties_and_caps():
    rrf = api('retrieval').rrf
    result = rrf([['z', 'z', 'a'], ['a', 'z']], k=100)
    assert [doc_id for doc_id, score in result] == ['a', 'z']
    assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert len(rrf([[str(i) for i in range(100)]], k=100)) == 50
    with pytest.raises(ValueError):
        rrf([['a']], k=0)


def test_faiss_ids_vectors_and_original_text_roundtrip(tmp_path, corpus):
    index_type = api('index').CorpusIndex
    index = index_type.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={'raw_row_count': 2})
    loaded = index_type.load(tmp_path, expected_provenance=TinyEmbedder.provenance)
    assert loaded.ids == ['z', 'a']
    np.testing.assert_array_equal(loaded.vectors, index.vectors)
    assert loaded.corpus['z'].text == corpus['z'].text
    assert 'answer_text' not in loaded.corpus['a'].metadata
    assert loaded.dense_search(np.array([1, 0, 0], dtype=np.float32), 50)[0] == ('z', 1.0)
    assert loaded.manifest['count'] == 2
    assert loaded.manifest['data_sha256'] == 'fixture'


def test_index_detects_corrupt_and_missing_files(tmp_path, corpus):
    index_type = api('index').CorpusIndex
    index_type.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    with pytest.raises(ValueError, match='provenance'):
        index_type.load(tmp_path, expected_provenance={'revision': 'other'})
    (tmp_path / 'ids.json').write_text(json.dumps(['z', 'missing']), encoding='utf-8')
    with pytest.raises(ValueError, match='integrity'):
        index_type.load(tmp_path)
    (tmp_path / 'vectors.npy').unlink()
    with pytest.raises(ValueError, match='missing'):
        index_type.load(tmp_path)


def test_contextual_representation_never_rewrites_evidence(tmp_path, corpus):
    module = api('index')
    representation = module.contextual_text(corpus['a'])
    assert representation.endswith(corpus['a'].text)
    assert 'Actual heading' in representation and 'FORBIDDEN' not in representation
    index = module.CorpusIndex.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={}, contextual=True)
    assert index.corpus['a'].text == 'ordinary document'


@pytest.mark.parametrize('variant,kind', [('B0', 'cosine'), ('B1', 'BM25'), ('H0', 'RRF'), ('H1', 'RRF')])
def test_shared_search_types_bounds_and_candidate_ids(tmp_path, corpus, variant, kind):
    index = api('index').CorpusIndex.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    retriever = api('retrieval').Retriever(index, TinyEmbedder(), variant=variant)
    hits = retriever.search('JSONLoader', k=99)
    assert hits[0].evidence.doc_id == 'z'
    assert all(hit.score_kind == kind for hit in hits)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    assert retriever.pool_ids == [hit.evidence.doc_id for hit in hits]
    assert retriever.timings and len(hits) <= 50
    assert retriever.search('  ') == []
    with pytest.raises(TypeError):
        retriever.search(123)
    with pytest.raises(ValueError):
        retriever.search('JSONLoader', k=0)


def test_length_sorted_encoding_restores_caller_order_and_stats():
    module = api('embedding')
    # Fake only the expensive tokenizer/model protocol, retaining production ordering/pooling.
    class ProtocolEmbedder(module.Embedder):
        dimension = 3

        def _load(self):
            pass

        def _token_lengths(self, texts):
            return [len(text) for text in texts]

        def _encode_batch(self, texts):
            return np.array([[len(text), 1, 0] for text in texts], dtype=np.float32)

    embedder = ProtocolEmbedder(batch_size=2)
    texts = ['x' * 2100, 'x', 'xxxx']
    result = embedder.encode(texts)
    np.testing.assert_allclose(result[:, 0] / result[:, 1], [2100, 1, 4], rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1)
    assert embedder.stats['original_token_counts'] == [2100, 1, 4]
    assert embedder.stats['truncated_count'] == 1 and embedder.stats['max_length'] == 2048


def test_resume_rejects_changed_provenance_and_keeps_completed_prefix(tmp_path, corpus):
    index_type = api('index').CorpusIndex
    class Interrupted(TinyEmbedder):
        calls = 0

        def encode(self, texts, is_query=False):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('artificial interruption')
            return super().encode(texts, is_query)

    with pytest.raises(RuntimeError, match='interruption'):
        index_type.build(corpus, tmp_path, Interrupted(), data_sha256='fixture', audit={}, chunk_size=1)
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['completed_ids'] == ['z']
    with pytest.raises(ValueError, match='resume'):
        index_type.build(corpus, tmp_path, TinyEmbedder(), data_sha256='changed', audit={}, chunk_size=1)
    completed = index_type.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={}, chunk_size=1)
    assert completed.ids == ['z', 'a']


def test_model_archive_checksums_and_pinned_revision(tmp_path):
    module = api('embedding')
    import hashlib
    (tmp_path / 'model.safetensors').write_bytes(b'artificial weight file')
    manifest = {'repo': module.MODEL_ID, 'revision': module.MODEL_REVISION,
                'files': {'model.safetensors': {'sha256': hashlib.sha256(b'artificial weight file').hexdigest()}}}
    (tmp_path / 'artifact-manifest.json').write_text(json.dumps(manifest))
    embedder = module.Embedder(tmp_path)
    assert embedder.provenance['model_files_sha256'] == manifest['files']
    embedder.validate_local_artifacts()
    (tmp_path / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        embedder.validate_local_artifacts()


def test_last_nonpadding_pooling_with_left_or_right_padding():
    module = api('embedding')
    import torch
    hidden = torch.tensor([[[1.], [2.], [3.]], [[4.], [5.], [6.]]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    torch.testing.assert_close(module.last_token_pool(hidden, mask), torch.tensor([[3.], [5.]]))


def test_query_prefix_exact_and_oom_retry_observable():
    module = api('embedding')
    class RetryEmbedder(module.Embedder):
        dimension = 3
        calls = []

        def _load(self):
            pass

        def _token_lengths(self, texts):
            assert all(text.startswith(module.QUERY_INSTRUCTION) for text in texts)
            return [len(text) for text in texts]

        def _encode_batch(self, texts):
            self.calls.append(len(texts))
            if len(texts) > 1:
                raise RuntimeError('CUDA out of memory artificial fixture')
            return np.array([[1, 0, 0]], dtype=np.float32)

    embedder = RetryEmbedder(batch_size=2)
    output = embedder.encode(['first', 'second'], is_query=True)
    assert output.shape == (2, 3)
    assert embedder.calls == [2, 1, 1]
    assert embedder.stats['oom_retries'][0]['batch_size'] == 2
    assert embedder.stats['effective_batch_size'] == 1


def test_build_cli_has_full_corpus_defaults_and_configs_are_fixed():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('build_index_cli', root / 'scripts/build_index.py')
    assert spec.origin and Path(spec.origin).is_file(), 'missing production index CLI'
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = cli.parser().parse_args([])
    assert args.expected_raw_count == 49514 and args.expected_count == 49505
    assert args.batch_size == 16 and args.chunk_size == 1024 and not args.contextual
    config = api('config')
    for variant in ('B0', 'B1', 'H0', 'H1', 'C'):
        assert config.RetrievalConfig.load(root / f'devsupport/configs/{variant}.json').variant == variant


@pytest.mark.parametrize('vectors', [np.array([[0., 0., 0.]]), np.array([[np.nan, 1., 0.]]), np.ones((2, 4))])
def test_nonfinite_zero_and_wrong_dimension_vectors_rejected(vectors):
    with pytest.raises(ValueError):
        api('embedding').normalized(vectors, 3)


def test_native_faiss_is_never_deserialized_and_forged_artifact_detected(tmp_path, corpus, monkeypatch):
    import hashlib
    import faiss
    index_type = api('index').CorpusIndex
    index_type.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    def unsafe(*args):
        raise AssertionError('unsafe native deserialization attempted')
    monkeypatch.setattr(faiss, 'read_index', unsafe)
    assert index_type.load(tmp_path).ids == ['z', 'a']
    (tmp_path / 'dense.faiss').write_bytes(b'untrusted fake native file')
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['files']['dense.faiss'] = hashlib.sha256(b'untrusted fake native file').hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='FAISS vector integrity'):
        index_type.load(tmp_path)


def test_retrieval_rejects_unknown_ids_and_wrong_query_embedder(tmp_path, corpus):
    index = api('index').CorpusIndex.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    class Wrong(TinyEmbedder):
        provenance = dict(TinyEmbedder.provenance, revision='different')
    with pytest.raises(ValueError, match='provenance'):
        api('retrieval').Retriever(index, Wrong())
    del index.corpus['z']
    retriever = api('retrieval').Retriever(index, TinyEmbedder())
    with pytest.raises(ValueError, match='unknown'):
        retriever.search('JSONLoader')


def test_resume_rejects_reordered_ids_and_corrupt_checkpoint(tmp_path, corpus):
    index_type = api('index').CorpusIndex
    class Interrupted(TinyEmbedder):
        calls = 0
        def encode(self, texts, is_query=False):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('artificial interruption')
            return super().encode(texts, is_query)
    with pytest.raises(RuntimeError):
        index_type.build(corpus, tmp_path, Interrupted(), data_sha256='fixture', audit={}, chunk_size=1)
    with pytest.raises(ValueError, match='resume'):
        index_type.build(dict(reversed(list(corpus.items()))), tmp_path, TinyEmbedder(),
                         data_sha256='fixture', audit={}, chunk_size=1)
    (tmp_path / 'chunk-00000000.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='resume chunk integrity'):
        index_type.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={}, chunk_size=1)


def test_ordinary_bm25_empty_documents_keep_real_lengths():
    import bm25s
    lexical = api('lexical').LexicalIndex(['empty', 'word']).build(['', 'word'])
    reference = bm25s.BM25(**api('config').BM25_PARAMETERS)
    reference.index([[], ['word']], show_progress=False)
    assert lexical.search('word')[0][1] == pytest.approx(float(reference.get_scores(['word'])[1]))
    empty = api('lexical').LexicalIndex(['a', 'b']).build(['', ''])
    assert empty.search('__devsupport_empty_document__') == []


def test_rebuild_lexical_preserves_dense_and_corpus_checksums(tmp_path, corpus):
    index_type = api('index').CorpusIndex
    original = index_type.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    rebuilt = index_type.rebuild_lexical(tmp_path)
    assert rebuilt.manifest['files']['vectors.npy'] == original.manifest['files']['vectors.npy']
    assert rebuilt.manifest['files']['corpus.json'] == original.manifest['files']['corpus.json']
    assert rebuilt.manifest['lexical_rebuild_seconds'] >= 0
    assert rebuilt.ordinary.search('JSONLoader')[0][0] == 'z'


def test_interrupted_lexical_rebuild_keeps_v1_readable_and_cli_retry_repairs(tmp_path, monkeypatch):
    import bm25s
    from pathlib import Path
    module = api('index')
    tiny = {'empty': Evidence('empty', '', {}), 'word': Evidence('word', 'word', {})}
    module.CorpusIndex.build(tiny, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    # Reproduce the published v1 sentinel tokenizer used by the already-running build.
    prior = bm25s.BM25(**api('config').BM25_PARAMETERS)
    prior.index([['__devsupport_empty_document__'], ['word']], show_progress=False)
    prior.save(str(tmp_path / 'ordinary'), allow_pickle=False)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    for name in manifest['files']:
        if name.startswith('ordinary/'):
            manifest['files'][name] = module._file_hash(tmp_path / name)
    module._write_json(tmp_path / 'manifest.json', manifest)
    original_manifest = (tmp_path / 'manifest.json').read_bytes()
    dense_bytes = {name: (tmp_path / name).read_bytes() for name in
                   ('vectors.npy', 'dense.faiss', 'corpus.json', 'ids.json', 'audit.json')}
    prior_score = module.CorpusIndex.load(tmp_path).ordinary.search('word')[0][1]
    write_json = module._write_json

    def interrupt_publication(path, value):
        if path.name == 'manifest.json':
            raise RuntimeError('artificial interruption before manifest publication')
        return write_json(path, value)

    with monkeypatch.context() as interruption:
        interruption.setattr(module, '_write_json', interrupt_publication)
        with pytest.raises(RuntimeError, match='interruption'):
            module.CorpusIndex.rebuild_lexical(tmp_path)
    assert (tmp_path / 'manifest.json').read_bytes() == original_manifest
    assert module.CorpusIndex.load(tmp_path).ordinary.search('word')[0][1] == prior_score
    assert all((tmp_path / name).read_bytes() == data for name, data in dense_bytes.items())

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('repair_index_cli', root / 'scripts/build_index.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(['--output', str(tmp_path), '--rebuild-lexical']) == 0
    repaired = module.CorpusIndex.load(tmp_path)
    reference = bm25s.BM25(**api('config').BM25_PARAMETERS)
    reference.index([[], ['word']], show_progress=False)
    assert repaired.ordinary.search('word')[0][1] == pytest.approx(float(reference.get_scores(['word'])[1]))
    assert repaired.ordinary.search('word')[0][1] != prior_score
    assert all((tmp_path / name).read_bytes() == data for name, data in dense_bytes.items())
    # Published v1 lexical files are immutable even after successful migration.
    assert all(module._file_hash(tmp_path / name) == digest for name, digest in manifest['files'].items())


@pytest.mark.parametrize('unsafe', ['../outside', '/outside', 'C:/outside', 'ordinary/../identifier'])
def test_lexical_manifest_rejects_unsafe_directory_pointer(tmp_path, corpus, unsafe):
    module = api('index')
    module.CorpusIndex.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['lexical_paths'] = {'ordinary': unsafe, 'identifier': 'identifier'}
    module._write_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='unsafe'):
        module.CorpusIndex.load(tmp_path)


def directory_link_or_skip(link, target):
    """Create a real directory symlink, using a Windows junction if privileges prevent it."""
    import os
    import subprocess
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name != 'nt':
            pytest.skip(f'directory symlinks unavailable: {error}')
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                                capture_output=True, text=True)
        if result.returncode:
            pytest.skip(f'directory symlinks/junctions unavailable: {error}; {result.stderr}')
    assert link.resolve() == target.resolve()


def test_lexical_rebuild_rejects_staging_directory_escape_before_any_write(tmp_path, corpus):
    module = api('index')
    root = tmp_path / 'index'
    outside = tmp_path / 'outside'
    outside.mkdir()
    module.CorpusIndex.build(corpus, root, TinyEmbedder(), data_sha256='fixture', audit={})
    original_manifest = (root / 'manifest.json').read_bytes()
    original_vectors = (root / 'vectors.npy').read_bytes()
    directory_link_or_skip(root / 'lexical-versions', outside)
    with pytest.raises(ValueError, match='unsafe'):
        module.CorpusIndex.rebuild_lexical(root)
    assert list(outside.iterdir()) == []
    assert (root / 'manifest.json').read_bytes() == original_manifest
    assert (root / 'vectors.npy').read_bytes() == original_vectors
    assert module.CorpusIndex.load(root).ordinary.search('JSONLoader')[0][0] == 'z'


def test_lexical_rebuild_rechecks_staged_hashes_before_publication(tmp_path, corpus, monkeypatch):
    module = api('index')
    module.CorpusIndex.build(corpus, tmp_path, TinyEmbedder(), data_sha256='fixture', audit={})
    original_manifest = (tmp_path / 'manifest.json').read_bytes()
    file_hash = module._file_hash
    changed = False

    def change_previously_hashed_stage(path):
        nonlocal changed
        digest = file_hash(path)
        if 'lexical-versions' in path.parts and path.parent.name == 'identifier' and not changed:
            (path.parent.parent / 'ordinary' / 'data.csc.index.npy').write_bytes(b'artificial staged corruption')
            changed = True
        return digest

    monkeypatch.setattr(module, '_file_hash', change_previously_hashed_stage)
    with pytest.raises(ValueError, match='integrity'):
        module.CorpusIndex.rebuild_lexical(tmp_path)
    assert changed
    assert (tmp_path / 'manifest.json').read_bytes() == original_manifest
    assert module.CorpusIndex.load(tmp_path).ordinary.search('JSONLoader')[0][0] == 'z'


def test_lexical_rebuild_rechecks_directory_link_after_staging(tmp_path, corpus, monkeypatch):
    module = api('index')
    root = tmp_path / 'index'
    outside = tmp_path / 'outside'
    outside.mkdir()
    module.CorpusIndex.build(corpus, root, TinyEmbedder(), data_sha256='fixture', audit={})
    original_manifest = (root / 'manifest.json').read_bytes()
    real_save = module.LexicalIndex.save

    def redirect_saved_directory(index, staging):
        real_save(index, staging)
        if staging.name == 'ordinary':
            staging.rename(staging.with_name('ordinary-saved'))
            directory_link_or_skip(staging, outside)

    monkeypatch.setattr(module.LexicalIndex, 'save', redirect_saved_directory)
    with pytest.raises(ValueError, match='unsafe'):
        module.CorpusIndex.rebuild_lexical(root)
    assert list(outside.iterdir()) == []
    assert (root / 'manifest.json').read_bytes() == original_manifest
    assert module.CorpusIndex.load(root).ordinary.search('JSONLoader')[0][0] == 'z'
