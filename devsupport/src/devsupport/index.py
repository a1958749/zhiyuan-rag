"""Safe corpus-only FAISS/BM25 artifacts with checksummed resumable embedding chunks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from uuid import uuid4

import numpy as np

from .config import BM25_PARAMETERS, LEG_CAP
from .embedding import normalized
from .lexical import LexicalIndex
from .types import Evidence


def _json_hash(value: object) -> str:
    """Hash canonical JSON without altering the saved original corpus text."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def _file_hash(path: Path) -> str:
    """Hash file bytes with bounded memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object):
    """Atomically replace one safe JSON file so interrupted writes do not look complete."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def _write_array(path: Path, array: np.ndarray):
    """Atomically write a non-object NumPy array with pickle disabled."""
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def _artifact_path(root: Path, name: str) -> Path:
    """Resolve a canonical relative artifact path without traversal or symlink escape."""
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
            PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or
            PurePosixPath(name).as_posix() != name):
        raise ValueError('integrity: unsafe index file path')
    source = root / name
    if not source.resolve().is_relative_to(root.resolve()):
        raise ValueError('integrity: unsafe index file path')
    return source


def contextual_text(evidence: Evidence) -> str:
    """Prefix factual existing URL/path/repository/heading only; keep Evidence unchanged."""
    metadata = evidence.to_corpus_dict()['metadata']
    context = [f'{key}: {metadata[key]}' for key in ('url', 'path', 'repository', 'heading')
               if isinstance(metadata.get(key), str) and metadata[key]]
    return '\n'.join(context + [evidence.text])


@dataclass
class CorpusIndex:
    """Exact official ID/text map and normalized dense + two lexical representations."""

    corpus: dict[str, Evidence]
    ids: list[str]
    vectors: np.ndarray
    dense: object
    ordinary: LexicalIndex
    identifier: LexicalIndex
    manifest: dict

    @classmethod
    def build(cls, corpus: dict[str, Evidence], path: str | Path, embedder: object,
              *, data_sha256: str, audit: dict, contextual: bool = False, chunk_size: int = 1024):
        """Build all raw retrieval artifacts, resuming verified ordered-ID embedding prefixes.

        Checkpoints contain arrays and JSON IDs/statistics only. A stale config, model,
        corpus or ordered prefix fails explicitly instead of reusing incompatible vectors.
        """
        import faiss

        if not corpus or chunk_size <= 0:
            raise ValueError('nonempty corpus and positive chunk_size required')
        ids = list(corpus)
        if any(key != evidence.doc_id for key, evidence in corpus.items()):
            raise ValueError('corpus key/ID mismatch')
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        rows = [corpus[doc_id].to_corpus_dict() for doc_id in ids]
        contract = {'schema': 1, 'count': len(ids), 'dimension': embedder.dimension,
                    'provenance': embedder.provenance, 'data_sha256': data_sha256,
                    'canonical_corpus_sha256': _json_hash(rows), 'audit_sha256': _json_hash(audit),
                    'ordered_ids_sha256': _json_hash(ids), 'contextual': contextual,
                    'bm25': BM25_PARAMETERS,
                    'tokenizers': {'ordinary': 'lowercase-word-no-stemming-v1',
                                   'identifier': 'exact-case-plus-lowercase-code-subterms-v1'}}
        progress_path = root / 'progress.json'
        if (root / 'manifest.json').exists():
            completed = cls.load(root, expected_provenance=embedder.provenance)
            if any(completed.manifest.get(key) != value for key, value in contract.items()):
                raise ValueError('resume completed index contract mismatch')
            return completed
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding='utf-8'))
            if progress.get('contract') != contract or progress.get('completed_ids') != ids[:len(progress['completed_ids'])]:
                raise ValueError('resume contract or ordered-ID prefix mismatch')
        else:
            progress = {'contract': contract, 'completed_ids': [], 'chunks': [], 'elapsed_seconds': 0}
        chunks = []
        verified_ids = []
        for item in progress['chunks']:
            chunk_path = root / item['file']
            if chunk_path.parent != root or not chunk_path.is_file() or _file_hash(chunk_path) != item['sha256']:
                raise ValueError('resume chunk integrity mismatch')
            values = np.load(chunk_path, allow_pickle=False)
            cls._validate_vectors(values, len(item['ids']), embedder.dimension)
            verified_ids.extend(item['ids'])
            chunks.append(values)
        if verified_ids != progress['completed_ids']:
            raise ValueError('resume checkpoint IDs mismatch')
        offset = len(verified_ids)
        while offset < len(ids):
            selected = ids[offset:offset + chunk_size]
            texts = [contextual_text(corpus[doc_id]) if contextual else corpus[doc_id].text for doc_id in selected]
            started = time.perf_counter()
            vectors = normalized(embedder.encode(texts), embedder.dimension)
            if vectors.shape[0] != len(selected):
                raise ValueError('embedding count mismatch')
            chunk_file = f'chunk-{offset:08d}.npy'
            _write_array(root / chunk_file, vectors)
            elapsed = time.perf_counter() - started
            progress['chunks'].append({'file': chunk_file, 'sha256': _file_hash(root / chunk_file),
                                       'ids': selected, 'stats': dict(embedder.stats), 'elapsed_seconds': elapsed})
            progress['completed_ids'].extend(selected)
            progress['elapsed_seconds'] += elapsed
            _write_json(progress_path, progress)
            chunks.append(vectors)
            offset += len(selected)
            print(json.dumps({'stage': 'embedding', 'completed': offset, 'total': len(ids),
                              'seconds': round(progress['elapsed_seconds'], 3),
                              'truncated_count': embedder.stats.get('truncated_count', 0)}), flush=True)
        all_vectors = np.concatenate(chunks)
        cls._validate_vectors(all_vectors, len(ids), embedder.dimension)
        dense = faiss.IndexFlatIP(embedder.dimension)
        dense.add(all_vectors)
        dense_temp = root / 'dense.faiss.tmp'
        faiss.write_index(dense, str(dense_temp))
        dense_temp.replace(root / 'dense.faiss')
        _write_array(root / 'vectors.npy', all_vectors)
        _write_json(root / 'ids.json', ids)
        _write_json(root / 'corpus.json', rows)
        _write_json(root / 'audit.json', audit)
        texts = [contextual_text(corpus[doc_id]) if contextual else corpus[doc_id].text for doc_id in ids]
        ordinary = LexicalIndex(ids).build(texts)
        identifier = LexicalIndex(ids, identifier=True).build(texts)
        ordinary.save(root / 'ordinary')
        identifier.save(root / 'identifier')
        files = [root / name for name in ('dense.faiss', 'vectors.npy', 'ids.json', 'corpus.json', 'audit.json')]
        files.extend(file for directory in ('ordinary', 'identifier') for file in (root / directory).iterdir() if file.is_file())
        manifest = dict(contract, files={file.relative_to(root).as_posix(): _file_hash(file) for file in files},
                        embedding_seconds=progress['elapsed_seconds'],
                        original_token_count=sum(sum(c['stats'].get('original_token_counts', [])) for c in progress['chunks']),
                        truncated_count=sum(c['stats'].get('truncated_count', 0) for c in progress['chunks']),
                        truncated_token_count=sum(c['stats'].get('truncated_token_count', 0) for c in progress['chunks']),
                        oom_retries=[event for c in progress['chunks'] for event in c['stats'].get('oom_retries', [])])
        _write_json(root / 'manifest.json', manifest)
        # Return the same detached safe corpus representation as a later load.
        safe_corpus = {row['doc_id']: Evidence(**row) for row in rows}
        return cls(safe_corpus, ids, all_vectors, dense, ordinary, identifier, manifest)

    @staticmethod
    def _validate_vectors(vectors: np.ndarray, count: int, dimension: int):
        """Reject wrong count/dimension, nonfinite data and non-unit saved vectors."""
        if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
            raise ValueError('vector count/dimension/finite mismatch')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError('vectors must be normalized')

    @classmethod
    def rebuild_lexical(cls, path: str | Path):
        """Rebuild only lexical artifacts of a verified index using current tokenizers.

        This offline maintenance operation preserves dense/corpus bytes and never
        re-embeds. Immutable lexical generations stage beside the original artifacts;
        one atomic manifest replacement publishes both legs and their hashes together.
        Interruption leaves the prior generation readable and a fresh retry safe. Old
        generations remain on disk so readers holding an earlier manifest stay valid.
        """
        root = Path(path)
        index = cls.load(root)
        started = time.perf_counter()
        texts = [contextual_text(index.corpus[doc_id]) if index.manifest['contextual']
                 else index.corpus[doc_id].text for doc_id in index.ids]
        generation = f'lexical-versions/{uuid4().hex}'
        lexical_paths = {name: f'{generation}/{name}' for name in ('ordinary', 'identifier')}
        # Existing unreferenced directory links must never redirect maintenance writes.
        staging_paths = {name: _artifact_path(root, directory) for name, directory in lexical_paths.items()}
        old_paths = index.manifest.get('lexical_paths', {'ordinary': 'ordinary', 'identifier': 'identifier'})
        files = {name: digest for name, digest in index.manifest['files'].items()
                 if not any(name.startswith(directory + '/') for directory in old_paths.values())}
        staged_files = {}
        for directory, identifier in (('ordinary', False), ('identifier', True)):
            lexical = LexicalIndex(index.ids, identifier=identifier).build(texts)
            staging = staging_paths[directory]
            staging.mkdir(parents=True, exist_ok=False)
            _artifact_path(root, lexical_paths[directory])
            lexical.save(staging)
            _artifact_path(root, lexical_paths[directory])
            for source in staging.iterdir():
                if source.is_file():
                    name = source.relative_to(root).as_posix()
                    staged_files[name] = _file_hash(_artifact_path(root, name))
        # Recheck containment and every staged digest before replacing the published
        # manifest, so a rejected generation cannot make a valid prior index unreadable.
        for directory in lexical_paths.values():
            _artifact_path(root, directory)
        for name, digest in staged_files.items():
            source = _artifact_path(root, name)
            if not source.is_file() or _file_hash(source) != digest:
                raise ValueError(f'staged index integrity mismatch: {name}')
        files.update(staged_files)
        index.manifest['files'] = files
        index.manifest['lexical_paths'] = lexical_paths
        index.manifest['lexical_rebuild_seconds'] = time.perf_counter() - started
        _write_json(root / 'manifest.json', index.manifest)
        return cls.load(root)

    @classmethod
    def load(cls, path: str | Path, *, expected_provenance: dict | None = None,
             expected_data_sha256: str | None = None):
        """Validate required files and integrity before safe NumPy/FAISS/BM25S loading."""
        import faiss

        root = Path(path)
        if not (root / 'manifest.json').is_file():
            raise ValueError('missing index manifest')
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != 1:
            raise ValueError('unsupported index schema')
        # Legacy schema-1 builds use fixed directories; maintenance adds only a
        # lexical pointer, leaving the dense contract and build-resume contract intact.
        lexical_paths = manifest.get('lexical_paths', {'ordinary': 'ordinary', 'identifier': 'identifier'})
        if not isinstance(lexical_paths, dict) or set(lexical_paths) != {'ordinary', 'identifier'}:
            raise ValueError('integrity: unsafe lexical directory mapping')
        lexical_directories = {name: _artifact_path(root, directory) for name, directory in lexical_paths.items()}
        required = {'dense.faiss', 'vectors.npy', 'ids.json', 'corpus.json', 'audit.json'}
        required.update(f'{directory}/{name}' for directory in lexical_paths.values() for name in
                        ('data.csc.index.npy', 'indices.csc.index.npy', 'indptr.csc.index.npy',
                         'vocab.index.json', 'params.index.json'))
        if not required <= manifest.get('files', {}).keys():
            raise ValueError('missing required manifest files')
        for name, digest in manifest['files'].items():
            source = _artifact_path(root, name)
            if not source.is_file():
                raise ValueError(f'missing index file: {name}')
        for name, digest in manifest['files'].items():
            source = root / name
            if _file_hash(source) != digest:
                raise ValueError(f'index integrity mismatch: {name}')
        if expected_provenance and any(manifest['provenance'].get(key) != value for key, value in expected_provenance.items()):
            raise ValueError('index provenance mismatch')
        if expected_data_sha256 and manifest['data_sha256'] != expected_data_sha256:
            raise ValueError('data checksum mismatch')
        ids = json.loads((root / 'ids.json').read_text(encoding='utf-8'))
        rows = json.loads((root / 'corpus.json').read_text(encoding='utf-8'))
        audit = json.loads((root / 'audit.json').read_text(encoding='utf-8'))
        if (len(ids) != manifest['count'] or len(set(ids)) != len(ids) or
                ids != [row['doc_id'] for row in rows] or _json_hash(ids) != manifest['ordered_ids_sha256'] or
                _json_hash(rows) != manifest['canonical_corpus_sha256'] or _json_hash(audit) != manifest['audit_sha256']):
            raise ValueError('canonical count/ID/audit integrity mismatch')
        corpus = {row['doc_id']: Evidence(**row) for row in rows}
        vectors = np.load(root / 'vectors.npy', allow_pickle=False)
        cls._validate_vectors(vectors, len(ids), manifest['dimension'])
        # Native FAISS input is not safely deserializable when supplied by an untrusted
        # source. Reconstruct IndexFlatIP from validated non-object NumPy vectors and
        # require its canonical native serialization to equal the saved native artifact.
        dense = faiss.IndexFlatIP(manifest['dimension'])
        dense.add(np.ascontiguousarray(vectors, dtype=np.float32))
        if hashlib.sha256(faiss.serialize_index(dense).tobytes()).hexdigest() != manifest['files']['dense.faiss']:
            raise ValueError('FAISS vector integrity mismatch')
        ordinary = LexicalIndex.load(lexical_directories['ordinary'], ids)
        identifier = LexicalIndex.load(lexical_directories['identifier'], ids, identifier=True)
        return cls(corpus, ids, vectors, dense, ordinary, identifier, manifest)

    def dense_search(self, query_vector: np.ndarray, k: int = LEG_CAP) -> list[tuple[str, float]]:
        """Search exact cosine/IP, ordering boundary ties by official ID deterministically."""
        if k <= 0:
            raise ValueError('k must be positive')
        query = normalized(np.asarray(query_vector).reshape(1, -1), self.manifest['dimension'])
        # IndexFlatIP has arbitrary ordering for ties. Full exact scores keep cap-boundary
        # membership deterministic too, rather than only sorting an arbitrary top-k subset.
        scores, positions = self.dense.search(query, len(self.ids))
        candidates = [(self.ids[int(position)], float(score)) for position, score in zip(positions[0], scores[0])]
        return sorted(candidates, key=lambda row: (-row[1], row[0]))[:min(k, LEG_CAP)]
