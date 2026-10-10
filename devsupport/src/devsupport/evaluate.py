"""Frozen offline evaluation with query-only runtime and fail-closed prepaid accounting."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit

from .metrics import evaluate_query


def digest(value: object) -> str:
    """Hash a canonical JSON contract, rejecting nonfinite values."""
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))


def text_hash(text: str) -> str:
    """Hash original UTF-8 text bytes."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_json(path: str | Path) -> dict:
    """Read public JSON configuration."""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path: str | Path, value: object) -> None:
    """Create an artifact exclusively; never silently replace old evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def freeze(path: str | Path, contract: dict) -> None:
    """Write the preregistration before any test execution; output is immutable by convention."""
    write_new(path, {'contract': contract, 'sha256': digest(contract)})


def verify_lock(path: str | Path, contract: dict) -> str:
    """Reject edited lock content or any changed source, configuration or data identity."""
    saved = read_json(path)
    if saved != {'contract': contract, 'sha256': digest(contract)}:
        raise ValueError('evaluation lock mismatch')
    return saved['sha256']


def normalize_profiles(profiles: dict) -> dict:
    """Resolve isolated ablations while forbidding per-group generator/prompt changes."""
    from .pipeline import PipelineConfig
    result = {}
    allowed = {'retrieval', 'reranker', 'adaptive', 'gate_enabled'}
    for name, settings in profiles.items():
        if not name or not name.replace('-', '').replace('_', '').isalnum() or set(settings) - allowed:
            raise ValueError('invalid profile or per-group generation override')
        value = {'retrieval': 'H1', 'reranker': None, 'adaptive': 'never', 'gate_enabled': False, **deepcopy(settings)}
        if value['retrieval'] not in ('B0', 'B1', 'H0', 'H1', 'U', 'C') or value['reranker'] not in (None, 'R1', 'R2'):
            raise ValueError('unsupported retrieval/reranking variant')
        if value['reranker'] and value['retrieval'] != 'H1':
            raise ValueError('rerankers require the identical H1 pool')
        PipelineConfig(adaptive=value['adaptive'], gate_enabled=value['gate_enabled'])
        result[name] = value
    if not result:
        raise ValueError('empty profiles')
    return result


def build_contract(config_path: str | Path, root: str | Path) -> dict:
    """Bind all runtime/scoring sources, model settings, prompts and artifact bytes."""
    from .data import sha256_file
    from .answer import ANSWER_PROMPT, TOKENIZER_PATH, TOKENIZER_SHA256
    from .adaptive import DECOMPOSITION_PROMPT
    from .gate import GATE_PROMPT
    root = Path(root).resolve()
    config = read_json(config_path)
    profiles = normalize_profiles(config['profiles'])
    paths = {key: str((root / config[key]).resolve()) for key in ('queries', 'corpus', 'index', 'split', 'provider')}
    split = read_json(paths['split'])
    dev, test = split['dev_ids'], split['test_ids']
    if len(dev) != 53 or len(test) != 150 or len(set(dev + test)) != 203:
        raise ValueError('expected immutable disjoint dev53/test150 split')
    selected_dev = config.get('dev_ids', dev)
    if not selected_dev or len(set(selected_dev)) != len(selected_dev) or not set(selected_dev) <= set(dev):
        raise ValueError('invalid development cohort')
    seeds = config.get('seeds', [20261008])
    if len(seeds) != 1 or type(seeds[0]) is not int:
        raise ValueError('one deterministic experiment/bootstrap seed required')
    generation = config.get('generation_max_tokens', 2048)
    if type(generation) is not int or not 1 <= generation <= 2048:
        raise ValueError('generation cap must be 1..2048')
    source = {str(p.relative_to(root)).replace('\\', '/'): sha256_file(p)
              for folder in ('devsupport/src/devsupport', 'devsupport/scripts')
              for p in sorted((root / folder).glob('*.py'))}
    artifacts = {key: sha256_file(path) for key, path in paths.items() if key != 'index'}
    artifacts['context_tokenizer'] = sha256_file(TOKENIZER_PATH)
    if artifacts['context_tokenizer'] != TOKENIZER_SHA256:
        raise ValueError('context tokenizer checksum mismatch')
    artifacts['index'] = sha256_file(Path(paths['index']) / 'manifest.json')
    artifacts['model_manifests'] = {
        name: sha256_file(root / 'devsupport/model-cache' / name / 'artifact-manifest.json')
        for name in ('embedding', 'pairwise', 'listwise')}
    artifacts['component_configs'] = {
        name: sha256_file(root / 'devsupport/configs' / (name + '.json'))
        for name in ('B0', 'B1', 'H0', 'H1', 'R1', 'R2')}
    for key in ('controls', 'budget_start'):
        if config.get(key):
            paths[key] = str((root / config[key]).resolve())
            artifacts[key] = sha256_file(paths[key])
    bind_optional_indexes(root, config, profiles, paths, artifacts, source)
    return {'schema': 1, 'source': source, 'config': config, 'config_sha256': sha256_file(config_path),
            'artifacts': artifacts, 'paths': paths, 'profiles': profiles,
            'provider': read_json(paths['provider']),
            'prompts': {k: text_hash(v) for k, v in {'answer': ANSWER_PROMPT, 'gate': GATE_PROMPT,
                                                    'decomposition': DECOMPOSITION_PROMPT}.items()},
            'split_ids': {'dev': selected_dev, 'test': test}, 'generation_ids': test[:30], 'seeds': seeds,
            'context_counter': 'cl100k_base_proxy_v1; 4000 proxy tokens, not exact gateway generation tokens',
            'generation_max_tokens': generation}



def bind_optional_indexes(root: Path, config: dict, profiles: dict, paths: dict,
                          artifacts: dict, source: dict) -> None:
    """Bind complete optional artifacts to the same raw corpus and embedding model."""
    from .data import sha256_file
    from .index import _artifact_path
    variants = {settings['retrieval'] for settings in profiles.values()}
    for variant, key, filename in [('U', 'native_u_index', 'native-manifest.json'),
                                    ('C', 'contextual_index', 'manifest.json')]:
        if variant not in variants:
            continue
        if not config.get(key):
            raise ValueError(f'{variant} requires {key}')
        folder = (root / config[key]).resolve()
        manifest_path = folder / filename
        if not manifest_path.is_file():
            raise ValueError(f'{key} requires a complete {filename}')
        manifest, raw = read_json(manifest_path), read_json(Path(paths['index']) / 'manifest.json')
        if (manifest.get('schema') != 1 or raw.get('contextual') is not False
                or raw.get('data_sha256') != artifacts['corpus']
                or any(manifest.get(k) != raw.get(k) for k in
                       ('provenance', 'canonical_corpus_sha256', 'count', 'dimension'))):
            raise ValueError(f'{key} corpus/model identity mismatch')
        if variant == 'U':
            if config.get('gpu_url', 'http://127.0.0.1:8093').rstrip('/') != 'http://127.0.0.1:8093':
                raise ValueError('native U requires the shared local embedding endpoint')
            required = {'index.faiss', 'index.json'}
            if manifest.get('profile') != 'U' or manifest.get('raw_manifest_sha256') != artifacts['index']:
                raise ValueError('native_u_index raw manifest mismatch')
            native_sources = sorted((root / 'docsgpt').rglob('*.py'))
            if not (root / 'docsgpt/retriever/classic_rag.py').is_file():
                raise ValueError('native U source unavailable')
            source.update({p.relative_to(root).as_posix(): sha256_file(p) for p in native_sources})
        else:
            if manifest.get('contextual') is not True or manifest.get('data_sha256') != artifacts['corpus']:
                raise ValueError('contextual_index must contain the same contextualized corpus')
            required = {'dense.faiss', 'vectors.npy', 'ids.json', 'corpus.json', 'audit.json'}
            lexical = manifest.get('lexical_paths', {'ordinary': 'ordinary', 'identifier': 'identifier'})
            if not isinstance(lexical, dict) or set(lexical) != {'ordinary', 'identifier'}:
                raise ValueError('invalid contextual lexical paths')
            required.update(f'{directory}/{name}' for directory in lexical.values() for name in
                            ('data.csc.index.npy', 'indices.csc.index.npy', 'indptr.csc.index.npy',
                             'vocab.index.json', 'params.index.json'))
        files = manifest.get('files', {})
        if not isinstance(files, dict) or not required <= files.keys():
            raise ValueError(f'{key} incomplete manifest')
        for name, expected in files.items():
            target = _artifact_path(folder, name)
            if not target.is_file() or sha256_file(target) != expected:
                raise ValueError(f'{key} artifact checksum mismatch')
        paths[key], artifacts[key] = str(folder), sha256_file(manifest_path)



def prepare_native_environment() -> None:
    """Select the documented shared-Qwen protocol in the standalone native Python runtime."""
    import sys
    from .config import MODEL_ID
    if sys.version_info < (3, 12):
        raise RuntimeError('native U requires the native Python 3.12 environment')
    os.environ.update(VECTOR_STORE='faiss', EMBEDDINGS_NAME=MODEL_ID,
                      EMBEDDINGS_BASE_URL='http://127.0.0.1:8093', EMBEDDINGS_MAX_QUERY_TOKENS='0',
                      EMBEDDINGS_MAX_INPUT_TOKENS='0')


def pipeline_result(query: str, pipeline: object, phase: str) -> dict:
    """Use the unchanged shared generator and context packer, retaining native search traces."""
    if isinstance(pipeline.retriever, TracedRetriever):
        pipeline.retriever.search_traces.clear()
    payload = pipeline.run(query) if phase == 'pipeline' else pipeline.retrieve(query)
    if isinstance(pipeline.retriever, TracedRetriever):
        payload['trace']['retrieval'] = deepcopy(pipeline.retriever.search_traces)
    return {**payload, 'evaluation_phase': phase}


def build_retriever(contract: dict, variant: str, raw_index: object, embedder: object) -> object:
    """Construct the genuine optional adapter before any paid provider is initialized."""
    from .index import CorpusIndex
    from .retrieval import Retriever
    if variant == 'U':
        from .native_baseline import NativeClassicRetriever
        folder = contract['paths']['native_u_index']
        return NativeClassicRetriever(contract['config'].get('native_u_source_id', Path(folder).name),
                                      folder, doc_token_limit=50000)
    index = raw_index
    if variant == 'C':
        index = CorpusIndex.load(contract['paths']['contextual_index'],
                                 expected_provenance=raw_index.manifest['provenance'])
    return Retriever(index, embedder, variant)


def retrieval_trace(query: str, retriever: object, hits: list) -> dict:
    """Retain genuine native trace plus the exact returned IDs and query bytes."""
    trace = {'query': query, 'ids': [h.evidence.doc_id for h in hits],
             'timings': deepcopy(getattr(retriever, 'timings', {})),
             'leg_ids': deepcopy(getattr(retriever, 'leg_ids', {}))}
    if hasattr(retriever, 'last_trace'):
        trace['native'] = deepcopy(retriever.last_trace)
    return trace


def retrieval_result(query: str, retriever: object) -> dict:
    """Execute one query-only local search and preserve original evidence/distance ordering."""
    started = time.perf_counter()
    hits = retriever.search(query, k=50)
    return {'ranked_ids': [h.evidence.doc_id for h in hits], 'scores': [h.score for h in hits],
            'score_kinds': [h.score_kind for h in hits],
            'trace': {'retrieval': [retrieval_trace(query, retriever, hits)]},
            'retrieval_elapsed_seconds': time.perf_counter() - started,
            'source_rows': [h.evidence.to_corpus_dict() for h in hits]}


class TracedRetriever:
    """Capture every real search for shared pipeline execution, including adaptive rounds."""

    def __init__(self, retriever: object):
        self.retriever, self.search_traces = retriever, []

    def __getattr__(self, name: str):
        return getattr(self.retriever, name)

    def search(self, query: str, k: int = 50) -> list:
        """Forward only query text and retain the native adapter trace after each search."""
        try:
            hits = self.retriever.search(query, k=k)
        except Exception:
            self.search_traces.append({'query': query, 'ids': [], 'error': 'retrieval_failed'})
            raise
        self.search_traces.append(retrieval_trace(query, self.retriever, hits))
        return hits


def select_ids(contract: dict, split: str, phase: str, lock: str | Path | None = None) -> list[str]:
    """Use first30 TEST IDs for pipeline generation; retrieval phases use all150."""
    if split not in ('dev', 'test') or phase not in ('pipeline', 'retrieve-only', 'retrieval', 'rerank'):
        raise ValueError('invalid split or phase')
    if split == 'test':
        if lock is None:
            raise ValueError('test requires a prior frozen evaluation lock')
        verify_lock(lock, contract)
        if contract['generation_ids'] != contract['split_ids']['test'][:30]:
            raise ValueError('generation cohort must be first30 holdout IDs')
    return list(contract['generation_ids'] if split == 'test' and phase == 'pipeline'
                else contract['split_ids'][split])


def numeric_balance(body: dict) -> dict:
    """Keep only numeric quota counters, never token names, keys or raw responses."""
    data = body.get('data', body)
    result = {k: v for k, v in data.items() if k in ('remain_quota', 'used_quota', 'total_granted',
              'total_used', 'total_available') and type(v) is int and v >= 0}
    if 'remain_quota' not in result and 'total_available' in result:
        result['remain_quota'] = result['total_available']
    if 'remain_quota' not in result:
        raise ValueError('balance unavailable')
    return result


def check_balance(provider_config: object, credential_loader=None, transport=None) -> dict:
    """GET the authorized token-usage endpoint with memory-only DPAPI and no redirects."""
    from .credentials import load_credential, validate_host
    from .llm import NoRedirect
    host = urlsplit(validate_host(provider_config.base_url))
    loader = credential_loader or (lambda: load_credential(provider_config.credential_path, False))
    opener = transport or urllib.request.build_opener(NoRedirect()).open
    try:
        request = urllib.request.Request(f'{host.scheme}://{host.netloc}/api/usage/token/',
                                         headers={'Authorization': 'Bearer ' + loader()}, method='GET')
        with opener(request, timeout=30) as response:
            raw = response.read(100001)
        if len(raw) > 100000:
            raise ValueError()
        return numeric_balance(json.loads(raw))
    except Exception:
        raise ValueError('balance unavailable') from None


class BudgetStopped(RuntimeError):
    """Finite prepaid allowance/reserve or unavailable balance prevents any further call."""


class BudgetGuard:
    """Charge all shared-key quota usage conservatively against root-authorized allowance."""

    def __init__(self, authorization: dict, checker, state_path: str | Path | None = None):
        for key in ('starting_quota', 'allowance_quota', 'reserve_quota'):
            value = authorization.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('finite numeric budget authorization required')
        if not 0 < authorization['allowance_quota'] <= authorization['starting_quota']:
            raise ValueError('invalid allowance')
        if authorization['reserve_quota'] <= 0:
            raise ValueError('positive finite reserve required')
        self.authorization, self.checker = deepcopy(authorization), checker
        self.observations, self.stopped = [], False
        self.previous_quota = authorization['starting_quota']
        self.consumed = 0
        self.state_path = Path(state_path) if state_path is not None else None
        self._restore()

    def _restore(self) -> None:
        """Restore shared accounting; reject edited authorization or corrupt state."""
        if self.state_path is None or not self.state_path.exists():
            return
        saved = read_json(self.state_path)
        checksum = saved.pop('sha256', None)
        if (checksum != digest(saved) or saved.get('authorization') != digest(self.authorization)
                or set(saved) != {'authorization', 'previous_quota', 'consumed'}
                or any(type(saved[k]) not in (int, float) or not math.isfinite(saved[k]) or saved[k] < 0
                       for k in ('previous_quota', 'consumed'))):
            raise ValueError('budget continuity mismatch')
        self.previous_quota, self.consumed = saved['previous_quota'], saved['consumed']

    def _persist(self) -> None:
        if self.state_path is None:
            return
        saved = {'authorization': digest(self.authorization), 'previous_quota': self.previous_quota,
                 'consumed': self.consumed}
        temporary = self.state_path.with_suffix(self.state_path.suffix + '.tmp')
        write_new(temporary, {**saved, 'sha256': digest(saved)})
        os.replace(temporary, self.state_path)

    def check(self) -> dict:
        """Fail closed before each provider call; allowance cannot grow after a top-up."""
        if self.state_path is None:
            return self._check_once()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.state_path.with_suffix(self.state_path.suffix + '.lock')
        try:
            handle = lock.open('x', encoding='utf-8')
        except FileExistsError:
            self.stopped = True
            raise BudgetStopped('budget_accounting_locked') from None
        try:
            self._restore()
            return self._check_once()
        except (OSError, ValueError):
            self.stopped = True
            raise BudgetStopped('budget_accounting_unavailable') from None
        finally:
            handle.close()
            lock.unlink()

    def _check_once(self) -> dict:
        if self.stopped:
            raise BudgetStopped('budget_stopped')
        try:
            balance = numeric_balance(self.checker())
        except Exception:
            self.stopped = True
            raise BudgetStopped('balance_unavailable') from None
        remaining = balance['remain_quota']
        self.consumed += max(0, self.previous_quota - remaining)
        self.previous_quota = remaining
        used = self.consumed
        self.observations.append({**balance, 'shared_usage_quota': used, 'quota_unit': 500000})
        self._persist()
        if (remaining <= self.authorization['reserve_quota'] or
                used >= self.authorization['allowance_quota']):
            self.stopped = True
            raise BudgetStopped('prepaid_allowance_or_reserve_reached')
        return balance


class GuardedProvider:
    """Wrap the real provider without changing generation prompts or replaying billed responses."""

    def __init__(self, provider, guard: BudgetGuard):
        self.provider, self.guard, self.config = provider, guard, provider.config

    def complete(self, *args, **kwargs):
        """Check before every decomposition, gate, generation and citation-repair call."""
        self.guard.check()
        try:
            result = self.provider.complete(*args, **kwargs)
        except Exception:
            # No further paid groups after transport, quota, model or truncation failure.
            self.guard.stopped = True
            raise
        # Do not perform fallible post-call work before the billed result can be checkpointed.
        return result


def rerank_pool(query: str, hits: list, ranker: object) -> list:
    """Require exact original50 evidence membership, including raw text and metadata."""
    original = {h.evidence.doc_id: h.evidence for h in hits}
    if len(hits) != 50 or len(original) != 50:
        raise ValueError('reranking requires exactly50 unique H1 candidates')
    ranked = ranker.rank(query, deepcopy(hits))
    if (len(ranked) != 50 or len({h.evidence.doc_id for h in ranked}) != 50 or
            any(h.evidence.doc_id not in original or h.evidence != original[h.evidence.doc_id] for h in ranked)):
        raise ValueError('reranker changed candidate membership or source bytes')
    return ranked


def make_record(question, result: dict, elapsed: float, error: str | None = None) -> dict:
    """Score offline only after the query-only runtime returns, preserving complete evidence."""
    trace = result.get('trace', {})
    prepack = result.get('ranked_ids', trace.get('final_candidate_ids', []))
    packed = result.get('evidence_ids', [])
    for ids in (prepack, packed):
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
            raise ValueError('invalid result ranking')
    if not set(packed) <= set(prepack):
        raise ValueError('packed evidence is outside ranking')
    packed_scores = {c['doc_id']: c.get('score') for c in result.get('context', [])}
    scores = result.get('scores', [packed_scores.get(i) for i in prepack])
    if len(scores) != len(prepack) or any(v is not None and
            (type(v) not in (int, float) or not math.isfinite(v)) for v in scores):
        raise ValueError('invalid ranking scores')
    row = {'query_id': question.query_id, 'query_sha256': text_hash(question.query), 'prepack_ids': prepack,
           'packed_ids': packed, 'prepack_scores': scores, 'metrics_prepack': evaluate_query(prepack, question),
           'metrics_packed': evaluate_query(packed, question), 'result': deepcopy(result),
           'context_sha256': {c['doc_id']: text_hash(c['text']) for c in result.get('context', [])},
           'elapsed_seconds': elapsed, 'error': error}
    row['record_sha256'] = digest(row)
    return row


def dependency_unavailable(exc: Exception) -> bool:
    """Identify local HTTPX transport/5xx failures without exposing diagnostic payloads."""
    try:
        import httpx
    except ImportError:
        return False  # Core-only installs can still checkpoint ordinary exceptions.
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, httpx.HTTPStatusError) and 500 <= exc.response.status_code <= 599
    )


def run(questions: list, execute, output: str | Path, manifest: dict, resume: bool = False,
        guard: BudgetGuard | None = None) -> list[dict]:
    """Append durable records in fixed order; a started but unrecorded call needs manual reconciliation.

    Runtime exceptions retain denominator with sanitized exception type. Corrupt contracts and
    checkpoints abort. An in-flight marker prevents a crash from silently replaying a paid call.
    """
    output = Path(output)
    meta = output.with_suffix('.manifest.json')
    pending = output.with_suffix('.inflight.json')
    ids = [q.query_id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate query IDs')
    bound = {**manifest, 'query_ids': ids, 'query_hashes': [text_hash(q.query) for q in questions]}
    if pending.exists():
        raise ValueError('unreconciled in-flight call; inspect billing/result before any resume')
    if meta.exists() or output.exists():
        if not resume:
            raise FileExistsError('existing experiment requires explicit resume')
        if not meta.exists() or read_json(meta) != bound:
            raise ValueError('experiment contract changed')
    else:
        write_new(meta, bound)
    rows = [json.loads(line) for line in output.read_bytes().splitlines()] if output.exists() else []
    if [r.get('query_id') for r in rows] != ids[:len(rows)] or len(rows) > len(ids):
        raise ValueError('nonmatching, duplicated or out-of-order checkpoint')
    for row, q in zip(rows, questions):
        expected = make_record(q, row['result'], row['elapsed_seconds'], row['error'])
        if row != expected:
            raise ValueError('checkpoint/query/metric integrity mismatch')
    for q in questions[len(rows):]:
        if guard is not None and guard.stopped:
            break
        write_new(pending, {'query_id': q.query_id, 'manifest_sha256': digest(bound)})
        start, error, result = time.perf_counter(), None, {}
        halt_after_record = False
        try:
            result = execute(q.query)
        except BudgetStopped:
            pending.unlink()
            break
        except Exception as exc:
            error = type(exc).__name__
            halt_after_record = dependency_unavailable(exc)
        if guard is not None:
            result['balance_observations'] = deepcopy(guard.observations)
        row = make_record(q, result, time.perf_counter() - start, error)
        with output.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        pending.unlink()
        rows.append(row)
        if halt_after_record:
            if guard is not None:
                guard.stopped = True
            break
    return rows


def cached_pools(rows: list[dict], manifest: dict, questions: list, corpus: dict,
                 questions_sha256: str, index_sha256: str, expected_contract: dict | None = None) -> dict:
    """Accept only intact current H1 caches bound to the exact frozen experiment."""
    from .types import Hit
    if (manifest.get('variant') != 'H1' or manifest.get('phase', 'retrieval') != 'retrieval' or
            manifest.get('questions_sha256') != questions_sha256 or
            manifest.get('index_manifest_sha256') != index_sha256):
        raise ValueError('candidate/query/index provenance mismatch')
    if (expected_contract is None or manifest.get('evaluation_contract') != expected_contract
            or manifest.get('contract_sha256') != digest(expected_contract)
            or manifest.get('settings') != expected_contract.get('profiles', {}).get('H1')
            or manifest.get('settings', {}).get('retrieval') != 'H1'
            or manifest.get('settings', {}).get('reranker') is not None
            or manifest.get('settings', {}).get('adaptive') != 'never'):
        raise ValueError('candidate frozen source/config provenance mismatch; legacy caches are unsupported')
    requested = {q.query_id: q for q in questions}
    if len(rows) != len(requested) or {r.get('query_id') for r in rows} != set(requested):
        raise ValueError('candidate cohort mismatch or duplicate')
    result = {}
    for row in rows:
        q = requested[row['query_id']]
        if row != make_record(q, row['result'], row['elapsed_seconds'], row['error']):
            raise ValueError('candidate record integrity mismatch')
        payload = row.get('result', row)
        ids = payload.get('ranked_ids', row.get('prepack_ids', []))
        scores, kinds = payload.get('scores', []), payload.get('score_kinds', [])
        if (row['query_sha256'] != text_hash(q.query) or row.get('error') or len(ids) != 50 or
                len(set(ids)) != 50 or not set(ids) <= set(corpus) or len(scores) != 50 or len(kinds) != 50 or
                any(type(s) not in (int, float) or not math.isfinite(s) for s in scores)):
            raise ValueError('invalid or mixed candidate pool')
        hits = [Hit(corpus[i], score, n, kind) for n, (i, score, kind) in enumerate(zip(ids, scores, kinds), 1)]
        if q.query in result and result[q.query] != hits:
            raise ValueError('same question has nonidentical cached pools')
        result[q.query] = hits
    return result
