"""Shared query-only adaptive retrieval, gate and citation-validated answer pipeline."""

import hashlib
import json
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable

from .adaptive import (
    DECOMPOSITION_PROMPT,
    IDENTIFIER_ANCHORING_POLICY,
    IDENTIFIER_CAPACITY_POLICY,
    format_decomposition_request,
    fuse_pools,
    prepare_subqueries,
    should_expand,
    validate_subqueries,
)
from .answer import ANSWER_PROMPT, ProxyTokenCounter, format_question_context, pack_context, validate_answer
from .gate import GATE_PROMPT, validate_gate
from .llm import ProviderError
from .types import Hit


def _hash(value: object) -> str:
    """Hash canonical JSON for stable public configuration and prompt identities."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen experiment controls; expansion and final evidence gate are independent."""

    adaptive: str = 'selective'
    gate_enabled: bool = True
    context_budget: int = 4000
    candidate_limit: int = 50
    generation_max_tokens: int = 2048

    def __post_init__(self) -> None:
        if (self.adaptive not in ('never', 'always', 'selective') or type(self.gate_enabled) is not bool
                or self.context_budget != 4000 or self.candidate_limit != 50
                or not 1 <= self.generation_max_tokens <= 2048):
            raise ValueError('invalid pipeline bounds; shared context budget is fixed at 4000')


class Pipeline:
    """Inject corpus-only retrieval/reranking/provider/counter; never accept offline Question objects."""

    def __init__(self, retriever: object, provider: object, reranker: object = None,
                 token_counter: Callable[[str], int] | None = None,
                 config: PipelineConfig = PipelineConfig()) -> None:
        """Bind the shared components and initialize a bounded, instance-local prepared-search cache."""
        self.retriever, self.provider, self.reranker = retriever, provider, reranker
        self.token_counter, self.config = token_counter or ProxyTokenCounter(), config
        self._prepared = OrderedDict()

    def run(self, query: str) -> dict:
        """Return detached evidence, answer decision, accounting and reproducible prompt/config hashes."""
        return self.answer(query, self.retrieve(query))

    def retrieve(self, query: str) -> dict:
        """Retrieve, adapt and pack once; return a JSON envelope referencing server-held evidence."""
        result, packed, user = self._execute(query, retrieve_only=True)
        search_id = uuid.uuid4().hex
        result['search_id'] = search_id
        result['query'] = query
        self._prepared[search_id] = (deepcopy(result), deepcopy(packed), user)
        while len(self._prepared) > 32:
            self._prepared.popitem(last=False)
        return deepcopy(result)

    def answer(self, query: str, search_envelope: dict) -> dict:
        """Consume one server-held search; ignore all client-supplied evidence and trace fields."""
        if not isinstance(query, str):
            raise TypeError('query must be a plain string')
        search_id = search_envelope.get('search_id') if isinstance(search_envelope, dict) else None
        prepared = self._prepared.get(search_id) if isinstance(search_id, str) else None
        if prepared is None or prepared[0]['query'] != query:
            raise ValueError('unknown, consumed, expired or query-mismatched search')
        del self._prepared[search_id]
        return self._execute(query, prepared=prepared)

    def _execute(self, query: str, prepared: tuple | None = None, retrieve_only: bool = False) -> dict | tuple:
        """Execute the shared stages; prepared state is created only by retrieve on this instance."""
        if not isinstance(query, str):
            raise TypeError('query must be a plain string')
        if not query.strip() or len(query.encode('utf-8')) > 32000:
            raise ValueError('query must be nonempty and at most 32000 UTF-8 bytes')
        started = time.perf_counter()
        trace = {'retrieval': [], 'reranking': [], 'adaptive': {
                     'additional_rounds': 0, 'subqueries': [], 'anchoring_policy': IDENTIFIER_ANCHORING_POLICY,
                     'preflight_policy': IDENTIFIER_CAPACITY_POLICY},
                 'generation': {'attempts': 0}, 'provider_calls': []}
        result = {'decision': 'insufficient', 'answer_text': '', 'citation_ids': [], 'evidence_ids': [],
                  'context': [], 'missing_information': [], 'trace': trace, 'usage': {}, 'cost': None,
                  'prompt_hash': _hash(ANSWER_PROMPT),
                  'runtime_hash': _hash({'pipeline': asdict(self.config),
                                        'identifier_anchoring': IDENTIFIER_ANCHORING_POLICY,
                                        'identifier_capacity': IDENTIFIER_CAPACITY_POLICY,
                                        'counter': getattr(self.token_counter, 'name', 'injected'),
                                        'retriever': {'class': type(self.retriever).__name__,
                                                      'variant': getattr(getattr(self.retriever, 'config', None),
                                                                         'variant', None),
                                                      'index': getattr(getattr(self.retriever, 'index', None),
                                                                       'manifest', None)},
                                        'reranker': {'class': type(self.reranker).__name__,
                                                     'variant': getattr(self.reranker, 'variant', None)},
                                        'provider': asdict(self.provider.config)
                                        if is_dataclass(getattr(self.provider, 'config', None)) else
                                        type(self.provider).__name__})}
        trace['prompt_versions'] = {'answer': _hash(ANSWER_PROMPT), 'gate': _hash(GATE_PROMPT),
                                    'decomposition': _hash(DECOMPOSITION_PROMPT)}
        if prepared is not None:
            result, packed, user = prepared
            trace = result['trace']
            started -= trace.get('elapsed_seconds', 0)

        def finish() -> dict:
            calls = trace['provider_calls']
            result['usage'] = {}
            for call in calls:
                for key, value in call.get('usage', {}).items():
                    if type(value) is int:
                        result['usage'][key] = result['usage'].get(key, 0) + value
            costs = [call.get('cost') for call in calls]
            result['cost'] = sum(costs) if all(c is not None for c in costs) else None
            result['cost_status'] = ('no_provider_calls' if not calls else
                                     'estimated' if result['cost'] is not None else 'unverified')
            result['cost_scope'] = 'provider_only'
            trace['elapsed_seconds'] = time.perf_counter() - started
            return result

        def complete(stage: str, system: str, user: str, max_tokens: int) -> str:
            prompt_record = {'prompt_hash': _hash({'system': system, 'user': user}), 'max_tokens': max_tokens}
            try:
                response = self.provider.complete(system, user, max_tokens=max_tokens)
            except ProviderError as exc:
                record = {key: val for key, val in exc.record.items() if key != 'text'}
                trace['provider_calls'].append(dict(record, **prompt_record, stage=stage, error=exc.code))
                if exc.code == 'model_mismatch':
                    trace['excluded_failure'] = 'model_mismatch'
                raise ValueError('provider_failure') from None
            except Exception:
                trace['provider_calls'].append(dict(prompt_record, stage=stage, error='provider_failure', cost=None))
                raise ValueError('provider_failure') from None
            record = {key: response[key] for key in ('usage', 'cost', 'cost_status', 'requested_model',
                      'actual_model', 'elapsed_seconds', 'attempts', 'events', 'stop_reason', 'truncated',
                      'known_response_cost', 'thinking', 'protocol', 'expected_reported_model',
                      'identity_status', 'reasoning_prefix_removed', 'pacing_wait_seconds') if key in response}
            trace['provider_calls'].append(dict(record, **prompt_record, stage=stage))
            provider_config = getattr(self.provider, 'config', None)
            expected_model = (getattr(provider_config, 'expected_reported_model', None)
                              or getattr(provider_config, 'model', None) or response.get('requested_model'))
            identity_mismatch = (response.get('actual_model') != expected_model
                                 or response.get('requested_model') != getattr(
                                     provider_config, 'model', response.get('requested_model')))
            if identity_mismatch:
                trace['excluded_failure'] = 'model_mismatch'
            if (response.get('truncated') or response.get('stop_reason') == 'max_tokens'
                    or identity_mismatch):
                raise ValueError('provider_identity_or_truncation_failure')
            return response['text']

        def retrieve(search_query: str) -> list[Hit]:
            try:
                rows = self.retriever.search(search_query, k=50)[:50]
            except Exception:
                trace['retrieval'].append({'query': search_query, 'ids': [], 'error': 'retrieval_failed'})
                raise
            trace['retrieval'].append({'query': search_query, 'ids': [h.evidence.doc_id for h in rows],
                                       'timings': dict(getattr(self.retriever, 'timings', {})),
                                       'leg_ids': dict(getattr(self.retriever, 'leg_ids', {}))})
            return rows

        def rerank(rows: list[Hit]) -> list[Hit]:
            if self.reranker is None:
                return rows
            originals = {h.evidence.doc_id: h.evidence for h in rows}
            ranked = self.reranker.rank(query, deepcopy(rows[:50]))
            if (len(ranked) != len(rows) or len({h.evidence.doc_id for h in ranked}) != len(rows)
                    or any(h.evidence.doc_id not in originals or h.evidence != originals[h.evidence.doc_id]
                           for h in ranked)):
                raise ValueError('reranker changed evidence identity')
            trace['reranking'].append(dict(getattr(self.reranker, 'last_trace', {})))
            return ranked

        if prepared is not None:
            return self._generate(result, packed, user, complete, finish)
        first_pool = retrieve(query)
        if not first_pool:
            trace['gate'] = {'decision': 'insufficient', 'reason': 'no_evidence'}
            trace['final_candidate_ids'] = []
            return (finish(), [], '') if retrieve_only else finish()
        ranked = rerank(first_pool)
        trigger, reason = should_expand(self.config.adaptive, query, ranked)
        trace['adaptive'].update({'triggered': trigger, 'reason': reason})
        if trigger:
            failure_stage, failure_code = 'expansion_preparation', 'unknown_failure'
            try:
                _, summary, _ = pack_context(ranked, 4000, self.token_counter)
                failure_stage, failure_code = 'decomposition_provider', 'provider_failure'
                reply = complete('decomposition', DECOMPOSITION_PROMPT,
                                 format_decomposition_request(query, summary),
                                 self.config.generation_max_tokens)
                failure_stage, failure_code = 'decomposition_validation', 'unknown_validation_failure'
                preparation = prepare_subqueries(reply, query)
                trace['adaptive'].update(preparation)
                if preparation.get('anchoring_error'):
                    raise ValueError(preparation['anchoring_error'])
                subqueries = validate_subqueries(json.dumps({'subqueries': preparation['effective_subqueries']
                                                            or preparation['model_subqueries']}), query)
                trace['adaptive']['effective_subqueries'] = subqueries
                if not subqueries:
                    trace['adaptive'].update(appended_identifiers=[], anchored_query_count=0)
                if subqueries:
                    trace['adaptive'].update({'additional_rounds': 1, 'subqueries': subqueries})
                failure_stage, failure_code = 'subquery_retrieval', 'retrieval_failed'
                pools = [first_pool] + [retrieve(subquery) for subquery in subqueries]
                if subqueries:
                    failure_stage, failure_code = 'expanded_rerank', 'reranking_failed'
                    ranked = rerank(fuse_pools(pools))
            except Exception as exc:
                if failure_stage == 'decomposition_validation' and 'model_subqueries' in trace['adaptive']:
                    trace['adaptive'].update(effective_subqueries=[], appended_identifiers=[], anchored_query_count=0)
                if failure_stage == 'decomposition_validation' and isinstance(exc, ValueError):
                    code = str(exc)
                    if code in ('invalid_json', 'invalid_subqueries', 'missing_exact_identifiers',
                                'identifier_anchoring_overflow'):
                        failure_code = code
                trace['adaptive'].update(fallback='first_round_preserved', failure_stage=failure_stage,
                                         failure_code=failure_code)
        trace['final_candidate_ids'] = [hit.evidence.doc_id for hit in ranked]
        packed, context, packing = pack_context(ranked, self.config.context_budget, self.token_counter)
        result['context'] = [dict(hit.evidence.to_corpus_dict(), score=hit.score, rank=hit.rank,
                                  score_kind=hit.score_kind, citation_number=n) for n, hit in enumerate(packed, 1)]
        result['evidence_ids'] = [hit.evidence.doc_id for hit in packed]
        trace['packing'] = packing
        user = format_question_context(query, context)
        result['prompt_hash'] = _hash({'system': ANSWER_PROMPT, 'user': user})
        if retrieve_only:
            return finish(), packed, user
        return self._generate(result, packed, user, complete, finish)

    def _generate(self, result: dict, packed: list[Hit], user: str, complete: Callable, finish: Callable) -> dict:
        """Run the same final gate and generation logic for both UI and offline callers."""
        trace = result['trace']
        if trace.get('excluded_failure'):
            result['decision'] = 'insufficient'
            return finish()
        if not packed:
            trace.setdefault('gate', {'decision': 'insufficient', 'reason': 'no_evidence_fits_budget'})
            return finish()
        if self.config.gate_enabled:
            try:
                gate = validate_gate(complete('gate', GATE_PROMPT, user, self.config.generation_max_tokens))
            except Exception:
                trace['gate'] = {'decision': 'insufficient', 'reason': 'gate_validation_or_provider_failure'}
                return finish()
            trace['gate'] = gate
            result['decision'], result['missing_information'] = gate['decision'], gate['missing_information']
            if gate['decision'] != 'answer':
                return finish()
        else:
            trace['gate'] = {'enabled': False, 'reason': 'ablation_citations_still_required'}
        result['decision'] = 'insufficient'
        for attempt in range(2):
            trace['generation']['attempts'] = attempt + 1
            prompt = user if attempt == 0 else user + '\nPrevious output failed citation/schema validation. '
            if attempt:
                prompt += 'Return corrected JSON with exact source IDs and matching [n] citations only.'
            try:
                text = complete('generation' if attempt == 0 else 'citation_repair', ANSWER_PROMPT,
                                prompt, self.config.generation_max_tokens)
            except Exception:
                trace['generation']['error'] = 'provider_failure'
                break  # Transport owns retries; do not replay a billed response or failed identity.
            try:
                answer = validate_answer(text, packed)
                result.update(answer, decision='answer')
                return finish()
            except ValueError as exc:
                trace['generation']['error'] = 'citation_or_schema_validation_failed'
                code = str(exc)
                if code not in ('invalid_json', 'invalid_answer_schema', 'invalid_citations',
                                'invalid_citation_number_format'):
                    code = 'invalid_answer_validation'
                # complete() returns the provider's post-reasoning-stripped answer text.
                # At most two bounded records; never retain provider thoughts or exception details.
                trace['generation'].setdefault('validation_failures', []).append(
                    {'error': code, 'response_text': text[:16384] if isinstance(text, str) else ''})
        return finish()
