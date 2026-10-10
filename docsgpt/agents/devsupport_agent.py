"""Native guarded answer adapter for the frozen shared DevSupport pipeline."""
from devsupport.src.devsupport.bridge_client import BridgeClient
from docsgpt.agents.classic_agent import ClassicAgent
from docsgpt.guardrails.types import Stage


class DevSupportAgent(ClassicAgent):
    """Inherit native input/output controls; never invoke the native generator."""

    def __init__(self, devsupport_context=None, **kwargs):
        self.devsupport_context = devsupport_context
        super().__init__(**kwargs)

    def _gen_inner(self, query, log_context):
        """Buffer shared gate/citation validation and native output scan before release."""
        from docsgpt.api.answer.services.prompt_renderer import format_docs_for_prompt
        context = self.devsupport_context
        if not context or query != context.get('query') or not context.get('run_id'):
            raise ValueError('Missing or changed DevSupport prepared question')
        if self.attachments or self.earlier_attachments or self.skipped_attachments:
            raise ValueError('DevSupport snapshot profiles do not support attachments; select an ordinary source')
        decision = self._guardrail_stage(format_docs_for_prompt(self.retrieved_docs), Stage.RETRIEVAL)
        if decision is not None and (decision.blocked or decision.redacted):
            self.retrieved_docs = []
            yield from self._guarded_complete_answer(
                'Retrieved evidence was withheld by the configured content policy.', lambda text: {'answer': text})
            yield {'sources': []}
            return
        result = BridgeClient().answer(query, context['mapping'], context['session'], context['run_id'])
        if result['evidence_ids'] != [doc['doc_id'] for doc in self.retrieved_docs]:
            raise ValueError('Shared answer evidence differs from native prepared evidence')
        text = result['answer_text']
        if not text:
            trace = result.get('trace', {})
            missing = result.get('missing_information') or []
            failed = (trace.get('excluded_failure') or trace.get('retrieval_error')
                      or trace.get('gate', {}).get('reason') == 'gate_validation_or_provider_failure'
                      or trace.get('generation', {}).get('error'))
            if failed:
                text = 'The answer request could not complete or pass validation. Please try again.'
            else:
                text = ('Please clarify: ' + '; '.join(missing)) if missing else (
                    'The October 2024 snapshot does not provide sufficient evidence to answer this question.')
        self.devsupport_result = result
        log_context.stacks.append({'component': 'devsupport', 'data': result})
        yield from self._guarded_complete_answer(text, lambda value: {'answer': value})
        yield {'sources': self.retrieved_docs}
        yield {'tool_calls': []}
