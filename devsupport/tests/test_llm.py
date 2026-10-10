import json
import urllib.error

import pytest


class Response:
    def __init__(self, model='claude-opus-5-5', stop='end_turn'):
        self.model, self.stop = model, stop

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size=-1):
        return json.dumps({'model': self.model, 'content': [{'type': 'text', 'text': '{}'}],
                           'usage': {'input_tokens': 12, 'output_tokens': 2},
                           'stop_reason': self.stop}).encode()


def test_provider_caps_model_and_usage():
    from devsupport.llm import AnthropicProvider, ProviderConfig

    requests = []

    def send(request, timeout):
        requests.append((request, timeout))
        return Response()

    provider = AnthropicProvider(ProviderConfig(), credential_loader=lambda: 'test-secret', transport=send)
    result = provider.complete('system', 'hello', max_tokens=999999)
    assert json.loads(requests[0][0].data)['max_tokens'] <= 2048
    # Real smoke: default request truncated; explicit disabled thinking completed.
    assert json.loads(requests[0][0].data)['thinking'] == {'type': 'disabled'}
    assert result['thinking'] == 'disabled'
    assert result['actual_model'] == result['requested_model'] == 'claude-opus-5-5'
    assert result['usage']['input_tokens'] == 12
    assert result['cost'] is None
    with pytest.raises(ValueError):
        provider.complete('s', 'x' * 200001)


def test_foreign_host_redirect_and_mismatch_rejected():
    from devsupport.llm import AnthropicProvider, ProviderConfig, ProviderError, NoRedirect

    with pytest.raises(ValueError):
        AnthropicProvider(ProviderConfig(base_url='https://evil.test/v1'))
    with pytest.raises(ProviderError):
        NoRedirect().redirect_request(None, None, 302, '', {}, 'https://evil.test')
    provider = AnthropicProvider(ProviderConfig(), credential_loader=lambda: 'secret',
                                 transport=lambda *a, **k: Response('different-model'))
    with pytest.raises(ProviderError, match='model_mismatch') as exc:
        provider.complete('s', 'u')
    assert exc.value.record['actual_model'] == 'different-model'
    assert exc.value.record['usage']['input_tokens'] == 12


def test_timeout_bounded_and_sanitized():
    from devsupport.llm import AnthropicProvider, ProviderConfig, ProviderError

    calls = []

    def send(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError('test-secret full response')

    provider = AnthropicProvider(ProviderConfig(), credential_loader=lambda: 'test-secret', transport=send)
    with pytest.raises(ProviderError) as exc:
        provider.complete('s', 'u')
    assert len(calls) == 2
    assert 'test-secret' not in str(exc.value)
    assert exc.value.record['attempts'] == 2
    assert exc.value.__cause__ is None


def test_retry_cost_unknown_and_truncation_not_replayed():
    from devsupport.llm import AnthropicProvider, ProviderConfig, ProviderError

    replies = iter([TimeoutError('secret'), Response()])

    def send(*a, **k):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    provider = AnthropicProvider(ProviderConfig(input_per_million=1, output_per_million=2),
                                 credential_loader=lambda: 'secret', transport=send)
    result = provider.complete('s', 'u')
    assert result['cost'] is None
    assert result['known_response_cost'] > 0
    assert result['cost_status'] == 'partial_unknown_retry_usage'
    provider = AnthropicProvider(ProviderConfig(), credential_loader=lambda: 'secret',
                                 transport=lambda *a, **k: Response(stop='max_tokens'))
    with pytest.raises(ProviderError, match='output_truncated') as exc:
        provider.complete('s', 'u')
    assert exc.value.record['attempts'] == 1


def test_credential_environment_override_requires_explicit_opt_in(monkeypatch):
    from devsupport.credentials import load_credential
    import devsupport.credentials as credentials

    monkeypatch.setenv('DEVSUPPORT_OPUS_API_KEY', 'test-private')
    monkeypatch.setattr(credentials.os, 'name', 'posix')
    with pytest.raises(RuntimeError):
        load_credential()
    assert load_credential(env_override=True) == 'test-private'


def test_chat_gateway_alias_is_explicit_and_never_relabels_returned_model():
    from devsupport.llm import AnthropicProvider, ProviderConfig, ProviderError

    class ChatResponse(Response):
        def read(self, size=-1):
            return json.dumps({'model': self.model,
                               'choices': [{'message': {'content': '<think>private reasoning</think>\n{"ok":true}'},
                                            'finish_reason': 'stop'}],
                               'usage': {'prompt_tokens': 12, 'completion_tokens': 7}}).encode()

    requests = []
    def send(request, timeout):
        requests.append(request)
        return ChatResponse('claude-opus-4-8')

    config = ProviderConfig(protocol='openai_chat', model='deepseek-v4-pro',
                            expected_reported_model='claude-opus-4-8')
    provider = AnthropicProvider(config, credential_loader=lambda: 'test-key', transport=send)
    result = provider.complete('system', 'question', 512)
    assert requests[0].full_url.endswith('/v1/chat/completions')
    assert requests[0].get_header('Authorization') == 'Bearer test-key'
    assert json.loads(requests[0].data)['messages'][0] == {'role': 'system', 'content': 'system'}
    assert result['requested_model'] == 'deepseek-v4-pro'
    assert result['actual_model'] == result['expected_reported_model'] == 'claude-opus-4-8'
    assert result['identity_status'] == 'configured_gateway_alias_underlying_model_unverified'
    assert result['text'] == '{"ok":true}'
    assert result['usage'] == {'input_tokens': 12, 'output_tokens': 7}
    strict = AnthropicProvider(ProviderConfig(protocol='openai_chat', model='deepseek-v4-pro'),
                               credential_loader=lambda: 'test-key', transport=send)
    with pytest.raises(ProviderError, match='model_mismatch'):
        strict.complete('system', 'question')
