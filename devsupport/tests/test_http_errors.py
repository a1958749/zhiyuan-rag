"""Synthetic HTTP errors never access providers, credentials or evaluation data."""
import io
import json
import urllib.error
from email.utils import formatdate

import pytest

from devsupport.llm import AnthropicProvider, ProviderConfig, ProviderError


NOW = 1_700_000_000
CANARY = 'secret-canary-do-not-persist'


@pytest.fixture
def http_provider():
    """Build a provider with a bounded-read spy and entirely local error transport."""
    def build(status=429, body=b'{}', retry_after=None, read_failure=False, clock=None):
        calls, sleeps, sizes = [], [], []

        class Body(io.BytesIO):
            def read(self, size=-1):
                sizes.append(size)
                assert 0 < size <= 16 * 1024
                if read_failure:
                    raise OSError(CANARY)
                return super().read(size)

        def send(*args, **kwargs):
            calls.append(1)
            headers = {'X-Secret': CANARY}
            if retry_after is not None:
                headers['Retry-After'] = retry_after
            raise urllib.error.HTTPError('https://example.invalid/' + CANARY, status, CANARY,
                                         headers, Body(body))

        provider = AnthropicProvider(ProviderConfig(), credential_loader=lambda: CANARY,
                                     transport=send, clock=clock or (lambda: NOW), sleep=sleeps.append)
        return provider, calls, sleeps, sizes
    return build


def test_http_403_keeps_only_safe_provider_diagnostics(http_provider):
    body = json.dumps({'error': {'code': '1007', 'type': 'permission_error', 'message': CANARY},
                       'request_id': CANARY}).encode()
    provider, calls, sleeps, sizes = http_provider(status=403, body=body)
    with pytest.raises(ProviderError, match='http_403') as caught:
        provider.complete('s', 'u')
    record = caught.value.record
    expected = {'provider_error_code': 1007, 'provider_error_type': 'permission_error'}
    assert record['http_error'] == expected
    assert record['events'][0]['http_error'] == expected
    assert len(calls) == 1 and not sleeps and sizes == [16 * 1024]
    assert CANARY not in json.dumps(record) + str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize('header,seconds,attempts', [
    ('120', 120, 1), ('86400', 86400, 1), ('0', 0, 2),
    (formatdate(NOW + 300, usegmt=True), 300, 1),
    (formatdate(NOW - 300, usegmt=True), 0, 2),
    (formatdate(NOW + 86401, usegmt=True), None, 2),
    (None, None, 2), ('garbage-' + CANARY, None, 2), ('86401', None, 2),
    ('-1', None, 2), ('NaN', None, 2), ('Infinity', None, 2), ('1.5', None, 2),
    ('9' * 20000, None, 2),
], ids=['delta', 'max-delta', 'zero', 'future-date', 'past-date', 'distant-date', 'absent',
         'malformed', 'over-limit', 'negative', 'nan', 'infinite', 'fractional', 'oversized'])
def test_retry_after_is_bounded_and_positive_wait_stops_transport_retry(http_provider, header, seconds, attempts):
    provider, calls, sleeps, _ = http_provider(retry_after=header)
    with pytest.raises(ProviderError, match='http_429') as caught:
        provider.complete('s', 'u')
    record = caught.value.record
    diagnostics = {} if seconds is None else {'retry_after_seconds': seconds}
    assert record.get('http_error', {}) == diagnostics
    assert all(event.get('http_error', {}) == diagnostics for event in record['events'])
    assert len(calls) == record['attempts'] == attempts
    assert not sleeps
    assert CANARY not in json.dumps(record)


@pytest.mark.parametrize('body,expected', [
    ({'code': 1007, 'type': 'rate_limit_error'},
     {'provider_error_code': 1007, 'provider_error_type': 'rate_limit_error'}),
    ({'error': {'code': 0}}, {'provider_error_code': 0}),
    ({'error': {'code': 1_000_000}}, {'provider_error_code': 1_000_000}),
    ({'error': {'code': 1.5}}, {'provider_error_code': 1.5}),
    ({'error': {'code': True, 'type': CANARY}}, {}),
    ({'error': {'code': -1}}, {}), ({'error': {'code': 1_000_001}}, {}),
    ({'error': {'code': 'nan'}}, {}), ({'error': {'code': 'inf'}}, {}),
    ({'error': {'code': [1007], 'type': ['permission_error']}}, {}),
    ({'error': CANARY}, {}), ([{'code': 1007}], {}),
])
def test_provider_error_fields_are_numeric_or_allowlisted(http_provider, body, expected):
    provider, _, _, _ = http_provider(status=403, body=json.dumps(body).encode())
    with pytest.raises(ProviderError) as caught:
        provider.complete('s', 'u')
    assert caught.value.record.get('http_error', {}) == expected
    assert CANARY not in json.dumps(caught.value.record)


@pytest.mark.parametrize('kind', ['rate_limit_error', 'authentication_error', 'permission_error',
                                 'invalid_request_error', 'overloaded_error', 'server_error'])
def test_known_provider_error_types_are_preserved(http_provider, kind):
    provider, _, _, _ = http_provider(status=403, body=json.dumps({'error': {'type': kind}}).encode())
    with pytest.raises(ProviderError) as caught:
        provider.complete('s', 'u')
    assert caught.value.record['http_error'] == {'provider_error_type': kind}


@pytest.mark.parametrize('body,read_failure', [
    (b'not json', False), (b'\xff', False),
    (json.dumps({'code': 1007, 'message': CANARY * 2000}).encode(), False),
    (b'{"code": 1007}' + b' ' * (16 * 1024) + CANARY.encode(), False),
    (b'{}', True),
], ids=['invalid-json', 'invalid-encoding', 'oversized', 'oversized-valid-prefix', 'read-failure'])
def test_bad_or_oversized_body_does_not_hide_retry_after(http_provider, body, read_failure):
    provider, calls, _, sizes = http_provider(body=body, retry_after='60', read_failure=read_failure)
    with pytest.raises(ProviderError, match='http_429') as caught:
        provider.complete('s', 'u')
    assert caught.value.record['http_error'] == {'retry_after_seconds': 60}
    assert len(calls) == 1 and sizes == [16 * 1024]
    assert CANARY not in json.dumps(caught.value.record)


def test_safe_http_diagnostics_survive_shared_pipeline_trace(http_provider):
    from test_adaptive import Retriever, hits
    from devsupport.pipeline import Pipeline, PipelineConfig

    body = json.dumps({'error': {'code': 1007, 'type': 'rate_limit_error', 'message': CANARY}}).encode()
    provider, calls, sleeps, _ = http_provider(body=body, retry_after='120')
    result = Pipeline(Retriever(hits()), provider, token_counter=lambda text: len(text.split()),
                      config=PipelineConfig(adaptive='never', gate_enabled=False)).run('Widget.render')
    trace = result['trace']['provider_calls'][0]
    assert trace['error'] == 'http_429'
    assert trace['http_error'] == {'provider_error_code': 1007, 'provider_error_type': 'rate_limit_error',
                                   'retry_after_seconds': 120}
    assert trace['events'][0]['http_error'] == trace['http_error']
    assert len(calls) == 1 and not sleeps
    assert CANARY not in json.dumps(result)


def test_instance_cooldown_blocks_credentials_and_transport_until_explicit_later_call(http_provider):
    now, credentials = [NOW], []
    provider, calls, sleeps, _ = http_provider(retry_after='120', clock=lambda: now[0])
    provider.credential_loader = lambda: credentials.append(1) or CANARY
    with pytest.raises(ProviderError, match='http_429'):
        provider.complete('s', 'u')
    now[0] += 60
    with pytest.raises(ProviderError, match='provider_cooldown') as caught:
        provider.complete('s', 'u')
    assert caught.value.record['http_error'] == {'retry_after_seconds': 60}
    assert caught.value.record['attempts'] == 0
    assert len(calls) == len(credentials) == 1 and not sleeps
    assert CANARY not in json.dumps(caught.value.record)
    now[0] += 60
    with pytest.raises(ProviderError, match='http_429'):
        provider.complete('s', 'u')
    assert len(calls) == len(credentials) == 2 and not sleeps


def test_adaptive_fallback_cannot_call_transport_again_during_cooldown(http_provider):
    from test_adaptive import Retriever, hits
    from devsupport.pipeline import Pipeline, PipelineConfig

    provider, calls, sleeps, _ = http_provider(retry_after='120')
    result = Pipeline(Retriever(hits()), provider, token_counter=lambda text: len(text.split()),
                      config=PipelineConfig(adaptive='always', gate_enabled=False)).run('Widget.render')
    trace = result['trace']['provider_calls']
    assert [(call['stage'], call['error']) for call in trace] == [
        ('decomposition', 'http_429'), ('generation', 'provider_cooldown')]
    assert trace[1]['http_error'] == {'retry_after_seconds': 120}
    assert len(calls) == 1 and not sleeps
    assert CANARY not in json.dumps(result)


def test_absent_retry_after_does_not_create_instance_cooldown(http_provider):
    provider, calls, sleeps, _ = http_provider(status=403)
    for _ in range(2):
        with pytest.raises(ProviderError, match='http_403'):
            provider.complete('s', 'u')
    assert len(calls) == 2 and not sleeps


@pytest.mark.parametrize('header', [
    'Tue, 14 Nov 2023 22:15:20 GMT ' + CANARY,
    '14 Nov 2023 22:15:20',
    'Tue, 14 Nov 2023 22:15:20 ' + CANARY,
    'Tue, 14 Nov 2023 22:15:20 +0000',
    'Tuesday, 14-Nov-23 22:15:20 GMT',
    'Mon, 14 Nov 2023 22:15:20 GMT',
    'Tue, 14 Nov 2023 22:15:20 GMT\r\n',
], ids=['trailing-secret', 'missing-day-and-zone', 'unrecognized-zone', 'numeric-zone',
         'obsolete-form', 'wrong-weekday', 'trailing-newline'])
def test_invalid_http_dates_do_not_create_retry_diagnostics_or_cooldown(http_provider, header):
    provider, calls, sleeps, _ = http_provider(retry_after=header)
    for _ in range(2):
        with pytest.raises(ProviderError, match='http_429') as caught:
            provider.complete('s', 'u')
        assert 'http_error' not in caught.value.record
        assert all('http_error' not in event for event in caught.value.record['events'])
        assert caught.value.record['attempts'] == 2
        assert CANARY not in json.dumps(caught.value.record)
    assert len(calls) == 4 and not sleeps
