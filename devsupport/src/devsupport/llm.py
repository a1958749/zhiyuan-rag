"""Bounded gateway transports with explicit reported identity and usage records."""

import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import format_datetime, parsedate_to_datetime
from typing import Callable

from .credentials import load_credential, validate_host
from .pacing import PacingError, RequestPacer


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable public provider settings; prices are estimates only when explicitly supplied."""

    base_url: str = 'https://www.xiaoyaoapi.com/v1'
    model: str = 'claude-opus-5-5'
    timeout_seconds: float = 90
    retries: int = 1
    max_output_tokens: int = 2048
    max_input_bytes: int = 200000
    input_per_million: float | None = None
    output_per_million: float | None = None
    env_override: bool = False
    credential_path: str = 'F:/DevSupportCopilot/.secrets/opus.dpapi'
    thinking: str = 'disabled'
    protocol: str = 'anthropic_messages'
    expected_reported_model: str | None = None
    min_request_interval_seconds: float = 0
    rate_state_path: str | None = None

    def __post_init__(self) -> None:
        if (type(self.min_request_interval_seconds) not in (int, float)
                or not 0 <= self.min_request_interval_seconds <= 120
                or (self.rate_state_path is not None
                    and (not isinstance(self.rate_state_path, str) or not self.rate_state_path.strip()))):
            raise ValueError('invalid request pacing configuration')
        validate_host(self.base_url)
        if (self.protocol not in ('anthropic_messages', 'openai_chat')
                or (self.expected_reported_model is not None
                    and not re.fullmatch(r'[A-Za-z0-9._-]{1,100}', self.expected_reported_model))):
            raise ValueError('invalid provider protocol or reported model')
        if (self.thinking != 'disabled' or not re.fullmatch(r'[A-Za-z0-9._-]{1,100}', self.model)
                or not 0 < self.timeout_seconds <= 90 or self.retries not in (0, 1)
                or not 1 <= self.max_output_tokens <= 2048 or not 1 <= self.max_input_bytes <= 200000):
            raise ValueError('invalid bounded provider configuration')
        for price in (self.input_per_million, self.output_per_million):
            if price is not None and (not isinstance(price, (int, float)) or not 0 <= price < 100000):
                raise ValueError('invalid estimated price')


class ProviderError(RuntimeError):
    """A sanitized failure with usage retained when the server returned a billed response."""

    def __init__(self, code: str, record: dict | None = None) -> None:
        super().__init__(code)
        self.code, self.record = code, record or {}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent credential forwarding even to another path on the trusted host."""

    def redirect_request(self, req: object, fp: object, code: int, msg: str, headers: object,
                         newurl: str) -> None:
        raise ProviderError('redirect_refused')


def parse_json(text: str) -> dict:
    """Read a JSON object, optionally wrapped by one Markdown JSON fence."""
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError('invalid_json')
    value = text.strip()
    if value.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', value, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError('invalid_json')
        value = match.group(1)
    try:
        result = json.loads(value)
    except (ValueError, TypeError):
        raise ValueError('invalid_json') from None
    if not isinstance(result, dict):
        raise ValueError('invalid_json')
    return result


def http_error_diagnostics(error: urllib.error.HTTPError, clock: Callable) -> dict:
    """Extract bounded public error fields, discarding raw bodies and headers."""
    diagnostics = {}
    try:
        value = error.headers.get('Retry-After') if error.headers is not None else None
        if isinstance(value, str) and len(value) <= 128:
            value = value.strip(' \t')
            if re.fullmatch(r'[0-9]+', value):
                seconds = int(value)
            else:
                if not re.fullmatch(
                    r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} '
                    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{4} '
                    r'[0-9]{2}:[0-9]{2}:[0-9]{2} GMT', value,
                ):
                    raise ValueError('invalid HTTP date')
                date = parsedate_to_datetime(value)
                if format_datetime(date, usegmt=True) != value:
                    raise ValueError('invalid HTTP date')
                seconds = date.timestamp() - clock()
                if math.isfinite(seconds):
                    seconds = max(0, seconds)
            if math.isfinite(seconds) and 0 <= seconds <= 86400:
                diagnostics['retry_after_seconds'] = seconds
    except Exception:
        pass
    try:
        raw = error.read(16 * 1024)
        # Reject a full buffer conservatively: it may be a truncated larger body.
        body = json.loads(raw) if len(raw) < 16 * 1024 else None
        detail = body.get('error', body) if isinstance(body, dict) else None
        if isinstance(detail, dict):
            code = detail.get('code')
            if isinstance(code, str) and len(code) <= 32:
                try:
                    code = float(code)
                except ValueError:
                    code = None
            if type(code) in (int, float) and 0 <= code <= 1_000_000:
                diagnostics['provider_error_code'] = code
            kind = detail.get('type')
            if isinstance(kind, str) and kind in {
                'rate_limit_error', 'authentication_error', 'permission_error',
                'invalid_request_error', 'overloaded_error', 'server_error',
            }:
                diagnostics['provider_error_type'] = kind
    except Exception:
        pass
    finally:
        try:
            error.close()
        except Exception:
            pass
    return diagnostics


class AnthropicProvider:
    """Call only the authorized host; injectable transport supports entirely offline verification."""

    def __init__(self, config: ProviderConfig = ProviderConfig(), credential_loader: Callable | None = None,
                 transport: Callable | None = None, clock: Callable = time.time,
                 sleep: Callable = time.sleep) -> None:
        self.config = config
        self.clock, self.sleep = clock, sleep
        self._cooldown_until = 0.0
        self.base_url = validate_host(config.base_url)
        self.credential_loader = credential_loader or (
            lambda: load_credential(config.credential_path, config.env_override))
        self.transport = transport or urllib.request.build_opener(NoRedirect()).open

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        """Return text and accounting, or raise a sanitized error after at most one retry."""
        if self._cooldown_until:
            remaining = self._cooldown_until - self.clock()
            if remaining > 0:
                raise ProviderError('provider_cooldown', {
                    'attempts': 0, 'cost': None, 'http_error': {'retry_after_seconds': remaining}})
        if not isinstance(system, str) or not isinstance(user, str):
            raise TypeError('messages must be strings')
        if len((system + user).encode('utf-8')) > self.config.max_input_bytes:
            raise ValueError('provider input byte cap exceeded')
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        payload = {'model': self.config.model, 'temperature': 0,
                   'thinking': {'type': self.config.thinking},
                   'max_tokens': min(max_tokens, self.config.max_output_tokens), 'system': system,
                   'messages': [{'role': 'user', 'content': user}]}
        if self.config.protocol == 'openai_chat':
            payload.pop('system')
            payload['messages'].insert(0, {'role': 'system', 'content': system})
        started = time.perf_counter()
        credential_failed = False
        try:
            secret = self.credential_loader()
        except Exception:
            credential_failed = True
        if credential_failed:
            raise ProviderError('credential_unavailable')
        headers = {'Content-Type': 'application/json'}
        if self.config.protocol == 'openai_chat':
            endpoint = '/chat/completions'
            headers['Authorization'] = 'Bearer ' + secret
        else:
            endpoint = '/messages'
            headers.update({'x-api-key': secret, 'anthropic-version': '2023-06-01'})
        request = urllib.request.Request(self.base_url + endpoint, data=json.dumps(payload).encode('utf-8'),
                                         headers=headers, method='POST')
        del secret
        events = []
        pacing_wait = 0.0
        pacer = RequestPacer(self.config.credential_path, self.base_url + endpoint,
                             self.config.min_request_interval_seconds, self.config.rate_state_path,
                             self.clock, self.sleep)
        for attempt in range(self.config.retries + 1):
            error, retryable, raw = None, False, None
            http_error = {}
            try:
                attempt_wait = pacer.acquire()
                pacing_wait += attempt_wait
            except PacingError:
                raise ProviderError('pacing_unavailable', {
                    'attempts': attempt, 'events': events, 'cost': None,
                    'elapsed_seconds': time.perf_counter() - started,
                    'pacing_wait_seconds': pacing_wait + pacer.wait_seconds}) from None
            try:
                with self.transport(request, timeout=self.config.timeout_seconds) as response:
                    raw = response.read(2_000_001)
            except ProviderError:
                error = 'redirect_refused'
            except urllib.error.HTTPError as exc:
                error = 'http_' + str(exc.code)
                http_error = http_error_diagnostics(exc, self.clock)
                if http_error.get('retry_after_seconds', 0) > 0:
                    self._cooldown_until = max(self._cooldown_until,
                                               self.clock() + http_error['retry_after_seconds'])
                retryable = exc.code in (429, 500, 502, 503, 504) and http_error.get('retry_after_seconds', 0) == 0
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                error, retryable = 'network_error', True
            except Exception:
                error = 'transport_error'
            finally:
                pacing_finished = pacer.finish()
            pacing_status = ('ready' if pacing_finished else 'completion_persistence_failed')
            if error:
                diagnostics = {'http_error': http_error} if http_error else {}
                events.append({'attempt': attempt + 1, 'status': error, 'usage': None, 'cost': None,
                               'pacing_wait_seconds': attempt_wait, **diagnostics})
                if pacing_finished and retryable and attempt < self.config.retries:
                    continue
                raise ProviderError(error, {'attempts': attempt + 1, 'events': events, **diagnostics,
                                           'elapsed_seconds': time.perf_counter() - started, 'cost': None,
                                           'pacing_wait_seconds': pacing_wait, 'pacing_status': pacing_status})
            record = {'requested_model': self.config.model, 'actual_model': None, 'usage': {}, 'cost': None,
                      'cost_status': 'unverified', 'attempts': attempt + 1, 'events': events,
                      'elapsed_seconds': time.perf_counter() - started, 'temperature': 0,
                      'pacing_wait_seconds': pacing_wait, 'pacing_status': pacing_status,
                      'thinking': self.config.thinking, 'protocol': self.config.protocol,
                      'expected_reported_model': self.config.expected_reported_model or self.config.model,
                      'identity_status': 'unverified'}
            invalid = False
            try:
                if len(raw) > 2_000_000:
                    raise ValueError()
                body = json.loads(raw)
                actual = body.get('model')
                record['actual_model'] = actual if isinstance(actual, str) and re.fullmatch(
                    r'[A-Za-z0-9._-]{1,100}', actual) else None
                usage = body.get('usage', {})
                if self.config.protocol == 'openai_chat':
                    usage = {'input_tokens': usage.get('prompt_tokens'),
                             'output_tokens': usage.get('completion_tokens'),
                             'cache_read_input_tokens': usage.get('prompt_tokens_details', {}).get('cached_tokens')}
                for key in ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'):
                    value = usage.get(key)
                    if type(value) is int and value >= 0:
                        record['usage'][key] = value
                if self.config.protocol == 'openai_chat':
                    choice = body['choices'][0]
                    record['stop_reason'] = {'stop': 'end_turn', 'length': 'max_tokens',
                                             'content_filter': 'refusal', 'tool_calls': 'tool_use'}.get(
                                                 choice.get('finish_reason'), 'unknown')
                    content = choice['message']['content']
                    if not isinstance(content, str):
                        raise ValueError()
                    # Never persist gateway-provided reasoning, including an incomplete prefix.
                    content = content.strip()
                    record['reasoning_prefix_removed'] = content.startswith('<think>')
                    if content.startswith('<think>'):
                        match = re.match(r'^<think>.*?</think>\s*', content, re.DOTALL)
                        if not match:
                            raise ValueError()
                        content = content[match.end():]
                    record['text'] = content
                else:
                    record['stop_reason'] = body.get('stop_reason') if body.get('stop_reason') in (
                        'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'pause_turn', 'refusal') else 'unknown'
                    text_parts = [part['text'] for part in body['content'] if part.get('type') == 'text']
                    record['text'] = ''.join(text_parts)
            except Exception:
                invalid = True
            if invalid:
                raise ProviderError('invalid_provider_response', record)
            if (self.config.input_per_million is not None and self.config.output_per_million is not None
                    and 'input_tokens' in record['usage'] and 'output_tokens' in record['usage']
                    and not any(record['usage'].get(k) for k in ('cache_creation_input_tokens', 'cache_read_input_tokens'))):
                record['cost'] = (record['usage']['input_tokens'] * self.config.input_per_million
                                  + record['usage']['output_tokens'] * self.config.output_per_million) / 1_000_000
                record['cost_status'] = 'estimated_configured_rates'
            record['known_response_cost'] = record['cost']
            if events:
                record['cost'] = None
                record['cost_status'] = 'partial_unknown_retry_usage'
            if record['actual_model'] != record['expected_reported_model']:
                record.pop('text', None)
                raise ProviderError('model_mismatch', record)
            record['identity_status'] = ('reported_identity_matches' if record['actual_model'] == self.config.model
                                         else 'configured_gateway_alias_underlying_model_unverified')
            record['truncated'] = record['stop_reason'] == 'max_tokens'
            if record['truncated']:
                record.pop('text', None)
                raise ProviderError('output_truncated', record)
            if not pacing_finished:
                raise ProviderError('pacing_unavailable', record)
            return record
