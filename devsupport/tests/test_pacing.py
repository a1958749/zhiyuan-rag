import json

import pytest

from devsupport.llm import AnthropicProvider, ProviderConfig, ProviderError
from test_llm import Response


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def provider(tmp_path, clock, transport, **kwargs):
    config = ProviderConfig(min_request_interval_seconds=65,
                            credential_path=str(tmp_path / 'credential.dpapi'),
                            rate_state_path=str(tmp_path / 'rate.json'), **kwargs)
    return AnthropicProvider(config, credential_loader=lambda: 'test-key', transport=transport,
                             clock=clock.time, sleep=clock.sleep)


def test_independent_instances_and_retry_share_spacing(tmp_path):
    clock, starts = Clock(), []

    def send(*args, **kwargs):
        starts.append(clock.now)
        if len(starts) == 2:
            raise TimeoutError()
        return Response()

    first = provider(tmp_path, clock, send).complete('s', 'u')
    second = provider(tmp_path, clock, send).complete('s', 'u')
    assert starts == [1000, 1065, 1130]
    assert first['pacing_wait_seconds'] == 0
    assert second['pacing_wait_seconds'] == 130
    assert second['events'][0]['pacing_wait_seconds'] == 65
    assert 'test-key' not in (tmp_path / 'rate.json').read_text()


@pytest.mark.parametrize('value', [-1, 121, float('nan'), float('inf'), True, '65'])
def test_interval_is_bounded(value):
    with pytest.raises(ValueError):
        ProviderConfig(min_request_interval_seconds=value)


@pytest.mark.parametrize('state', [{}, {'api_key': 'secret'},
                                  {'version': 1, 'scope': 'foreign', 'next_call': 1000}])
def test_unknown_foreign_state_never_overwritten_or_dispatched(tmp_path, state):
    clock, calls = Clock(), []
    path = tmp_path / 'rate.json'
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    with pytest.raises(ProviderError, match='pacing_unavailable') as exc:
        provider(tmp_path, clock, lambda *a, **k: calls.append(1)).complete('s', 'u')
    assert calls == []
    assert path.read_bytes() == before
    assert not (tmp_path / 'rate.json.lock').exists()
    assert exc.value.record['attempts'] == 0


def test_stale_lock_fails_closed_without_deletion(tmp_path):
    clock, calls = Clock(), []
    lock = tmp_path / 'rate.json.lock'
    lock.write_text('abandoned')
    with pytest.raises(ProviderError, match='pacing_unavailable'):
        provider(tmp_path, clock, lambda *a, **k: calls.append(1)).complete('s', 'u')
    assert calls == []
    assert lock.read_text() == 'abandoned'
    assert clock.now <= 1150


def test_default_disabled_does_not_create_state(tmp_path):
    config = ProviderConfig(rate_state_path=str(tmp_path / 'rate.json'))
    result = AnthropicProvider(config, credential_loader=lambda: 'test',
                               transport=lambda *a, **k: Response()).complete('s', 'u')
    assert result['pacing_wait_seconds'] == 0
    assert not (tmp_path / 'rate.json').exists()


def child_request(directory, queue):
    from pathlib import Path
    clock = Clock()
    starts = []
    result = provider(Path(directory), clock,
                      lambda *a, **k: starts.append(clock.now) or Response()).complete('s', 'u')
    queue.put((starts, result['pacing_wait_seconds']))


def test_state_coordinates_separate_processes(tmp_path):
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    results = []
    for _ in range(2):
        child = context.Process(target=child_request, args=(str(tmp_path), queue))
        child.start()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join()
            pytest.fail('offline child did not terminate')
        assert child.exitcode == 0
        results.append(queue.get(timeout=2))
    assert results == [([1000.0], 0), ([1065.0], 65)]
    queue.close()


def test_failed_atomic_write_never_calls_transport(tmp_path, monkeypatch):
    import devsupport.pacing as pacing
    calls, clock = [], Clock()
    def fail(*args):
        raise OSError('private filename')
    monkeypatch.setattr(pacing.os, 'replace', fail)
    with pytest.raises(ProviderError, match='pacing_unavailable') as exc:
        provider(tmp_path, clock, lambda *a, **k: calls.append(1)).complete('s', 'u')
    assert calls == []
    assert exc.value.record['attempts'] == 0
    assert not list(tmp_path.glob('.pacing-*'))
    assert not (tmp_path / 'rate.json.lock').exists()


def test_lock_is_held_through_transport(tmp_path):
    from devsupport.pacing import PacingError, RequestPacer
    clock = Clock()
    def send(*args, **kwargs):
        contender = RequestPacer(str(tmp_path / 'credential.dpapi'),
                                 'https://www.xiaoyaoapi.com/v1/messages', 65,
                                 str(tmp_path / 'rate.json'), clock.time, clock.sleep)
        with pytest.raises(PacingError):
            contender.acquire()
        assert (tmp_path / 'rate.json.lock').exists()
        return Response()
    provider(tmp_path, clock, send).complete('s', 'u')
    assert not (tmp_path / 'rate.json.lock').exists()


def test_credential_path_cannot_be_state_even_when_missing(tmp_path):
    path = tmp_path / 'credential.dpapi'
    config = ProviderConfig(credential_path=str(path), rate_state_path=str(path), min_request_interval_seconds=65)
    calls = []
    with pytest.raises(ProviderError, match='pacing_unavailable'):
        AnthropicProvider(config, credential_loader=lambda: 'test',
                          transport=lambda *a, **k: calls.append(1)).complete('s', 'u')
    assert calls == []
    assert not path.exists()


def test_default_state_scoped_by_credential_and_endpoint(tmp_path):
    from devsupport.pacing import RequestPacer
    a = RequestPacer(str(tmp_path / 'a'), 'https://www.xiaoyaoapi.com/v1/messages', 65)
    same = RequestPacer(str(tmp_path / '.' / 'a'), 'https://www.xiaoyaoapi.com/v1/messages', 65)
    other_key = RequestPacer(str(tmp_path / 'b'), 'https://www.xiaoyaoapi.com/v1/messages', 65)
    other_endpoint = RequestPacer(str(tmp_path / 'a'), 'https://www.xiaoyaoapi.com/v1/chat/completions', 65)
    assert a.path == same.path
    assert len({a.path, other_key.path, other_endpoint.path}) == 3


def test_403_not_retried_and_next_instance_still_waits(tmp_path):
    import urllib.error
    clock, calls = Clock(), []
    def fail(*args, **kwargs):
        calls.append(clock.now)
        raise urllib.error.HTTPError('test', 403, 'blocked', {}, None)
    with pytest.raises(ProviderError, match='http_403') as exc:
        provider(tmp_path, clock, fail).complete('s', 'u')
    assert exc.value.record['pacing_wait_seconds'] == 0
    assert exc.value.record['attempts'] == 1
    provider(tmp_path, clock, lambda *a, **k: calls.append(clock.now) or Response()).complete('s', 'u')
    assert calls == [1000, 1065]


def child_hold_lock(directory, ready, release):
    from pathlib import Path
    from devsupport.pacing import RequestPacer
    path = Path(directory)
    clock = Clock()
    pacer = RequestPacer(str(path / 'credential.dpapi'), 'https://www.xiaoyaoapi.com/v1/messages',
                        65, str(path / 'rate.json'), clock.time, clock.sleep)
    pacer.acquire()
    ready.set()
    try:
        release.wait(10)
    finally:
        pacer.release()


def test_other_process_lock_blocks_dispatch(tmp_path):
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    ready, release = context.Event(), context.Event()
    child = context.Process(target=child_hold_lock, args=(str(tmp_path), ready, release))
    child.start()
    calls = []
    try:
        assert ready.wait(5)
        with pytest.raises(ProviderError, match='pacing_unavailable'):
            provider(tmp_path, Clock(), lambda *a, **k: calls.append(1)).complete('s', 'u')
        assert calls == []
    finally:
        release.set()
        child.join(5)
        if child.is_alive():
            child.terminate()
            child.join()
    assert child.exitcode == 0


def test_delayed_reservation_cannot_burst_next_request(tmp_path, monkeypatch):
    import devsupport.pacing as pacing
    clock, starts = Clock(), []
    replace = pacing.os.replace
    writes = []
    def delayed_replace(*args):
        replace(*args)
        writes.append(1)
        if len(writes) == 1:
            clock.sleep(70)
    monkeypatch.setattr(pacing.os, 'replace', delayed_replace)
    def send(*args, **kwargs):
        starts.append(clock.now)
        return Response()
    provider(tmp_path, clock, send).complete('s', 'u')
    result = provider(tmp_path, clock, send).complete('s', 'u')
    assert starts == [1070, 1135]
    assert result['pacing_wait_seconds'] == 65


def test_interval_starts_after_long_transport_finishes(tmp_path):
    clock, starts = Clock(), []
    def send(*args, **kwargs):
        starts.append(clock.now)
        clock.sleep(90)
        return Response()
    provider(tmp_path, clock, send).complete('s', 'u')
    result = provider(tmp_path, clock, send).complete('s', 'u')
    assert starts == [1000, 1155]
    assert result['pacing_wait_seconds'] == 65


@pytest.mark.parametrize('transport_fails', [False, True])
def test_completion_write_failure_retains_accounting_and_lock_without_replay(tmp_path, monkeypatch, transport_fails):
    import devsupport.pacing as pacing
    clock, calls, writes = Clock(), [], []
    replace = pacing.os.replace
    def fail_completion(*args):
        writes.append(1)
        if len(writes) == 2:
            raise OSError('private filesystem detail')
        replace(*args)
    monkeypatch.setattr(pacing.os, 'replace', fail_completion)
    def send(*args, **kwargs):
        calls.append(clock.now)
        if transport_fails:
            raise TimeoutError('private transport detail')
        return Response()
    with pytest.raises(ProviderError) as exc:
        provider(tmp_path, clock, send, input_per_million=1, output_per_million=2).complete('s', 'u')
    assert len(calls) == 1
    record = exc.value.record
    assert record['attempts'] == 1
    assert record['pacing_status'] == 'completion_persistence_failed'
    if transport_fails:
        assert exc.value.code == 'network_error'
        assert record['events'][0]['status'] == 'network_error'
        assert record['cost'] is None
    else:
        assert exc.value.code == 'pacing_unavailable'
        assert record['usage'] == {'input_tokens': 12, 'output_tokens': 2}
        assert record['text'] == '{}'
        assert record['cost'] == record['known_response_cost'] == 0.000016
    assert (tmp_path / 'rate.json.lock').exists()
    assert 'private' not in str(exc.value)
    with pytest.raises(ProviderError, match='pacing_unavailable') as blocked:
        provider(tmp_path, clock, send).complete('s', 'u')
    assert blocked.value.record['attempts'] == 0
    assert len(calls) == 1
    assert (tmp_path / 'rate.json.lock').exists()
