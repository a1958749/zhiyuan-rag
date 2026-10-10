"""Fail-closed, process-shared request pacing without storing credentials."""

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable


class PacingError(RuntimeError):
    """Pacing could not safely reserve a tracked request."""


class RequestPacer:
    """Serialize dispatches with an exclusive lock and atomic next-call timestamp.

    A crashed owner leaves its lock in place: recovery must be explicit. The lock
    remains held until a completion-time reservation is persisted, so even a
    delayed dispatch is followed by a full interval before the next dispatch.
    """

    def __init__(self, credential_path: str, endpoint: str, interval: float,
                 state_path: str | None = None, clock: Callable = time.time,
                 sleep: Callable = time.sleep) -> None:
        self.interval, self.clock, self.sleep = interval, clock, sleep
        credential = Path(credential_path).resolve()
        identity = os.path.normcase(str(credential)) + '\n' + endpoint
        self.scope = hashlib.sha256(identity.encode()).hexdigest()
        self.path = Path(state_path) if state_path else credential.parent / '.rate-state' / (self.scope + '.json')
        self.lock = self.path.with_name(self.path.name + '.lock')
        self.forbidden = self.path.resolve() == credential or self.lock.resolve() == credential
        self.owned = False
        self.wait_seconds = 0.0

    def acquire(self) -> float:
        """Wait for a safe dispatch slot; never reclaim an abandoned lock."""
        self.wait_seconds = 0.0
        if not self.interval:
            return 0.0
        try:
            if self.forbidden or self.path.is_symlink() or self.lock.is_symlink():
                raise PacingError()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                try:
                    descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    self.owned = True
                    os.close(descriptor)
                    break
                except FileExistsError:
                    if self.wait_seconds >= 150:
                        raise PacingError()
                    self.sleep(0.25)
                    self.wait_seconds += 0.25
            now = self.clock()
            next_call = now
            if self.path.exists():
                if self.path.is_symlink() or self.path.stat().st_nlink != 1 or self.path.stat().st_size > 1024:
                    raise PacingError()
                state = json.loads(self.path.read_text(encoding='utf-8'))
                if (not isinstance(state, dict) or set(state) != {'version', 'scope', 'next_call'}
                        or type(state['version']) is not int or state['version'] != 1 or state['scope'] != self.scope
                        or type(state['next_call']) not in (int, float)
                        or not math.isfinite(state['next_call']) or state['next_call'] < 0
                        or state['next_call'] > now + 120):
                    raise PacingError()
                next_call = state['next_call']
            # Recheck the wall clock after sleep; an early wake must not dispatch.
            while next_call > self.clock():
                delay = next_call - self.clock()
                if delay > 120 or self.wait_seconds + delay > 270:
                    raise PacingError()
                self.sleep(delay)
                self.wait_seconds += delay
            self._persist_next_call()
            return self.wait_seconds
        except Exception:
            self.release()
            raise PacingError('pacing_unavailable') from None

    def _persist_next_call(self) -> None:
        """Atomically persist a reservation while this caller owns the lock."""
        state = {'version': 1, 'scope': self.scope, 'next_call': self.clock() + self.interval}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix='.pacing-', delete=False) as stream:
                temporary = stream.name
                json.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def finish(self) -> bool:
        """Reserve a full interval after transport; retain the lock on failure.

        Returns:
            Whether completion was persisted. False must stop retries, while the
            caller retains any already-received response and billed accounting.
        """
        if not self.interval:
            return True
        if not self.owned:
            return False
        try:
            self._persist_next_call()
        except Exception:
            return False
        self.release()
        return True

    def release(self) -> None:
        """Release only this instance's lock; an unlink failure stays closed."""
        if self.owned:
            self.owned = False
            try:
                self.lock.unlink()
            except OSError:
                pass
