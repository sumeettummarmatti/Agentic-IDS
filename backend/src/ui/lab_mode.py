"""
Attack Lab mode — while the Attack Lab page is open, background (non-demo) traffic is
skipped so only the injected demo flows are analysed.

The page sends a heartbeat every few seconds; if they stop (tab closed, crash, laptop
sleep) lab mode expires on its own so normal analysis always resumes.
"""

import threading
import time

_TTL_SECONDS = 30.0
_lock = threading.Lock()
_last_beat = 0.0
_skipped = 0


def touch():
    """Enter / keep lab mode alive."""
    global _last_beat, _skipped
    with _lock:
        if time.monotonic() - _last_beat > _TTL_SECONDS:
            _skipped = 0
        _last_beat = time.monotonic()


def release():
    global _last_beat
    with _lock:
        _last_beat = 0.0


def is_active() -> bool:
    return (time.monotonic() - _last_beat) < _TTL_SECONDS


def count_skipped():
    global _skipped
    with _lock:
        _skipped += 1


def status() -> dict:
    return {"active": is_active(), "skipped_flows": _skipped}
