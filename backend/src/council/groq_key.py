"""
Runtime Groq key manager.

Lets the UI replace the Groq API key while the system is running. LLM calls that hit a
rate limit or an invalid key wait here for a new key (or for the limit to reset) and
then continue, instead of failing the analysis that was in flight.

The new key lives in memory (and os.environ) until the process restarts; the key in
backend/config/.env is used again after a restart.
"""

import logging
import os
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_cond = threading.Condition()
_generation = 0
_listeners = []
_status = {"state": "ok", "message": "", "since": 0.0, "last_seen": 0.0}   # ok | rate_limited | invalid | missing
_STALE_AFTER = 75.0   # a problem nobody has hit for this long is treated as cleared
_applied_at = 0.0


def hint() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    return ("····" + key[-4:]) if len(key) >= 8 else ""


def status() -> dict:
    with _cond:
        st = dict(_status)
        if st["state"] != "ok" and time.time() - st.get("last_seen", 0.0) > _STALE_AFTER:
            st.update(state="ok", message="")
        return {**st, "key_hint": hint(), "generation": _generation, "key_updated_at": _applied_at}


def register(callback):
    """callback(new_key) is called whenever a new valid key is applied."""
    _listeners.append(callback)


def mark_ok():
    with _cond:
        if _status["state"] != "ok":
            _status.update(state="ok", message="", since=time.time())


def mark_problem(state: str, message: str):
    with _cond:
        if _status["state"] != state:
            _status.update(state=state, since=time.time())
        _status.update(message=message[:240], last_seen=time.time())
    logger.warning(f"[GroqKey] {state}: {message[:240]}")


def generation() -> int:
    with _cond:
        return _generation


def wait_for_new_key(seen_generation: int, timeout: float) -> bool:
    """Block until a new key is applied (True) or the timeout passes (False)."""
    with _cond:
        return _cond.wait_for(lambda: _generation != seen_generation, timeout=timeout)


def apply(new_key: str):
    """Validate a key against Groq and, if good, hot-swap it. Returns (ok, message)."""
    global _generation, _applied_at
    new_key = (new_key or "").strip()
    if not new_key.startswith("gsk_") or len(new_key) < 20:
        return False, "That doesn't look like a Groq key. Keys start with gsk_."
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {new_key}", "User-Agent": "curl/8.4"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status != 200:
                return False, f"Groq answered with status {r.status}."
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "Groq rejected this key. Check that you copied all of it."
        return False, f"Groq answered with status {e.code}."
    except Exception as e:
        return False, f"Could not reach Groq to check the key ({type(e).__name__})."

    os.environ["GROQ_API_KEY"] = new_key
    for cb in list(_listeners):
        try:
            cb(new_key)
        except Exception as e:
            logger.warning(f"[GroqKey] listener failed: {e}")
    with _cond:
        _generation += 1
        _applied_at = time.time()
        _status.update(state="ok", message="", since=time.time(), last_seen=0.0)
        _cond.notify_all()
    logger.info(f"[GroqKey] Key updated at runtime ({hint()})")
    return True, "Key updated. Any paused analysis is continuing."
