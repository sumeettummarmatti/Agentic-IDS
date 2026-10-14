"""Check that every configured council model answers. Run before a demo: python backend/check_llm.py

Calls each provider directly (groq / lmstudio / ollama) with NO fallback, so a failure here
means the council would silently fall back to groq:llama-3.1-8b-instant for that agent.
"""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Resolve .env next to this file so it works from the repo root or from backend/
_HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_HERE / ".env", override=False)
load_dotenv(dotenv_path=_HERE / "config" / ".env", override=False)

PROMPT = "Reply with the word OK."
LMSTUDIO_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def ask_groq(model):
    from groq import Groq
    r = Groq(api_key=os.environ.get("GROQ_API_KEY")).chat.completions.create(
        model=model, max_tokens=20, messages=[{"role": "user", "content": PROMPT}])
    return r.choices[0].message.content


def ask_lmstudio(model):
    r = requests.post(f"{LMSTUDIO_URL}/chat/completions", timeout=60, json={
        "model": model, "max_tokens": 20, "stream": False,
        "messages": [{"role": "user", "content": PROMPT}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_ollama(model):
    # same request shape as the council: thinking off, otherwise qwen3 returns empty content
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=120, json={
        "model": model, "stream": False, "think": False, "options": {"num_predict": 20},
        "messages": [{"role": "user", "content": PROMPT}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


ASK = {"groq": ask_groq, "lmstudio": ask_lmstudio, "ollama": ask_ollama}

failed = 0
for var in ("ANALYST_MODEL", "ENGINEER_MODEL", "INTEL_MODEL"):
    spec = os.getenv(var, "")
    provider, model = spec.split(":", 1) if ":" in spec else ("groq", spec)
    fn = ASK.get(provider)
    if fn is None:
        print(f"{var:15} {spec:40} SKIPPED (provider '{provider}' not checked)")
        continue
    try:
        reply = fn(model).strip().replace("\n", " ")
        if not reply:
            raise RuntimeError("empty response")
        print(f"{var:15} {spec:40} OK      -> {reply[-60:]!r}")
    except Exception as e:
        failed += 1
        print(f"{var:15} {spec:40} FAILED  -> {str(e)[:110]}")

# The council's fallback for any failed local call
try:
    ask_groq(os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant"))
    print(f"{'(fallback)':15} {'groq:llama-3.1-8b-instant':40} OK")
except Exception as e:
    print(f"{'(fallback)':15} {'groq:llama-3.1-8b-instant':40} FAILED  -> {str(e)[:110]}")

raise SystemExit(1 if failed else 0)
