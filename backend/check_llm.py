"""Quick check that the configured Groq models answer. Run before a demo: python check_llm.py"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env next to this file so it works from the repo root or from backend/
_HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_HERE / ".env", override=False)
load_dotenv(dotenv_path=_HERE / "config" / ".env", override=False)

from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
for var in ("ANALYST_MODEL", "ENGINEER_MODEL", "INTEL_MODEL"):
    spec = os.getenv(var, "")
    provider, model = spec.split(":", 1) if ":" in spec else ("groq", spec)
    if provider != "groq":
        print(f"{var:15} {spec:28} SKIPPED (provider '{provider}', not Groq)")
        continue
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the word OK."}],
        )
        print(f"{var:15} {model:28} OK  -> {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"{var:15} {model:28} FAILED -> {e}")
