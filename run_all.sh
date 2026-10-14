#!/usr/bin/env bash
# One-command launcher for Agentic-IDS. Works on macOS, Ubuntu/Linux, and Windows (WSL 2 or Git Bash).
#
#   bash run_all.sh       start everything and open the browser
#   ./run_all.sh          start everything and open the browser
#   ./run_all.sh stop     stop the Docker stack (and Ollama if this script started it)
#   ./run_all.sh stop -v  same, and wipe Kafka data
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$PWD"
ENV_FILES=("backend/.env" "backend/config/.env")
PIDFILE="$ROOT/.run_all_ollama.pid"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*"; exit 1; }

# ── Which operating system? ──
case "$(uname -s)" in
  Darwin)               OS=mac ;;
  Linux)                if grep -qi microsoft /proc/version 2>/dev/null; then OS=wsl; else OS=linux; fi ;;
  MINGW*|MSYS*|CYGWIN*) OS=gitbash ;;
  *)                    OS=linux ;;
esac

# Open a URL in the default browser (never fails the script).
open_url() {
  case "$OS" in
    mac)     open "$1" >/dev/null 2>&1 ;;
    wsl)     { command -v wslview >/dev/null && wslview "$1"; } >/dev/null 2>&1 || cmd.exe /c start "" "$1" >/dev/null 2>&1 || explorer.exe "$1" >/dev/null 2>&1 ;;
    gitbash) start "" "$1" >/dev/null 2>&1 || cmd.exe /c start "" "$1" >/dev/null 2>&1 ;;
    *)       xdg-open "$1" >/dev/null 2>&1 ;;
  esac
  return 0
}

# Read KEY from the env files (first file wins, like the backend does).
env_val() {
  local key="$1" f v
  for f in "${ENV_FILES[@]}"; do
    [ -f "$f" ] || continue
    v=$(grep -E "^[[:space:]]*${key}=" "$f" | head -1 | tr -d '\r' | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//' -e 's/[[:space:]]*#.*$//')
    [ -n "$v" ] && { printf '%s' "$v"; return; }
  done
}

wait_for() { # wait_for <seconds> <description> <command...>
  local secs="$1" desc="$2"; shift 2
  local i
  for ((i = 0; i < secs; i += 2)); do
    "$@" >/dev/null 2>&1 && return 0
    sleep 2
  done
  warn "timed out waiting for $desc"
  return 1
}

# ───────────────────────── stop ─────────────────────────
if [ "${1:-}" = "stop" ]; then
  say "Stopping Docker stack"
  docker compose down ${2:-}
  if [ -f "$PIDFILE" ]; then
    say "Stopping Ollama started by this script"
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  say "Done. (LM Studio / Docker Desktop left alone — quit them manually if you want.)"
  exit 0
fi

# ───────────────────────── 1. Docker ─────────────────────────
command -v docker >/dev/null || die "docker not found. Install it first (see README, Part A, step 1 for your system)."
command -v curl >/dev/null || die "curl not found. On Ubuntu/WSL run: sudo apt install -y curl"
if ! docker info >/dev/null 2>&1; then
  case "$OS" in
    mac)
      say "Starting Docker Desktop"
      open -a Docker ;;
    linux)
      say "Docker is not running. Trying: sudo systemctl start docker"
      sudo systemctl start docker || true ;;
    wsl|gitbash)
      warn "Docker is not running. Open Docker Desktop on Windows and wait for 'Engine running'." ;;
  esac
  wait_for 180 "Docker daemon" docker info || die "Docker did not start. On Linux, make sure your user is in the 'docker' group (README, Ubuntu step 1)."
fi
docker compose version >/dev/null 2>&1 || die "'docker compose' (v2) not found. Update Docker Desktop, or on Ubuntu install the docker-compose-plugin package."
say "Docker is running ($OS)"

# ───────────────────────── 2. iCloud offloaded files (macOS only) ─────────────────────────
if [ "$OS" = "mac" ]; then
  OFFLOADED=$(find . -path ./.git -prune -o -type f -flags +dataless -print 2>/dev/null | wc -l | tr -d ' ')
  if [ "$OFFLOADED" != "0" ]; then
    say "Downloading $OFFLOADED iCloud-offloaded files (Docker can't read them otherwise)"
    find . -path ./.git -prune -o -type f -flags +dataless -exec cat {} + >/dev/null 2>&1
  fi
fi

# ───────────────────────── 3. Local LLMs ─────────────────────────
MODELS="$(env_val ANALYST_MODEL) $(env_val ENGINEER_MODEL) $(env_val INTEL_MODEL)"

if [[ "$MODELS" == *ollama:* ]]; then
  if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    if command -v ollama >/dev/null; then
      say "Starting Ollama"
      nohup ollama serve >/tmp/agentic-ids-ollama.log 2>&1 &
      echo $! > "$PIDFILE"
      wait_for 30 "Ollama" curl -sf http://localhost:11434/api/tags || warn "Ollama not reachable (see /tmp/agentic-ids-ollama.log)"
    else
      warn "ollama not installed (brew install ollama) — council will fall back to Groq"
    fi
  else
    say "Ollama already running"
  fi
fi

if [[ "$MODELS" == *lmstudio:* ]]; then
  LMS="$HOME/.lmstudio/bin/lms"
  if curl -sf http://localhost:1234/v1/models >/dev/null 2>&1; then
    say "LM Studio server already running"
  elif [ -x "$LMS" ]; then
    say "Starting LM Studio server"
    "$LMS" server start >/dev/null 2>&1 || true
    LMS_MODEL=$(echo "$MODELS" | tr ' ' '\n' | grep '^lmstudio:' | head -1 | cut -d: -f2-)
    "$LMS" load "$LMS_MODEL" >/dev/null 2>&1 || warn "could not load $LMS_MODEL in LM Studio"
  else
    warn "LM Studio configured but its CLI wasn't found — start its server manually (Developer tab) or the council falls back to Groq"
  fi
fi

# ───────────────────────── 3b. Required files ─────────────────────────
[ -f backend/data/raw/filtered_nowebatt.csv ] || die "Missing backend/data/raw/filtered_nowebatt.csv (the dataset). See README step 4."
[ -f backend/config/.env ] || die "Missing backend/config/.env. See README step 5 (copy .env.example to .env and paste your Groq key)."

# ───────────────────────── 4. Groq key ─────────────────────────
GROQ_API_KEY="${GROQ_API_KEY:-$(env_val GROQ_API_KEY)}"
if [ "$GROQ_API_KEY" = "YOUR_KEY_HERE" ]; then
  die "backend/config/.env still says YOUR_KEY_HERE. Paste your Groq key there (README step 5)."
fi
if [ -z "$GROQ_API_KEY" ]; then
  warn "No GROQ_API_KEY found in backend/.env or backend/config/.env — Groq calls will fail"
fi
export GROQ_API_KEY

# ───────────────────────── 5. LLM check (optional) ─────────────────────────
# Off by default: on a fresh Mac, running python can pop up an "install developer tools" dialog.
# Run with CHECK_LLM=1 ./run_all.sh to test every LLM provider before starting.
if [ "${CHECK_LLM:-0}" = "1" ] && [ -f backend/check_llm.py ]; then
  PY=""
  for c in python python3; do command -v "$c" >/dev/null && { PY="$c"; break; }; done
  if [ -n "$PY" ]; then
    say "Checking LLM providers (informational)"
    "$PY" backend/check_llm.py 2>&1 | sed 's/^/    /' || true
  fi
fi

# ───────────────────────── 6. Stack ─────────────────────────
say "Starting stack (first build can take 10+ minutes)"
docker compose up --build -d || die "docker compose failed"

say "Waiting for the dashboard on http://localhost:5001"
if ! wait_for 300 "dashboard" curl -sf http://localhost:5001/api/stats; then
  warn "Dashboard not responding yet. Recent consumer logs:"
  docker logs --tail 25 agentic-ids-consumer 2>&1 | sed 's/^/    /'
  warn "Opening it anyway; see MANUAL.md section 9 if it stays blank."
fi

docker compose ps -a

# ───────────────────────── 7. Open browser ─────────────────────────
say "Opening dashboard and Kafka UI"
open_url http://localhost:5001
open_url http://localhost:8080

cat <<EOF

  Dashboard : http://localhost:5001
  Kafka UI  : http://localhost:8080
  Live log  : docker logs -f agentic-ids-consumer 2>&1 | grep --line-buffered '\[Defender\]'
  Stop all  : ./run_all.sh stop
EOF
