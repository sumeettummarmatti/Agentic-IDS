# Agentic-IDS — Step-by-Step Manual

Every step to go from "everything off" to "whole project working and verified", then back off again.
Each step says **what to run** and **what you should see**. If you don't see it, jump to [Troubleshooting](#9-troubleshooting).

```
 3 simulators (CSV replay) ─┐
 AegisFlow C engine (pcap) ─┴─► Kafka: raw-flows ──► ids-consumer ──► Kafka: threats, decisions
        └─► Kafka: server-registry        │ 1. ML detect (XGBoost + LSTM)
                                          │ 2. LLM council (Analyst, ML Engineer, Threat Intel)
                                          │ 3. PPO defender picks an action
                                          └─► dashboard :5001  +  backend/reports/actions/*.jsonl
```

| Port | What |
|---|---|
| 5001 | IDS dashboard |
| 8080 | Kafka UI |
| 9092 | Kafka (from your Mac) |
| 11434 | Ollama |
| 1234 | LM Studio server |

**Conventions:** every command runs from the repo root `~/Desktop/agentic-ids` unless the step says `cd backend`.
Use **one terminal window** ("Terminal 1") for commands. Some steps open extra terminals for things that keep running.

---

## 1. One-time setup (skip once done)

### 1.1 Apps
1. **Docker Desktop**: installed at `/Applications/Docker.app`. In Settings → Resources, give it **at least 6 GB RAM**.
2. **Ollama**: `brew install ollama` (already installed: `/opt/homebrew/bin/ollama`).
3. **LM Studio**: installed at `/Applications/LM Studio.app`.
4. **Build tools** (only for the C engine, step 7): `brew install cmake libpcap librdkafka`.

### 1.2 Python environment
```bash
conda activate agentic-ids          # Python 3.12 env that already exists
python --version                    # → Python 3.12.x
pip install -r backend/requirements.txt   # only the first time / after requirements change
```

### 1.3 Data and trained models
```bash
ls -la backend/data/raw/filtered_nowebatt.csv      # ~270 MB, 300,030 rows (not in git)
ls backend/models/trained/                         # class_mapping.pkl lstm_detector.pt preprocessor.pkl xgboost_multiclass.pkl
ls backend/models/defender_ppo.zip
```
All must exist. The dataset holds BENIGN, DDoS and PortScan, about 100k each.

### 1.4 iCloud (important on this Mac)
Your Desktop syncs to iCloud, which can offload files to the cloud. Docker can't read offloaded files, so the
IDS container then silently fails to load the models. Before every Docker session, download everything locally:
```bash
find . -path ./.git -prune -o -type f -flags +dataless -print | wc -l        # how many files are offloaded
find . -path ./.git -prune -o -type f -flags +dataless -exec cat {} + > /dev/null   # download them
find . -path ./.git -prune -o -type f -flags +dataless -print | wc -l        # → 0
```
Permanent fix: move the repo out of `~/Desktop` (e.g. `~/code/agentic-ids`), or turn off
System Settings → Apple ID → iCloud → iCloud Drive → "Desktop & Documents Folders".

### 1.5 Config files and the Groq key
There are two env files and both are read (`backend/.env` first, then `backend/config/.env`; the first value wins).
Real shell variables win over both.

1. Create a new Groq key at https://console.groq.com/keys (the current one returns **401 Invalid API Key**).
2. Put it in **both** files (open them in an editor):
   - `backend/.env` → `GROQ_API_KEY=gsk_...`
   - `backend/config/.env` → `GROQ_API_KEY=gsk_...`
3. Choose the council models in `backend/config/.env`. The current setup:
   ```
   ANALYST_MODEL=lmstudio:qwen2.5-coder-7b-instruct
   ENGINEER_MODEL=ollama:qwen3:8b
   INTEL_MODEL=groq:llama-3.1-8b-instant
   ```
   **Recommended on 16 GB RAM** (one local model instead of two; LM Studio has crashed when both were loaded):
   ```
   ANALYST_MODEL=ollama:qwen3:8b
   ENGINEER_MODEL=ollama:qwen3:8b
   INTEL_MODEL=groq:llama-3.1-8b-instant
   ```
   Any agent whose local model fails is retried on `groq:llama-3.1-8b-instant`, so the Groq key matters either way.

### 1.6 Local models (download once)
```bash
ollama serve &              # temporarily, just to pull
ollama pull qwen3:8b        # ~5 GB; already downloaded on this Mac
ollama list                 # → qwen3:8b
kill %1
```
LM Studio: open the app → search `qwen2.5-coder-7b-instruct` → Download (already downloaded on this Mac).

---

## 2. Start the services (every session)

### 2.1 Docker Desktop
```bash
open -a Docker
docker info --format '{{.ServerVersion}}'     # repeat until it prints a version (e.g. 27.3.1), ~30 s
```
In the Docker Desktop window, the whale icon in the menu bar should say "Docker Desktop is running".

### 2.2 Ollama (Terminal 2, leave open)
```bash
ollama serve
```
Expected: `Listening on 127.0.0.1:11434`. Check from Terminal 1:
```bash
ollama list          # → qwen3:8b
```

### 2.3 LM Studio (skip if you switched the Analyst to Ollama)
GUI: open LM Studio → **Developer** tab (left bar, `>_` icon) → load `qwen2.5-coder-7b-instruct` → toggle **Start Server**
(status: "Running on port 1234").
CLI alternative:
```bash
~/.lmstudio/bin/lms server start
~/.lmstudio/bin/lms load qwen2.5-coder-7b-instruct
~/.lmstudio/bin/lms ps            # → qwen2.5-coder-7b-instruct loaded
```

### 2.4 Check every LLM
```bash
python backend/check_llm.py
```
Expected, every line `OK`:
```
ANALYST_MODEL   lmstudio:qwen2.5-coder-7b-instruct   OK      -> 'OK'
ENGINEER_MODEL  ollama:qwen3:8b                      OK      -> 'OK.'
INTEL_MODEL     groq:llama-3.1-8b-instant            OK      -> 'OK'
(fallback)      groq:llama-3.1-8b-instant            OK
```
This calls each provider directly, with no fallback. Fix any FAILED line before continuing (see Troubleshooting).
The pipeline still runs with failed LLMs, but the council's reasoning is then empty.

---

## 3. Level 1: tests without Kafka (≈1 min)

### 3.1 Unit tests
```bash
cd backend
python -m unittest tests.test_agents
```
Expected: `Ran 2 tests ... OK`

### 3.2 Offline pipeline (CSV → ML → council → defender, all in one process)
```bash
python main.py --mode test --n-servers 3 --fps 5 --max-flows 40
cd ..
```
Expected while running:
- `✓ All components loaded from disk`
- lines like `[Defender] DEL-01 | DDoS (99.1%) → DEEP_PACKET_INSPECTION | latency=...`
- council blocks `[Security Analyst - ...]`, `[ML Engineer - OLLAMA:qwen3:8b]`

Expected at the end:
```
PIPELINE COMPLETE
  count  ...   mean_ms ...   p95_ms ...   p99_ms ...   throughput_per_sec ...
```
`--max-flows 40` = 40 flows per virtual server. Without it, the run replays all 300k rows (hours).

This run writes `backend/reports/actions/{DEL-01,SIN-02,TOK-03}_actions.jsonl` (see step 8.3 to clean up).

---

## 4. Level 2: the full stack in Docker

### 4.1 Start it
```bash
export GROQ_API_KEY="$(python -c "from dotenv import dotenv_values;print(dotenv_values('backend/config/.env')['GROQ_API_KEY'])")"
docker compose up --build -d
```
- First build: 10+ minutes (the IDS image installs torch). Later starts: seconds.
- Optional knobs, put before `docker compose`: `FLOWS_PER_SECOND=5` (per simulator, default 10),
  `AUTO_MONITOR=MUM-01,LON-01` (default `all`; empty = nothing is analysed until you click Monitor in the dashboard).
- Inside Docker the LLMs on your Mac are reached via `host.docker.internal`, so keep Ollama and LM Studio running.

### 4.2 Containers are up
```bash
docker compose ps -a
```
Expected:

| NAME | STATUS | Role |
|---|---|---|
| agentic-ids-kafka | Up (healthy) | Kafka broker |
| agentic-ids-kafka-init | **Exited (0)** | creates the 4 topics, then exits (this is correct) |
| agentic-ids-kafka-ui | Up | web UI on :8080 |
| agentic-ids-server-mum | Up | simulator MUM-01 (Mumbai) |
| agentic-ids-server-use | Up | simulator USE-01 (Virginia) |
| agentic-ids-server-lon | Up | simulator LON-01 (London) |
| agentic-ids-consumer | Up | the IDS + dashboard |

Also visible in **Docker Desktop → Containers → agentic-ids** (click a container → **Logs**, **Stats**).
```bash
docker logs agentic-ids-kafka-init | tail -6                # → "=== Topics ready ===" + decisions raw-flows server-registry threats
docker inspect -f '{{.RestartCount}}' agentic-ids-consumer  # → 0 (higher = crash-looping)
docker stats --no-stream                                     # CPU / RAM per container
```

### 4.3 The IDS loaded correctly
```bash
docker logs agentic-ids-consumer 2>&1 | grep -E '✓|Monitoring STARTED|Error loading|deadlock'
```
Expected:
```
✓ Loaded XGBoost model
✓ Loaded class mapping: [0 1 2]
✓ Loaded LSTM checkpoint (input=50, classes=3)
✓ Loaded 3 prompts from config/prompts/threat_analysis.txt
✓ Loaded PPO Defender from models/defender_ppo.zip
[Consumer] ▶ Monitoring STARTED for ALL servers
```
If you see `Error loading models` or `Resource deadlock avoided` → step 1.4 (iCloud), then `docker restart agentic-ids-consumer`.

### 4.4 Simulators are sending flows
```bash
docker logs agentic-ids-server-mum | tail -4
```
Expected: `[MUM-01] Registered on 'server-registry'` and `[MUM-01] 300/100009 flows  (10 flows/s)` counting up.

### 4.5 Kafka: in the browser
Open **http://localhost:8080** → cluster **agentic-ids-local** → **Topics**.

| Topic | Expect | Click → Messages to see |
|---|---|---|
| `raw-flows` | message count rising ~30/s (3 × 10) | JSON with `server_id` and 79 `features`; key = MUM-01 / USE-01 / LON-01 |
| `server-registry` | 3 `register` + 3 `heartbeat` every 30 s | server name, lat/lon, `event` |
| `threats` | rising | `attack_type` DDoS/PortScan, `confidence`, `severity` High/Medium |
| `decisions` | rising | `action` (MONITOR, RATE_LIMIT, BLOCK_SOURCE, DEEP_PACKET_INSPECTION), `latency_ms` |

Then **Consumers** → the group `agentic-ids-group-…` that has members → lag should stay small (it keeps up).

### 4.6 Kafka: from the terminal
Define the helper once per terminal (works in zsh and bash):
```bash
kt() { docker exec agentic-ids-kafka /opt/kafka/bin/"$@"; }
```
Then:
```bash
kt kafka-topics.sh --bootstrap-server localhost:9092 --list
#   → decisions  raw-flows  server-registry  threats  (+ __consumer_offsets)

kt kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic server-registry --from-beginning --property print.key=true --max-messages 3
#   → LON-01 {"server_id": "LON-01", ... "event": "register"} (one per server)

kt kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic threats --max-messages 2
kt kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic decisions --max-messages 2

kt kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --all-groups
#   look at the group whose CONSUMER-ID column is filled; groups with "no active members" are old runs
```
Always pass `--max-messages`; otherwise the consumer waits forever on a live topic (Ctrl-C stops it).

### 4.7 The pipeline deciding in real time
```bash
docker logs -f agentic-ids-consumer 2>&1 | grep --line-buffered '\[Defender\]'
```
Expected stream: `[Defender] LON-01 | PortScan (97.0%) → MONITOR | latency=7.7ms`, with DDoS, PortScan and BENIGN mixed. Ctrl-C to stop watching.

Count so far:
```bash
docker logs agentic-ids-consumer 2>&1 | grep '\[Defender\]' | awk '{print $4}' | sort | uniq -c
#   → roughly equal BENIGN / DDoS / PortScan
```

### 4.8 The LLM council
```bash
docker logs agentic-ids-consumer 2>&1 | grep -E 'Security Analyst -|ML Engineer -|Threat Intel|falling back|401' | tail -12
docker logs agentic-ids-consumer 2>&1 | grep -A4 'Engineer raw: {' | head -10
```
Expected: `[ML Engineer - OLLAMA:qwen3:8b]` followed by a JSON analysis (`model_decision_valid`, `false_positive_risk`, …).
Problems: `falling back to groq` means that local model wasn't reachable; `401` means a bad Groq key.
The council only runs on threats and is slow (seconds per call), so it lags behind the ML detections.

### 4.9 Dashboard
Open **http://localhost:5001**
1. The world map shows MUM-01, USE-01 and LON-01.
2. Flow and threat counters tick up; threats appear in the list.
3. Click a DDoS/PortScan threat → see the council's reasoning (signature match, threat actor, false-positive risk) and the chosen action.
4. The **Live Pipeline** tab animates the ingestion → detection → council → defender stages.

Raw data behind it (open in the browser):
- http://localhost:5001/api/servers: registered servers
- http://localhost:5001/api/monitored: servers being analysed
- http://localhost:5001/api/stats: counters and latency
- http://localhost:5001/api/reports: threats with council output

### 4.10 Actions written to disk
```bash
ls -la backend/reports/actions/                                # MUM-01/LON-01/USE-01 files growing
tail -n 1 backend/reports/actions/MUM-01_actions.jsonl | python -m json.tool
```
Each line is one defender action with its **dry-run** firewall command. Nothing touches your real firewall.

### 4.11 Live controls
- Stop analysing a server: dashboard server list → click its green **⏸** button (or its dot on the map). A toast says
  "⏸ Paused MUM-01" and its flows are dropped. Click **▶** to resume ("▶ Monitoring MUM-01").
- Slower/faster traffic: `docker compose down`, then `FLOWS_PER_SECOND=3 docker compose up -d`.
- Simulate a server going down: `docker stop agentic-ids-server-lon` (its heartbeats stop); `docker start agentic-ids-server-lon`.

---

## 5. Level 3: real packets with the AegisFlow C engine

Needs the stack from step 4 running (at least `kafka`).

### 5.1 Build (once)
```bash
cmake -S AegisFlow -B AegisFlow/build -DAEGISFLOW_KAFKA=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build AegisFlow/build -j8
ls AegisFlow/build/aegisflow_kafka          # binary exists
```
The configure step must print `Kafka exporter: ON`.

### 5.2 Get a pcap
Record 20 seconds of your own traffic (browse a website while it runs):
```bash
sudo tcpdump -i en0 -w /tmp/sample.pcap -G 20 -W 1
ls -la /tmp/sample.pcap
```

### 5.3 Replay it into Kafka
```bash
AegisFlow/build/aegisflow_kafka -r /tmp/sample.pcap -b localhost:9092 -s TEST -g lab
```
Expected at the end: `[kafka_exporter] summary: produced=N errors=0` with N > 0.
(The `Kafka produced : 0` line in the summary box above it is printed before the final flush, so ignore it.)

### 5.4 See the C-engine flows in Kafka
- Kafka UI → `raw-flows` → Messages → filter by key `TEST`: flat JSON with snake_case fields
  (`src_ip`, `dst_port`, `total_fwd_packets`, `flow_duration`, `server_id: "TEST"`, `captured_at_us`).
- Terminal:
  ```bash
  kt kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic raw-flows --property print.key=true --max-messages 2000 --timeout-ms 10000 | grep '^TEST' | head -2
  ```

### 5.5 The IDS analyses them
With `AUTO_MONITOR=all` (the default), flows from `TEST` are analysed too:
```bash
docker logs agentic-ids-consumer 2>&1 | grep 'TEST' | tail -5
```
Expected: `[Defender] TEST | BENIGN (...) → ALLOW` (normal browsing should come out BENIGN).
Limit: 14 of the 63 model columns aren't computed by AegisFlow and are zero-filled, so accuracy on live traffic is not validated.

### 5.6 Live capture (optional)
```bash
sudo AegisFlow/build/aegisflow_kafka -i en0 -b localhost:9092 -s MAC-01 -g home
```
Ctrl-C to stop.

---

## 6. Alternative: dashboard on the Mac without the IDS container

Useful for debugging the Python code with your local LLMs directly.
```bash
docker compose up -d kafka kafka-init kafka-ui server-mum server-use server-lon
cd backend
python main.py --mode ui --auto-monitor all        # dashboard on :5001, Ctrl-C to stop
# headless variant: python main.py --mode production   (analyses all servers, no dashboard)
cd ..
```
Don't run this while the `agentic-ids-consumer` container is up: both use port 5001.

---

## 7. Five-minute demo script
1. `python backend/check_llm.py` → all OK.
2. `docker compose up -d` → `docker compose ps -a` → everything Up, kafka-init Exited (0).
3. Kafka UI → `server-registry` (3 servers + heartbeats) → `raw-flows` growing.
4. Dashboard :5001 → map → open a DDoS threat → council reasoning → chosen action.
5. Kafka UI → `threats` and `decisions` messages.
6. `tail -n 1 backend/reports/actions/MUM-01_actions.jsonl | python -m json.tool` → dry-run command.
7. Optional: replay a pcap (step 5.3) and show the `TEST` flows.

---

## 8. Turn everything off

### 8.1 Docker stack
```bash
docker compose down            # stop and remove containers (Kafka data is kept)
docker compose down -v         # same, and also wipe Kafka data (fresh topics next time)
docker compose ps -a           # → empty
```

### 8.2 LLMs and apps
- Ollama: Ctrl-C in Terminal 2 (or `pkill -f 'ollama serve'`).
- LM Studio: `~/.lmstudio/bin/lms server stop`, then quit the app (⌘Q).
- Docker Desktop: menu-bar whale → **Quit Docker Desktop**.

Confirm nothing is left listening:
```bash
for p in 9092 8080 5001 11434 1234; do nc -z localhost $p && echo "$p still OPEN" || echo "$p closed"; done
```

### 8.3 Clean up generated files (optional)
The action logs for MUM-01, LON-01 and USE-01 are tracked in git, and both Docker and test mode append to them:
```bash
git status --short backend/reports
git checkout -- backend/reports                         # restore the tracked logs
rm -f backend/reports/actions/{DEL-01,SIN-02,TOK-03}_actions.jsonl   # remove the test-mode logs
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `check_llm`: `401 Invalid API Key` | Groq key wrong or revoked | new key in **both** `.env` files (step 1.5); re-`export GROQ_API_KEY` before `docker compose up` |
| `check_llm`: `Connection refused` on 11434 | Ollama not running | step 2.2 |
| `check_llm`: `Connection refused/reset` on 1234 | LM Studio server off or app crashed | step 2.3; on 16 GB use Ollama for both local agents |
| `check_llm`: Ollama `empty response` | model spends its tokens "thinking" | the code sends `think: false`; update Ollama (`brew upgrade ollama`) if it still happens |
| `docker info` errors | Docker Desktop not started yet | `open -a Docker`, wait ~30 s |
| consumer log: `Error loading models` / `Resource deadlock avoided` / `[Detection] Error ... 'NoneType'` | iCloud-offloaded files | step 1.4, then `docker restart agentic-ids-consumer` |
| consumer log: `falling back to groq` | local LLM unreachable from Docker | keep Ollama / LM Studio running on the Mac |
| `threats` / `decisions` stay at 0 | consumer not analysing | check step 4.3 for `Monitoring STARTED for ALL servers`; `AUTO_MONITOR` must not be empty |
| only `raw-flows` topic exists | old kafka-init | `docker compose up -d --force-recreate kafka-init` |
| port 5001 already in use | local `main.py --mode ui` still running | stop it (Ctrl-C) or `docker compose stop ids-consumer` |
| Mac very slow / LM Studio quits | RAM (16 GB) | one local model; `FLOWS_PER_SECOND=3`; close other apps |
| `cmake` error about `cmake_minimum_required` | CMake 4.x vs the fetched cJSON | keep `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` |
| `kt: command not found` | helper not defined in this terminal | run the `kt() { ... }` line from step 4.6 |
