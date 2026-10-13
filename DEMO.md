# Agentic-IDS — Run & Demo Guide (branch v2-kafka-nexus)

## 1. One-time setup
1. Put the dataset at `backend/data/raw/filtered_nowebatt.csv` (it is gitignored).
2. `cp backend/config/.env.example backend/config/.env` and set `GROQ_API_KEY`.
3. Trained models must exist in `backend/models/` (`trained/*.pkl|pt`, `defender_ppo.zip`). If not, the first
   run trains them (slow) — do this *before* the demo: `cd backend && python main.py --mode test --fps 0`.
4. Verify the LLMs answer: `python backend/check_llm.py` (change `*_MODEL` in `.env` if one fails; non-Groq
   models such as `ollama:`/`lmstudio:` are skipped).
5. Precedence: real environment variables win over `.env` files; `backend/.env` is read before `backend/config/.env`.

## 2. Run the full stack (Docker)
```bash
export GROQ_API_KEY=...          # or put it in a root .env next to docker-compose.yml
docker compose up --build
```
| URL | What |
|---|---|
| http://localhost:5001 | IDS dashboard (map, live flows, threats, actions) |
| http://localhost:8080 | Kafka UI — watch topics `raw-flows`, `threats`, `decisions` fill |

The container mounts `backend/config` and reads `*_MODEL` from `config/.env` (compose does not set them).
Use `groq:` models there for Docker — `ollama:`/`lmstudio:` point at `localhost` inside the container and fall back to Groq.
Action logs land on the host in `backend/reports/actions/` (mounted).

Knobs: `FLOWS_PER_SECOND=10` (per simulated server), `AUTO_MONITOR=all|MUM-01,LON-01|` (empty = click Monitor in the UI).
Keep the rate low: every threat goes to 3 LLM calls and Groq's free tier rate-limits.

## 3. Run without Docker
- Offline pipeline: `cd backend && python main.py --mode test --n-servers 3 --fps 5 --max-flows 40`
  (without `--max-flows` it replays the whole ~300k-row dataset: hours at 5 fps)
- Dashboard + Kafka: start Kafka, then `python main.py --mode ui` (or `--mode production`)
- Real packets: build and run the C engine
  ```bash
  cmake -S AegisFlow -B AegisFlow/build -DAEGISFLOW_KAFKA=ON   # CMake 4.x: add -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  cmake --build AegisFlow/build
  AegisFlow/build/aegisflow_kafka -r capture.pcap -b localhost:9092 -s MUM-01 -g asia-south1
  ```
  Messages are flat JSON (snake_case keys) on `raw-flows`, keyed by server_id; the IDS wraps and bridges them.

## 4. Data flow (what to say in the demo)
`AegisFlow (C: libpcap → flow table → features)` → Kafka `raw-flows` (keyed by server_id) →
**Ingestion** (bridge column names, scale) → **Detection** (XGBoost 0.7 + LSTM 0.3, threshold 0.6; BENIGN/DDoS/PortScan) →
publishes `threats` → **Council** (analyst, ML engineer, threat-intel LLMs vote) → **Defender** (PPO picks MONITOR / BLOCK_SOURCE /
DPI / RATE_LIMIT) → publishes `decisions` + dry-run iptables command in `reports/actions/*.jsonl`.

## 5. Suggested demo script
1. Show Kafka UI: 3 servers heartbeat on `server-registry`.
2. Open the dashboard: servers on the map, flows ticking.
3. Point at a DDoS detection → council reasoning → chosen action.
4. Show `threats` and `decisions` topics in Kafka UI.
5. Show `reports/actions` (commands are dry-run; nothing touches the host firewall).

## 6. Known limits (be upfront)
- Of the 63 columns the preprocessor was fitted on, AegisFlow input covers 49 (direct or derived); 14 are zero-filled
  (directional PSH/URG flags, header lengths, CWE/ECE, Fwd Act Data Pkts, Fwd Seg Size Min, Active*), so accuracy on *live* capture is unvalidated. CSV replay is the reliable demo.
- Council quality/latency depends on Groq model availability and rate limits.
