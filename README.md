# Agentic-IDS — Setup & Run Guide

## Project Structure

```
agentic-ids/
├── AegisFlow/                     # C packet capture engine (submodule)
│   ├── src/
│   │   ├── exporter/
│   │   │   ├── kafka_exporter.c   ← NEW: publishes flows to Kafka
│   │   │   ├── json_exporter.c
│   │   │   └── csv_exporter.c
│   │   ├── kafka_main.c           ← NEW: kafka-enabled entry point
│   │   └── main.c
│   ├── include/
│   │   ├── kafka_exporter.h       ← NEW
│   │   └── ...
│   └── CMakeLists.txt             ← patched: -DAEGISFLOW_KAFKA=ON
│
├── backend/                       # Python ML + Kafka pipeline
│   ├── src/
│   │   ├── agents/                Attacker/Defender RL agents
│   │   ├── council/               LLM multi-agent council
│   │   ├── detector/              XGBoost + LSTM ensemble
│   │   ├── ingestion/             AegisFlow → IDS feature bridge
│   │   ├── kafka/                 Producer / Consumer / Registry
│   │   ├── pipeline/              Async IDS pipeline stages
│   │   ├── simulation/            Dataset slicer + replay engine
│   │   └── ui/                    FastAPI Server Manager API
│   ├── data/                      Raw datasets (CSV/XLSX)
│   ├── models/                    Trained model checkpoints
│   ├── config/                    .env, .env.example
│   ├── main.py                    Entry point (test/production/ui modes)
│   ├── server_simulator.py        Docker server simulator script
│   ├── benchmark_latency.py       Latency + load balancing benchmark
│   └── requirements.txt
│
├── frontend/                      # Dashboard UI (pure HTML/CSS/JS)
│   └── index.html                 Server Manager dashboard
│
├── Dockerfile.server              Lightweight server simulator container
├── Dockerfile.ids                 Full IDS pipeline container
├── docker-compose.yml             Complete stack (Kafka + 3 servers + IDS)
├── build_aegisflow.sh             Build script for AegisFlow C binary
└── README.md
```

---

## How Data Flows (Production Mode)

```
Network Interface (eth0)
        │
        ▼
  AegisFlow (C binary)          ← libpcap capture → CICFlowMeter features
  aegisflow_kafka -i eth0
        │  librdkafka
        ▼
  Kafka  raw-flows topic         ← partitioned by server_id
        │
        ▼
  IDS Pipeline (Python async)   ← XGBoost + LSTM ensemble
  → LLM Council (Groq/Ollama)
  → RL Defender Agent
        │
        ▼
  Kafka  threats / decisions topics
```

```
[Simulation / Test Mode — no real capture needed]

  Dataset CSV
        │  server_simulator.py (Docker)
        │  or DatasetSlicer + ReplayEngine (local Python)
        ▼
  Kafka  raw-flows topic
        │
        ▼
  IDS Pipeline  (same as above)
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | IDS pipeline |
| Docker + Docker Compose | Latest | Kafka + server containers |
| cmake | ≥ 3.16 | Build AegisFlow C binary |
| libpcap-dev | Any | Packet capture |
| librdkafka-dev | Any | Kafka C producer |

---

## Quick Start (Simulation — no real traffic needed)

```bash
# 1. Install Python deps
cd backend
pip install -r requirements.txt

# 2. Set your Groq API key
cp config/.env.example config/.env
# edit config/.env → set GROQ_API_KEY=gsk_...

# 3. Start Kafka + 3 geo-servers + IDS consumer (all in Docker)
cd ..
docker-compose up --build

# Done! Open:
#   http://localhost:8080  → Kafka UI (watch flows arrive)
#   http://localhost:5001  → Server Manager dashboard
```

---

## Step-by-Step: Local Python (No Docker for IDS)

```bash
# Terminal 1 — Kafka + simulated servers only
docker-compose up kafka kafka-init kafka-ui server-mum server-use server-lon

# Terminal 2 — IDS pipeline (local Python, connects to Kafka)
cd backend
conda activate agentic-ids
python main.py --mode production

# Or with the dashboard UI:
python main.py --mode ui    # → http://localhost:5001
```

---

## Step-by-Step: Test Mode (No Kafka at all)

```bash
cd backend
conda activate agentic-ids
python main.py --mode test \
  --data data/raw/filtered_nowebatt.csv \
  --n-servers 3
```

---

## Build AegisFlow (Production Packet Capture)

```bash
# Standard build (JSON/CSV output only)
./build_aegisflow.sh

# With Kafka support (publishes flows directly to Kafka)
./build_aegisflow.sh --kafka
```

Requires on macOS:
```bash
brew install libpcap librdkafka cmake
```

Requires on Ubuntu/Debian:
```bash
sudo apt install libpcap-dev librdkafka-dev cmake build-essential
```

### Run AegisFlow → Kafka (production on real hardware)

```bash
# Make sure Kafka is running first (docker-compose up kafka)

# Live capture (needs sudo for raw socket)
sudo ./AegisFlow/build/aegisflow_kafka \
  -i eth0 \
  -b localhost:9092 \
  -T raw-flows \
  -s MUM-01 \
  -g asia-south1

# Offline PCAP replay (no sudo needed)
./AegisFlow/build/aegisflow_kafka \
  -r capture.pcap \
  -b localhost:9092 \
  -T raw-flows \
  -s MUM-01 \
  -g asia-south1

# Or use env vars (great for Docker/systemd):
export SERVER_ID=MUM-01
export GEO_REGION=asia-south1
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
sudo ./AegisFlow/build/aegisflow_kafka -i eth0
```

---

## Docker Stack Details

| Service | Image | Port | Purpose |
|---|---|---|---|
| `kafka` | apache/kafka:3.7.0 | 9092 (host), 29092 (internal) | Message broker |
| `kafka-ui` | provectuslabs/kafka-ui | 8080 | Topic inspector |
| `kafka-init` | apache/kafka | — | Creates topics once |
| `server-mum` | Dockerfile.server | — | Mumbai geo-server sim |
| `server-use` | Dockerfile.server | — | US-East geo-server sim |
| `server-lon` | Dockerfile.server | — | London geo-server sim |
| `ids-consumer` | Dockerfile.ids | 5001 | Full IDS + dashboard |

### Useful docker-compose commands

```bash
# Full stack
docker-compose up --build

# Infra only (Kafka + UI)
docker-compose up kafka kafka-init kafka-ui

# Watch logs of one server
docker-compose logs -f server-mum

# Watch IDS consumer decisions
docker-compose logs -f ids-consumer

# Scale up a server (add more Mumbai traffic)
docker-compose up --scale server-mum=2

# Add a 4th server (Singapore) without changing compose file
docker-compose run --rm \
  -e SERVER_ID=SGP-01 \
  -e GEO_REGION=asia-southeast1 \
  -e SLICE_INDEX=0 \
  -e TOTAL_SLICES=1 \
  -e FLOWS_PER_SECOND=200 \
  server-mum

# Tear down everything
docker-compose down -v
```

---

## Latency + Load Balancing Benchmark

```bash
cd backend
conda activate agentic-ids

# Full benchmark suite (A + B + C)
python benchmark_latency.py \
  --data data/raw/filtered_nowebatt.csv \
  --n 300 \
  --lb-servers 4

# Load balancing only
python benchmark_latency.py --n 300 --lb-only
```

Tests:
- **A** — Direct (no Kafka) baseline latency
- **B** — Async pipeline latency + throughput
- **C1** — Balanced load: all servers same rate
- **C2** — Unbalanced: 10% / 30% / 60% split → partition isolation check
- **C3** — Spike: one server 5× burst → does it affect others?

---

## Quick Reference

```bash
# From repo root
docker-compose up --build          # full stack
docker-compose up kafka kafka-ui   # infra only
./build_aegisflow.sh --kafka       # build C binary

# From backend/
python main.py --mode test         # test pipeline (no Kafka)
python main.py --mode production   # live Kafka consumer
python main.py --mode ui           # dashboard at :5001
python benchmark_latency.py --n 300   # benchmark
```
