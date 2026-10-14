# Agentic-IDS: Demo Guide (start here)

An AI-powered intrusion detection system. Network traffic flows through Kafka, two ML models classify it, a council of three LLMs reviews each threat, and a reinforcement-learning defender picks the response. You watch all of it live in a browser.

This guide assumes **a Mac where only Docker Desktop is installed**. Follow the steps in order and do not skip any. Anything in a grey box is typed into the **Terminal** app, then Enter.

> **Time needed:** about 30 minutes the first time (most of it is downloading). Later starts take about 2 minutes.

---

## Part A: One-time setup

### Step 1. Check Docker Desktop is installed and open it
1. Press `Cmd + Space`, type `Docker`, press Enter.
2. Wait until the whale icon in the top menu bar stops animating and Docker says **"Engine running"** (bottom-left corner of the Docker window).
3. Docker Desktop > Settings (gear icon) > Resources: set **Memory to 6 GB or more**, click **Apply & restart**.

### Step 2. Open Terminal
Press `Cmd + Space`, type `Terminal`, press Enter. A window with a text prompt opens. Leave it open for the whole demo.

### Step 3. Download the project
**Easy way (no extra tools):**
1. Open this link in your browser: https://github.com/sumeettummarmatti/Agentic-IDS-/archive/refs/heads/demo.zip
2. When the download finishes, open your **Downloads** folder and double-click the zip. A folder named `Agentic-IDS--demo` appears.
3. In Terminal, go into that folder:
```bash
cd ~/Downloads/Agentic-IDS--demo
```

**Or, if you already have git:**
```bash
git clone -b demo https://github.com/sumeettummarmatti/Agentic-IDS-.git
cd Agentic-IDS-
```

Check you are in the right place. This should list `run_all.sh`, `docker-compose.yml` and `backend`:
```bash
ls
```

### Step 4. Add the dataset file
The attack data (117 MB) is too big for GitHub, so it is shared separately. The presenter gives you a file named **`filtered_nowebatt.csv`**.
1. Put that file inside the folder `backend/data/raw/` (create the folders if they do not exist):
```bash
mkdir -p backend/data/raw
```
2. Drag `filtered_nowebatt.csv` into that folder in Finder, or copy it with:
```bash
cp ~/Downloads/filtered_nowebatt.csv backend/data/raw/
```
3. Check it is there:
```bash
ls -lh backend/data/raw/
```
You should see `filtered_nowebatt.csv` at about 112M.

### Step 5. Add your free Groq API key (the "brain" of the three AI experts)
1. Open https://console.groq.com/keys and sign up (free).
2. Click **Create API Key**, give it any name, and **copy** the key (it starts with `gsk_`).
3. Create your settings file from the template:
```bash
cp backend/config/.env.example backend/config/.env
```
4. Open it in TextEdit:
```bash
open -e backend/config/.env
```
5. On line 2, replace `YOUR_KEY_HERE` with your key, so it reads `GROQ_API_KEY=gsk_xxxxxxxx...`. No spaces, no quotes.
6. Press `Cmd + S` to save, then close TextEdit.

> You can also skip this and paste the key later in the dashboard (bottom-left "Groq key" button), but the first start is smoother if it is set now.

---

## Part B: Start everything (one command)

In Terminal, still inside the project folder, run:
```bash
bash run_all.sh
```
What happens (you just watch):
1. Docker is checked (started if it was closed).
2. All containers are built. **The first time takes 5 to 15 minutes**; lines of text scroll by, which is normal.
3. The script waits until the dashboard is ready, then **opens two browser tabs by itself**:
   - **Dashboard:** http://localhost:5001
   - **Kafka UI:** http://localhost:8080

It is ready when Terminal prints `Dashboard : http://localhost:5001` and a table of containers that all say `running` or `Up`.

If a message in yellow or red appears, read it. It tells you exactly what is missing (for example the dataset file or the Groq key).

---

## Part C: The demo, step by step

### Step 6. Look at the live dashboard (http://localhost:5001)
- Three servers (Mumbai, Virginia, London) are sending network traffic. The numbers and table update by themselves.
- Click the **Pipeline** tab to see each stage light up as flows move through: *Servers > Kafka > ML detection > AI council > Defender*.

### Step 7. Launch an attack in the Attack Lab
1. At the top of the dashboard, click **Attack Lab**.
2. A warning box appears. It says live analysis pauses while you are here. Click **Enter Attack Lab**.
3. Pick a **server** (for example Mumbai).
4. Pick an attack: **DDoS**, **Port scan**, or **Normal** traffic.
5. Click **Launch**. This sends 30 real labelled attack flows through the whole system.

### Step 8. Watch the AI work
Scroll down the Attack Lab page:
- **Pipeline stages** at the top light up one by one as flows move through.
- **Who said what** compares three opinions: the ML models, the expert majority, and the defender.
- **The AI experts at work** shows three experts reading each flow at the same time:
  - **Security Analyst** classifies the attack,
  - **ML Engineer** checks whether the ML model could be wrong (false alarm?),
  - **Threat Intel** matches the traffic to known attacks such as Mirai.
  Each shows its answer, reasoning, and the exact prompt it received (click to expand).
- The **step-by-step log** at the bottom narrates everything in plain English.

Sometimes an expert takes 5 to 10 seconds. That is the free Groq tier being rate limited. It recovers by itself; a bar at the bottom explains it.

### Step 9. Go back to normal
Click **Back to dashboard**. Live analysis resumes.

---

## Part D: Stop everything
```bash
bash run_all.sh stop
```
To start again later, just run `bash run_all.sh` again (about 2 minutes now).

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `docker: command not found` | Open Docker Desktop (Step 1) and wait for "Engine running", then rerun. |
| `Missing backend/data/raw/filtered_nowebatt.csv` | Do Step 4. |
| `still says YOUR_KEY_HERE` or `Missing backend/config/.env` | Do Step 5. |
| `no such file or directory: run_all.sh` | You are not in the project folder. Run `cd ~/Downloads/Agentic-IDS--demo` (or the folder you cloned) and try again. |
| Dashboard page is blank | Wait 1 minute and refresh. Still blank: `docker restart agentic-ids-consumer`. |
| Amber bar "Groq rate limit" | Wait. It clears itself. Or paste a different Groq key using the bottom-left "Groq key" button. |
| Experts never answer | Free Groq daily limit reached. Click the "Groq key" button and paste a key from another Groq account. |
| Port 5001 / 8080 already in use | Close the other program using it, or run `bash run_all.sh stop` first. |

Useful extras:
- Live logs: `docker logs -f agentic-ids-consumer`
- All containers: `docker ps`
- Full reset (also wipes Kafka data): `bash run_all.sh stop -v`

---

# Developer reference

(The sections below are the original technical notes. Newer details are in `MANUAL.md`.)

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
