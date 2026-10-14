# Agentic-IDS: Demo Guide (start here)

An AI-powered intrusion detection system. Network traffic flows through Kafka, two ML models classify it, a council of three LLMs reviews each threat, and a reinforcement-learning defender picks the response. You watch all of it live in a browser.

This guide works on **Mac, Ubuntu and Windows**. It assumes a fresh laptop with nothing installed. Follow the steps for **your system only**, in order, and do not skip any.

> **Time needed:** about 30 minutes the first time (most of it is downloading). Later starts take about 2 minutes.
> **You also need:** internet, and about 10 GB of free disk space.

**Jump to your system:** [Mac](#part-a-mac) | [Ubuntu](#part-a-ubuntu) | [Windows](#part-a-windows). After that, everyone continues at [Part B](#part-b-start-everything-one-command).

Commands in grey boxes are typed into a terminal, then Enter. The terminal is called **Terminal** on Mac and Ubuntu, and **Ubuntu (WSL)** on Windows.

---

## Part A (Mac)

### A1. Install and open Docker Desktop
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/ (pick Apple Silicon for M1/M2/M3/M4 Macs, Intel otherwise) and drag it into Applications.
2. Press `Cmd + Space`, type `Docker`, press Enter. Accept the terms.
3. Wait until the whale icon in the menu bar stops moving and the window says **Engine running**.
4. Settings (gear) > Resources: set **Memory to 6 GB or more**, then **Apply & restart**.

### A2. Open Terminal
Press `Cmd + Space`, type `Terminal`, press Enter. Leave it open.

### A3. Download the project
**Easy way (no extra tools):** open https://github.com/sumeettummarmatti/Agentic-IDS-/archive/refs/heads/demo.zip in your browser. Double-click the zip in **Downloads**, then:
```bash
cd ~/Downloads/Agentic-IDS--demo
```
**Or with git:**
```bash
git clone -b demo https://github.com/sumeettummarmatti/Agentic-IDS-.git
cd Agentic-IDS-
```
Check you are in the right place. This should list `run_all.sh`, `docker-compose.yml` and `backend`:
```bash
ls
```

### A4. Add the dataset file
The attack data (117 MB) is too big for GitHub, so it is shared separately. You are given a file named **`filtered_nowebatt.csv`**.
```bash
mkdir -p backend/data/raw
cp ~/Downloads/filtered_nowebatt.csv backend/data/raw/
ls -lh backend/data/raw/
```
You should see `filtered_nowebatt.csv` at about 112M.

### A5. Add your free Groq API key
1. Open https://console.groq.com/keys, sign up (free), click **Create API Key**, and **copy** the key (it starts with `gsk_`).
2. Create your settings file and open it:
```bash
cp backend/config/.env.example backend/config/.env
open -e backend/config/.env
```
3. On line 2, replace `YOUR_KEY_HERE` with your key, so it reads `GROQ_API_KEY=gsk_xxxxxxxx...`. No spaces, no quotes. Press `Cmd + S` and close TextEdit.

Now go to [Part B](#part-b-start-everything-one-command).

---

## Part A (Ubuntu)

Tested target: Ubuntu 22.04 or newer on a normal PC (64-bit).

### A1. Install the tools
```bash
sudo apt update
sudo apt install -y git curl unzip nano
```

### A2. Install Docker
This is Docker's own installer script. It also installs `docker compose`.
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```
**Now log out and log back in** (or restart), so your user can use Docker without `sudo`. Then check it works:
```bash
docker run --rm hello-world
docker compose version
```
Both should print without errors.

### A3. Download the project
```bash
git clone -b demo https://github.com/sumeettummarmatti/Agentic-IDS-.git
cd Agentic-IDS-
ls
```
`ls` should list `run_all.sh`, `docker-compose.yml` and `backend`.

### A4. Add the dataset file
You are given a file named **`filtered_nowebatt.csv`** (117 MB, shared separately because it is too big for GitHub).
```bash
mkdir -p backend/data/raw
cp ~/Downloads/filtered_nowebatt.csv backend/data/raw/
ls -lh backend/data/raw/
```
You should see `filtered_nowebatt.csv` at about 112M.

### A5. Add your free Groq API key
1. Open https://console.groq.com/keys, sign up (free), click **Create API Key**, and **copy** the key (it starts with `gsk_`).
2. Create and edit your settings file:
```bash
cp backend/config/.env.example backend/config/.env
nano backend/config/.env
```
3. On line 2, replace `YOUR_KEY_HERE` with your key, so it reads `GROQ_API_KEY=gsk_xxxxxxxx...`. No spaces, no quotes. Save with `Ctrl + O`, Enter, then leave with `Ctrl + X`.

Now go to [Part B](#part-b-start-everything-one-command).

---

## Part A (Windows)

On Windows we run the project inside **WSL 2** (a real Ubuntu inside Windows). This avoids path and line-ending problems. Needs Windows 10 (version 2004 or newer) or Windows 11.

> **What is WSL, in plain words?** WSL is a small Linux (Ubuntu) that lives inside Windows. You open it like any app, and you get a black terminal window. The project's scripts are written for Linux, so we run them there. Docker Desktop (a normal Windows app) does the heavy lifting in the background, and the two talk to each other.
>
> **Two windows, two jobs. Do not mix them up:**
>
> | Window | Used for |
> |---|---|
> | **PowerShell** (blue) | Only step A1: installing WSL. Nothing else. |
> | **Ubuntu** (black) | **Every other command in this guide.** Its prompt looks like `name@PC:~$`. |
>
> **Where files live:** inside Ubuntu, your home folder is `~`. Keep the project there (`~/Agentic-IDS-`). To see it in Windows File Explorer, run `explorer.exe .` in Ubuntu, or type `\\wsl$\Ubuntu` in the Explorer address bar. Avoid keeping the project under `/mnt/c/...` (your C: drive): it is slower and more likely to break.
>
> **Browser:** the dashboard opens in your normal Windows browser at http://localhost:5001. WSL and Windows share `localhost`.

### A1. Install WSL with Ubuntu
1. Click Start, type `PowerShell`, right-click **Windows PowerShell**, choose **Run as administrator**.
2. Run:
```powershell
wsl --install -d Ubuntu
```
3. **Restart the computer** when asked. After the restart an **Ubuntu** window opens by itself. Choose a username and password (remember them) and wait until the prompt appears.

### A2. Install Docker Desktop and connect it to Ubuntu
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/ and install it. Keep **Use WSL 2** ticked. Restart if asked.
2. Open Docker Desktop and wait for **Engine running**.
3. Settings (gear) > **Resources > WSL integration**: switch on **Ubuntu**, then **Apply & restart**.
4. Settings > Resources: if you see a Memory setting, set it to **6 GB or more**.

### A3. Open Ubuntu and check Docker works
Click Start, type `Ubuntu`, open it. This is the terminal for **all** remaining steps. Type:
```bash
sudo apt update
sudo apt install -y git curl unzip nano
docker run --rm hello-world
docker compose version
```
The last two must print without errors. If Docker is "not found", redo A2 step 3.

### A3b. Give Docker a working internet connection (do this now, it prevents the most common error)
On many Windows PCs, especially on college, office or VPN networks, Docker cannot reach the Python package server while building, and the build fails with `Could not find a version that satisfies the requirement numpy (from versions: none)`. Prevent it:
1. Open **Docker Desktop**, click the **gear** (Settings), then **Docker Engine** on the left.
2. You will see a box of text (JSON). Make it look like this. Keep any lines that are already there, and put a comma after the line above the new one:
```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "dns": ["8.8.8.8", "1.1.1.1"]
}
```
3. Click **Apply & restart** and wait for **Engine running**.
4. Back in **Ubuntu**, test it (it should finish with `Successfully installed numpy`):
```bash
docker run --rm python:3.11-slim pip install numpy
```

### A4. Download the project (into the Ubuntu side)
Do this in your Ubuntu home folder, **not** inside `/mnt/c`:
```bash
cd ~
git clone -b demo https://github.com/sumeettummarmatti/Agentic-IDS-.git
cd Agentic-IDS-
ls
```
`ls` should list `run_all.sh`, `docker-compose.yml` and `backend`.

### A5. Add the dataset file
You are given a file named **`filtered_nowebatt.csv`** (117 MB, shared separately). Save it in your Windows **Downloads** folder first, then copy it across (replace `YOURNAME` with your Windows user name):
```bash
mkdir -p backend/data/raw
cp /mnt/c/Users/YOURNAME/Downloads/filtered_nowebatt.csv backend/data/raw/
ls -lh backend/data/raw/
```
You should see `filtered_nowebatt.csv` at about 112M. Not sure of your user name? Type `ls /mnt/c/Users`.

### A6. Add your free Groq API key
1. Open https://console.groq.com/keys, sign up (free), click **Create API Key**, and **copy** the key (it starts with `gsk_`).
2. Create and edit your settings file:
```bash
cp backend/config/.env.example backend/config/.env
nano backend/config/.env
```
3. On line 2, replace `YOUR_KEY_HERE` with your key, so it reads `GROQ_API_KEY=gsk_xxxxxxxx...`. No spaces, no quotes. Paste with a right-click. Save with `Ctrl + O`, Enter, then leave with `Ctrl + X`.

Now go to [Part B](#part-b-start-everything-one-command).

---

## Part B: Start everything (one command)

In your terminal (Terminal on Mac and Ubuntu, Ubuntu on Windows), inside the project folder, run:
```bash
bash run_all.sh
```
What happens (you just watch):
1. Docker is checked. On Mac it is started if closed. On Ubuntu it tries to start the Docker service. On Windows, open Docker Desktop yourself first.
2. All containers are built. **The first time takes 5 to 15 minutes.** Lines of text scroll by, which is normal.
3. The script waits until the dashboard is ready, then tries to **open two browser tabs by itself**:
   - **Dashboard:** http://localhost:5001
   - **Kafka UI:** http://localhost:8080

If the tabs do not open (this can happen on Ubuntu servers or WSL), open those two addresses yourself in your browser. On Windows, use your normal Windows browser.

It is ready when the terminal prints `Dashboard : http://localhost:5001` and a table of containers that all say `running` or `Up`.

If a message in yellow or red appears, read it. It says exactly what is missing (for example the dataset file or the Groq key).

---

## Part C: The demo, step by step

### Step 1. Look at the live dashboard (http://localhost:5001)
- Three servers (Mumbai, Virginia, London) send network traffic. The numbers and table update by themselves.
- Click the **Pipeline** tab to see each stage light up as flows move through: *Servers > Kafka > ML detection > AI council > Defender*.

### Step 2. Launch an attack in the Attack Lab
1. At the top of the dashboard, click **Attack Lab**.
2. A warning box says live analysis pauses while you are here. Click **Enter Attack Lab**.
3. Pick a **server** (for example Mumbai).
4. Pick an attack: **DDoS**, **Port scan**, or **Normal** traffic.
5. Click **Launch**. This sends 30 real labelled flows through the whole system.

### Step 3. Watch the AI work
Scroll down the Attack Lab page:
- **Pipeline stages** light up one by one as flows move through.
- **Who said what** compares three opinions: the ML models, the expert majority, and the defender.
- **The AI experts at work** shows three experts reading each flow at the same time:
  - **Security Analyst** classifies the attack,
  - **ML Engineer** checks whether the ML model could be wrong (false alarm?),
  - **Threat Intel** matches the traffic to known attacks such as Mirai.
  Each shows its answer, reasoning, and the exact prompt it received (click to expand).
- The **step-by-step log** at the bottom narrates everything in plain English.

Sometimes an expert takes 5 to 10 seconds. That is the free Groq tier being rate limited. It recovers by itself, and a bar at the bottom explains it.

### Step 4. Go back to normal
Click **Back to dashboard**. Live analysis resumes.

### Optional: the Kafka UI (http://localhost:8080)
Click **Topics**, then `raw-flows`, then **Messages** to see each flow arrive, with the server ID as its key. `threats` shows flagged attacks and `decisions` shows what the defender chose. Do not delete topics or clear messages during a demo.

---

## Part D: Stop everything
```bash
bash run_all.sh stop
```
To start again later, just run `bash run_all.sh` again (about 2 minutes now). Docker Desktop (Mac and Windows) stays open; quit it from its icon if you want.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `docker: command not found` | Do Docker install (Part A, step 1 or 2 for your system), open Docker Desktop, wait for "Engine running", then rerun. |
| Windows: `docker` not found inside Ubuntu | Docker Desktop > Settings > Resources > WSL integration > switch on Ubuntu > Apply. |
| Ubuntu: `permission denied ... docker.sock` | You skipped the `usermod` step or did not log out and in. Run `sudo usermod -aG docker $USER`, then log out and back in. |
| `'docker compose' (v2) not found` | Update Docker Desktop. On Ubuntu use the install script in Part A (Ubuntu) step 2. |
| `curl not found` | Ubuntu or WSL: `sudo apt install -y curl`. |
| `Missing backend/data/raw/filtered_nowebatt.csv` | Do the "Add the dataset file" step for your system. |
| `still says YOUR_KEY_HERE` or `Missing backend/config/.env` | Do the Groq key step for your system. |
| `no such file or directory: run_all.sh` | You are not in the project folder. `cd` into it (the `ls` check in the download step should list `run_all.sh`). |
| `Could not find a version that satisfies the requirement numpy (from versions: none)` or any `pip install` failing during the build | Docker has no working DNS. Do Windows step **A3b** (add `"dns": ["8.8.8.8", "1.1.1.1"]` in Docker Desktop > Settings > Docker Engine > Apply & restart). Still failing: turn off any VPN, try your phone hotspot (college Wi-Fi often blocks PyPI), or run `wsl --shutdown` in PowerShell and reopen Docker Desktop and Ubuntu. On Ubuntu Linux, test with the same `docker run --rm python:3.11-slim pip install numpy` command. |
| `ReadTimeoutError ... files.pythonhosted.org` during the build | The download was too slow and timed out. Make sure you have the latest project (`git pull`), because the build now uses a much smaller CPU-only PyTorch and retries automatically. Then run `bash run_all.sh` again. It resumes from what is already downloaded. If it keeps failing, use a faster network or a phone hotspot. |
| Windows: commands like `bash run_all.sh` do not work in PowerShell | You are in the wrong window. Open the **Ubuntu** app (black window, prompt like `name@PC:~$`). Only the WSL install in A1 uses PowerShell. |
| Windows: project is slow or acts strange under `/mnt/c/...` | Move it to the Ubuntu home folder: `cd ~` then `git clone -b demo https://github.com/sumeettummarmatti/Agentic-IDS-.git`, and copy `filtered_nowebatt.csv` and your `.env` again. |
| `bad interpreter` or `\r: command not found` (Windows) | The files got Windows line endings. Re-clone the project inside Ubuntu (`git clone` in `~`), not from a Windows folder. |
| Browser tabs did not open | Open http://localhost:5001 and http://localhost:8080 yourself. |
| Dashboard page is blank | Wait 1 minute and refresh. Still blank: `docker restart agentic-ids-consumer`. |
| Amber bar "Groq rate limit" | Wait. It clears itself. Or paste a different Groq key with the bottom-left "Groq key" button. |
| Experts never answer | Free Groq daily limit reached. Click the "Groq key" button and paste a key from another Groq account. |
| Port 5001 / 8080 already in use | Close the other program using it, or run `bash run_all.sh stop` first. |
| First build is very slow or fails | Check internet. Make sure Docker has at least 6 GB of memory and 10 GB of free disk. |

Useful extras (all systems):
- Live logs: `docker logs -f agentic-ids-consumer`
- All containers: `docker ps`
- Full reset (also wipes Kafka data): `bash run_all.sh stop -v`

> **Not tested:** the Mac steps were tested end to end. The Ubuntu and Windows steps follow the standard Docker and WSL 2 setup but have not been run on a fresh machine. If a step fails, the table above covers the common causes.

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
