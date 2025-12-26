# Agentic-IDS: Multi-Agent Intrusion Detection System

An advanced, autonomous Intrusion Detection System (IDS) that combines high-speed Machine Learning with the reasoning capabilities of Large Language Models (LLMs) and the adaptive decision-making of Reinforcement Learning (RL).

![Agentic IDS Architecture](https://via.placeholder.com/800x400?text=Agentic+IDS+Architecture)

## 🚀 Key Features

*   **Ensemble Detection**: Combines **XGBoost** (for tabular accuracy) and **LSTM** (for sequence analysis) to detect attacks with **99.9% accuracy**.
*   **Threat Analysis Council**: A multi-agent LLM system (Security Analyst, ML Engineer, Threat Intel) that investigates high-confidence threats.
*   **Adaptive Defense**: A **PPO (Reinforcement Learning)** agent that autonomously executes mitigation actions (Block, Rate Limit, DPI).
*   **Dynamic Configuration**: Customize agent personas and logic via simple text files without changing code.
*   **Live Monitoring**: Replay real network traffic flows to test the system in real-time.

## 🛠️ Architecture

The system operates in three phases:
1.  **Detection Layer**: Analyzes raw network flows using a pre-trained Ensemble Model.
2.  **Reasoning Layer**: If a threat is detected (>60% confidence), the **Council** convenes to correlate features, match signatures, and assess severity.
3.  **Response Layer**: The **Defender Agent** receives the Council's report and executes the optimal mitigation strategy.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/agentic-ids.git
    cd agentic-ids
    ```

2.  **Set up Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Create a `.env` file in the `config/` directory:
    ```bash
    cp config/.env.example config/.env
    ```
    Edit `config/.env` and add your API keys:
    ```ini
    GROQ_API_KEY=gsk_...
    ANALYST_MODEL=groq:llama-3.1-8b-instant
    ENGINEER_MODEL=groq:llama-3.3-70b-versatile
    INTEL_MODEL=ollama:qwen2.5:8b
    ```

5.  **Set up Ollama (Local LLM)**
    Ensure [Ollama](https://ollama.com/) is installed and running:
    ```bash
    ollama serve
    ollama pull qwen2.5:8b  # Or your preferred model
    ```

## 🚦 Usage

### 1. Run the Full Pipeline
Starts the simulation, trains the models on your data, and begins the monitoring loop.
```bash
python main.py
```

### 2. Live Data Replay
Test the system with your own network traffic file (CSV/Excel).
```bash
python main.py --live-data data/raw/filtered_nowebatt.csv
```
*The system will automatically map your columns and handle feature selection.*

### 3. Run Accuracy Benchmark
Verify the detection performance against the test set.
```bash
python benchmark.py
```

## ⚙️ Configuration

### Dynamic Prompts
You can tune the behavior of the AI Council **without coding**.
Edit `config/prompts/threat_analysis.txt` to modify the instructions for:
*   `[SECURITY_ANALYST_PROMPT]`
*   `[ML_ENGINEER_PROMPT]`
*   `[THREAT_INTEL_PROMPT]`

The system reloads these prompts automatically every time it runs.

## 📊 Dataset
The system is optimized for key network flow features including:
*   `Total Fwd Packet`, `Total Bwd packets`
*   `Flow Duration`, `Flow Bytes/s`, `Flow Packets/s`
*   `Fwd/Bwd Header Length`
*   `SYN/RST Flag Counts`

## 🤝 Contributing
1.  Fork the repo
2.  Create a feature branch
3.  Commit your changes
4.  Push to the branch
5.  Create a Pull Request

## 📄 License
MIT License