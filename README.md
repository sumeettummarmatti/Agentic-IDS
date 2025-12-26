# Agentic-IDS: Multi-Agent Intrusion Detection System

Advanced threat detection using:
- **Attacker Agents**: Generate synthetic edge-case data (DDoS, PortScan, Evasion)
- **Defender Agents**: RL-based defense optimization
- **LLM Council**: Karpathy's framework for multi-LLM consensus reasoning
- **Ensemble Detector**: XGBoost + LSTM classification

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/agentic-ids.git
cd agentic-ids

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp config/.env.example config/.env
# Edit config/.env with your Groq API key

# Start Ollama (separate terminal)
ollama serve

# Run pipeline
python main.py
```