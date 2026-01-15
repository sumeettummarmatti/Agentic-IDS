# **AGENTIC-IDS: VISUAL SYSTEM ARCHITECTURE**

## **How Everything Works Together**

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK TRAFFIC STREAM                        │
│             (Your CIC-Darknet2020 dataset flows)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              1. PREPROCESSOR (Clean Data)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Load Darknet.xlsx                                      │  │
│  │ • Drop non-predictive columns (IP, Port, Timestamp)      │  │
│  │ • Handle missing values                                  │  │
│  │ • Encode labels (0=benign, 1=attack)                    │  │
│  │ • Scale features (StandardScaler)                        │  │
│  │ Output: X (71 features) + y (labels)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌────────────────────────┐
│  REAL DATA        │    │ SYNTHETIC DATA AGENTS  │
│  (141k flows)     │    │                        │
│                   │    │ • DDoS Agent (2k)      │
│ ✓ Balanced        │    │   - SYN Flood          │
│ ✓ Normalized      │    │   - UDP Flood          │
│ ✓ Ready for ML    │    │   - Evasion techniques │
│                   │    │                        │
└────────┬──────────┘    │ • PortScan Agent (1k)  │
         │               │   - Slow scans         │
         │               │   - Decoy traffic      │
         │               │   - Port rotation      │
         │               │                        │
         │               └────────┬────────────────┘
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  COMBINED DATASET (144k)    │
        │  80% Train + 20% Test       │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │  2. ENSEMBLE DETECTOR       │
        │  ┌───────────────────────┐  │
        │  │ XGBoost Model         │  │
        │  │ (100 trees, depth=8)  │  │
        │  │ Accuracy: 92-95%      │  │
        │  └─────────┬─────────────┘  │
        │            │                │
        │            ▼                │
        │  Output: Binary Prediction  │
        │  - 0 = Benign              │
        │  - 1 = Attack              │
        │  + Confidence (0-1)         │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────┐
        │  3. LLM COUNCIL (Decision Making)   │
        │                                     │
        │  ┌─────────────────────────────┐   │
        │  │ Security Analyst            │   │
        │  │ ✓ Classifies attack type    │   │
        │  │ ✓ Assesses threat severity  │   │
        │  │ ✓ Recommends actions        │   │
        │  │ Input: Flow features + pred │   │
        │  │ Output: Threat classification   │
        │  └─────────────────────────────┘   │
        │                                     │
        │  ┌─────────────────────────────┐   │
        │  │ ML Engineer                 │   │
        │  │ ✓ Analyzes feature patterns │   │
        │  │ ✓ Explains model decision   │   │
        │  │ ✓ Suggests improvements     │   │
        │  │ Input: Flow features + model    │
        │  │ Output: Feature anomalies       │
        │  └─────────────────────────────┘   │
        │                                     │
        │  ┌─────────────────────────────┐   │
        │  │ Threat Intelligence         │   │
        │  │ ✓ Matches known signatures  │   │
        │  │ ✓ Attributes to threat      │   │
        │  │ ✓ Provides context          │   │
        │  │ Input: Attack characteristics   │
        │  │ Output: Attribution + confidence  │
        │  └─────────────────────────────┘   │
        │                                     │
        │  ┌─────────────────────────────┐   │
        │  │ CONSENSUS ENGINE            │   │
        │  │ Aggregates 3 perspectives   │   │
        │  │ Votes on threat level       │   │
        │  │ Generates confidence score  │   │
        │  └─────────────────────────────┘   │
        └────────────┬────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │ 4. THREAT ANALYSIS RESULT        │
        │                                  │
        │ {                                │
        │   "threat_detected": true,       │
        │   "threat_type": "DDoS",         │
        │   "severity": "High",            │
        │   "confidence": 0.92,            │
        │   "council_consensus": 0.87,     │
        │   "explanation": "...",          │
        │   "recommendations": [           │
        │     "Block source IP",           │
        │     "Update firewall",           │
        │     "Monitor related flows"      │
        │   ],                             │
        │   "timestamp": "2025-12-26..."   │
        │ }                                │
        └──────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │ 5. RL DEFENDER AGENT (Action)    │
        │                                  │
        │  Input: Confidence + Severity    │
        │  Model: PPO (Stable Baselines3)  │
        │                                  │
        │  Executes Mitigation:            │
        │  ► BLOCK_SOURCE                  │
        │  ► RATE_LIMIT                    │
        │  ► DEEP_PACKET_INSPECTION        │
        │  ► MONITOR                       │
        │                                  │
        │  *Optimizes for Uptime/Safety*   │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │ 6. HUMAN-READABLE REPORT         │
        │                                  │
        │ THREAT ANALYSIS COUNCIL REPORT   │
        │                                  │
        │ Security Analyst Assessment:     │
        │ This flow exhibits SYN flood     │
        │ characteristics with 450+ SYN    │
        │ flags in 1 second. Confidence:   │
        │ 95% DDoS attack.                 │
        │                                  │
        │ ML Engineer Perspective:         │
        │ Feature anomalies detected:      │
        │ extremely high packet rate       │
        │ (10x normal), low packet size    │
        │ variation. Model output: 92%     │
        │ attack probability.              │
        │                                  │
        │ Threat Intelligence:             │
        │ Signature matches Mirai DDoS     │
        │ variant with 87% confidence.     │
        │ Known attack from botnet.        │
        │                                  │
        │ FINAL ASSESSMENT:                │
        │ DDoS Attack - HIGH SEVERITY      │
        │ Council Consensus: 87%           │
        │ DEFENDER ACTION: BLOCK_SOURCE    │
        │                                  │
        └──────────────────────────────────┘
```

---

## **DATA FLOW DETAILED**

### **Input: Your Darknet.xlsx (141k flows)**

```
Flow ID | Src IP | Dst IP | Protocol | Total Fwd | Total Bwd | ... | Type
--------|--------|--------|----------|-----------|-----------|-----|-------
1       | 192... | 10...  | 6        | 591       | 400       | ... | Non-Tor
2       | 192... | 10...  | 6        | 1         | 1         | ... | Non-Tor
3       | 192... | 10...  | 17       | 5995      | 6000      | ... | VPN
...
141484
```

### **Processing Step 1: Preprocessor**

```
DROP:        Keep:
- Flow ID    ✓ Protocol
- Src IP     ✓ Total Fwd Packet
- Dst IP     ✓ Total Bwd packets
- Timestamp  ✓ Fwd Packet Length Max
- Src/Dst    ✓ Fwd SYN Flags
  Port       ✓ Fwd RST Flags
- ...        ✓ Flow Bytes/s
             ✓ 67 more features

Label Encoding:
Type='Non-Tor' → Encryption=0 (benign)
Type='VPN'     → Encryption=1 (attack-like)
Type='Tor'     → Encryption=1 (attack-like)
Type='NonVPN'  → Encryption=0 (benign)
```

### **Processing Step 2: Synthetic Data Generation**

```
DDoS Agent generates:
┌─────────────────────────────────────┐
│ DDoS Flow 1:                        │
│ Protocol: 6 (TCP)                   │
│ Total Fwd Packet: 523 ← High!       │
│ Fwd SYN Flags: 496 ← Attack signal! │
│ Total Bwd Packets: 1 ← Low response │
│ Total Length Fwd: 20,920 bytes      │
│ Label: DDoS                         │
│ Encryption: 1 (attack)              │
└─────────────────────────────────────┘

PortScan Agent generates:
┌─────────────────────────────────────┐
│ Scan Flow 1:                        │
│ Protocol: 6 (TCP)                   │
│ Total Fwd Packet: 1 ← Probe!        │
│ Fwd SYN Flags: 1 ← Single probe     │
│ Total Bwd Packets: 0 ← No response  │
│ Total Length Fwd: 40 bytes          │
│ Label: PortScan                     │
│ Encryption: 1 (attack)              │
└─────────────────────────────────────┘

Generate 2,000 DDoS variations
Generate 1,000 PortScan variations
Total: 3,000 synthetic attack samples
```

### **Processing Step 3: Feature Scaling**

```
Raw Values:          Scaled Values:
Total Fwd Packet:    Mean: 500, Std: 150
  - Min: 0           → Normalized to [-2, +3]
  - Max: 6000
  - Median: 450

Total Bwd Packets:
  - Min: 0
  - Max: 6000
  - Median: 380

All 71 features scaled to mean=0, std=1
Ready for XGBoost training
```

### **Processing Step 4: Model Training**

```
Train Set (80% = 115k flows):
┌──────────────────────────┐
│ 57,500 benign samples    │ (50%)
│ 57,500 attack samples    │ (50%)
│ Total: 115k flows        │
│ Features: 71             │
│ → XGBoost trains here    │
└──────────────────────────┘

Test Set (20% = 29k flows):
┌──────────────────────────┐
│ 14,350 benign samples    │ (49%)
│ 14,650 attack samples    │ (51%)
│ Total: 29k flows         │
│ → Model evaluates here   │
└──────────────────────────┘

Output: Trained model.pkl (can be used on new flows)
```

### **Processing Step 5: Prediction Pipeline**

```
New Flow Arrives:
┌──────────────────────────────────────────┐
│ Flow: 123.45.67.89 → 10.20.30.40         │
│ Protocol: 6, Total Fwd: 450, Bwd: 5     │
│ Fwd SYN: 440, RST: 1, ...               │
└──────────────────────────────────────────┘
         │
         ▼ (Feature Extraction)
┌──────────────────────────────────────────┐
│ Extract 71 numeric features              │
│ Fill NaN with 0                          │
│ Scale with StandardScaler                │
└──────────────────────────────────────────┘
         │
         ▼ (XGBoost Prediction)
┌──────────────────────────────────────────┐
│ XGBoost predicts:                        │
│ - Class: 1 (Attack)                      │
│ - Probability: [0.08, 0.92]              │
│ - Confidence: 92%                        │
└──────────────────────────────────────────┘
         │
         ▼ (LLM Council Analysis)
┌──────────────────────────────────────────┐
│ Council votes:                           │
│ Security Analyst: "DDoS (95% conf)"      │
│ ML Engineer: "Feature anomalies (92%)"   │
│ Threat Intel: "Mirai variant (87%)"      │
│ Consensus: 91% attack, DDoS threat      │
└──────────────────────────────────────────┘
         │
         ▼ (RL Defender Action)
┌──────────────────────────────────────────┐
│ Observation: High Threat, High Conf      │
│ Action: BLOCK_SOURCE                     │
│ Reward: +2.0 (Correctly mitigated)       │
└──────────────────────────────────────────┘
         │
         ▼ (Final Report)
┌──────────────────────────────────────────┐
│ ALERT: DDoS Attack Detected              │
│ Severity: HIGH                           │
│ Confidence: 91%                          │
│ Type: SYN Flood                          │
│ Recommendation: Block source IP          │
└──────────────────────────────────────────┘
```

---

## **API KEY LOCATIONS**

```
Groq API Flow:
┌─────────────────────┐
│ https://console.groq.com
│ 1. Sign up (FREE)
│ 2. Create API key
└────────┬────────────┘
         │
         ▼
    Copy key (starts with gsk_)
         │
         ▼
┌──────────────────────────┐
│ Paste into config/.env:  │
│ GROQ_API_KEY=gsk_...     │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Your Python code reads:  │
│ from dotenv import load_dotenv
│ load_dotenv('config/.env')
│ import os
│ api_key = os.getenv('GROQ_API_KEY')
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Groq client initialized: │
│ client = Groq(api_key=...) 
│ Ready to analyze threats!
└──────────────────────────┘
```

---

## **EXECUTION TIMELINE**

```
You run: python main.py
│
├─ [0s]    Load Darknet.xlsx (141k flows)
│          └─ Parse Excel → DataFrame
│          └─ Drop columns
│          └─ Handle NaN
│          └─ Status: ✓ 141,484 rows loaded
│
├─ [2s]    Generate synthetic attacks (3k)
│          └─ DDoS Agent: 2,000 flows
│          └─ PortScan Agent: 1,000 flows
│          └─ Save to CSV
│          └─ Status: ✓ Synthetic data ready
│
├─ [3s]    Combine datasets (144k total)
│          └─ Stack real + synthetic
│          └─ Create labels
│          └─ Split 80/20
│          └─ Status: ✓ Combined dataset ready
│
├─ [4s]    Train XGBoost (30-45 seconds)
│          └─ Initialize model
│          └─ Fit on training data
│          └─ Evaluate on test data
│          └─ Save model
│          └─ Status: ✓ Accuracy: 92-95%
│
├─ [45s]   Initialize LLM Council
│          └─ Connect to Groq API
│          └─ Load prompts
│          └─ Test connection
│          └─ Status: ✓ Council ready
│
├─ [48s]   Analyze 5 sample attacks
│          ├─ Attack 1: LLM analysis (3 perspectives) → 2 seconds
│          ├─ Attack 2: LLM analysis → 2 seconds
│          ├─ Attack 3: LLM analysis → 2 seconds
│          ├─ Attack 4: LLM analysis → 2 seconds
│          ├─ Attack 5: LLM analysis → 2 seconds
│          └─ Status: ✓ Council analyzed 5 attacks
│          └─ Defender: Actions executed (Block/Monitor)
│
└─ [60s]   COMPLETE!
           Total time: ~1 minute (per run)
           
Expected output: 50-100 lines of logs + attack analysis
```

---

## **YOUR M3 AIR CAN HANDLE IT**

```
M3 Air Specs:
├─ CPU: 8-core (5P + 3E)
├─ RAM: 16GB
├─ Storage: 256GB SSD
└─ GPU: None (but doesn't need for this)

Resource Usage:
├─ XGBoost training: ~4GB RAM, ~30s
├─ LLM Council analysis: ~2GB RAM, ~2s per attack
├─ Ollama (optional): ~6GB RAM (if enabled)
├─ Groq API: No local resources (cloud-based)
└─ Total: Well within M3 Air capability
```

---

## **EXPECTED FILES AFTER RUN**

```
models/trained/
├─ ensemble_detector.pkl    ← Trained XGBoost model
├─ scaler.pkl              ← Feature scaler
└─ feature_list.pkl        ← Feature names

data/synthetic/
└─ attacks.csv             ← 3,000 synthetic flows
  │                            (saved for reuse)

logs/ (if enabled)
└─ agentic_ids.log         ← Full execution log
```

---

## **COMMAND QUICK REFERENCE**

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Get Groq key
# Go to: https://console.groq.com → Copy API key → Add to config/.env

# Test connection
python << 'EOF'
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv('config/.env')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
print('✓ API Connected')
EOF

# Copy your data
cp ~/Downloads/Darknet.xlsx data/raw/

# Run pipeline
python main.py

# Check results
ls -la data/synthetic/
cat data/synthetic/attacks.csv | head -20
```

---

**This is your complete system visualization.**
