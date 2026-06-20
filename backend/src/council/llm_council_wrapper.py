"""
LLM Council - Hybrid multi-provider support
Each council member can use different provider/model
"""

import logging
import os
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
import requests
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


@dataclass
class ThreatAnalysis:
    """Result of threat analysis"""
    threat_detected: bool
    confidence: float
    threat_type: str
    severity: str
    explanation: str
    recommendations: List[str]
    council_consensus: float
    timestamp: str
    
    def to_dict(self):
        return {
            'threat_detected': self.threat_detected,
            'confidence': self.confidence,
            'threat_type': self.threat_type,
            'severity': self.severity,
            'explanation': self.explanation,
            'recommendations': self.recommendations,
            'council_consensus': self.council_consensus,
            'timestamp': self.timestamp
        }


class LLMClient:
    """Universal LLM client supporting Groq, Ollama, LM Studio, and HuggingFace"""
    
    def __init__(self):
        self.groq_client = None
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.lmstudio_base_url = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
        
        # Initialize providers
        self._init_groq()
        self._init_ollama()
        self._init_lmstudio()
        self._init_huggingface()
    
    def _init_groq(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                self.groq_client = Groq(api_key=api_key)
                logger.info("✓ Groq client initialized")
            else:
                logger.warning("GROQ_API_KEY not found in environment")
        except ImportError:
            logger.warning("Groq not installed. Run: pip install groq")
    
    def _init_huggingface(self):
        """Initialize Hugging Face client"""
        self.hf_token = os.getenv('HF_API_KEY')
        if self.hf_token:
            self.hf_client = InferenceClient(token=self.hf_token)
            logger.info("✓ Hugging Face client initialized")
        else:
            self.hf_client = None
            logger.warning("HF_API_KEY not found in environment (required for HF models)")
    
    def _init_ollama(self):
        """Initialize Ollama connection"""
        try:
            response = requests.get(f'{self.ollama_base_url}/api/tags', timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"✓ Ollama connected - {len(models)} models available")
            else:
                logger.warning("Ollama not responding properly")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")

    def _init_lmstudio(self):
        """Check LM Studio connection (OpenAI-compatible at localhost:1234)"""
        try:
            response = requests.get(f'{self.lmstudio_base_url}/models', timeout=2)
            if response.status_code == 200:
                models = response.json().get('data', [])
                loaded = [m['id'] for m in models]
                logger.info(f"✓ LM Studio connected - loaded: {loaded}")
            else:
                logger.warning("LM Studio not responding")
        except Exception as e:
            logger.warning(f"LM Studio not available at {self.lmstudio_base_url}: {e}")
    
    def generate(self, prompt: str, provider: str, model: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        """
        Universal generate method.
        provider: 'groq' | 'ollama' | 'lmstudio' | 'hf'
        """
        p = provider.lower()
        if p == 'groq':
            return self._generate_groq(prompt, model, max_tokens, temperature)
        elif p == 'ollama':
            return self._generate_ollama(prompt, model, max_tokens, temperature)
        elif p == 'lmstudio':
            return self._generate_lmstudio(prompt, model, max_tokens, temperature)
        elif p == 'hf':
            return self._generate_huggingface(prompt, model, max_tokens, temperature)
        else:
            logger.error(f"Unknown provider: {provider}")
            return f"Error: Invalid LLM provider '{provider}'"
    
    def _generate_groq(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Generate using Groq"""
        if not self.groq_client:
            return "Error: Groq client not initialized"
        
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return f"Error: {str(e)}"
    
    def _generate_ollama(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Generate using Ollama."""
        try:
            # Disable thinking mode for qwen3 — /nothink prevents the model from
            # entering its extended reasoning phase which can take minutes per call.
            # We need fast structured JSON, not a lengthy chain-of-thought.
            adjusted_prompt = prompt + "\n/nothink" if "qwen3" in model.lower() else prompt

            response = requests.post(
                f'{self.ollama_base_url}/api/generate',
                json={
                    'model': model,
                    'prompt': adjusted_prompt,
                    'stream': False,
                    'options': {
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    },
                },
                timeout=30,  # was 180s — fail fast so the semaphore slot frees up
            )
            if response.status_code == 200:
                return response.json()['response']
            else:
                logger.error(f"Ollama error: {response.text}")
                return f"Error: Ollama returned {response.status_code}"
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: {str(e)}"

    def _generate_lmstudio(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """
        Generate using LM Studio (OpenAI-compatible REST at localhost:1234/v1).
        Does a fast connectivity check first (1s) so we fail in ~1s instead of
        waiting 120s when LM Studio isn't running.
        """
        # Fast connectivity probe — avoids burning 120s on a dead port
        try:
            requests.get(f'{self.lmstudio_base_url}/models', timeout=1)
        except Exception:
            return "Error: LM Studio not reachable (not running or wrong port)"

        try:
            payload = {
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stream': False,
            }
            response = requests.post(
                f'{self.lmstudio_base_url}/chat/completions',
                json=payload,
                timeout=8,   # was 120s — LM Studio responds fast or not at all
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                logger.error(f"LM Studio error {response.status_code}: {response.text[:200]}")
                return f"Error: LM Studio returned {response.status_code}"
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            return f"Error: {str(e)}"

    def _generate_huggingface(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Generate using Hugging Face InferenceClient"""
        if not self.hf_client:
             return "Error: HF_API_KEY not found or client not initialized"
        
        try:
            # Use chat completion API for compatibility
            messages = [{"role": "user", "content": prompt}]
            
            completion = self.hf_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"HF generation failed: {e}")
            return f"Error: {str(e)}"

    def invoke_model(self, model_id: str, prompt: str) -> str:
        """
        Public API to invoke a model by ID (e.g., 'groq:llama-3.1-8b-instant')
        """
        if ':' not in model_id:
            logger.error(f"Invalid model_id format: {model_id}. Expected 'provider:model'")
            return "Error: Invalid model ID"
            
        provider, model_name = model_id.split(':', 1)
        return self.generate(prompt, provider, model_name, max_tokens=1024, temperature=0.7)


class ThreatAnalysisCouncil:
    """
    Hybrid Multi-LLM council for threat analysis
    Each member can use different provider + model
    """
    
    def __init__(self, provider: str = 'groq'):
        self.llm_client = LLMClient()
        self.provider = provider.lower()

        if self.provider == 'hybrid':
            # ── Hybrid mode: each agent uses its own provider ──────────────
            # Format: "provider:model"  e.g. "lmstudio:mistral-7b"
            # Agent 1 (Security Analyst)  → LM Studio  (local)
            # Agent 2 (ML Engineer)       → Ollama     (local)
            # Agent 3 (Threat Intel)      → Groq       (cloud, small ctx)
            self.analyst_config = self._parse_model_config(
                os.getenv('ANALYST_MODEL', 'lmstudio:loaded'),
                default_provider='lmstudio'
            )
            self.engineer_config = self._parse_model_config(
                os.getenv('ENGINEER_MODEL', 'ollama:llama3.2:3b'),
                default_provider='ollama'
            )
            self.intel_config = self._parse_model_config(
                os.getenv('INTEL_MODEL', 'groq:llama-3.1-8b-instant'),
                default_provider='groq'
            )
        elif self.provider == 'hf':
            self.analyst_config = self._parse_model_config(
                os.getenv('HF_ANALYST_MODEL', os.getenv('ANALYST_MODEL', 'meta-llama/Meta-Llama-3-8B-Instruct')),
                default_provider='hf'
            )
            self.engineer_config = self._parse_model_config(
                os.getenv('HF_ENGINEER_MODEL', os.getenv('ENGINEER_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')),
                default_provider='hf'
            )
            self.intel_config = self._parse_model_config(
                os.getenv('HF_INTEL_MODEL', os.getenv('INTEL_MODEL', 'google/gemma-7b-it')),
                default_provider='hf'
            )
        elif self.provider == 'ollama':
            self.analyst_config = self._parse_model_config(
                os.getenv('OLLAMA_ANALYST_MODEL', os.getenv('ANALYST_MODEL', 'llama3.2:3b')),
                default_provider='ollama'
            )
            self.engineer_config = self._parse_model_config(
                os.getenv('OLLAMA_ENGINEER_MODEL', os.getenv('ENGINEER_MODEL', 'llama3.2:3b')),
                default_provider='ollama'
            )
            self.intel_config = self._parse_model_config(
                os.getenv('OLLAMA_INTEL_MODEL', os.getenv('INTEL_MODEL', 'qwen2.5:3b')),
                default_provider='ollama'
            )
        else:
            # Default: all Groq
            self.analyst_config = self._parse_model_config(
                os.getenv('GROQ_ANALYST_MODEL', os.getenv('ANALYST_MODEL', 'llama-3.1-8b-instant')),
                default_provider='groq'
            )
            self.engineer_config = self._parse_model_config(
                os.getenv('GROQ_ENGINEER_MODEL', os.getenv('ENGINEER_MODEL', 'llama-3.1-8b-instant')),
                default_provider='groq'
            )
            self.intel_config = self._parse_model_config(
                os.getenv('GROQ_INTEL_MODEL', os.getenv('INTEL_MODEL', 'llama-3.1-8b-instant')),
                default_provider='groq'
            )
        
        logger.info("=" * 60)
        logger.info("THREAT ANALYSIS COUNCIL INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Security Analyst:  {self.analyst_config['provider']} - {self.analyst_config['model']}")
        logger.info(f"ML Engineer:       {self.engineer_config['provider']} - {self.engineer_config['model']}")
        logger.info(f"Threat Intel:      {self.intel_config['provider']} - {self.intel_config['model']}")
        logger.info("=" * 60)
        
        # Load prompts from config file
        self.prompts = self._load_prompts('config/prompts/threat_analysis.txt')

    def _load_prompts(self, filepath: str) -> Dict[str, str]:
        """Load and parse prompt templates from file"""
        prompts = {}
        current_section = None
        current_content = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # Save previous section
                        if current_section:
                            prompts[current_section] = '\n'.join(current_content).strip()
                        
                        # Start new section
                        current_section = line[1:-1] # Remove []
                        current_content = []
                    else:
                        if current_section:
                            current_content.append(line)
                            
            # Save last section
            if current_section:
                prompts[current_section] = '\n'.join(current_content).strip()
                
            logger.info(f"✓ Loaded {len(prompts)} prompts from {filepath}")
            return prompts
            
        except Exception as e:
            logger.error(f"Failed to load prompts from {filepath}: {e}")
            # Fallback to hardcoded defaults (simplified) if file fails
            return {}

    def _parse_model_config(self, config_string: str, default_provider: str = 'groq') -> Dict[str, str]:
        """
        Parse model config string in format: 'provider:model'
        If provider is missing, use default_provider.
        """
        if ':' in config_string:
            parts = config_string.split(':', 1)
            return {
                'provider': parts[0],
                'model': parts[1]
            }
        else:
            return {
                'provider': default_provider,
                'model': config_string
            }
    
    def analyze_threat(self, flow_data: Dict, detector_prediction: Dict) -> dict:
        """
        Run the three-agent council and return a plain dict (not ThreatAnalysis)
        so the caller gets all parsed fields, including agent votes and indicators.
        """
        logger.info("\n" + "=" * 60)
        logger.info("THREAT ANALYSIS COUNCIL CONVENES")
        logger.info("=" * 60)

        # ── 3 LLM calls ───────────────────────────────────────────────
        analyst_raw  = self._security_analyst_perspective(flow_data, detector_prediction)
        engineer_raw = self._ml_engineer_perspective(flow_data, detector_prediction)
        intel_raw    = self._threat_intel_perspective(flow_data, detector_prediction)

        # ── Parse structured JSON from each response ────────────────────
        analyst  = self._parse_analyst(analyst_raw.get('analysis', ''))
        engineer = self._parse_engineer(engineer_raw.get('analysis', ''))
        intel    = self._parse_intel(intel_raw.get('analysis', ''))

        # ── Real vote-based consensus ───────────────────────────────
        consensus_score, recommendation = self._reach_consensus(
            analyst, engineer, intel
        )

        result = {
            # Consensus outputs (used by defender + executor)
            "threat_type":          recommendation["threat_type"],
            "severity":             recommendation["severity"],
            "confidence":           recommendation["confidence"],
            "recommended_action":   recommendation["actions"][0],   # voted action
            "recommendations":      recommendation["actions"],
            "council_consensus":    consensus_score,
            "false_positive_risk":  recommendation["false_positive_risk"],
            "model_decision_valid": recommendation["model_decision_valid"],
            "signature_match":      recommendation["signature_match"],
            "threat_actor_type":    recommendation["threat_actor_type"],
            "all_indicators":       recommendation["all_indicators"],
            "agent_votes":          recommendation["agent_votes"],
            # Full human-readable report
            "explanation":          self._generate_explanation(analyst, engineer, intel),
            "timestamp":            __import__('datetime').datetime.now().isoformat(),
            # Per-agent parsed data (useful for UI / debugging)
            "analyst":  analyst,
            "engineer": engineer,
            "intel":    intel,
        }

        logger.info("\u2713 Council complete")
        return result
    
    # ── Key flow features for LLM prompts (keeps prompts small to avoid TPM limits) ──
    _KEY_FEATURES = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Flow Packets/s", "Flow Bytes/s",
        "Fwd Packets/s", "Bwd Packets/s",
        "Flow IAT Mean", "Flow IAT Std",
        "SYN Flag Count", "ACK Flag Count", "URG Flag Count",
        "Down/Up Ratio", "Average Packet Size",
    ]

    @classmethod
    def _slim_flow(cls, flow_data: Dict, max_features: int = 12) -> str:
        """Return only the most diagnostic features as a compact string."""
        items = [
            (k, v) for k, v in flow_data.items()
            if any(k.startswith(kf) or kf in k for kf in cls._KEY_FEATURES)
        ]
        if not items:  # fallback: first N entries
            items = list(flow_data.items())[:max_features]
        # Round floats for brevity
        parts = []
        for k, v in items[:max_features]:
            try:
                parts.append(f"{k}: {float(v):.2f}")
            except (TypeError, ValueError):
                parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def _security_analyst_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """Security analyst: independent attack classification."""
        logger.info(
            f"\n[Security Analyst - "
            f"{self.analyst_config['provider'].upper()}:{self.analyst_config['model']}]"
        )
        flow_str = self._slim_flow(flow_data)  # compact — avoids TPM overflow
        pred_str = (
            f"- Classification: {'ATTACK' if prediction.get('prediction')==1 else 'BENIGN'}\n"
            f"- Confidence: {prediction.get('confidence', 0.5):.2%}\n"
            f"- Type: {prediction.get('attack_type', 'Unknown')}"
        )
        template = self.prompts.get('SECURITY_ANALYST_PROMPT', '')
        prompt = template.format(flow_data=flow_str, prediction=pred_str) if template else (
            f'You are a cybersecurity analyst. Return ONLY JSON with fields '
            f'attack_type, confidence(0-100), severity, indicators(list), '
            f'recommended_action, reasoning. Flow: {flow_str}. Prediction: {pred_str}'
        )
        try:
            analysis = self.llm_client.generate(
                prompt=prompt,
                provider=self.analyst_config['provider'],
                model=self.analyst_config['model'],
                max_tokens=350,   # JSON response only needs ~150-250 tokens
                temperature=0.2,
            )
            logger.info(f"Analyst raw: {analysis[:200]}...")
            return {'role': 'security_analyst', 'analysis': analysis}
        except Exception as e:
            logger.error(f"Analyst call failed: {e}")
            return {'role': 'security_analyst', 'analysis': ''}
    
    def _ml_engineer_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """ML Engineer: audits model decision validity + false positive risk."""
        logger.info(
            f"\n[ML Engineer - "
            f"{self.engineer_config['provider'].upper()}:{self.engineer_config['model']}]"
        )
        features_str  = self._slim_flow(flow_data)  # compact — avoids TPM overflow
        model_out_str = (
            f"{prediction.get('confidence', 0.5):.2%} confidence "
            f"for {prediction.get('attack_type', 'Unknown')}"
        )
        template = self.prompts.get('ML_ENGINEER_PROMPT', '')
        prompt = template.format(features=features_str, model_output=model_out_str) if template else (
            f'You are an ML engineer. Return ONLY JSON with fields '
            f'model_decision_valid(bool), false_positive_risk(Low/Medium/High), '
            f'anomalous_features(list), normal_features(list), '
            f'recommended_action, reasoning. Features: {features_str}. '
            f'Model: {model_out_str}'
        )
        try:
            analysis = self.llm_client.generate(
                prompt=prompt,
                provider=self.engineer_config['provider'],
                model=self.engineer_config['model'],
                max_tokens=350,   # JSON response only needs ~150-250 tokens
                temperature=0.2,
            )
            logger.info(f"Engineer raw: {analysis[:200]}...")
            return {'role': 'ml_engineer', 'analysis': analysis}
        except Exception as e:
            logger.error(f"Engineer call failed: {e}")
            return {'role': 'ml_engineer', 'analysis': ''}

    def _threat_intel_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """Threat Intel: signature attribution + IOC matching."""
        logger.info(
            f"\n[Threat Intel - "
            f"{self.intel_config['provider'].upper()}:{self.intel_config['model']}]"
        )
        char_str     = self._slim_flow(flow_data)  # compact — avoids TPM overflow
        patterns_str = (
            "Mirai (high-volume UDP/TCP, small fixed-size packets, many IPs), "
            "Slowloris (low packet rate, many partial HTTP connections), "
            "UDP Flood (large UDP bursts, single destination port), "
            "TCP SYN Flood (high SYN count, low ACK ratio), "
            "Masscan/Nmap (sequential IPs, multiple destination ports, short flows), "
            "Generic PortScan (many unique dst ports, low bytes-per-flow)"
        )
        template = self.prompts.get('THREAT_INTEL_PROMPT', '')
        prompt = template.format(characteristics=char_str, known_patterns=patterns_str) if template else (
            f'You are a threat intel analyst. Return ONLY JSON with fields '
            f'signature_match, attribution_confidence(0-100), threat_actor_type, '
            f'ioc_pattern, recommended_action, reasoning. '
            f'Flow: {char_str}. Patterns: {patterns_str}'
        )
        try:
            analysis = self.llm_client.generate(
                prompt=prompt,
                provider=self.intel_config['provider'],
                model=self.intel_config['model'],
                max_tokens=350,   # JSON response only needs ~150-250 tokens
                temperature=0.2,
            )
            logger.info(f"Intel raw: {analysis[:200]}...")
            return {'role': 'threat_intel', 'analysis': analysis}
        except Exception as e:
            logger.error(f"Intel call failed: {e}")
            return {'role': 'threat_intel', 'analysis': ''}
    
    def _reach_consensus(self, analyst, engineer, intel) -> tuple:
        """Aggregate council perspectives into consensus"""
        logger.info("\n[Council Consensus]")
        
        # Simple consensus based on threat type matches
        threat_types = [
            analyst.get('threat_type', 'Unknown'),
            'DDoS' if 'DDoS' in engineer.get('analysis', '') else 'Unknown',
            'DDoS' if 'DDoS' in intel.get('analysis', '') else 'Unknown'
        ]
        
        # Count votes
        from collections import Counter
        votes = Counter(threat_types)
        final_threat = votes.most_common(1)[0][0]
        
        # Consensus score: % of council in agreement
        consensus_score = votes[final_threat] / 3.0
        
        recommendation = {
            'threat_type': final_threat,
            'severity': 'High' if consensus_score > 0.66 else 'Medium' if consensus_score > 0.33 else 'Low',
            'actions': [
                'Log flow for detailed forensic analysis',
                f'Block source if {final_threat} pattern persists',
                'Update IDS signatures based on findings',
                'Monitor related network flows'
            ]
        }
        
        logger.info(f"Consensus Score: {consensus_score:.1%}")
        logger.info(f"Final Threat Type: {recommendation['threat_type']}")
        logger.info(f"Severity: {recommendation['severity']}")
        
        return consensus_score, recommendation
    
    def _generate_explanation(self, analyst, engineer, intel) -> str:
        """Generate human-readable explanation"""
        return f"""
THREAT ANALYSIS COUNCIL REPORT
{'=' * 60}

SECURITY ANALYST ASSESSMENT ({self.analyst_config['provider'].upper()}):
{analyst.get('analysis', 'No analysis available')[:500]}

ML ENGINEER PERSPECTIVE ({self.engineer_config['provider'].upper()}):
{engineer.get('analysis', 'No analysis available')[:500]}

THREAT INTELLIGENCE ({self.intel_config['provider'].upper()}):
{intel.get('analysis', 'No analysis available')[:500]}
{'=' * 60}
"""
    
    # ─────────────────────────────────────────────────────────────
    #  Parsers — each reads the JSON the LLM was asked to return
    # ─────────────────────────────────────────────────────────────

    def _parse_json_response(self, text: str) -> dict:
        """
        Robustly extract JSON from an LLM response.
        LLMs often wrap JSON in markdown code fences or add prose before/after.
        We strip all that and try multiple extraction strategies.
        """
        import re, json

        if not text or text.startswith("Error:"):
            return {}

        # Strategy 1: find first {...} block (handles markdown fences)
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 2: strip common markdown prefixes and try again
        cleaned = re.sub(r'```(?:json)?', '', text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return {}

    def _parse_analyst(self, text: str) -> dict:
        """Parse Security Analyst JSON response into normalised fields."""
        raw = self._parse_json_response(text)
        return {
            "role":               "security_analyst",
            "raw_text":          text,
            "attack_type":       str(raw.get("attack_type", "Unknown")),
            "confidence":        float(raw.get("confidence", 50)) / 100.0,
            "severity":          str(raw.get("severity", "Medium")),
            "indicators":        list(raw.get("indicators", [])),
            "recommended_action": str(raw.get("recommended_action", "MONITOR")),
            "reasoning":         str(raw.get("reasoning", "")),
            "parse_ok":          bool(raw),
        }

    def _parse_engineer(self, text: str) -> dict:
        """Parse ML Engineer JSON response into normalised fields."""
        raw = self._parse_json_response(text)
        return {
            "role":               "ml_engineer",
            "raw_text":          text,
            "model_decision_valid": bool(raw.get("model_decision_valid", True)),
            "false_positive_risk": str(raw.get("false_positive_risk", "Low")),
            "anomalous_features": list(raw.get("anomalous_features", [])),
            "normal_features":   list(raw.get("normal_features", [])),
            "recommended_action": str(raw.get("recommended_action", "MONITOR")),
            "reasoning":         str(raw.get("reasoning", "")),
            "parse_ok":          bool(raw),
        }

    def _parse_intel(self, text: str) -> dict:
        """Parse Threat Intel JSON response into normalised fields."""
        raw = self._parse_json_response(text)
        return {
            "role":                "threat_intel",
            "raw_text":           text,
            "signature_match":    str(raw.get("signature_match", "Unknown")),
            "attribution_confidence": float(raw.get("attribution_confidence", 50)) / 100.0,
            "threat_actor_type": str(raw.get("threat_actor_type", "unknown")),
            "ioc_pattern":        str(raw.get("ioc_pattern", "")),
            "recommended_action": str(raw.get("recommended_action", "MONITOR")),
            "reasoning":         str(raw.get("reasoning", "")),
            "parse_ok":          bool(raw),
        }

    # ─────────────────────────────────────────────────────────────
    #  Consensus  — real aggregation across agent outputs
    # ─────────────────────────────────────────────────────────────

    def _reach_consensus(self, analyst: dict, engineer: dict, intel: dict) -> tuple:
        """
        Aggregate the three parsed agent outputs into a final decision.

        Consensus rules:
          1. Recommended action  → majority vote across all three agents.
             Tie-break: analyst > intel > engineer (domain hierarchy).
          2. Attack type         → analyst is primary; intel signature match
             is used as confirmation / override if confidence > 80%.
          3. Severity            → escalate if engineer flags false_positive_risk=High;
             de-escalate if model_decision_valid=False.
          4. Confidence          → weighted average:
               analyst 50% + intel.attribution_confidence 30% + engineer 20%
             Engineer weight is lower because they assess model validity,
             not attack confidence directly.
          5. False positive gate → if engineer says model_decision_valid=False
             AND false_positive_risk=High, override action to MONITOR regardless
             of other votes.
        """
        from collections import Counter
        import logging
        log = logging.getLogger(__name__)

        # ── 1. Action vote — exclude agents that failed to parse ─────────────
        # A timed-out or error'd agent returns parse_ok=False with all defaults.
        # Counting its default MONITOR as a real vote corrupts the consensus.
        # Only include agents whose JSON actually parsed successfully.
        votes = [
            a.get("recommended_action")
            for a in (analyst, engineer, intel)
            if a.get("parse_ok", False)
        ]

        if not votes:
            # All 3 failed (e.g. all providers down) — fall back to MONITOR
            final_action = "MONITOR"
            log.warning("[Council] All agents failed — defaulting to MONITOR")
        elif len(votes) == 1:
            # Only one agent responded — trust it directly
            final_action = votes[0]
        else:
            vote_counts = Counter(votes)
            if vote_counts.most_common(1)[0][1] > 1:
                final_action = vote_counts.most_common(1)[0][0]
            else:
                # Tie among successful agents — analyst > intel > engineer
                for preferred in (analyst, intel, engineer):
                    if preferred.get("parse_ok"):
                        final_action = preferred.get("recommended_action", "MONITOR")
                        break
                else:
                    final_action = "MONITOR"

        # Rebuild vote_counts for logging (include only successful agents)
        vote_counts = Counter(
            a.get("recommended_action")
            for a in (analyst, engineer, intel)
            if a.get("parse_ok", False)
        )
        log.info(
            f"[Council] Action votes (parse_ok agents only): "
            f"{dict(vote_counts)} → {final_action} "
            f"[{sum(a.get('parse_ok', False) for a in (analyst, engineer, intel))}/3 agents responded]"
        )

        # ── 2. False positive gate ────────────────────────────────────────────
        fp_risk = engineer.get("false_positive_risk", "Low")
        model_valid = engineer.get("model_decision_valid", True)
        if not model_valid and fp_risk == "High":
            final_action = "MONITOR"
            log.warning("[Council] FP gate triggered — overriding action to MONITOR")

        # ── 3. Attack type ────────────────────────────────────────────────────
        final_attack_type = analyst.get("attack_type", "Unknown")
        intel_sig = intel.get("signature_match", "Unknown")
        intel_conf = intel.get("attribution_confidence", 0.5)
        if intel_sig != "Unknown" and intel_conf > 0.8:
            # High-confidence signature match overrides analyst classification
            final_attack_type = intel_sig
            log.info(f"[Council] Intel override: {intel_sig} @ {intel_conf:.0%}")

        # ── 4. Severity ───────────────────────────────────────────────────────
        analyst_sev = analyst.get("severity", "Medium")
        if not model_valid:
            # Engineer suspects FP — downgrade severity by one level
            sev_map = {"Critical": "High", "High": "Medium", "Medium": "Low", "Low": "Low"}
            final_severity = sev_map.get(analyst_sev, "Medium")
        elif fp_risk == "High":
            sev_map = {"Critical": "High", "High": "Medium", "Medium": "Low", "Low": "Low"}
            final_severity = sev_map.get(analyst_sev, "Medium")
        else:
            final_severity = analyst_sev

        # ── 5. Confidence ─────────────────────────────────────────────────────
        analyst_conf = analyst.get("confidence", 0.5)
        intel_attr   = intel.get("attribution_confidence", 0.5)
        # Engineer contributes 20% as a validity weight (not an attack confidence)
        engineer_weight = 0.8 if model_valid else 0.3
        final_confidence = (
            analyst_conf * 0.5
            + intel_attr * 0.3
            + engineer_weight * 0.2
        )

        # ── Build consensus record ────────────────────────────────────────────
        all_indicators = (
            analyst.get("indicators", []) +
            engineer.get("anomalous_features", []) +
            ([intel.get("ioc_pattern")] if intel.get("ioc_pattern") else [])
        )
        # Deduplicate while preserving order
        seen, unique_indicators = set(), []
        for ind in all_indicators:
            if ind and ind not in seen:
                seen.add(ind)
                unique_indicators.append(ind)

        consensus_score = vote_counts[final_action] / 3.0
        log.info(
            f"[Council] Consensus: action={final_action} | "
            f"type={final_attack_type} | severity={final_severity} | "
            f"confidence={final_confidence:.1%} | agreement={consensus_score:.0%}"
        )

        recommendation = {
            "threat_type":  final_attack_type,
            "severity":     final_severity,
            "confidence":   final_confidence,
            "actions":      [
                # Actual voted action first, then agent-specific reasoning
                f"RL action: {final_action} (voted {vote_counts[final_action]}/3 agents)",
                f"Analyst: {analyst.get('reasoning', 'no reasoning extracted')}",
                f"Engineer: {engineer.get('reasoning', 'no reasoning extracted')} "
                f"[FP risk: {fp_risk}]",
                f"Intel: {intel.get('reasoning', 'no reasoning extracted')} "
                f"[signature: {intel_sig} @ {intel_conf:.0%}]",
            ],
            "false_positive_risk":   fp_risk,
            "model_decision_valid":  model_valid,
            "signature_match":       intel_sig,
            "threat_actor_type":     intel.get("threat_actor_type", "unknown"),
            "all_indicators":        unique_indicators,
            "agent_votes":           {
                "analyst":  analyst.get("recommended_action", "MONITOR"),
                "engineer": engineer.get("recommended_action", "MONITOR"),
                "intel":    intel.get("recommended_action", "MONITOR"),
            },
        }

        return consensus_score, recommendation

    def _generate_explanation(self, analyst: dict, engineer: dict, intel: dict) -> str:
        """Human-readable council report built from parsed agent fields."""
        fp_warning = (
            "\n⚠ ML Engineer flagged HIGH false-positive risk — "
            "model decision validity is questionable."
            if not engineer.get("model_decision_valid", True)
               and engineer.get("false_positive_risk") == "High"
            else ""
        )
        parse_warnings = []
        for agent in (analyst, engineer, intel):
            if not agent.get("parse_ok"):
                parse_warnings.append(
                    f"{agent['role']}: JSON parsing failed — raw text used as fallback"
                )

        return f"""\
THREAT ANALYSIS COUNCIL REPORT
{'=' * 60}

SECURITY ANALYST ({self.analyst_config['provider'].upper()}:{self.analyst_config['model']})
  Classification : {analyst.get('attack_type', 'Unknown')} @ {analyst.get('confidence', 0):.0%}
  Severity       : {analyst.get('severity', '?')}
  Recommended    : {analyst.get('recommended_action', '?')}
  Key Indicators : {'; '.join(analyst.get('indicators', [])) or 'none extracted'}
  Reasoning      : {analyst.get('reasoning', 'none extracted')}

ML ENGINEER ({self.engineer_config['provider'].upper()}:{self.engineer_config['model']})
  Model Valid    : {analyst.get('model_decision_valid', True)}
  FP Risk        : {engineer.get('false_positive_risk', '?')}
  Recommended    : {engineer.get('recommended_action', '?')}
  Anomalous Feat : {'; '.join(engineer.get('anomalous_features', [])) or 'none extracted'}
  Reasoning      : {engineer.get('reasoning', 'none extracted')}

THREAT INTELLIGENCE ({self.intel_config['provider'].upper()}:{self.intel_config['model']})
  Signature Match: {intel.get('signature_match', 'Unknown')} @ {intel.get('attribution_confidence', 0):.0%}
  Threat Actor   : {intel.get('threat_actor_type', '?')}
  Recommended    : {intel.get('recommended_action', '?')}
  IOC Pattern    : {intel.get('ioc_pattern', 'none extracted')}
  Reasoning      : {intel.get('reasoning', 'none extracted')}
{fp_warning}
{'[PARSE WARNINGS] ' + ' | '.join(parse_warnings) if parse_warnings else ''}
{'=' * 60}"""

