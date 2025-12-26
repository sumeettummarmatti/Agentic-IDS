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
    """Universal LLM client supporting Groq and Ollama"""
    
    def __init__(self):
        self.groq_client = None
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Initialize both providers
        self._init_groq()
        self._init_ollama()
    
    def _init_groq(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                logger.info("✓ Groq client initialized")
            else:
                logger.warning("GROQ_API_KEY not found in environment")
        except ImportError:
            logger.warning("Groq not installed. Run: pip install groq")
    
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
    
    def generate(self, prompt: str, provider: str, model: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        """
        Universal generate method
        
        Args:
            prompt: Input prompt
            provider: 'groq' or 'ollama'
            model: Model name
            max_tokens: Max response length
            temperature: Sampling temperature
            
        Returns:
            str: Generated response
        """
        if provider.lower() == 'groq':
            return self._generate_groq(prompt, model, max_tokens, temperature)
        elif provider.lower() == 'ollama':
            return self._generate_ollama(prompt, model, max_tokens, temperature)
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
        """Generate using Ollama"""
        try:
            response = requests.post(
                f'{self.ollama_base_url}/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': temperature,
                        'num_predict': max_tokens
                    }
                },
                timeout=60  # Ollama can be slower
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                logger.error(f"Ollama error: {response.text}")
                return f"Error: Ollama returned {response.status_code}"
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: {str(e)}"


class ThreatAnalysisCouncil:
    """
    Hybrid Multi-LLM council for threat analysis
    Each member can use different provider + model
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        
        # Load council member configurations from env
        self.analyst_config = self._parse_model_config(
            os.getenv('ANALYST_MODEL', 'groq:llama-3.1-8b-instant')
        )
        self.engineer_config = self._parse_model_config(
            os.getenv('ENGINEER_MODEL', 'groq:llama-3.3-70b-versatile')
        )
        self.intel_config = self._parse_model_config(
            os.getenv('INTEL_MODEL', 'ollama:qwen3:8b')
        )
        
        logger.info("=" * 60)
        logger.info("THREAT ANALYSIS COUNCIL INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Security Analyst:  {self.analyst_config['provider']} - {self.analyst_config['model']}")
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

    def _parse_model_config(self, config_string: str) -> Dict[str, str]:
        """
        Parse model config string in format: 'provider:model'
        
        Examples:
            'groq:llama-3.1-8b-instant'
            'ollama:qwen2.5:8b'
            'llama-3.1-70b-versatile' (defaults to groq)
        
        Returns:
            dict: {'provider': str, 'model': str}
        """
        if ':' in config_string:
            parts = config_string.split(':', 1)
            return {
                'provider': parts[0],
                'model': parts[1]
            }
        else:
            # Default to groq if no provider specified
            return {
                'provider': 'groq',
                'model': config_string
            }
    
    def analyze_threat(self, flow_data: Dict, detector_prediction: Dict) -> ThreatAnalysis:
        """Analyze detected threat using council consensus"""
        
        logger.info("\n" + "=" * 60)
        logger.info("THREAT ANALYSIS COUNCIL CONVENES")
        logger.info("=" * 60)
        
        # Get analysis from each council member
        analyst_view = self._security_analyst_perspective(flow_data, detector_prediction)
        engineer_view = self._ml_engineer_perspective(flow_data, detector_prediction)
        intel_view = self._threat_intel_perspective(flow_data, detector_prediction)
        
        # Aggregate perspectives
        consensus, final_recommendation = self._reach_consensus(
            analyst_view, engineer_view, intel_view
        )
        
        # Generate final analysis
        result = ThreatAnalysis(
            threat_detected=detector_prediction.get('prediction') == 1,
            confidence=detector_prediction.get('confidence', 0.5),
            threat_type=final_recommendation['threat_type'],
            severity=final_recommendation['severity'],
            explanation=self._generate_explanation(
                analyst_view, engineer_view, intel_view
            ),
            recommendations=final_recommendation['actions'],
            council_consensus=consensus,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("✓ Council analysis complete\n")
        return result
    
    def _security_analyst_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """Security analyst: Attack classification"""
        logger.info(f"\n[Security Analyst - {self.analyst_config['provider'].upper()}:{self.analyst_config['model']}]")
        
        # Prepare data string for template
        flow_str = "\n".join([f"- {k}: {v}" for k, v in flow_data.items()])
        pred_str = f"- Classification: {'ATTACK' if prediction.get('prediction')==1 else 'BENIGN'}\n- Confidence: {prediction.get('confidence', 0.5):.2%}\n- Type: {prediction.get('attack_type', 'Unknown')}"
        
        # Get template or legacy default
        template = self.prompts.get('SECURITY_ANALYST_PROMPT', "")
        
        if template:
            prompt = template.format(flow_data=flow_str, prediction=pred_str)
        else:
            # Legacy fallback
            prompt = f"""You are a senior cybersecurity analyst. Analyze this network flow:
            Flow Data: {flow_str}
            Prediction: {pred_str}
            Respond with Attack Type, Confidence, Indicators, Actions."""
        
        try:
            analysis = self.llm_client.generate(
                prompt=prompt,
                provider=self.analyst_config['provider'],
                model=self.analyst_config['model'],
                max_tokens=300,
                temperature=0.3
            )
            
            logger.info(f"Response: {analysis[:150]}...")
            
            return {
                'role': 'security_analyst',
                'analysis': analysis,
                'threat_type': self._extract_threat_type(analysis)
            }
            
        except Exception as e:
            logger.error(f"Analyst analysis failed: {e}")
            return {'role': 'security_analyst', 'analysis': '', 'threat_type': 'Unknown'}
    
    def _ml_engineer_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """ML Engineer: Feature pattern analysis"""
        logger.info(f"\n[ML Engineer - {self.engineer_config['provider'].upper()}:{self.engineer_config['model']}]")
        
        # Prepare data strings
        features_str = "\n".join([f"- {k}: {v}" for k, v in flow_data.items()])
        model_out_str = f"{prediction.get('confidence', 0.5):.2%} confidence for {prediction.get('attack_type', 'Unknown')}"
        
        # Get template
        template = self.prompts.get('ML_ENGINEER_PROMPT', "")
        
        if template:
            prompt = template.format(features=features_str, model_output=model_out_str)
        else:
            # Legacy fallback
            prompt = f"Analyze ML features: {features_str}. Model says: {model_out_str}"
        
        try:
            analysis = self.llm_client.generate(
                prompt=prompt,
                provider=self.engineer_config['provider'],
                model=self.engineer_config['model'],
                max_tokens=300,
                temperature=0.3
            )
            
            logger.info(f"Response: {analysis[:150]}...")
            
            return {
                'role': 'ml_engineer',
                'analysis': analysis,
                'feature_anomalies': self._extract_anomalies(analysis)
            }
            
        except Exception as e:
            logger.error(f"Engineer analysis failed: {e}")
            return {'role': 'ml_engineer', 'analysis': '', 'feature_anomalies': []}
    
    def _threat_intel_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """Threat Intel: Signature matching"""
        logger.info(f"\n[Threat Intel - {self.intel_config['provider'].upper()}:{self.intel_config['model']}]")
        
        # Prepare data strings
        char_str = "\n".join([f"- {k}: {v}" for k, v in flow_data.items()])
        patterns_str = "Known signatures: Mirai, Slowloris, UDP Flood, TCP SYN Flood, XSS, SQLi"
        
        # Get template
        template = self.prompts.get('THREAT_INTEL_PROMPT', "")
        
        if template:
            prompt = template.format(characteristics=char_str, known_patterns=patterns_str)
        else:
            # Legacy fallback
            prompt = f"Correlate flow: {char_str} with known patterns."
        
        try:
            analysis = self.llm_client.generate(
                prompt=prompt,
                provider=self.intel_config['provider'],
                model=self.intel_config['model'],
                max_tokens=250,
                temperature=0.3
            )
            
            logger.info(f"Response: {analysis[:150]}...")
            
            return {
                'role': 'threat_intel',
                'analysis': analysis,
                'attribution': self._extract_attribution(analysis)
            }
            
        except Exception as e:
            logger.error(f"Intel analysis failed: {e}")
            return {'role': 'threat_intel', 'analysis': '', 'attribution': ''}
    
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
    
    def _extract_threat_type(self, text: str) -> str:
        """Extract threat type from analyst text"""
        text_upper = text.upper()
        if 'DDOS' in text_upper:
            return 'DDoS'
        elif 'PORT' in text_upper and 'SCAN' in text_upper:
            return 'PortScan'
        elif 'EVASION' in text_upper:
            return 'Evasion'
        elif 'NONE' in text_upper or 'BENIGN' in text_upper:
            return 'Benign'
        return 'Unknown'
    
    def _extract_anomalies(self, text: str) -> List[str]:
        """Extract feature anomalies"""
        anomalies = []
        if 'packet rate' in text.lower():
            anomalies.append('High packet rate')
        if 'syn' in text.lower():
            anomalies.append('SYN flag concentration')
        if 'packet size' in text.lower():
            anomalies.append('Unusual packet sizes')
        return anomalies or ['Pattern analysis in progress']
    
    def _extract_attribution(self, text: str) -> str:
        """Extract threat attribution"""
        text_lower = text.lower()
        if 'mirai' in text_lower:
            return 'Mirai botnet'
        elif 'slowloris' in text_lower:
            return 'Slowloris attack'
        elif 'scanner' in text_lower or 'scan' in text_lower:
            return 'Network scanner'
        return 'Unknown/Generic attack pattern'
