"""
LLM Council for threat analysis and explanation
Coordinates multiple LLMs to analyze detected threats
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers"""
    GROQ = "groq"
    OLLAMA = "ollama"
    OPENAI = "openai"

@dataclass
class ThreatAnalysis:
    """Result of threat analysis"""
    threat_detected: bool
    confidence: float
    threat_type: str
    severity: str  # Low, Medium, High
    explanation: str
    recommendations: List[str]
    council_consensus: float  # 0-1
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

class ThreatAnalysisCouncil:
    """
    Multi-LLM council for analyzing network threats
    
    Workflow:
    1. Detector identifies suspicious flow
    2. Council analyzes from 3 perspectives:
       - Security Analyst: Attack classification
       - ML Engineer: Feature patterns
       - Threat Intel: Known signatures
    3. Council votes (consensus)
    4. Generate explanation
    """
    
    def __init__(self, 
                 primary_provider: LLMProvider = LLMProvider.GROQ,
                 enable_ollama: bool = True,
                 enable_openai: bool = False):
        """
        Initialize threat analysis council
        
        Args:
            primary_provider: Main LLM provider
            enable_ollama: Use local Ollama
            enable_openai: Use OpenAI API
        """
        self.primary_provider = primary_provider
        self.enable_ollama = enable_ollama
        self.enable_openai = enable_openai
        
        self.clients = {}
        self.council_members = [
            'security_analyst',
            'ml_engineer',
            'threat_intel_specialist'
        ]
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize LLM clients"""
        
        if self.primary_provider == LLMProvider.GROQ or True:
            try:
                from groq import Groq
                api_key = os.getenv('GROQ_API_KEY')
                if api_key:
                    self.clients['groq'] = Groq(api_key=api_key)
                    logger.info("✓ Groq client initialized")
                else:
                    logger.warning("GROQ_API_KEY not set")
            except ImportError:
                logger.warning("Groq not installed. Install: pip install groq")
        
        if self.enable_ollama:
            try:
                import requests
                # Test Ollama connection
                response = requests.get('http://localhost:11434/api/tags')
                if response.status_code == 200:
                    self.clients['ollama'] = True
                    logger.info("✓ Ollama connection established")
                else:
                    logger.warning("Ollama not running. Start: ollama serve")
            except Exception as e:
                logger.warning(f"Ollama not available: {e}")
    
    def analyze_threat(self, flow_data: Dict, detector_prediction: Dict) -> ThreatAnalysis:
        """
        Analyze detected threat using council consensus
        
        Args:
            flow_data: Network flow features
            detector_prediction: Output from ensemble detector
            
        Returns:
            ThreatAnalysis: Structured analysis result
        """
        
        logger.info("=" * 60)
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
        
        logger.info("✓ Council analysis complete")
        
        return result
    
    def _security_analyst_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """
        Security analyst: Attack classification
        """
        logger.info("\n[Security Analyst Perspective]")
        
        prompt = f"""
You are a senior cybersecurity analyst with 10 years experience.
Analyze this network flow for attack characteristics:

Flow Data:
- Protocol: {flow_data.get('Protocol', 'Unknown')}
- Fwd Packets: {flow_data.get('Total Fwd Packet', 0)}
- Bwd Packets: {flow_data.get('Total Bwd packets', 0)}
- SYN Flags: {flow_data.get('Fwd SYN Flags', 0)}
- RST Flags: {flow_data.get('Fwd RST Flags', 0)}
- Data Rate (Bytes/s): {flow_data.get('Flow Bytes/s', 0)}

Detector Prediction:
- Classification: {"ATTACK" if prediction.get('prediction') == 1 else "BENIGN"}
- Confidence: {prediction.get('confidence', 0.5):.2%}

Provide your assessment in this format:
1. Attack Type: [DDoS/PortScan/Evasion/None]
2. Confidence: [0-100]%
3. Key Indicators: [list indicators]
4. Recommended Actions: [immediate response]
"""
        
        try:
            if 'groq' in self.clients:
                response = self.clients['groq'].chat.completions.create(
                    model='mixtral-8x7b-32768',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                analysis = response.choices.message.content
            else:
                analysis = "Using Ollama fallback..."
            
            logger.info(f"Analyst: {analysis[:200]}...")
            
            return {
                'role': 'security_analyst',
                'analysis': analysis,
                'threat_type': self._extract_threat_type(analysis)
            }
            
        except Exception as e:
            logger.error(f"Analyst analysis failed: {e}")
            return {'role': 'security_analyst', 'analysis': '', 'threat_type': 'Unknown'}
    
    def _ml_engineer_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """
        ML Engineer: Feature pattern analysis
        """
        logger.info("\n[ML Engineer Perspective]")
        
        prompt = f"""
You are an ML engineer specializing in network intrusion detection.
Analyze this flow's feature patterns:

Key Features:
- Packet Ratio: {flow_data.get('Total Fwd Packet', 0) / max(1, flow_data.get('Total Bwd packets', 1)):.2f}
- Avg Packet Size: {(flow_data.get('Total Length of Fwd Packet', 0) / max(1, flow_data.get('Total Fwd Packet', 1))):.0f} bytes
- Flow Rate: {flow_data.get('Flow Packets/s', 0):.0f} packets/sec
- SYN/Total Ratio: {flow_data.get('Fwd SYN Flags', 0) / max(1, flow_data.get('Total Fwd Packet', 0)):.2%}

Detector Output: {prediction.get('confidence', 0.5):.2%} confidence for attack

Provide:
1. Feature Anomalies: [what's unusual]
2. Model Explanation: [why model thinks this is attack]
3. Suggested Features to Monitor: [new features for improvement]
"""
        
        try:
            if 'groq' in self.clients:
                response = self.clients['groq'].chat.completions.create(
                    model='mixtral-8x7b-32768',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                analysis = response.choices.message.content
            else:
                analysis = "ML analysis pending..."
            
            logger.info(f"Engineer: {analysis[:200]}...")
            
            return {
                'role': 'ml_engineer',
                'analysis': analysis,
                'feature_anomalies': self._extract_anomalies(analysis)
            }
            
        except Exception as e:
            logger.error(f"Engineer analysis failed: {e}")
            return {'role': 'ml_engineer', 'analysis': '', 'feature_anomalies': []}
    
    def _threat_intel_perspective(self, flow_data: Dict, prediction: Dict) -> Dict:
        """
        Threat Intel: Signature matching
        """
        logger.info("\n[Threat Intelligence Perspective]")
        
        prompt = f"""
You are a threat intelligence analyst tracking global attacks.
Correlate this flow with known attack signatures:

Flow Characteristics:
- Protocol: {flow_data.get('Protocol', 6)}
- SYN Flood Indicator: {flow_data.get('Fwd SYN Flags', 0) > 100}
- Port Scan Indicator: {flow_data.get('Total Fwd Packet', 0) == 1}
- Botnet Characteristics: TBD

Known Attack Patterns Matched:
1. Mirai DDoS: [match score]
2. Generic Port Scanner: [match score]
3. Evasion Technique: [match score]

Provide:
1. Known Attack Match: [which known attack]
2. Confidence in Attribution: [0-100]%
3. Threat Intelligence Summary: [attribution and context]
"""
        
        try:
            if 'groq' in self.clients:
                response = self.clients['groq'].chat.completions.create(
                    model='mixtral-8x7b-32768',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                analysis = response.choices.message.content
            else:
                analysis = "Intel analysis pending..."
            
            logger.info(f"Intel: {analysis[:200]}...")
            
            return {
                'role': 'threat_intel',
                'analysis': analysis,
                'attribution': self._extract_attribution(analysis)
            }
            
        except Exception as e:
            logger.error(f"Intel analysis failed: {e}")
            return {'role': 'threat_intel', 'analysis': '', 'attribution': ''}
    
    def _reach_consensus(self, analyst, engineer, intel) -> tuple:
        """
        Aggregate council perspectives into consensus
        
        Returns:
            tuple: (consensus_score, recommendation_dict)
        """
        logger.info("\n[Council Consensus]")
        
        # Simple consensus: average confidence
        consensus_score = 0.7  # Placeholder
        
        recommendation = {
            'threat_type': analyst.get('threat_type', 'Unknown'),
            'severity': 'High' if consensus_score > 0.7 else 'Medium' if consensus_score > 0.5 else 'Low',
            'actions': [
                'Log flow for analysis',
                'Update detection rules',
                'Monitor related flows'
            ]
        }
        
        logger.info(f"Consensus Score: {consensus_score:.2%}")
        logger.info(f"Threat Type: {recommendation['threat_type']}")
        
        return consensus_score, recommendation
    
    def _generate_explanation(self, analyst, engineer, intel) -> str:
        """Generate human-readable explanation"""
        return f"""
THREAT ANALYSIS COUNCIL REPORT
{self.clients}
Security Analyst Assessment:
{analyst.get('analysis', 'Pending...')[:500]}

ML Engineer Perspective:
{engineer.get('analysis', 'Pending...')[:500]}

Threat Intelligence:
{intel.get('analysis', 'Pending...')[:500]}
"""
    
    def _extract_threat_type(self, text: str) -> str:
        """Extract threat type from analyst text"""
        if 'DDoS' in text:
            return 'DDoS'
        elif 'PortScan' in text or 'Port Scan' in text:
            return 'PortScan'
        elif 'Evasion' in text:
            return 'Evasion'
        return 'Unknown'
    
    def _extract_anomalies(self, text: str) -> List[str]:
        """Extract feature anomalies"""
        return ['High packet rate', 'SYN flag concentration', 'Unusual packet sizes']
    
    def _extract_attribution(self, text: str) -> str:
        """Extract threat attribution"""
        return 'Unknown botnet/attack'
