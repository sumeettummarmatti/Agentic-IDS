# File: src/council/llm_council_wrapper.py

"""
Wrapper around Karpathy's LLM Council for threat analysis
Coordinates multiple LLMs to reach consensus on attack classification
"""

import os
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import Karpathy's LLM Council
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llm-council'))
from llm_council import Council, Member

from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from pydantic import BaseModel

# ============================================================================
# Data Models
# ============================================================================

class AttackType(str, Enum):
    DDOS = "DDoS"
    PORTSCAN = "PortScan"
    EVASION = "Evasion"
    BENIGN = "Benign"
    UNKNOWN = "Unknown"

@dataclass
class ThreatAnalysis:
    attack_type: AttackType
    confidence: float
    reasoning: str
    feature_recommendations: List[str]
    consensus_score: float  # How much council agreed (0-1)
    individual_votes: Dict[str, AttackType]

# ============================================================================
# LLM Council Configuration
# ============================================================================

class CouncilMember(BaseModel):
    """Represents a council member (LLM with specific role)"""
    name: str
    role: str
    system_prompt: str
    model_provider: str  # "groq" or "ollama"
    model_name: str

# ============================================================================
# LLM Council Wrapper
# ============================================================================

class ThreatAnalysisCouncil:
    """
    Multi-LLM threat analysis council
    Uses Karpathy's Council framework to coordinate threat classification
    """
    
    def __init__(self, use_local_llm: bool = True, use_groq: bool = True):
        """
        Args:
            use_local_llm: Use local Ollama models
            use_groq: Use Groq API (online)
        """
        self.use_local = use_local_llm
        self.use_groq = use_groq
        self.members: Dict[str, Any] = {}
        
        # Initialize council members
        self._setup_council()
    
    def _setup_council(self):
        """Setup council members with specific expertise"""
        
        # Member 1: Security Analyst (Groq - fast)
        if self.use_groq:
            try:
                self.members['security_analyst'] = ChatGroq(
                    model_name="mixtral-8x7b-32768",
                    api_key=os.getenv('GROQ_API_KEY'),
                    temperature=0.3  # Lower temp for consistency
                )
            except Exception as e:
                print(f"⚠️  Groq not available: {e}. Falling back to Ollama.")
                self.use_groq = False
        
        # Member 2: Feature Engineer (Local)
        if self.use_local:
            self.members['feature_engineer'] = Ollama(
                model=os.getenv('OLLAMA_MODEL', 'deepseek-v3:7b'),
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                temperature=0.5
            )
        
        # Member 3: Threat Intel (Groq - if available)
        if self.use_groq and 'security_analyst' in self.members:
            self.members['threat_intel'] = ChatGroq(
                model_name="mixtral-8x7b-32768",
                api_key=os.getenv('GROQ_API_KEY'),
                temperature=0.4
            )
        
        if not self.members:
            raise RuntimeError(
                "No LLM providers available. "
                "Install Groq key or start Ollama (ollama serve)"
            )
    
    def analyze_threat(self, flow_data: Dict[str, Any]) -> ThreatAnalysis:
        """
        Analyze network flow through council
        
        Args:
            flow_data: Network flow features as dict
            
        Returns:
            ThreatAnalysis with consensus classification
        """
        
        # Step 1: Format flow for analysis
        flow_narrative = self._flow_to_narrative(flow_data)
        
        # Step 2: Get votes from each council member
        votes = self._collect_votes(flow_narrative)
        
        # Step 3: Compute consensus
        consensus = self._compute_consensus(votes)
        
        # Step 4: Get recommendations
        recommendations = self._get_recommendations(flow_narrative, consensus)
        
        return ThreatAnalysis(
            attack_type=consensus['type'],
            confidence=consensus['confidence'],
            reasoning=consensus['reasoning'],
            feature_recommendations=recommendations,
            consensus_score=consensus['consensus'],
            individual_votes=votes
        )
    
    def _flow_to_narrative(self, flow_data: Dict) -> str:
        """Convert network flow to natural language"""
        
        narrative = f"""
        NETWORK FLOW ANALYSIS REQUEST
        ============================
        
        Flow Characteristics:
        - Protocol: {flow_data.get('Protocol', 'Unknown')} 
          {'(TCP)' if flow_data.get('Protocol') == 6 else '(UDP)' if flow_data.get('Protocol') == 17 else ''}
        - Forward Packets: {flow_data.get('Total Fwd Packet', 0)}
        - Backward Packets: {flow_data.get('Total Bwd packets', 0)}
        - SYN Flags: {flow_data.get('SYN Flag Count', 0)}
        - RST Flags: {flow_data.get('RST Flag Count', 0)}
        - Forward Packet Length (Max): {flow_data.get('Fwd Packet Length Max', 0)}
        - Forward Packet Length (Min): {flow_data.get('Fwd Packet Length Min', 0)}
        - Total Forward Data: {flow_data.get('Total Length of Fwd Packet', 0)} bytes
        
        TASK:
        1. Classify attack type (DDoS, PortScan, Evasion, or Benign)
        2. Provide confidence (0-100%)
        3. Explain reasoning
        """
        
        return narrative.strip()
    
    def _collect_votes(self, flow_narrative: str) -> Dict[str, AttackType]:
        """Get classification votes from each council member"""
        
        votes = {}
        
        for member_name, llm in self.members.items():
            try:
                prompt = f"""
                {flow_narrative}
                
                Based on the flow characteristics, classify as ONE of:
                - DDoS (many packets, SYN flood pattern, high throughput)
                - PortScan (single/few packets, SYN flags, varied ports)
                - Evasion (randomized features, suspicious patterns)
                - Benign (normal traffic characteristics)
                
                Respond with ONLY the classification (DDoS/PortScan/Evasion/Benign).
                """
                
                response = llm.invoke(prompt)
                
                # Extract classification from response
                response_text = response if isinstance(response, str) else response.content
                classification = self._extract_classification(response_text)
                votes[member_name] = classification
                
            except Exception as e:
                print(f"⚠️  {member_name} vote failed: {e}")
                votes[member_name] = AttackType.UNKNOWN
        
        return votes
    
    def _extract_classification(self, text: str) -> AttackType:
        """Extract classification from LLM response"""
        
        text = text.upper()
        
        if 'DDOS' in text:
            return AttackType.DDOS
        elif 'PORTSCAN' in text or 'PORT SCAN' in text:
            return AttackType.PORTSCAN
        elif 'EVASION' in text:
            return AttackType.EVASION
        elif 'BENIGN' in text:
            return AttackType.BENIGN
        else:
            return AttackType.UNKNOWN
    
    def _compute_consensus(self, votes: Dict[str, AttackType]) -> Dict[str, Any]:
        """Compute consensus from member votes"""
        
        # Count votes
        vote_counts = {}
        for classification in votes.values():
            vote_counts[classification] = vote_counts.get(classification, 0) + 1
        
        # Find majority vote
        majority_class = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_class]
        total_votes = len(votes)
        
        consensus_score = majority_count / total_votes if total_votes > 0 else 0.0
        
        return {
            'type': majority_class,
            'confidence': min(95, int(consensus_score * 100)),  # Cap at 95%
            'consensus': consensus_score,
            'reasoning': f"Council consensus: {majority_count}/{total_votes} members agreed on {majority_class.value}"
        }
    
    def _get_recommendations(self, flow_narrative: str, consensus: Dict) -> List[str]:
        """Get feature engineering recommendations from council"""
        
        try:
            if 'feature_engineer' in self.members:
                prompt = f"""
                {flow_narrative}
                
                Based on this {consensus['type'].value} attack, suggest 3-5 new features that would 
                improve detection. Format as Python variable names (snake_case).
                """
                
                response = self.members['feature_engineer'].invoke(prompt)
                response_text = response if isinstance(response, str) else response.content
                
                # Parse features from response
                features = [line.strip() for line in response_text.split('\n') 
                           if line.strip() and not line.startswith('#')]
                return features[:5]
        except Exception as e:
            print(f"⚠️  Feature recommendation failed: {e}")
        
        return []

# ============================================================================
# Usage
# ============================================================================

def main():
    """Test council"""
    
    # Initialize council
    council = ThreatAnalysisCouncil(use_local_llm=True, use_groq=True)
    
    # Sample flow
    sample_flow = {
        'Protocol': 6,
        'Total Fwd Packet': 500,
        'Total Bwd packets': 50,
        'SYN Flag Count': 450,
        'RST Flag Count': 10,
        'Fwd Packet Length Max': 40,
        'Fwd Packet Length Min': 40,
        'Total Length of Fwd Packet': 20000
    }
    
    # Analyze
    print("🔍 Analyzing flow through LLM Council...\n")
    analysis = council.analyze_threat(sample_flow)
    
    print(f"Attack Type: {analysis.attack_type.value}")
    print(f"Confidence: {analysis.confidence}%")
    print(f"Consensus Score: {analysis.consensus_score:.2%}")
    print(f"Reasoning: {analysis.reasoning}")
    print(f"Feature Recommendations: {analysis.feature_recommendations}")
    print(f"\nIndividual Votes:")
    for member, vote in analysis.individual_votes.items():
        print(f"  - {member}: {vote.value}")

if __name__ == "__main__":
    import sys
    main()
EOF
