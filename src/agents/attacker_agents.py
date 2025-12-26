"""
Attacker agents that generate synthetic attack flows
- DDoS Agent (SYN Flood, UDP Flood)
- PortScan Agent (slow scan, decoy traffic)
- Evasion Agent (adversarial examples)
"""
#syn data generator - just like ddos agent we had built - need to add more details
import pandas as pd
import numpy as np
import random
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class BaseAttackAgent:
    """Base class for attack agents"""
    #checkonce -hardcoding the pattern??
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.base_features = {
            'Protocol': 6,
            'Total Fwd Packet': 0,
            'Total Bwd packets': 0,
            'Total Length of Fwd Packet': 0,
            'Total Length of Bwd Packet': 0,
            'Fwd Packet Length Max': 0,
            'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 0.0,
            'Fwd Packet Length Std': 0.0,
            'Bwd Packet Length Max': 0,
            'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 0.0,
            'Bwd Packet Length Std': 0.0,
            'Flow Bytes/s': 0,
            'Flow Packets/s': 0,
            'Fwd IAT Total': 0,
            'Fwd IAT Mean': 0.0,
            'Fwd IAT Std': 0.0,
            'Bwd IAT Total': 0,
            'Bwd IAT Mean': 0.0,
            'Bwd IAT Std': 0.0,
            'Fwd PSH Flags': 0,
            'Fwd SYN Flags': 0,
            'Fwd RST Flags': 0,
            'Bwd PSH Flags': 0,
            'Bwd SYN Flags': 0,
            'Bwd RST Flags': 0,
            'Fwd Init Win Bytes': 0,
            'Bwd Init Win Bytes': 0,
            'Encryption': 1  # Attack label
        }
    
    def gaussian_random(self, mean=0, std_dev=1):
        """Box-Muller transform for Gaussian sampling"""
        u1, u2 = random.random(), random.random()
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return mean + z * std_dev

class DDoSAgent(BaseAttackAgent):
    """
    DDoS attack agent with variants:
    - SYN Flood: High SYN flags, small packets
    - UDP Flood: Many packets, large payload
    - HTTP Flood: Application layer attacks
    """
    
    def __init__(self, attack_type='syn_flood', evasion_level=0.5):
        super().__init__()
        self.attack_type = attack_type
        self.evasion_level = evasion_level
        self.current_dst_port = random.randint(1, 65535)
    
    def generate_flow(self, omega=1.5, use_evasion=True):
        """
        Generate DDoS attack flow
        
        Args:
            omega: Attack intensity multiplier
            use_evasion: Apply evasion techniques
            
        Returns:
            dict: Attack flow with features
        """
        flow = self.base_features.copy()
        
        if self.attack_type == 'syn_flood':
            flow['Protocol'] = 6  # TCP
            flow['Total Fwd Packet'] = int(self.gaussian_random(500 * omega, 100))
            flow['Fwd SYN Flags'] = int(flow['Total Fwd Packet'] * 0.95)
            flow['Total Length of Fwd Packet'] = flow['Total Fwd Packet'] * 40
            flow['Fwd Packet Length Max'] = 40
            flow['Fwd Packet Length Min'] = 40
            flow['Fwd Packet Length Mean'] = 40.0
            
            if use_evasion and self.evasion_level > 0.5:
                # Add RST to look normal
                flow['Fwd RST Flags'] = int(flow['Fwd SYN Flags'] * 0.1)
                # Variable packet sizes
                flow['Fwd Packet Length Max'] = int(self.gaussian_random(100, 30))
                flow['Fwd Packet Length Min'] = int(self.gaussian_random(40, 15))
        
        elif self.attack_type == 'udp_flood':
            flow['Protocol'] = 17  # UDP
            flow['Total Fwd Packet'] = int(self.gaussian_random(1000 * omega, 200))
            flow['Total Length of Fwd Packet'] = flow['Total Fwd Packet'] * 500
            flow['Fwd Packet Length Max'] = 500
            flow['Fwd Packet Length Min'] = 500
            
            if use_evasion and self.evasion_level > 0.5:
                flow['Fwd Packet Length Max'] = int(self.gaussian_random(600, 100))
        
        # Flow rate (packets per second)
        flow['Flow Packets/s'] = flow['Total Fwd Packet'] / 1.0  # 1 second
        flow['Flow Bytes/s'] = flow['Total Length of Fwd Packet'] / 1.0
        
        return flow
    
    def generate_attack_sequence(self, num_flows=100, omega=1.5):
        """
        Generate sequence of attack flows
        
        Args:
            num_flows: Number of flows in sequence
            omega: Attack intensity
            
        Returns:
            pd.DataFrame: Attack sequence
        """
        flows = []
        for i in range(num_flows):
            # Intensity varies (burst pattern)
            current_omega = omega * (1 + 0.3 * np.sin(2 * np.pi * i / num_flows))
            flow = self.generate_flow(omega=current_omega, use_evasion=True)
            flow['Flow Sequence'] = i
            flow['Timestamp'] = i * 0.001
            flows.append(flow)
        
        return pd.DataFrame(flows)

class PortScanAgent(BaseAttackAgent):
    """
    Stealthy port scanning agent
    - Slow scans (evade rate limiting)
    - Decoy traffic (look benign)
    - Protocol switching
    """
    
    def __init__(self, scan_speed='slow'):
        super().__init__()
        self.scan_speed = scan_speed
        self.target_ports = list(range(1, 1000)) + [3306, 5432, 27017]
        self.port_index = 0
    
    def generate_flow(self, decoy_ratio=0.3):
        """
        Generate port scan flow
        
        Args:
            decoy_ratio: Fraction of flows that are decoy
            
        Returns:
            dict: Scan or decoy flow
        """
        flow = self.base_features.copy()
        flow['Protocol'] = 6  # TCP
        
        if random.random() > decoy_ratio:
            # Real scan attempt
            flow['Total Fwd Packet'] = 1
            flow['Fwd SYN Flags'] = 1
            flow['Total Bwd packets'] = random.choice([0, 1])
            flow['Total Length of Fwd Packet'] = 40
            flow['Fwd Packet Length Max'] = 40
            flow['Fwd Packet Length Min'] = 40
            
            flow['Fwd IAT Total'] = self.gaussian_random(5000, 1000)
            flow['Fwd IAT Mean'] = flow['Fwd IAT Total']
        else:
            # Decoy: looks normal
            flow['Total Fwd Packet'] = int(self.gaussian_random(10, 2))
            flow['Total Bwd packets'] = int(self.gaussian_random(10, 2))
            flow['Total Length of Fwd Packet'] = int(self.gaussian_random(500, 100))
            flow['Flow Packets/s'] = self.gaussian_random(5, 1)
        
        return flow
    
    def generate_reconnaissance_sequence(self, num_probes=100):
        """
        Generate realistic port scan sequence
        
        Args:
            num_probes: Number of probes
            
        Returns:
            pd.DataFrame: Scan sequence
        """
        flows = []
        for i in range(num_probes):
            flow = self.generate_flow(decoy_ratio=0.2)
            flow['Probe Number'] = i
            flow['Timestamp'] = i * (5 if self.scan_speed == 'slow' else 0.1)
            flows.append(flow)
        
        return pd.DataFrame(flows)

def generate_balanced_synthetic_dataset(num_ddos=5000, num_portscan=2000):
    """
    Generate balanced synthetic dataset
    
    Args:
        num_ddos: Number of DDoS flows
        num_portscan: Number of PortScan flows
        
    Returns:
        pd.DataFrame: Synthetic attack data
    """
    logger.info("Generating synthetic attack data...")
    
    ddos_agent = DDoSAgent(attack_type='syn_flood', evasion_level=0.8)
    portscan_agent = PortScanAgent(scan_speed='slow')
    
    # Generate DDoS flows
    ddos_flows = []
    for i in range(num_ddos // 100):
        sequence = ddos_agent.generate_attack_sequence(num_flows=100, omega=1.5)
        ddos_flows.append(sequence)
    
    ddos_df = pd.concat(ddos_flows, ignore_index=True)
    ddos_df['Label'] = 'DDoS'
    ddos_df['Encryption'] = 1  # Attack
    
    # Generate PortScan flows
    portscan_flows = []
    for i in range(num_portscan // 100):
        sequence = portscan_agent.generate_reconnaissance_sequence(num_probes=100)
        portscan_flows.append(sequence)
    
    portscan_df = pd.concat(portscan_flows, ignore_index=True)
    portscan_df['Label'] = 'PortScan'
    portscan_df['Encryption'] = 1  # Attack
    
    # Combine
    synthetic_df = pd.concat([ddos_df, portscan_df], ignore_index=True)
    
    logger.info(f"✓ Generated {len(synthetic_df)} synthetic flows")
    logger.info(f"  - DDoS: {len(ddos_df)}")
    logger.info(f"  - PortScan: {len(portscan_df)}")
    
    return synthetic_df
