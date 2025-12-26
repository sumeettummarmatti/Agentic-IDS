"""Unit tests for attack agents"""

import unittest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.attacker_agents import DDoSAgent, PortScanAgent

class TestAttackerAgents(unittest.TestCase):
    
    def test_ddos_generation(self):
        """DDoS agent generates valid flows"""
        agent = DDoSAgent(attack_type='syn_flood', evasion_level=0.8)
        flow = agent.generate_flow(omega=1.5)
        
        self.assertIsInstance(flow, dict)
        self.assertGreater(flow['SYN Flag Count'], 0)
        self.assertEqual(flow['Protocol'], 6)
    
    def test_portscan_sequence(self):
        """PortScan agent generates valid sequences"""
        agent = PortScanAgent(scan_speed='slow')
        sequence = agent.generate_reconnaissance_sequence(num_probes=10)
        
        self.assertIsInstance(sequence, pd.DataFrame)
        self.assertEqual(len(sequence), 10)

if __name__ == '__main__':
    unittest.main()
