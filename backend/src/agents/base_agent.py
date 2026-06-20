"""Base classes for all agents"""

from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import numpy as np
import random

class BaseAttackAgent(ABC):
    """Abstract base class for attack simulation agents"""
    
    def __init__(self, seed: int = 42):
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
            'SYN Flag Count': 0,
            'RST Flag Count': 0,
            'Encryption': 'High'
        }
    
    @staticmethod
    def gaussian_random(mean: float = 0, std_dev: float = 1) -> float:
        """Box-Muller Gaussian sampling"""
        u1, u2 = random.random(), random.random()
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return mean + z * std_dev
    
    @abstractmethod
    def generate_flow(self, **kwargs) -> Dict:
        """Generate single attack flow"""
        pass
    
    @abstractmethod
    def generate_sequence(self, **kwargs) -> pd.DataFrame:
        """Generate sequence of flows"""
        pass

class BaseDefenderAgent(ABC):
    """Abstract base class for defense agents"""
    
    @abstractmethod
    def observe(self) -> Dict:
        """Observe network state"""
        pass
    
    @abstractmethod
    def act(self, observation: Dict) -> Dict:
        """Take defensive action"""
        pass
