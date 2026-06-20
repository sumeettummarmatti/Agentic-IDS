"""
Defender Agent using Reinforcement Learning
Uses Stable Baselines 3 (PPO) to learn optimal response strategies
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import logging
from .base_agent import BaseDefenderAgent

logger = logging.getLogger(__name__)

class IDSDefenseEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface
    The agent learns to mitigate threats while minimizing disruption
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(IDSDefenseEnv, self).__init__()
        
        # Actions: 0=Monitor, 1=Block Source, 2=Deep Packet Inspection, 3=Rate Limit
        self.action_space = spaces.Discrete(4)
        
        # Observation: [Confidence, Threat_Severity(0-2), Traffic_Volume_Normalized]
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(3,), 
            dtype=np.float32
        )
        
        self.current_state = None
        self.steps_left = 100
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.steps_left = 100
        return self.current_state, {}

    def step(self, action):
        self.steps_left -= 1
        
        # Simulate environment response (simplified)
        # Reward = (Threat Mitigated) - (Cost of Action)
        
        reward = 0
        terminated = False
        truncated = self.steps_left <= 0
        
        # Extract state info
        confidence = self.current_state[0]
        severity = self.current_state[1] # 0=Low, 0.5=Med, 1=High
        
        # Action costs (Disruption to user)
        action_costs = {
            0: 0.0,  # Monitor (No cost)
            1: 0.8,  # Block (High cost if false positive)
            2: 0.2,  # DPI (Latency)
            3: 0.4   # Rate Limit (Performance hit)
        }
        cost = action_costs[action]
        
        # Benefit (Mitigating actual threat)
        # Assume if confidence > 0.7 it's likely a real threat
        is_real_threat = confidence > 0.7
        
        if is_real_threat:
            if action == 1: # Block
                reward = 2.0 # High reward for blocking real threat
            elif action == 3: # Rate Limit
                reward = 1.0
            elif action == 0: # Monitor
                reward = -1.0 # Penalty for ignoring real threat
        else: # False positive or low confidence
            if action == 1:
                reward = -2.0 # High penalty for blocking benign
            elif action == 0:
                reward = 0.5 # Reward for doing nothing on benign
                
        reward -= cost
        
        # Randomly update next state
        self.current_state = np.random.rand(3).astype(np.float32)
        
        return self.current_state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass

class DefenderRLAgent(BaseDefenderAgent):
    """
    RL-based Defender Agent using PPO
    """
    
    def __init__(self, model_path="models/defender_ppo"):
        self.env = IDSDefenseEnv()
        self.model = None
        self.model_path = model_path
        
        # Map actions to names
        self.actions = {
            0: "MONITOR",
            1: "BLOCK_SOURCE",
            2: "DEEP_PACKET_INSPECTION",
            3: "RATE_LIMIT"
        }
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize or load PPO model"""
        try:
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            logger.info("Initialized new PPO Defender Agent")
        except Exception as e:
            logger.error(f"Failed to initialize PPO agent: {e}")
            
    def train(self, total_timesteps=1000):
        """Train the agent"""
        if self.model:
            logger.info(f"Training Defender Agent for {total_timesteps} steps...")
            self.model.learn(total_timesteps=total_timesteps)
            # self.model.save(self.model_path) # Optional: save
            logger.info("✓ Defender training complete")

    def observe(self, perception_dict: dict) -> np.ndarray:
        """
        Convert perception dictionary to observation vector
        Expected dict keys: 'confidence', 'severity_score', 'traffic_volume'
        """
        confidence = float(perception_dict.get('confidence', 0.5))
        
        # Map severity string to 0-1
        sev_map = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0}
        severity = sev_map.get(perception_dict.get('threat_level', 'Low'), 0.1)
        
        # Normalize volume (simplified)
        volume = min(perception_dict.get('flow_rate', 0) / 10000.0, 1.0)
        
        return np.array([confidence, severity, volume], dtype=np.float32)

    def act(self, observation: np.ndarray) -> dict:
        """Select action based on observation"""
        if self.model:
            action, _ = self.model.predict(observation)
            action_name = self.actions[int(action)]
            return {
                'action_id': int(action),
                'action': action_name,
                'status': 'executed'
            }
        else:
            return {'action': 'ERROR', 'status': 'failed'}
