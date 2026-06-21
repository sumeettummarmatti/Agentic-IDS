"""
Defender Agent using Reinforcement Learning
Uses Stable Baselines 3 (PPO) to learn optimal response strategies.

Key design decisions
--------------------
* **Reward tiers** — three distinct confidence bands drive different optimal actions:
    - High threat (conf > 0.7)   : BLOCK_SOURCE or RATE_LIMIT rewarded, MONITOR penalised
    - Medium threat (0.5–0.7)    : DEEP_PACKET_INSPECTION rewarded (verify before escalating)
    - Low / benign (conf < 0.5)  : MONITOR rewarded, BLOCK_SOURCE heavily penalised (FP cost)

* **Stateful transitions** — next state is a noisy blend of current + random, giving the
  LSTM-like autocorrelation a real IDS would see (bursts of similar traffic).

* **Entropy bonus** — ent_coef=0.01 ensures the agent keeps exploring even after it finds
  a good policy, preventing premature convergence to a single action.

* **Save / load** — model is saved after training and loaded on the next startup so the
  policy never resets to random weights.

* **Minimum 20 k timesteps** — PPO default n_steps=2048, so 500 steps (the old value)
  meant ZERO gradient updates ever occurred. 20 k = ~9 full rollouts + policy updates.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import logging

from .base_agent import BaseDefenderAgent

logger = logging.getLogger(__name__)


# ─── Environment ─────────────────────────────────────────────────────────────

class IDSDefenseEnv(gym.Env):
    """
    Custom Gymnasium environment for IDS defence policy learning.

    Observation space (3-D continuous Box):
        [confidence (0–1), threat_severity (0–1), traffic_volume_norm (0–1)]

    Action space (Discrete 4):
        0 = MONITOR              (no kernel change — passive watch)
        1 = BLOCK_SOURCE         (iptables DROP from source IP)
        2 = DEEP_PACKET_INSPECTION (NFQUEUE redirect for Suricata)
        3 = RATE_LIMIT           (hashlimit ≤50 pkt/s from source)
    """

    metadata = {'render.modes': ['human']}

    # Action costs model operational disruption to legitimate users
    ACTION_COSTS = {
        0: 0.0,   # MONITOR       — zero disruption
        1: 0.7,   # BLOCK_SOURCE  — hard block; high FP cost
        2: 0.15,  # DPI           — latency overhead
        3: 0.35,  # RATE_LIMIT    — performance hit
    }

    def __init__(self):
        super().__init__()
        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.current_state = None
        self.steps_left    = 200   # longer episodes = richer trajectories

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start from a random realistic state
        self.current_state = np.random.rand(3).astype(np.float32)
        self.steps_left    = 200
        return self.current_state, {}

    def step(self, action: int):
        self.steps_left -= 1

        confidence = float(self.current_state[0])
        severity   = float(self.current_state[1])

        # ── Reward tiers ──────────────────────────────────────────────────────
        #
        # HIGH THREAT  (conf > 0.70) — confirmed attack; act decisively
        #   BLOCK_SOURCE  +3.0   best mitigation
        #   RATE_LIMIT    +1.5   good for volumetric (DDoS)
        #   DPI           +0.8   acceptable but should have blocked already
        #   MONITOR       -2.5   dangerous inaction
        #
        # MEDIUM THREAT (conf 0.50–0.70) — ambiguous; gather info first
        #   DPI           +2.0   investigate before escalating
        #   RATE_LIMIT    +0.8   throttle while watching
        #   MONITOR       +0.3   acceptable wait-and-see
        #   BLOCK_SOURCE  -1.5   risky false positive
        #
        # LOW / BENIGN  (conf < 0.50) — probably benign; don't disrupt
        #   MONITOR       +1.0   correct: do nothing
        #   RATE_LIMIT    -0.8   unnecessary throttle
        #   DPI           -0.5   wasted resources
        #   BLOCK_SOURCE  -3.0   false positive; highest penalty

        if confidence > 0.70:
            reward_map = {0: -2.5, 1: 3.0, 2: 0.8, 3: 1.5}
        elif confidence > 0.50:
            reward_map = {0: 0.3, 1: -1.5, 2: 2.0, 3: 0.8}
        else:
            reward_map = {0: 1.0, 1: -3.0, 2: -0.5, 3: -0.8}

        reward = reward_map[action] - self.ACTION_COSTS[action]

        # Severity bonus: for high-severity + correct aggressive action
        if severity > 0.7 and action == 1:
            reward += 0.5

        # ── Stateful transition: blend current + noise ────────────────────────
        # Real traffic bursts have autocorrelation — this models that.
        noise = np.random.rand(3).astype(np.float32)
        self.current_state = np.clip(
            0.7 * self.current_state + 0.3 * noise, 0.0, 1.0
        ).astype(np.float32)

        terminated = False
        truncated  = self.steps_left <= 0
        return self.current_state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass


# ─── Agent ───────────────────────────────────────────────────────────────────

class DefenderRLAgent(BaseDefenderAgent):
    """
    PPO-based IDS Defender Agent (Stable Baselines 3).

    Policy: MlpPolicy (3-layer MLP, 64 units each)
    Hyperparameters chosen for a 3-D observation, 4-action discrete task:
      - n_steps=2048     : one full rollout buffer before each update
      - batch_size=256   : mini-batch size for PPO gradient steps
      - n_epochs=10      : number of passes over the rollout data
      - learning_rate=3e-4
      - ent_coef=0.01    : entropy bonus to sustain exploration
      - clip_range=0.2   : standard PPO clip
    """

    _DEFAULT_TRAIN_STEPS = 20_000   # ≥9 full rollouts → meaningful policy

    def __init__(self, model_path: str = "models/defender_ppo"):
        self.env        = IDSDefenseEnv()
        self.model      = None
        self.model_path = model_path

        self.actions = {
            0: "MONITOR",
            1: "BLOCK_SOURCE",
            2: "DEEP_PACKET_INSPECTION",
            3: "RATE_LIMIT",
        }

        self._initialize_model()

    def _initialize_model(self):
        """Load saved model if available, otherwise create a new PPO instance."""
        zip_path = self.model_path + ".zip"
        if os.path.exists(zip_path):
            try:
                self.model = PPO.load(zip_path, env=self.env)
                logger.info(f"✓ Loaded PPO Defender from {zip_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load PPO model ({e}) — will retrain")

        # Fresh model with tuned hyperparameters
        self.model = PPO(
            "MlpPolicy",
            self.env,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            ent_coef=0.01,        # entropy bonus for sustained exploration
            clip_range=0.2,
            verbose=0,            # suppress SB3 progress spam
        )
        logger.info("Initialized new PPO Defender Agent")

    def train(self, total_timesteps: int = _DEFAULT_TRAIN_STEPS):
        """
        Train the PPO policy.

        total_timesteps must be ≥ n_steps (2048) for any gradient update
        to occur. The default 20 k gives ~9 rollout→update cycles.
        """
        if not self.model:
            return

        # Skip expensive re-training if a saved model was already loaded
        zip_path = self.model_path + ".zip"
        if os.path.exists(zip_path):
            logger.info("PPO Defender already trained — skipping retraining")
            return

        logger.info(
            f"Training PPO Defender for {total_timesteps:,} timesteps "
            f"(n_steps=2048 → {total_timesteps // 2048} updates)…"
        )
        self.model.learn(total_timesteps=total_timesteps)
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"✓ PPO Defender trained and saved → {zip_path}")

    def observe(self, perception_dict: dict) -> np.ndarray:
        """
        Convert a perception dictionary to a 3-D observation vector.

        Expected keys:
            confidence   (float 0–1)
            threat_level (str: 'Low' | 'Medium' | 'High')
            flow_rate    (float, packets/s)
        """
        confidence = float(perception_dict.get('confidence', 0.5))
        sev_map    = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0}
        severity   = sev_map.get(perception_dict.get('threat_level', 'Low'), 0.1)
        volume     = min(float(perception_dict.get('flow_rate', 0)) / 10_000.0, 1.0)
        return np.array([confidence, severity, volume], dtype=np.float32)

    def act(self, observation: np.ndarray) -> dict:
        """Predict the best action for the given observation."""
        if not self.model:
            return {'action': 'MONITOR', 'action_id': 0, 'status': 'no_model'}
        action, _ = self.model.predict(observation, deterministic=True)
        action_id   = int(action)
        action_name = self.actions.get(action_id, "MONITOR")
        return {
            'action_id': action_id,
            'action':    action_name,
            'status':    'executed',
        }
