import torch
from torch import  nn
import torch.nn.functional as F
import pytorch_lightning as pl
import gymnasium as gym
from typing import Tuple, Any, Dict

from torch import Tensor
import numpy as np

"""
Vanilla Policy Gradient (PG) implementation using PyTorch Lightning
"""
class VanillaPG(nn.Module):
    def __init__(self, env: gym.Env, gamma: float = 0.99) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n))
        self.gamma = gamma
        self.configure_optimizers()

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

    """
    Get the action and the log probability of the action
    """
    def get_action(self, obs) -> Tuple[Any, Tensor]:
        state = torch.from_numpy(obs).float().unsqueeze(0)
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def forward(self, x: np.ndarray):
        self.get_action(x)

    def update(self, batch: Dict[str, list[np.ndarray]]) -> Dict[str, Any]:
        rewards = batch['rewards']
        gs = []
        # calculate rewards-to-go
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            gs.insert(0, R)
        returns = torch.tensor(gs)
        loss = 0
        for log_prob, R in zip(batch['log_probs'], returns):
            loss += log_prob.mean() * R * -1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss, "returns": returns}

    def get_optimizer(self): 
        return self.optimizer

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return self.optimizer
