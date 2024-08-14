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
class VanillaPG(pl.LightningModule):
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
    def get_action(self, obs: np.ndarray) -> Tuple[Any, Tensor]:
        state = torch.from_numpy(obs).float().unsqueeze(0)
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def forward(self, x: np.ndarray):
        self.get_action(x)

    def loss(self, log_prob: Tensor, reward: Tensor) -> Tensor:
        return -(log_prob * reward).mean()

    def training_step(self, batch: Dict[str, list[np.ndarray]]) -> Dict[str, Tensor]:
        lp = batch['log_probs']
        rewards = batch['rewards']
        returns = []
        # calculate rewards-to-goj
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, requires_grad=True)
        log_probs = torch.tensor(lp, requires_grad=True)

        return {"loss" : self.loss(log_probs, returns), "returns": returns.mean()}

    def backward(self, loss: Dict[str, Tensor]):
        self.optimizer.zero_grad()
        loss['loss'].backward()
        self.optimizer.step()

    def get_optimizer(self): 
        return self.optimizer

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return self.optimizer
