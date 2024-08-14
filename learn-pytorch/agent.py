import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import gym
from typing import Tuple

from torch.types import Number
from torch import Tensor

"""
Vanilla Policy Gradient (PG) implementation using PyTorch Lightning
"""
class VanillaPG(pl.LightningModule):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n))
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

    """
    Get the action and the log probability of the action
    """
    def get_action(self, obs: torch.Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def forward(self, x: torch.Tensor):
        self.get_action(x)

    def loss(self, log_prob, reward):
        return -(log_prob * reward).mean()

    def training_step(self, L, R) -> torch.Tensor:
        return self.loss(L, R)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
