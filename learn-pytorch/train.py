import gym
import lightning as L
import agent
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict
import numpy as np


env = gym.make('CartPole-v1')
agent = agent.VanillaPG(env)

fabric = L.Fabric()
fabric.launch()

optimizer = agent.configure_optimizers()
agent, optimizer = fabric.setup(agent, optimizer)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    state = env.reset()
    done = False
    while not done:
        action, log_prob = agent.get_action(Tensor(state[0]))
        next_state, reward, done, _, _ = env.step(action.squeeze().item())
        agent.store({'log_prob': log_prob, 'reward': reward})
        state = next_state

