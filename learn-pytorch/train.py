import gym
import agent
import torch
import tensordict
import numpy as np

def train_one_epoch(env, agent, optim):
    done = False
    obs = env.reset()
    obs = obs[0]
    log_probs = []
    rewards = []
    ts = 0
    while not done:
        ts += 1
        action, log_prob = agent.get_action(torch.tensor(obs))
        obs, reward, done, _, _ = env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        print(reward)
        if done or ts >= 1000:
            break
    R = torch.tensor([np.sum(rewards[i:]) * 0.99 ** i for i in range(len(rewards))])
    L = torch.tensor(log_probs)
    optim.zero_grad()
    agent.training_step(L, R)
    optim.step()

env = gym.make('MountainCar-v0')
agent = agent.VanillaPG(env)
optimizer = agent.configure_optimizers()
for i in range(100):
    train_one_epoch(env, agent, optimizer)
