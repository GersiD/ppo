import gymnasium as gym
import agent
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def train_one_epoch(env, agent):
    done = False
    obs = env.reset()
    obs = obs[0]
    batch = {
        'log_probs': [],
        'rewards': []
    }
    ts = 0
    while not done:
        ts += 1
        action, log_prob = agent.get_action(obs)
        obs, reward, done, _, _ = env.step(action)
        batch['log_probs'].append(log_prob)
        batch['rewards'].append(reward)
        if done or ts >= 1000:
            break
    l = agent.training_step(batch)
    agent.backward(l)
    return l['returns'].item()

env = gym.make('CartPole-v1')
agent = agent.VanillaPG(env)
returns = []
Epoch_rets = []
for i in range(10000):
    returns.append(train_one_epoch(env, agent))
    Epoch_rets.append(returns[-1])
    if i % 1000 == 0:
        print(f'Epoch {i}: {np.mean(returns)}')
        Epoch_rets = []

plt.plot(returns)
plt.xlabel('Epochs')
plt.ylabel('Returns')
plt.show()
plt.savefig('returns.png')
