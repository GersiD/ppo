import torch
import torch.nn.functional as F
import gym
from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric
from torch.nn import Module
from torch import Tensor
from typing import Tuple, Dict
import math

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class PPO(LightningModule):
    def __init__(self, envs: gym.vector.SyncVectorEnv, act_fun: Module = torch.nn.ReLU(), ortho_init: bool = True, clip_coef: float = 0.2, value_coef: float =
                 0.2, entropy_coef = 0.1, clip_vloss: bool = True, normalize_advantages: bool = True, **torchmetrics_kwargs) -> None:
        super().__init__()
        self.critic = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(int(math.prod(envs.single_observation_space.shape or [0])), 64),
                ortho_init=ortho_init,
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, 1), std=1.0, ortho_init=ortho_init),
        )
        self.actor = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(math.prod(envs.single_observation_space.shape or [0]), 64),
                ortho_init=ortho_init,
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, envs.single_action_space.n), std=0.01, ortho_init=ortho_init),
        )
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages
        self.avg_pg_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_value_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_ent_loss = MeanMetric(**torchmetrics_kwargs)

    def ppo_loss(self, advantages: Tensor, ratio: Tensor) -> Tensor:
        return torch.max(-advantages * ratio, -advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef))

    def entropy_loss(self, entropy: Tensor) -> Tensor:
        return -self.entropy_coef * entropy.mean()

    def value_loss(self, new_values: Tensor, old_values: Tensor, returns: Tensor) -> Tensor:
        if self.clip_vloss:
            return self.value_coef * F.mse_loss(old_values + torch.clamp(new_values - old_values, -self.clip_coef, self.clip_coef), returns)
        else:
            return self.value_coef * F.mse_loss(new_values, returns)

    def get_action(self, x: Tensor, action: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy() # return action and log prob and entropy

    def get_action_and_value(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        action, log_prob, entropy = self.get_action(x)
        return action, log_prob, entropy, self.get_value(x)

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    """
    Returns action, log_prob, entropy, value
    """ 
    def forward(self, x: torch.Tensor, action: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        action, log_prob, entropy = self.get_action(x, action)
        return action, log_prob, entropy, self.get_value(x)

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        _, new_log_prob, entropy, new_value = self(batch['obs'], batch['actions'].long())
        log_ratio = new_log_prob - batch['logprobs']

        # Policy loss
        advantages = batch['advantages']
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = self.ppo_loss(advantages, log_ratio.exp())
        # Value loss
        value_loss = self.value_loss(new_value, batch['values'], batch['returns'])
        # Entropy loss
        entropy_loss = self.entropy_loss(entropy)
        # Update metrics
        self.avg_pg_loss(policy_loss)
        self.avg_value_loss(value_loss)
        self.avg_ent_loss(entropy_loss)
        return policy_loss + value_loss + entropy_loss

    @torch.no_grad()
    def estimate_returns_and_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_obs: Tensor,
        next_done: Tensor,
        num_steps: int,
        gamma: float,
    ) -> Tuple[Tensor, Tensor]:
        next_value = self.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = torch.logical_not(next_done)
                nextvalues = next_value
            else:
                nextnonterminal = torch.logical_not(dones[t + 1])
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * nextnonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

    def on_train_epoch_end(self) -> None:
        metrics: Dict[str, float] = {
            "pg_loss": float(self.avg_pg_loss.compute()),
            "value_loss": float(self.avg_value_loss.compute()),
            "ent_loss": float(self.avg_ent_loss.compute()),
        }
        self.logger.log_metrics(metrics, step=self.global_step)
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr)

def layer_init(
    layer: torch.nn.Module,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
    ortho_init: bool = True,
):
    if ortho_init:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
