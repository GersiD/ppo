from lightning.fabric import Fabric
import gymnasium as gym
from typing import Optional
import PPO
import torch
from torch.utils.data import BatchSampler, RandomSampler
from typing import Dict, Tuple
from torch import Tensor


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: Optional[str] = None, prefix: str = ""):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0 and run_name is not None:
                env = gym.wrappers.RecordVideo(
                    env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
                )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def train(
    fabric: Fabric,
    agent,
    optimizer,
    data: Dict[str, Tensor],
    global_step: int,
):
    sampler = RandomSampler(list(range(data["obs"].shape[0])))
    sampler = BatchSampler(sampler, batch_size=1, drop_last=False)

    for _ in range(128):
        for batch_idxes in sampler:
            loss = agent.training_step({k: v[batch_idxes] for k, v in data.items()})
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent, optimizer, max_norm=0.5)
            optimizer.step()
        agent.on_train_epoch_end(global_step)

def main():
    # Initialize Fabric
    fabric = Fabric()
    rank = fabric.global_rank  # The rank of the current process
    world_size = fabric.world_size  # Number of processes spawned
    device = fabric.device
    fabric.seed_everything(42)  # We seed everything for reproduciability purpose
    num_envs = 5
    num_steps = 128
    total_timesteps = 100000
    anneal_lr = True
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 42, i, False) for i in range(num_envs)])
    agent = PPO.PPO(envs, act_fun=torch.nn.ReLU(), ortho_init=True, clip_coef=0.2, value_coef=0.2, entropy_coef=0.1, clip_vloss=True, normalize_advantages=True)
    optimizer = agent.configure_optimizers(1e-3)
    agent, optimizer = fabric.setup(agent, optimizer)


    with fabric.device:
        # with fabric.device is only supported in PyTorch 2.x+
        obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape)
        actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape)
        rewards = torch.zeros((num_steps, num_envs))
        dones = torch.zeros((num_steps, num_envs))

        # Log-probabilities of the action played are needed later on during the training phase
        logprobs = torch.zeros((num_steps, num_envs))

        # The same happens for the critic values
        values = torch.zeros((num_steps, num_envs))

    # Global variables
    global_step = 0
    single_global_rollout = int(num_envs * num_steps * world_size)
    num_updates = total_timesteps // single_global_rollout

    with fabric.device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(envs.reset(seed=42)[0])
        next_done = torch.zeros(num_envs)

    # Collect `num_steps` experiences `num_updates` times
    for update in range(1, num_updates + 1):
        print(f"Update {update}/{num_updates}")
        for step in range(0, num_steps):
            global_step += num_envs * world_size
            obs[step] = next_obs
            dones[step] = next_done

            # Sample an action given the observation received by the environment
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())

            # Check whether the game has finished or not
            done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))

            with fabric.device:
                rewards[step] = torch.tensor(reward).view(-1)
                next_obs, next_done = torch.tensor(next_obs), done
            # Estimate advantages and returns with GAE ()
            returns, advantages = agent.estimate_returns_and_advantages(
                rewards, values, dones, next_obs, next_done, num_steps, 0.98
            )
            # Flatten the batch
            local_data = {
                "obs": obs.reshape((-1,) + envs.single_observation_space.shape),
                "logprobs": logprobs.reshape(-1),
                "actions": actions.reshape((-1,) + envs.single_action_space.shape),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
                "values": values.reshape(-1),
            }

            # Train the agent
            train(fabric, agent, optimizer, local_data, global_step)




if __name__ == "__main__":
    main()
