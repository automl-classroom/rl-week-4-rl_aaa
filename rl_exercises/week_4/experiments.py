# most of this file is generated with chatgpt help
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_4.networks import QNetwork


def run_experiment(env, config, hidden_dim=64, num_frames=10000, seed=42):
    """Run a single experiment with given configuration."""
    set_seed(env, seed)
    rewards_history = []
    current_rewards = []

    # Create agent with default network
    agent = DQNAgent(env, **config)

    # If custom hidden_dim specified, replace the networks
    if hidden_dim != 64:  # Only replace if different from default
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        agent.q = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim)
        agent.target_q = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim)
        agent.target_q.load_state_dict(agent.q.state_dict())
        agent.optimizer = optim.Adam(agent.q.parameters(), lr=config.get("lr", 1e-3))

    state, _ = env.reset()
    ep_reward = 0.0

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state)
        next_state, reward, done, truncated, _ = env.step(action)

        agent.buffer.add(state, action, reward, next_state, done or truncated, {})
        state = next_state
        ep_reward += reward

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

        if done or truncated:
            state, _ = env.reset()
            current_rewards.append(ep_reward)
            if len(current_rewards) >= 10:  # Record average every 10 episodes
                rewards_history.append(np.mean(current_rewards))
                current_rewards = []
            ep_reward = 0.0

    return rewards_history


def main():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Base configuration
    base_config = {
        "buffer_capacity": 10000,
        "batch_size": 32,
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_final": 0.01,
        "epsilon_decay": 500,
        "target_update_freq": 1000,
        "seed": 42,
    }

    env = gym.make("CartPole-v1")

    # Experiment 1: Network Architecture
    architectures = [
        {"name": "Small", "hidden_dim": 32},
        {"name": "Medium", "hidden_dim": 64},
        {"name": "Large", "hidden_dim": 128},
    ]

    plt.figure(figsize=(10, 6))
    for arch in architectures:
        config = base_config.copy()
        rewards = run_experiment(env, config, hidden_dim=arch["hidden_dim"])
        plt.plot(rewards, label=f"{arch['name']} Network (h={arch['hidden_dim']})")

    plt.xlabel("Episodes (x10)")
    plt.ylabel("Average Reward (10 episodes)")
    plt.title("DQN Performance with Different Network Sizes")
    plt.legend()
    plt.savefig("plots/network_sizes.png")
    plt.close()

    # Experiment 2: Buffer Sizes
    buffer_sizes = [1000, 5000, 10000, 50000]

    plt.figure(figsize=(10, 6))
    for size in buffer_sizes:
        config = base_config.copy()
        config["buffer_capacity"] = size
        rewards = run_experiment(env, config)
        plt.plot(rewards, label=f"Buffer Size {size}")

    plt.xlabel("Episodes (x10)")
    plt.ylabel("Average Reward (10 episodes)")
    plt.title("DQN Performance with Different Buffer Sizes")
    plt.legend()
    plt.savefig("plots/buffer_sizes.png")
    plt.close()

    # Experiment 3: Batch Sizes
    batch_sizes = [16, 32, 64, 128]

    plt.figure(figsize=(10, 6))
    for size in batch_sizes:
        config = base_config.copy()
        config["batch_size"] = size
        rewards = run_experiment(env, config)
        plt.plot(rewards, label=f"Batch Size {size}")

    plt.xlabel("Episodes (x10)")
    plt.ylabel("Average Reward (10 episodes)")
    plt.title("DQN Performance with Different Batch Sizes")
    plt.legend()
    plt.savefig("plots/batch_sizes.png")
    plt.close()


if __name__ == "__main__":
    main()
