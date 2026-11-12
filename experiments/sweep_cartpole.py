"""Learning curve for from-scratch DQN on CartPole, averaged over seeds,
plus a comparison against disabling the target network (target_sync_every=1,
i.e. the target network tracks the online network every single step, which
is equivalent to having no separate target network at all) to measure
whether it actually matters here, rather than assuming it does because
the literature says so.

Generates results/cartpole_learning_curve.png and
results/cartpole_target_network_ablation.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deep_q_rl.agents.dqn import DQNAgent
from deep_q_rl.envs.cartpole import CartPole
from deep_q_rl.metrics import train

EPISODES = 150
SEEDS = range(10)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_agent(seed: int, target_sync_every: int = 100) -> DQNAgent:
    return DQNAgent(
        state_dim=4, n_actions=2, hidden_dim=32, lr=0.01, gamma=0.99,
        epsilon_decay_steps=3000, target_sync_every=target_sync_every, seed=seed,
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Learning curve with the target network, averaged over seeds.
    all_returns = []
    for seed in SEEDS:
        env = CartPole(seed=seed)
        agent = make_agent(seed)
        all_returns.append(train(env, agent, episodes=EPISODES, max_steps=500))
    all_returns = np.array(all_returns)
    mean_returns = all_returns.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mean_returns)
    ax.axhline(500, color="gray", linestyle="--", linewidth=1, label="max possible (500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (steps balanced)")
    ax.set_title(f"CartPole: from-scratch DQN learning curve (mean over {len(SEEDS)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cartpole_learning_curve.png", dpi=150)

    print(f"With target network: first 10 episodes mean={all_returns[:, :10].mean():.1f}, "
          f"last 10 episodes mean={all_returns[:, -10:].mean():.1f}")

    # 2. Ablation: target_sync_every=1 means the "target" network is
    # resynced every single update, i.e. it's not really a separate
    # target at all -- the online network is chasing a target that moves
    # under its own update, which is exactly the instability a target
    # network exists to prevent.
    no_target_returns = []
    for seed in SEEDS:
        env = CartPole(seed=seed)
        agent = make_agent(seed, target_sync_every=1)
        no_target_returns.append(train(env, agent, episodes=EPISODES, max_steps=500))
    no_target_returns = np.array(no_target_returns)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mean_returns, label="target_sync_every=100 (normal)")
    ax.plot(no_target_returns.mean(axis=0), label="target_sync_every=1 (no real target net)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (steps balanced)")
    ax.set_title(f"CartPole: effect of the target network (mean over {len(SEEDS)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cartpole_target_network_ablation.png", dpi=150)

    print(f"Without target network: first 10 episodes mean={no_target_returns[:, :10].mean():.1f}, "
          f"last 10 episodes mean={no_target_returns[:, -10:].mean():.1f}")


if __name__ == "__main__":
    main()
