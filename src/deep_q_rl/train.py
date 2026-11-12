"""CLI entry point: python -m deep_q_rl.train --episodes 300"""
from __future__ import annotations

import argparse

from deep_q_rl.agents.dqn import DQNAgent
from deep_q_rl.envs.cartpole import CartPole
from deep_q_rl.metrics import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CartPole(seed=args.seed)
    agent = DQNAgent(state_dim=4, n_actions=2, hidden_dim=args.hidden_dim, lr=args.lr,
                      gamma=0.99, epsilon_decay_steps=3000, target_sync_every=100, seed=args.seed)
    returns = train(env, agent, episodes=args.episodes)

    print(f"First 10 episodes mean return: {returns[:10].mean():.1f}")
    print(f"Last 20 episodes mean return:  {returns[-20:].mean():.1f}")


if __name__ == "__main__":
    main()
