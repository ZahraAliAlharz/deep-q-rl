from __future__ import annotations

import numpy as np

from deep_q_rl.agents.dqn import DQNAgent
from deep_q_rl.envs.cartpole import CartPole


def run_episode(env: CartPole, agent: DQNAgent, max_steps: int = 500, train: bool = True) -> float:
    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        if train:
            agent.store(state, action, reward, next_state, done)
            agent.update()
        total_reward += reward
        state = next_state
        if done:
            break
    return total_reward


def train(env: CartPole, agent: DQNAgent, episodes: int, max_steps: int = 500) -> np.ndarray:
    returns = np.zeros(episodes)
    for ep in range(episodes):
        returns[ep] = run_episode(env, agent, max_steps=max_steps, train=True)
    return returns
