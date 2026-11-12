"""DQN (Mnih et al., 2015): Q-learning with a function approximator, trained
off a replay buffer, using a separate target network to keep the TD target
from shifting under the same update that's chasing it.

Without the target network, the regression target `r + gamma * max_a' Q(s',a')`
moves every time the online network's weights move — including the weights
that produced the target a step ago — which is a major source of instability
in naive neural-network Q-learning. Freezing a copy of the weights for
`target_sync_every` steps at a time is the standard fix.
"""
from __future__ import annotations

import numpy as np

from deep_q_rl.agents.q_network import QNetwork
from deep_q_rl.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 32,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        lr: float = 0.01,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 3000,
        target_sync_every: int = 100,
        min_buffer_size: int = 500,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_every = target_sync_every
        self.min_buffer_size = min_buffer_size

        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(state_dim, hidden_dim, n_actions, seed=seed)
        self.target = QNetwork(state_dim, hidden_dim, n_actions, seed=seed)
        self.target.copy_from(self.online)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, seed=seed)

        self.total_steps = 0

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.online.q_values(state)))

    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    def update(self) -> float | None:
        self.total_steps += 1
        if self.target_sync_every and self.total_steps % self.target_sync_every == 0:
            self.target.copy_from(self.online)
        if len(self.buffer) < max(self.batch_size, self.min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        next_q = self.target.forward(next_states)
        targets = rewards + self.gamma * next_q.max(axis=1) * (~dones)
        return self.online.train_step(states, actions, targets, self.lr)
