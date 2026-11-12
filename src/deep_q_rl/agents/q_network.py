"""A two-layer (one hidden layer) MLP, with forward and backward passes
written out by hand — no autograd. Q(s, a) is the output unit for action a.

The whole point of this repo is to not reach for a deep learning framework:
writing the backward pass out by hand is the only way to really know what
"a neural network learns the Q-function" means mechanically. The companion
gradient-check test (tests/test_q_network.py) verifies this hand-derived
backward pass against numerical (finite-difference) gradients — that test
is the real correctness guarantee here, not just "training looks okay."
"""
from __future__ import annotations

import numpy as np


class QNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He-style init, scaled for ReLU hidden units.
        self.w1 = rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, n_actions))
        self.b2 = np.zeros(n_actions)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, input_dim) -> q: (batch, n_actions)."""
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        q = h @ self.w2 + self.b2
        return q

    def _forward_cache(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        q = h @ self.w2 + self.b2
        return q, z1, h

    def q_values(self, state) -> np.ndarray:
        x = np.asarray(state, dtype=float).reshape(1, -1)
        return self.forward(x)[0]

    def loss(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """Mean squared TD error for the taken actions, with params as-is."""
        batch_size = x.shape[0]
        q = self.forward(x)
        q_sa = q[np.arange(batch_size), actions]
        return float(np.mean(0.5 * (targets - q_sa) ** 2))

    def gradients(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        """Hand-derived backward pass. Returns (d_w1, d_b1, d_w2, d_b2, mse)."""
        batch_size = x.shape[0]
        q, z1, h = self._forward_cache(x)
        rows = np.arange(batch_size)
        q_sa = q[rows, actions]
        td_error = targets - q_sa  # (B,)

        d_q = np.zeros_like(q)
        d_q[rows, actions] = -td_error  # dL/dq, L = mean(0.5 * td_error^2)

        d_w2 = h.T @ d_q / batch_size
        d_b2 = d_q.mean(axis=0)

        d_h = d_q @ self.w2.T
        d_z1 = d_h * (z1 > 0)
        d_w1 = x.T @ d_z1 / batch_size
        d_b1 = d_z1.mean(axis=0)

        return d_w1, d_b1, d_w2, d_b2, float(np.mean(td_error**2))

    def train_step(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray, lr: float) -> float:
        """One mini-batch gradient-descent step regressing Q(s, a) toward
        `targets`, for only the taken action in each row — every other
        output unit gets zero gradient, exactly as in DQN's loss.

        Returns the mean squared TD error, for logging.
        """
        d_w1, d_b1, d_w2, d_b2, mse = self.gradients(x, actions, targets)

        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2

        return mse

    def copy_from(self, other: "QNetwork") -> None:
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()])

    def set_flat_params(self, flat: np.ndarray) -> None:
        shapes = [self.w1.shape, self.b1.shape, self.w2.shape, self.b2.shape]
        sizes = [int(np.prod(s)) for s in shapes]
        parts = np.split(flat, np.cumsum(sizes)[:-1])
        self.w1 = parts[0].reshape(shapes[0])
        self.b1 = parts[1].reshape(shapes[1])
        self.w2 = parts[2].reshape(shapes[2])
        self.b2 = parts[3].reshape(shapes[3])
