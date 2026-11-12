"""CartPole (Barto, Sutton & Anderson, 1983; the standard Gym CartPole-v1
formulation reproduced here from scratch).

A pole is hinged to a cart that moves along a frictionless track. The only
control is a fixed-magnitude push left or right on the cart; the agent has
to learn to use the cart's motion to keep the pole from falling, rather than
reacting directly to the pole. Reward is +1 per step the pole stays up.

This is the classic first environment for deep RL: the 4D continuous state
(cart position, cart velocity, pole angle, pole angular velocity) is small
enough to be solved by tabular discretization, but is normally used to show
that a function approximator can learn directly on the raw continuous state
with no hand-built tiling at all — which is the point of this repo.

Actions: 0 = push left, 1 = push right.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_LENGTH = 0.5  # half the pole's length
POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds per simulation step

X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12 * 2 * np.pi / 360


class CartPole:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.actions = (0, 1)
        self.state = np.zeros(4)

    def reset(self) -> tuple[float, float, float, float]:
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        return tuple(self.state)

    def step(self, action: int) -> tuple[tuple[float, float, float, float], float, bool]:
        x, x_dot, theta, theta_dot = self.state
        force = FORCE_MAG if action == 1 else -FORCE_MAG
        costheta, sintheta = np.cos(theta), np.sin(theta)

        temp = (force + POLE_MASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
        theta_acc = (GRAVITY * sintheta - costheta * temp) / (
            HALF_LENGTH * (4.0 / 3.0 - MASS_POLE * costheta**2 / TOTAL_MASS)
        )
        x_acc = temp - POLE_MASS_LENGTH * theta_acc * costheta / TOTAL_MASS

        x = x + TAU * x_dot
        x_dot = x_dot + TAU * x_acc
        theta = theta + TAU * theta_dot
        theta_dot = theta_dot + TAU * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        done = bool(
            x < -X_THRESHOLD
            or x > X_THRESHOLD
            or theta < -THETA_THRESHOLD_RADIANS
            or theta > THETA_THRESHOLD_RADIANS
        )
        reward = 1.0
        return tuple(self.state), reward, done
