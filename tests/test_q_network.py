"""The most important test in this repo: it verifies the hand-derived
backward pass in QNetwork.gradients() against numerical (finite-difference)
gradients of the loss. If the analytic backprop has a sign error, an
indexing bug, or a wrong chain-rule term, this is what catches it --
"training looks like it's working" is not a reliable way to tell.
"""
import numpy as np

from deep_q_rl.agents.q_network import QNetwork


def numerical_gradient(net: QNetwork, x, actions, targets, eps=1e-5) -> np.ndarray:
    base = net.get_flat_params().copy()
    grad = np.zeros_like(base)
    for i in range(len(base)):
        plus = base.copy()
        plus[i] += eps
        net.set_flat_params(plus)
        loss_plus = net.loss(x, actions, targets)

        minus = base.copy()
        minus[i] -= eps
        net.set_flat_params(minus)
        loss_minus = net.loss(x, actions, targets)

        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    net.set_flat_params(base)
    return grad


def test_analytic_gradient_matches_numerical_gradient():
    net = QNetwork(input_dim=3, hidden_dim=5, n_actions=2, seed=0)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 3))
    actions = rng.integers(0, 2, size=4)
    targets = rng.normal(size=4)

    d_w1, d_b1, d_w2, d_b2, _ = net.gradients(x, actions, targets)
    analytic = np.concatenate([d_w1.ravel(), d_b1.ravel(), d_w2.ravel(), d_b2.ravel()])
    numerical = numerical_gradient(net, x, actions, targets)

    assert np.allclose(analytic, numerical, atol=1e-4)


def test_forward_output_shape():
    net = QNetwork(input_dim=4, hidden_dim=8, n_actions=2, seed=0)
    q = net.forward(np.zeros((5, 4)))
    assert q.shape == (5, 2)


def test_train_step_reduces_loss_on_same_batch():
    net = QNetwork(input_dim=4, hidden_dim=8, n_actions=2, seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 4))
    actions = rng.integers(0, 2, size=16)
    targets = rng.normal(size=16)

    loss_before = net.loss(x, actions, targets)
    for _ in range(20):
        net.train_step(x, actions, targets, lr=0.05)
    loss_after = net.loss(x, actions, targets)

    assert loss_after < loss_before


def test_only_taken_action_column_changes():
    net = QNetwork(input_dim=4, hidden_dim=8, n_actions=3, seed=0)
    x = np.ones((1, 4))
    w2_before = net.w2.copy()
    net.train_step(x, actions=np.array([1]), targets=np.array([5.0]), lr=0.1)
    assert not np.allclose(net.w2[:, 1], w2_before[:, 1])
    assert np.allclose(net.w2[:, 0], w2_before[:, 0])
    assert np.allclose(net.w2[:, 2], w2_before[:, 2])
