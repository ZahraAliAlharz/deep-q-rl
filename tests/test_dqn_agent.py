import numpy as np

from deep_q_rl.agents.dqn import DQNAgent
from deep_q_rl.envs.cartpole import CartPole
from deep_q_rl.metrics import train


def test_select_action_in_range():
    agent = DQNAgent(state_dim=4, n_actions=2, seed=0)
    for _ in range(20):
        assert agent.select_action(np.zeros(4)) in (0, 1)


def test_epsilon_decays_toward_end_value():
    agent = DQNAgent(state_dim=4, n_actions=2, epsilon_start=1.0, epsilon_end=0.05,
                      epsilon_decay_steps=100, seed=0)
    assert agent.epsilon == 1.0
    agent.total_steps = 100
    assert np.isclose(agent.epsilon, 0.05)
    agent.total_steps = 1000  # past decay horizon: should clamp, not overshoot
    assert np.isclose(agent.epsilon, 0.05)


def test_update_returns_none_before_min_buffer_size():
    agent = DQNAgent(state_dim=4, n_actions=2, min_buffer_size=50, batch_size=8, seed=0)
    for _ in range(10):
        agent.store(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert agent.update() is None


def test_target_network_syncs_periodically():
    agent = DQNAgent(state_dim=4, n_actions=2, target_sync_every=5, min_buffer_size=1,
                      batch_size=4, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4):
        s = rng.normal(size=4)
        agent.store(s, 0, 1.0, rng.normal(size=4), False)
    for _ in range(4):
        agent.update()  # steps 1-4: online diverges from target (non-zero inputs)
    assert not np.allclose(agent.online.w1, agent.target.w1)
    agent.update()  # step 5: should trigger a sync before this update's gradient step
    # After the sync-then-update, target should be very close to the
    # online weights as they were right at the sync point (not identical,
    # since online takes one more gradient step after the sync).
    assert np.allclose(agent.online.w1, agent.target.w1, atol=0.5)


def test_training_improves_average_return_on_cartpole():
    # Real DQN-on-CartPole learning curves are flat/noisy (returns in the
    # teens-20s, a near-random bang-bang policy's survival time) for a
    # while before a sharp takeoff once the Q-function is good enough to
    # actually balance -- not a smooth monotonic climb. 150 episodes with
    # these hyperparameters reliably reaches that takeoff for this seed.
    env = CartPole(seed=0)
    agent = DQNAgent(state_dim=4, n_actions=2, hidden_dim=32, lr=0.01, gamma=0.99,
                      epsilon_decay_steps=3000, target_sync_every=100, seed=0)
    returns = train(env, agent, episodes=150, max_steps=500)
    assert returns[-10:].mean() > returns[:10].mean() * 3
