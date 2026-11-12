import numpy as np

from deep_q_rl.agents.replay_buffer import ReplayBuffer


def test_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=10, state_dim=4, seed=0)
    for i in range(5):
        buf.add(state=np.zeros(4) + i, action=i % 2, reward=1.0, next_state=np.zeros(4), done=False)
    assert len(buf) == 5
    states, actions, rewards, next_states, dones = buf.sample(batch_size=3)
    assert states.shape == (3, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 4)
    assert dones.shape == (3,)


def test_capacity_wraps_around():
    buf = ReplayBuffer(capacity=3, state_dim=2, seed=0)
    for i in range(5):
        buf.add(state=np.array([i, i]), action=0, reward=0.0, next_state=np.array([i, i]), done=False)
    assert len(buf) == 3  # capped at capacity, oldest entries overwritten
    # Only the last 3 inserted states (indices 2, 3, 4) should remain.
    remaining_values = set(buf.states[:, 0].tolist())
    assert remaining_values == {2.0, 3.0, 4.0}
