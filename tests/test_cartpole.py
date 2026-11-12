import numpy as np

from deep_q_rl.envs.cartpole import CartPole


def test_reset_state_in_range():
    env = CartPole(seed=0)
    state = env.reset()
    assert len(state) == 4
    assert all(-0.05 <= s <= 0.05 for s in state)


def test_pole_falls_without_control_input():
    # Starting the pole tipped over with no corrective push: it should
    # fall (terminate) well before the 500-step cap.
    env = CartPole(seed=0)
    env.reset()
    env.state = np.array([0.0, 0.0, 0.15, 0.0])  # ~8.6 degrees, already past upright
    for step in range(500):
        _, _, done = env.step(1)  # constant push, not actively balancing
        if done:
            break
    assert done
    assert step < 100


def test_episode_can_run_to_max_steps_with_alternating_action():
    # A naive bang-bang policy (alternate push direction) should at least
    # be able to run without erroring, terminating either by falling or
    # by reaching the step cap -- this is just a smoke test for step().
    env = CartPole(seed=1)
    env.reset()
    for step in range(500):
        action = step % 2
        _, reward, done = env.step(action)
        assert reward == 1.0
        if done:
            break
