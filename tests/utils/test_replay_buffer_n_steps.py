# Based on
# https://github.com/DLR-RM/stable-baselines3/blob/e206fc55cf7768b8adde3ed3b87c27faa4edf6fe/tests/test_n_step_replay.py

import sys
from collections.abc import Callable

import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.buffer import BatchTransition, ReplayBuffer, random_crop_vectorized
from tests.fixtures.constants import DUMMY_REPO_ID


def state_dims() -> list[str]:
    return ["observation.image", "observation.state"]


@pytest.fixture(scope="function")
def replay_buffer() -> ReplayBuffer:
    return create_empty_replay_buffer()


def clone_state(state: dict) -> dict:
    return {k: v.clone() for k, v in state.items()}

def create_empty_replay_buffer(
    optimize_memory: bool = False,
    use_drq: bool = False,
    image_augmentation_function: Callable | None = None,
) -> ReplayBuffer:
    buffer_capacity = 10
    device = "cpu"
    return ReplayBuffer(
        buffer_capacity,
        device,
        state_dims(),
        optimize_memory=optimize_memory,
        use_drq=use_drq,
        image_augmentation_function=image_augmentation_function,
    )

def fill_buffer(buffer: ReplayBuffer, length: int, done_at: int | None = None, truncated_at: int | None = None):
    """
    Fill the buffer with:
    - reward = 1.0
    - observation = index
    - optional `done` at index `done_at`
    - optional truncation at index `truncated_at`
    """
    for i in range(length):
        # obs = torch.full((1, 4), float(i), dtype=torch.float32)
        # next_obs = torch.full((1, 4), float(i + 1), dtype=torch.float32)
        # action = torch.zeros((1, 2), dtype=torch.float32)
        # reward = torch.tensor([1.0], dtype=torch.float32)
        # done = torch.tensor([1.0 if i == done_at else 0.0], dtype=torch.float32)
        # truncated = i == truncated_at
        # infos = [{"TimeLimit.truncated": truncated}]
        state = {
            "observation.image": torch.full((1, 4), float(i), dtype=torch.float32),
            "observation.state": torch.full((1, 4), float(i + 1), dtype=torch.float32),
        }
        next_state = {
            "observation.image": torch.full((1, 4), float(i + 1), dtype=torch.float32),
            "observation.state": torch.full((1, 4), float(i + 2), dtype=torch.float32),
        }
        action = torch.zeros((1, 2), dtype=torch.float32)
        reward = torch.tensor([1.0], dtype=torch.float32)
        done = torch.tensor([1.0 if i == done_at else 0.0], dtype=torch.bool)
        truncated = torch.tensor([1.0 if i == truncated_at else 0.0], dtype=torch.bool)
        complementary_info = {}
        # action 
        buffer.add(
            state=clone_state(state),
            next_state=clone_state(next_state),
            action=action,
            reward=reward,
            done=done,
            truncated=truncated,
            complementary_info=complementary_info,
        )

def compute_expected_nstep_reward(gamma, n_steps, stop_idx=None):
    """
    Compute the expected n-step reward for the test env (reward=1 for each step),
    optionally stopping early due to termination/truncation.
    Uses torch for computation.
    """
    rewards = torch.ones(n_steps, dtype=torch.float32)
    returns = torch.zeros(n_steps, dtype=torch.float32)
    last_sum = torch.tensor(0.0, dtype=torch.float32)
    for step in reversed(range(n_steps)):
        next_non_terminal = step != stop_idx
        last_sum = rewards[step] + gamma * next_non_terminal * last_sum
        returns[step] = last_sum
    return returns[0]

@pytest.mark.parametrize("done_at", [1, 2])
@pytest.mark.parametrize("n_steps", [3, 5])
@pytest.mark.parametrize("base_idx", [0, 2])
def test_nstep_early_termination(done_at, n_steps, base_idx, replay_buffer: ReplayBuffer):
    gamma = 0.98
    # buffer = create_buffer(n_steps=n_steps, gamma=gamma)
    fill_buffer(replay_buffer, length=10, done_at=done_at)

    batch = replay_buffer._sample(torch.tensor([base_idx]), batch_size=1, gamma=gamma, n_steps=n_steps)
    actual = batch["reward_nsteps"].item()
    expected = compute_expected_nstep_reward(gamma=gamma, n_steps=n_steps, stop_idx=done_at - base_idx)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    assert batch["done_nsteps"].item() == float(base_idx <= done_at)

@pytest.mark.parametrize("truncated_at", [1, 2])
@pytest.mark.parametrize("n_steps", [2, 5])
@pytest.mark.parametrize("base_idx", [0, 1])
def test_nstep_early_truncation(truncated_at, n_steps, base_idx, replay_buffer: ReplayBuffer):
    # buffer = create_buffer(n_steps=n_steps)
    fill_buffer(replay_buffer, length=10, truncated_at=truncated_at)

    batch = replay_buffer._sample(torch.tensor([base_idx]), batch_size=1, gamma=0.99, n_steps=n_steps)
    actual = batch["reward_nsteps"].item()

    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=truncated_at - base_idx)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    assert batch["done_nsteps"].item() == 0.0

@pytest.mark.parametrize("n_steps", [3, 5])
def test_nstep_no_terminations(n_steps, replay_buffer: ReplayBuffer):
    fill_buffer(replay_buffer, length=10)  # no done or truncation
    gamma = 0.99

    base_idx = 3
    batch = replay_buffer._sample(torch.tensor([base_idx]), batch_size=1, gamma=gamma, n_steps=n_steps)
    actual = batch["reward_nsteps"].item()
    # Discount factor for bootstrapping with target Q-Value
    torch.testing.assert_close(batch["discount_nsteps"].item(), gamma**n_steps, rtol=1e-6, atol=1e-6)
    expected = compute_expected_nstep_reward(gamma=gamma, n_steps=n_steps)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    assert batch["done_nsteps"].item() == 0.0

    # Check that self.pos-1 truncation is set when buffer is full
    # Note: buffer size is 10, here we are erasing past transitions
    fill_buffer(replay_buffer, length=2)
    # We create a tmp truncation to not sample across episodes
    base_idx = 0
    batch = replay_buffer._sample(torch.tensor([base_idx]), batch_size=1, gamma=0.99, n_steps=n_steps)
    actual = batch["reward_nsteps"].item()
    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=replay_buffer.position - 1)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    assert batch["done_nsteps"].item() == 0.0
    # Discount factor for bootstrapping with target Q-Value
    # (not equal to gamma ** n_steps because of truncation at n_steps=2)
    torch.testing.assert_close(batch["discount_nsteps"].item(), gamma**2, rtol=1e-6, atol=1e-6)

    # Set done=1 manually, the tmp truncation should not be set (it would set batch.done=False)
    replay_buffer.dones[replay_buffer.position - 1] = True
    batch = replay_buffer._sample(torch.tensor([base_idx]), batch_size=1, gamma=0.99, n_steps=n_steps)
    actual = batch["reward_nsteps"].item()
    expected = compute_expected_nstep_reward(gamma=0.99, n_steps=n_steps, stop_idx=replay_buffer.position - 1)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-6)
    assert batch["done_nsteps"].item() == 1.0
