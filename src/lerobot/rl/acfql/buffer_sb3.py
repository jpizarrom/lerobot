#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections.abc import Callable, Generator, Sequence
from contextlib import suppress
from typing import TypedDict

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, REWARD
from lerobot.utils.transition import Transition

from ..buffer import ReplayBuffer


class BatchTransitionNSteps(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    action_is_pad: torch.Tensor
    discount: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


class ReplayBufferNSteps(ReplayBuffer):
    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # Pre-allocate tensors for storage
        self.states = {
            key: torch.zeros((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.zeros((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.zeros((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
                key: torch.zeros((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.zeros((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.zeros((self.capacity,), dtype=torch.bool, device=self.storage_device)

        # Initialize storage for complementary_info
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # Pre-allocate tensors for each key in complementary_info
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.squeeze(0).shape
                    self.complementary_info[key] = torch.empty(
                        (self.capacity, *value_shape), device=self.storage_device
                    )
                elif isinstance(value, (int | float)):
                    # Handle scalar values similar to reward
                    self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def sample_nstep_full(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
    ) -> BatchTransitionNSteps:
        """Sample a random batch of transitions and collate them into batched tensors.

        Args:
            batch_size (int): Size of batches to sample
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor

        Yields:
            BatchTransitionNSteps: Batched transitions
        """
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1.")

        batch_size = min(batch_size, self.size)
        high = (
            max(0, self.size - n_steps - 1)
            if self.optimize_memory and self.size < self.capacity
            else self.size - n_steps
        )

        # Random indices for sampling - create on the same device as storage
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)
        return self.sample_nstep_full_for_indices(idx, batch_size, n_steps, gamma)

    def sample_nstep_full_for_indices(
        self,
        idx: torch.Tensor,
        batch_size: int,
        n_steps: int,
        gamma: float,
    ) -> BatchTransitionNSteps:
        # Based on https://github.com/DLR-RM/stable-baselines3/blob/30ceaf3ea1f29ca7213735eaa8460ca2fcfaf9c0/stable_baselines3/common/buffers.py#L924

        last_valid_index = self.position - 1
        original_truncated_values = self.truncateds[last_valid_index].clone()
        self.truncateds[last_valid_index] = torch.logical_or(
            original_truncated_values, torch.logical_not(self.dones[last_valid_index])
        )

        # Compute n-step indices with wrap-around
        steps = torch.arange(n_steps, device=self.storage_device).reshape(1, -1)  # shape: [1, n_steps]

        self.force_full_n_steps = False
        if self.force_full_n_steps:  # TODO: add tests
            # filter out indices that lead to done/truncated states before n_steps
            # TODO: support when episode ends before n_steps
            high = max(0, self.size - 1) if self.optimize_memory and self.size < self.capacity else self.size

            def filter_valid_indices(idx: torch.Tensor) -> torch.Tensor:
                indices = (idx[:, None] + steps) % self.capacity  # shape: [batch, n_steps]
                dones_seq = self.dones[indices]  # [batch, n_steps]
                truncated_seq = self.truncateds[indices]  # [batch, n_steps]

                # Compute masks: 1 until first done/truncation (inclusive)
                done_or_truncated = torch.logical_or(dones_seq, truncated_seq)
                done_idx = done_or_truncated.int().argmax(axis=1)
                # If no done/truncation, keep full sequence
                has_done_or_truncated = done_or_truncated.any(axis=1)
                done_idx = torch.where(has_done_or_truncated, done_idx, n_steps - 1)

                return idx[done_idx == n_steps - 1]

            idx = filter_valid_indices(idx)
            while len(idx) < batch_size:
                # If not enough valid indices, sample some again
                new_idx = torch.randint(
                    low=0, high=high, size=(batch_size - len(idx),), device=self.storage_device
                )
                idx = torch.cat((idx, new_idx), dim=0)
                idx = filter_valid_indices(idx)

        indices = (idx[:, None] + steps) % self.capacity  # shape: [batch, n_steps]

        # Retrieve sequences of transitions
        rewards_seq = self.rewards[indices]  # [batch, n_steps]
        dones_seq = self.dones[indices]  # [batch, n_steps]
        truncated_seq = self.truncateds[indices]  # [batch, n_steps]

        # Compute masks: 1 until first done/truncation (inclusive)
        done_or_truncated = torch.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.int().argmax(axis=1)
        # If no done/truncation, keep full sequence
        has_done_or_truncated = done_or_truncated.any(axis=1)
        done_idx = torch.where(has_done_or_truncated, done_idx, n_steps - 1)
        # done_idx = torch.where(has_done_or_truncated, done_idx, torch.full_like(done_idx, n_steps - 1))

        mask = (
            torch.arange(n_steps, device=self.storage_device).reshape(1, -1) <= done_idx[:, None]
        )  # shape: [batch, n_steps]
        # Compute discount factors for bootstrapping (using target Q-Value)
        # It is gamma ** n_steps by default but should be adjusted in case of early termination/truncation.
        target_q_discounts = gamma ** mask.sum(axis=1, keepdims=True)  # .astype(np.float32)  # [batch, 1]

        # Apply discount
        discounts = gamma ** torch.arange(
            n_steps, dtype=torch.float32, device=self.storage_device
        )  # .reshape(1, -1)  # [1, n_steps]
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=1, keepdims=True)  # [batch, 1]

        # Compute indices of next_obs/done at the final point of the n-step transition
        last_indices = (idx + done_idx) % self.capacity
        # next_obs = self._normalize_obs(self.next_observations[last_indices, env_indices], env)
        # next_dones = self.dones[last_indices].to(self.device).float()
        # next_truncateds = self.truncateds[last_indices].to(self.device).float()

        # batch_rewards = self.rewards[idx].to(self.device)
        batch_rewards_nsteps = n_step_returns.to(self.device)
        batch_discounts_nsteps = target_q_discounts.to(self.device)
        batch_dones_nsteps = self.dones[last_indices].to(self.device).float()
        # batch_dones_nsteps = torch.logical_or(self.dones[last_indices], self.truncateds[last_indices]).to(self.device).float()
        # batch_dones_nsteps = next_dones * (1.0 - next_truncateds)
        batch_truncateds_nsteps = self.truncateds[last_indices].to(self.device).float()
        # batch_mc_returns = mc_returns.to(self.device)

        # batch_actions [batch, n_steps, ...] copy actions until done_idx, then copy last action
        batch_actions = self.actions[indices]  # [batch, n_steps, ...]
        # Create a mask for valid steps (before done/truncation)
        mask = (
            torch.arange(n_steps, device=self.storage_device).reshape(1, -1) <= done_idx[:, None]
        )  # [batch, n_steps]
        # Expand mask to match action shape
        # Expand mask to match action shape using .view and .expand
        mask_expanded = mask.view(batch_size, n_steps, *([1] * (batch_actions.ndim - 2))).expand_as(
            batch_actions
        )
        # Mask out actions after done/truncation
        batch_actions = batch_actions * mask_expanded  # Zero out actions after done/truncation
        batch_actions_is_pad = ~mask

        batch_actions = batch_actions.to(self.device)  # Move actions to the target device
        batch_actions_is_pad = batch_actions_is_pad.to(self.device)  # Move action mask to the target device

        # Revert back tmp changes to avoid sampling across episodes
        self.truncateds[last_valid_index] = original_truncated_values

        # Gather observations and actions
        # obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        # actions = self.actions[batch_inds, env_indices]

        # Identify image keys that need augmentation
        image_keys = [k for k in self.states if k.startswith("observation.image")] if self.use_drq else []

        # Create batched state and next_state
        batch_state = {}
        batch_next_state = {}
        batch_next_state_nsteps = {}

        self.use_terminal_for_next_state = False
        # First pass: load all state tensors to target device
        for key in self.states:
            batch_state[key] = self.states[key][idx].to(self.device)

            if not self.optimize_memory:
                # Standard approach - load next_states directly
                batch_next_state[key] = self.next_states[key][idx].to(self.device)
                batch_next_state_nsteps[key] = self.next_states[key][last_indices].to(self.device)
            else:
                # Memory-optimized approach - get next_state from the next index
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx].to(self.device)

                if self.use_terminal_for_next_state:
                    # TODO: review this way of handling done/truncated for next_state_nsteps and add tests
                    next_state_nsteps_idx = (
                        torch.where(has_done_or_truncated, idx + done_idx, idx + done_idx + 1) % self.capacity
                    )
                else:
                    # TODO: review and add test: next_state_nsteps_idx can be out of bounds
                    next_state_nsteps_idx = (idx + done_idx + 1) % self.capacity

                batch_next_state_nsteps[key] = self.states[key][next_state_nsteps_idx].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])
                all_images.append(batch_next_state_nsteps[key])

            # Optimization: Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # Calculate offsets for the current image key:
                # For each key, we have 3*batch_size images (batch_size for states, batch_size for next_states)
                # States start at index i*2*batch_size and take up batch_size slots
                batch_state[key] = augmented_images[i * 3 * batch_size : (i * 3 + 1) * batch_size]
                # Next states start after the states at index (i*2+1)*batch_size and also take up batch_size slots
                batch_next_state[key] = augmented_images[(i * 3 + 1) * batch_size : (i * 3 + 2) * batch_size]
                # Next states n-steps start after the next states at index (i*3+2)*batch_size and also take up batch_size slots
                batch_next_state_nsteps[key] = augmented_images[
                    (i * 3 + 2) * batch_size : (i * 3 + 3) * batch_size
                ]

        # Sample other tensors
        # batch_actions = self.actions[idx].to(self.device)
        # batch_rewards = self.rewards[idx].to(self.device)
        # batch_dones = self.dones[idx].to(self.device).float()
        # batch_dones = torch.logical_or(self.dones[idx], self.truncateds[idx]).to(self.device).float()
        # batch_truncateds = self.truncateds[idx].to(self.device).float()

        # Sample complementary_info if available
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][idx].to(self.device)

        return BatchTransitionNSteps(
            state=batch_state,
            # action=batch_actions[:,0],
            action=batch_actions,
            action_is_pad=batch_actions_is_pad,
            # action_is_pad=torch.zeros((*batch_actions.shape[:-1],), dtype=torch.bool, device=self.device),
            # reward=batch_rewards,
            # next_state=batch_next_state,
            # done=batch_dones,
            # truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
            reward=batch_rewards_nsteps,
            next_state=batch_next_state_nsteps,
            done=batch_dones_nsteps,
            truncated=batch_truncateds_nsteps,
            discount=batch_discounts_nsteps,
        )

    def get_iterator_nstep(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)

        Yields:
            BatchTransitionNSteps: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator_nstep(batch_size, n_steps, gamma, queue_size)
            else:
                iterator = self._get_naive_iterator_nstep(batch_size, n_steps, gamma)

            # Yield all items from the iterator
            with suppress(StopIteration):
                yield from iterator

    def _get_naive_iterator_nstep(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
        queue_size: int = 2,
    ):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor
            queue_size (int): Number of initial batches to prefetch

        Yields:
            BatchTransitionNSteps: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample_nstep_full(batch_size, n_steps, gamma)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def _get_async_iterator_nstep(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
        queue_size: int = 2,
    ):
        """
        Create an iterator that continuously yields prefetched batches in a
        background thread. The design is intentionally simple and avoids busy
        waiting / complex state management.

        Args:
            batch_size (int): Size of batches to sample.
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor
            queue_size (int): Maximum number of prefetched batches to keep in
                memory.

        Yields:
            BatchTransitionNSteps: A batch sampled from the replay buffer.
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """Continuously put sampled batches into the queue until shutdown."""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample_nstep_full(batch_size, n_steps, gamma)
                    # The timeout ensures the thread unblocks if the queue is full
                    # and the shutdown event gets set meanwhile.
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # Queue is full – loop again (will re-check shutdown_event)
                    continue
                except Exception:
                    # Surface any unexpected error and terminate the producer.
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # If the producer already set the shutdown flag we exit.
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # Drain the queue quickly to help the thread exit if it's blocked on `put`.
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # Give the producer thread a bit of time to finish.
            producer_thread.join(timeout=1.0)

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity. If None, uses dataset length.
            action_mask (Sequence[int] | None): Indices of action dimensions to keep.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        # Convert dataset to transitions generator
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

        # TODO: handle empty dataset case
        first_transition = next(list_transition, None)

        # Initialize the buffer with the first transition to set up storage tensors
        if first_transition is not None:
            first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
            first_action = first_transition["action"].to(device)

            # Get complementary info if available
            first_complementary_info = None
            if (
                "complementary_info" in first_transition
                and first_transition["complementary_info"] is not None
            ):
                first_complementary_info = {
                    k: v.to(device) for k, v in first_transition["complementary_info"].items()
                }

            replay_buffer._initialize_storage(
                state=first_state, action=first_action, complementary_info=first_complementary_info
            )

        # Merge first transition with remaining transitions using itertools.chain

        for data in itertools.chain([first_transition], list_transition):
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(storage_device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(storage_device)

            action = data["action"]

            replay_buffer.add(
                state=data["state"],
                action=action,
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=data["truncated"],  # NOTE: Truncation are not supported yet in lerobot dataset
                complementary_info=data.get("complementary_info", None),
            )

        return replay_buffer

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
    ) -> Generator[Transition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.

        Args:
            dataset (LeRobotDataset):
                The dataset to convert. Each item in the dataset is expected to have
                at least the following keys:
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                plus whatever your 'state_keys' specify.

            state_keys (Sequence[str] | None):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = DONE in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample[ACTION].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = float(current_sample[REWARD].item())  # ensure float

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample[DONE].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            truncated = False
            if not done:
                #  This is important if the dataset has truncations, as it is likely that resuming training will have truncations.
                # If this is the last frame or if next frame is in a different episode, mark as truncated
                if i == num_frames - 1:
                    truncated = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        truncated = True

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # TODO: (azouitine) Check if it's necessary to convert to tensor
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )

            yield transition


def concatenate_batch_transitions_nstep(
    left_batch_transitions: BatchTransitionNSteps, right_batch_transition: BatchTransitionNSteps
) -> BatchTransitionNSteps:
    """
    Concatenates two BatchTransitionNSteps objects into one.

    This function merges the right BatchTransitionNSteps into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransitionNSteps): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransitionNSteps): The second batch to append to the first one.

    Returns:
        BatchTransitionNSteps: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # Concatenate basic fields
    left_batch_transitions[ACTION] = torch.cat(
        [left_batch_transitions[ACTION], right_batch_transition[ACTION]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]], dim=0
    )
    left_batch_transitions["action_is_pad"] = torch.cat(
        [left_batch_transitions["action_is_pad"], right_batch_transition["action_is_pad"]], dim=0
    )
    left_batch_transitions["discount"] = torch.cat(
        [left_batch_transitions["discount"], right_batch_transition["discount"]], dim=0
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # Only process if right_info exists
    if right_info is not None:
        # Initialize left complementary_info if needed
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # Concatenate each field
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions
