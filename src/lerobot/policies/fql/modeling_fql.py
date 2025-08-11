#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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

import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    TanhTransform,
    Transform,
    TransformedDistribution,
)

from lerobot.policies.fql.configuration_fql import FQLConfig, is_image_feature
from lerobot.policies.normalize import NormalizeBuffer, UnnormalizeBuffer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class FQLPolicy(
    PreTrainedPolicy,
):
    config_class = FQLConfig
    name = "fql"

    def __init__(
        self,
        config: FQLConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # # queues are populated during rollout of the policy, they contain the n latest observations and actions
        # self._queues = None

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0]
        self._init_normalization(dataset_stats)
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor_bc_flow(continuous_action_dim)
        self._init_actor_onestep_flow(continuous_action_dim)
        self._init_temperature()

        self.reset()

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor_bc_flow": [
                p
                for n, p in self.actor_bc_flow.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "actor_onestep_flow": [
                p
                for n, p in self.actor_onestep_flow.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            # "temperature": self.log_alpha,
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
            optim_params["discrete_bc_flow"] = [
                p
                for n, p in self.discrete_actor_bc_flow.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ]
            optim_params["discrete_onestep_flow"] = [
                p
                for n, p in self.discrete_actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ]
        raise ValueError("Not used")
        return optim_params

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.chunk_size)

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def compute_discrete_flow_actions(
        self, batch: dict[str, Tensor], discrete_noises: Tensor = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute discrete actions using masked discrete flow matching sampling exactly following reference."""
        observations_features = None
        if self.shared_encoder and self.discrete_actor_bc_flow.encoder.has_images:
            # Cache and normalize image features
            observations_features = self.discrete_actor_bc_flow.encoder.get_cached_image_features(
                batch, normalize=True
            )

        batch_size = batch["observation.state"].shape[0]
        chunk_size = self.config.chunk_size  # D in reference
        device = batch["observation.state"].device
        MASK_TOKEN = self.config.num_discrete_actions  # MASK token at index num_discrete_actions

        # Sampling exactly following the reference pattern with masking
        t = 0.0
        dt = 1.0 / self.config.flow_steps
        noise = 10.0  # noise parameter from reference (higher for masking)

        # Start from all masked tokens: xt = MASK_TOKEN * torch.ones((num_samples, D), dtype=torch.long)
        xt = MASK_TOKEN * torch.ones((batch_size, chunk_size), dtype=torch.long, device=device)

        # Flow matching sampling loop exactly following reference with masking
        while t < 1.0:
            # Get model predictions: logits = model(xt, t * torch.ones((num_samples,)))
            t_tensor = torch.full((batch_size,), t, device=device)
            t_expanded = t_tensor.unsqueeze(-1)  # [batch_size, 1] for network

            # Model outputs logits only over valid values (0 to num_discrete_actions-1), not including MASK token
            logits, _, _ = self.discrete_actor_bc_flow(
                batch, observations_features, xt, t_expanded
            )  # [batch_size, chunk_size, num_discrete_actions] -> (B, D, num_discrete_actions)

            # Convert to probabilities over valid actions: x1_probs = F.softmax(logits, dim=-1)
            x1_probs = F.softmax(logits, dim=-1)  # (B, D, num_discrete_actions)

            # Sample from valid actions: x1 = Categorical(x1_probs).sample()
            x1 = Categorical(x1_probs).sample()  # (B, D) - values in range [0, num_discrete_actions-1]

            # Determine which positions will unmask: will_unmask = torch.rand((B, D)) < (dt * (1 + noise * t) / (1-t))
            will_unmask = torch.rand((batch_size, chunk_size), device=device) < (
                dt * (1 + noise * t) / (1 - t)
            )
            # Only unmask positions that are currently masked
            will_unmask = will_unmask & (xt == MASK_TOKEN)  # (B, D)

            # Unmask selected positions with sampled values
            xt[will_unmask] = x1[will_unmask]

            t += dt

            # Add noise (re-mask) if not final step
            if t < 1.0:
                # Determine which unmasked positions will be re-masked: will_mask = torch.rand((B, D)) < dt * noise
                will_mask = torch.rand((batch_size, chunk_size), device=device) < dt * noise
                # Only mask positions that are currently unmasked
                will_mask = will_mask & (xt != MASK_TOKEN)  # (B, D)
                # Re-mask selected positions
                xt[will_mask] = MASK_TOKEN

        # Final forward pass to get final probabilities
        t_final = torch.ones((batch_size,), device=device)
        t_final_expanded = t_final.unsqueeze(-1)
        logits, log_probs, action_probs = self.discrete_actor_bc_flow(
            batch, observations_features, xt, t_final_expanded
        )

        return xt, log_probs, action_probs

    @torch.no_grad()
    def compute_flow_actions(self, batch: dict[str, Tensor], noises: Tensor) -> Tensor:
        observations_features = None
        if self.shared_encoder and self.actor_bc_flow.encoder.has_images:
            # Cache and normalize image features
            observations_features = self.actor_bc_flow.encoder.get_cached_image_features(
                batch, normalize=True
            )

        actions = noises
        flow_steps = self.config.flow_steps
        # dt = 1.0 / flow_steps

        # Euler method.
        for i in range(flow_steps):
            t_val = float(i) / flow_steps
            t = torch.full((actions.shape[0], 1), t_val, device=noises.device)
            vels, _, _ = self.actor_bc_flow(batch, observations_features, actions, t)
            actions = actions + vels / flow_steps

        actions = torch.clamp(actions, -1.0, 1.0)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""

        observations_features = None
        if self.shared_encoder and self.actor_onestep_flow.encoder.has_images:
            # Cache and normalize image features
            observations_features = self.actor_onestep_flow.encoder.get_cached_image_features(
                batch, normalize=True
            )

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # batch_shape = list(observations[list(observations.keys())[0]].shape[:-1])
            batch_shape = batch["observation.state"].shape[0]
            action_dim = self.actor_onestep_flow.action_dim  # self.config['action_dim']
            device = batch["observation.state"].device

            noises = torch.randn(batch_shape, action_dim, device=device)
            actions, _, _ = self.actor_onestep_flow(batch, observations_features, noises)

            actions = actions.reshape(batch_shape, -1, 3)
            actions = torch.clamp(actions, -1.0, 1.0)
            # actions = self.unnormalize_targets({"action": actions})["action"]
            if self.config.num_discrete_actions is not None:
                # Use discrete flow matching for discrete actions with masking approach
                chunk_size = self.config.chunk_size
                MASK_TOKEN = self.config.num_discrete_actions  # MASK token at index num_discrete_actions

                # Start with masked tokens for one-step prediction
                discrete_tokens = MASK_TOKEN * torch.ones(
                    (batch_shape, chunk_size), dtype=torch.long, device=device
                )

                # One-step prediction using discrete actor (no time input for one-step)
                # This outputs logits over valid actions (0 to num_discrete_actions-1), not including MASK
                discrete_logits, _, _ = self.discrete_actor(batch, observations_features, discrete_tokens)
                discrete_dist = Categorical(logits=discrete_logits.view(-1, self.config.num_discrete_actions))
                discrete_actions = discrete_dist.sample().view(batch_shape, chunk_size)

                # Add as last dimension
                discrete_actions = discrete_actions.unsqueeze(-1).float()
                actions = torch.cat([actions, discrete_actions], dim=-1)

            self._action_queue.extend(actions.transpose(0, 1))

        actions = self._action_queue.popleft()

        # if self.config.num_discrete_actions is not None:
        #     discrete_action, _, _ = self.discrete_actor(batch, observations_features)
        #     # discrete_action_value = self.discrete_critic(batch, observations_features)
        #     # discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
        #     actions = torch.cat([actions, discrete_action.unsqueeze(-1)], dim=-1)

        return actions

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
        do_output_normalization: bool = True,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features, do_output_normalization)
        return q_values

    def discrete_critic_forward(
        self,
        observations,
        actions: Tensor,
        use_target=False,
        observation_features=None,
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, actions, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal[
            "actor",
            "critic",
            "discrete_critic",
            "actor_bc_flow",
            "actor_onestep_flow",
            "discrete_bc_flow",
            "discrete_onestep_flow",
        ] = "critic",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch["action"]
        actions_is_pad = batch["actions_is_pad"]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward_nsteps"]
            discounts: Tensor = batch["discount_nsteps"]
            next_observations: dict[str, Tensor] = batch["next_state_nsteps"]
            done: Tensor = batch["done_nsteps"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic, info = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                actions_is_pad=actions_is_pad,
                rewards=rewards,
                discounts=discounts,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {"loss_critic": loss_critic, "info": info}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward_nsteps"]
            discounts: Tensor = batch["discount_nsteps"]
            next_observations: dict[str, Tensor] = batch["next_state_nsteps"]
            done: Tensor = batch["done_nsteps"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic, info = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                actions_is_pad=actions_is_pad,
                rewards=rewards,
                discounts=discounts,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic, "info": info}
        # if model == "actor":
        #     return {
        #         "loss_actor": self.compute_loss_actor(
        #             observations=observations,
        #             observation_features=observation_features,
        #             actions=actions,
        #         )
        #     }
        if model == "actor_bc_flow":
            loss_actor_bc_flow, info = self.compute_loss_actor_bc_flow(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                actions_is_pad=actions_is_pad,
            )
            return {"loss_actor_bc_flow": loss_actor_bc_flow, "info": info}
        if model == "actor_onestep_flow":
            loss_actor_onestep_flow, info = self.compute_loss_actor_onestep_flow(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                actions_is_pad=actions_is_pad,
            )
            return {"loss_actor_onestep_flow": loss_actor_onestep_flow, "info": info}

        if model == "discrete_bc_flow":
            if self.config.num_discrete_actions is None:
                raise ValueError("Discrete BC flow is not configured for this policy.")

            loss_discrete_bc_flow, info = self.compute_loss_discrete_bc_flow(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                actions_is_pad=actions_is_pad,
            )
            return {"loss_discrete_bc_flow": loss_discrete_bc_flow, "info": info}

        if model == "discrete_onestep_flow":
            if self.config.num_discrete_actions is None:
                raise ValueError("Discrete one-step flow is not configured for this policy.")

            loss_discrete_onestep_flow, info = self.compute_loss_discrete_onestep_flow(
                observations=observations,
                observation_features=observation_features,
                actions_is_pad=actions_is_pad,
            )
            return {"loss_discrete_onestep_flow": loss_discrete_onestep_flow, "info": info}

        # if model == "temperature":
        #     return {
        #         "loss_temperature": self.compute_loss_temperature(
        #             observations=observations,
        #             observation_features=observation_features,
        #         )
        #     }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )

        # Update discrete critic target networks if they exist
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        actions_is_pad: Tensor,
        rewards,
        discounts,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        # actions = self.normalize_targets({"action": actions})["action"]

        with torch.no_grad():
            batch_shape = next_observations["observation.state"].shape[0]
            action_dim = self.actor_onestep_flow.action_dim  # self.config['action_dim']

            noises = torch.randn(
                batch_shape, action_dim, device=next_observations["observation.state"].device
            )
            next_actions, _, _ = self.actor_onestep_flow(next_observations, next_observation_features, noises)
            # next_actions = self.select_action(next_observations)

            next_actions = torch.clamp(next_actions, -1.0, 1.0)

            # 2- compute q targets
            next_qs = self.critic_forward(
                observations=next_observations,
                actions=next_actions,
                use_target=True,
                observation_features=next_observation_features,
                do_output_normalization=False,
            )  # (critic_ensemble_size, batch_size)

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                raise NotImplementedError(
                    "Subsampling critics is not implemented yet. "
                    "Please set num_subsample_critics to None or implement the subsampling logic."
                )
                # indices = torch.randperm(self.config.num_critics)
                # indices = indices[: self.config.num_subsample_critics]
                # next_qs = next_qs[indices]

            # critics subsample size
            if self.config.q_agg == "min":
                next_q, _ = next_qs.min(dim=0)  # Get values from min operation
            else:
                next_q = next_qs.mean(dim=0)

            # if self.config.use_backup_entropy:
            #     min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards.squeeze(-1) + (1 - done) * discounts.squeeze(-1) * next_q
            # td_target = rewards + (1 - done) * self.config.discount * next_q

        # 3- compute predicted qs
        if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :, :DISCRETE_DIMENSION_INDEX]

        # actions = actions[:, 0, :3] # TODO: use all chunks
        # actions = actions * (~actions_is_pad).unsqueeze(-1)
        actions = actions[:, :, :].reshape(actions.shape[0], -1)  # [32, 150]

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
            do_output_normalization=False,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up

        q_preds = q_preds[:, ~actions_is_pad[:, -1]]
        td_target_duplicate = td_target_duplicate[:, ~actions_is_pad[:, -1]]

        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()

        info = {
            # "critic_loss": critics_loss,
            "predicted_qs": torch.mean(q_preds),
            "target_qs": torch.mean(td_target_duplicate),
            "rewards": rewards.mean(),
            "actions_is_pad": torch.mean(actions_is_pad.float()),
            # "discrete_critic_loss": discrete_critic_loss,
            # "discrete_predicted_qs": torch.mean(predicted_discrete_qs),
            # "discrete_target_qs": torch.mean(target_discrete_q_duplicate),
            # "discrete_rewards": rewards_discrete.mean(),
        }

        return critics_loss, info

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        actions_is_pad,
        rewards,
        discounts,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, :, DISCRETE_DIMENSION_INDEX:].clone()

        # Ensure discrete actions are integers and in valid range
        actions_discrete = torch.clamp(actions_discrete, 0, self.config.num_discrete_actions - 1)
        actions_discrete = torch.round(actions_discrete).long()

        # Validate action shape
        expected_shape = (actions.shape[0], actions.shape[1], 1)  # batch_size, chunk_size, 1
        assert actions_discrete.shape == expected_shape, (
            f"Expected discrete actions shape {expected_shape}, got {actions_discrete.shape}"
        )

        with torch.no_grad():
            # Use discrete flow matching for next action sampling with masking
            batch_size = next_observations["observation.state"].shape[0]
            device = next_observations["observation.state"].device
            chunk_size = self.config.chunk_size
            MASK_TOKEN = self.config.num_discrete_actions  # MASK token at index num_discrete_actions

            # Start with masked tokens for discrete actor (one-step flow)
            discrete_tokens = MASK_TOKEN * torch.ones(
                (batch_size, chunk_size), dtype=torch.long, device=device
            )

            # Get next action probabilities using discrete_actor (one-step flow)
            # This outputs logits over valid actions (0 to num_discrete_actions-1), not including MASK
            next_action_logits, next_log_probs, next_action_probs = self.discrete_actor(
                next_observations, next_observation_features, discrete_tokens
            )
            # next_action_logits [batch_size, chunk_size, num_discrete_actions]
            # next_log_probs [batch_size, chunk_size, num_discrete_actions]
            # next_action_probs [batch_size, chunk_size, num_discrete_actions]

            # Sample actions from the distribution for next state Q-value computation
            next_actions_sampled = Categorical(probs=next_action_probs).sample()  # [batch_size, chunk_size]
            
            # Convert to one-hot encoding for discrete critic input
            next_actions_one_hot = F.one_hot(
                next_actions_sampled,
                num_classes=self.config.num_discrete_actions,
            ).float()  # [batch_size, chunk_size, num_discrete_actions]
            
            next_actions_flat = next_actions_one_hot.view(
                batch_size, -1
            )  # [batch_size, chunk_size * num_discrete_actions]

            # Compute next Q-values using sampled actions
            next_q_values = self.discrete_critic_forward(
                observations=next_observations,
                actions=next_actions_flat,
                use_target=True,
                observation_features=next_observation_features,
            )  # (num_critics, batch_size)

            if self.config.num_subsample_critics is not None:
                raise NotImplementedError(
                    "Subsampling critics is not implemented yet. "
                    "Please set num_subsample_critics to None or implement the subsampling logic."
                )

            # Compute expected Q-value under next action distribution
            if self.config.q_agg == "min":
                next_q_min, _ = next_q_values.min(dim=0)
            else:
                next_q_min = next_q_values.mean(dim=0)

            # if self.config.use_backup_entropy:
            #     next_q_min = next_q_min - (self.temperature * next_log_probs)

            td_target = rewards.squeeze(-1) + (1 - done) * discounts.squeeze(-1) * next_q_min

        # Compute current Q-values for discrete actions
        # Convert discrete actions to one-hot for critic input (only for valid actions 0 to num_discrete_actions-1)
        # This maintains consistency with the probabilistic representation used for next actions
        actions_discrete_flat = actions_discrete.view(batch_size, -1)  # (batch_size, chunk_size)
        actions_one_hot = F.one_hot(
            actions_discrete_flat,
            num_classes=self.config.num_discrete_actions,  # Valid actions only
        ).float()
        actions_one_hot = actions_one_hot.view(
            batch_size, -1
        )  # (batch_size, chunk_size * num_discrete_actions)

        # Get current Q-values using the one-hot encoded actual actions
        q_preds = self.discrete_critic_forward(
            observations=observations,
            actions=actions_one_hot,
            use_target=False,
            observation_features=observation_features,
        )  # (num_critics, batch_size)

        # Compute TD loss
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])

        # Apply padding mask to Q-predictions and targets
        if actions_is_pad is not None:
            # Use the last timestep padding mask as overall episode mask
            episode_mask = ~actions_is_pad[:, -1]  # (batch_size,)
            q_preds = q_preds[:, episode_mask]
            td_target_duplicate = td_target_duplicate[:, episode_mask]

        # Compute MSE loss for each critic and sum
        discrete_critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()

        info = {
            "predicted_qs": torch.mean(q_preds),
            "target_qs": torch.mean(td_target_duplicate),
            "rewards": rewards.mean(),
            # "discrete_expected_next_q": torch.mean(expected_next_q_episode),
        }

        return discrete_critics_loss, info

    # def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
    #     """Compute the temperature loss"""
    #     # calculate temperature loss
    #     with torch.no_grad():
    #         discrete_noises = torch.randn(
    #             observations["observation.state"].shape[0],
    #             self.discrete_actor.action_dim,
    #             device=observations["observation.state"].device,
    #         )
    #         _, log_probs, _ = self.discrete_actor(observations, observation_features, discrete_noises)
    #     temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
    #     return temperature_loss

    def compute_loss_actor_bc_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions: Tensor | None,
        actions_is_pad: Tensor | None,
    ) -> Tensor:
        # actions = self.normalize_targets({"action": actions})["action"]
        actions = actions[:, :, :DISCRETE_DIMENSION_INDEX]

        batch_size = actions.shape[0]
        action_dim = self.actor_bc_flow.action_dim  # self.config['action_dim']

        # BC flow loss.
        x_0 = torch.randn(batch_size, action_dim, device=observations["observation.state"].device)
        x_1 = actions  # .clone()  # Use the provided actions as x_1
        x_1 = x_1.reshape(batch_size, -1)  # Flatten the action dimension
        t = torch.rand(batch_size, 1, device=observations["observation.state"].device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        vel = vel.reshape(batch_size, -1, 3)  # Reshape to match action dimensions

        vel_pred, _, _ = self.actor_bc_flow(observations, observation_features, x_t, t)
        vel_pred = vel_pred.reshape(batch_size, actions_is_pad.shape[1], -1)

        bc_flow_loss = F.mse_loss(input=vel_pred, target=vel, reduction="none")  # (128, 10, 3)
        bc_flow_loss = bc_flow_loss * (~actions_is_pad).unsqueeze(-1)
        bc_flow_loss = bc_flow_loss.mean()

        info = {
            # "bc_flow_loss": bc_flow_loss,
            # "discrete_bc_flow_loss": discrete_bc_flow_loss,
        }

        return bc_flow_loss, info

    def compute_loss_actor_onestep_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions: Tensor | None,
        actions_is_pad: Tensor | None,
    ) -> Tensor:
        batch_size = actions.shape[0]
        action_dim = self.actor_onestep_flow.action_dim  # self.config['action_dim']

        # Distillation loss.
        noises = torch.randn(batch_size, action_dim, device=observations["observation.state"].device)
        target_flow_actions = self.compute_flow_actions(observations, noises)
        actor_actions, _, _ = self.actor_onestep_flow(observations, observation_features, noises)
        distill_loss = F.mse_loss(input=actor_actions, target=target_flow_actions)

        # Q loss.
        actor_actions = torch.clamp(actor_actions, -1.0, 1.0)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actor_actions,
            use_target=False,
            observation_features=observation_features,
            do_output_normalization=False,
        )
        # min_q_preds = q_preds.min(dim=0)[0]
        min_q_preds = q_preds.mean(dim=0, keepdim=True)
        q_loss = -min_q_preds.mean()

        if self.config.normalize_q_loss:
            lam = 1.0 / q_preds.abs().mean().detach()
            q_loss = lam * q_loss

        actor_onestep_flow_loss = self.config.alpha * distill_loss + q_loss

        info = {
            "q_loss": q_loss,
            "predicted_qs": torch.mean(q_preds),
            "distill_loss": distill_loss,
            "q": torch.mean(min_q_preds),
            # "discrete_q_loss": discrete_q_loss,
            # "discrete_predicted_qs": torch.mean(discrete_q_preds),
        }

        return actor_onestep_flow_loss, info

    def compute_loss_discrete_bc_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions: Tensor | None,
        actions_is_pad: Tensor | None,
    ) -> Tensor:
        """Compute discrete behavioral cloning flow matching loss with masking exactly following the reference pattern."""
        # Extract discrete actions from the action tensor
        actions_discrete: Tensor = actions[:, :, DISCRETE_DIMENSION_INDEX:].clone()

        # Ensure discrete actions are integers and in valid range (0 to num_discrete_actions-1, where num_discrete_actions is MASK)
        actions_discrete = torch.clamp(actions_discrete, 0, self.config.num_discrete_actions - 1)
        x1 = torch.round(actions_discrete).long().squeeze(-1)  # [batch_size, chunk_size] - this is our target

        batch_size = x1.shape[0]
        chunk_size = x1.shape[1]  # D in reference
        device = observations["observation.state"].device
        S = self.config.num_discrete_actions + 1  # S in reference (valid actions + MASK token)
        MASK_TOKEN = self.config.num_discrete_actions  # MASK token at index num_discrete_actions

        # Sample random time steps - following reference: t = torch.rand((B,))
        t = torch.rand(batch_size, device=device)  # [batch_size]

        # Create corrupted sequence exactly following the reference masking pattern
        xt = x1.clone()  # Start with target actions

        # Create corruption mask: corrupt_mask = torch.rand((B, D)) < (1 - t[:, None])
        corrupt_mask = torch.rand((batch_size, chunk_size), device=device) < (1 - t[:, None])

        # Apply corruption with MASK tokens: xt[corrupt_mask] = MASK_TOKEN
        xt[corrupt_mask] = MASK_TOKEN

        # Predict logits using discrete BC flow actor
        # Pass time with correct shape for the network
        t_expanded = t.unsqueeze(-1)  # [batch_size, 1] for network compatibility
        logits, _, _ = self.discrete_actor_bc_flow(
            observations, observation_features, xt, t_expanded
        )  # [batch_size, chunk_size, num_discrete_actions] -> (B, D, num_discrete_actions)

        # Prepare target for loss computation - following reference pattern
        x1_loss_target = x1.clone()
        # Don't compute loss on dimensions that are already revealed (not corrupted)
        x1_loss_target[xt != MASK_TOKEN] = -1  # Set to ignore_index for non-masked positions

        # Compute cross-entropy loss exactly following reference pattern
        # loss = F.cross_entropy(logits.transpose(1,2), x1_loss_target, reduction='mean', ignore_index=-1)
        bc_flow_loss = F.cross_entropy(
            logits.transpose(
                1, 2
            ),  # [batch_size, num_discrete_actions, chunk_size] -> (B, num_discrete_actions, D)
            x1_loss_target,  # [batch_size, chunk_size] -> (B, D)
            reduction="none",  # We'll handle reduction manually for padding
            ignore_index=-1,
        )  # -> (B, D)

        # Apply action padding mask if provided
        if actions_is_pad is not None:
            # Apply mask to valid timesteps
            chunk_mask = (~actions_is_pad).float()  # [batch_size, chunk_size]
            bc_flow_loss = bc_flow_loss * chunk_mask
            # Average over valid timesteps only
            bc_flow_loss = bc_flow_loss.sum() / chunk_mask.sum().clamp(min=1.0)
        else:
            # Standard mean reduction
            bc_flow_loss = bc_flow_loss.mean()

        info = {
            "corruption_rate": (1 - t).mean(),
            "mask_rate": (xt == MASK_TOKEN).float().mean(),
            "accuracy": (logits.argmax(dim=-1) == x1).float().mean(),
        }

        return bc_flow_loss, info

    def compute_loss_discrete_onestep_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions_is_pad: Tensor | None,
    ) -> Tensor:
        """Compute discrete one-step flow loss with distillation and Q-learning using masking approach."""
        batch_size = observations["observation.state"].shape[0]
        chunk_size = self.config.chunk_size
        device = observations["observation.state"].device
        MASK_TOKEN = self.config.num_discrete_actions  # MASK token at index num_discrete_actions

        # Get target action distribution from BC flow (distillation target)
        target_actions, target_log_probs, target_action_probs = self.compute_discrete_flow_actions(
            observations
        )
        # target_actions: [batch_size, chunk_size] - values in range [0, num_discrete_actions-1]
        # target_log_probs: [batch_size, chunk_size, num_discrete_actions]
        # target_action_probs: [batch_size, chunk_size, num_discrete_actions]

        # Start with masked tokens for one-step prediction
        discrete_tokens = MASK_TOKEN * torch.ones((batch_size, chunk_size), dtype=torch.long, device=device)

        # Predict action distribution using one-step flow (no time input)
        predicted_logits, predicted_log_probs, predicted_action_probs = self.discrete_actor(
            observations, observation_features, discrete_tokens
        )
        # predicted_logits: [batch_size, chunk_size, num_discrete_actions]
        # predicted_log_probs: [batch_size, chunk_size, num_discrete_actions]
        # predicted_action_probs: [batch_size, chunk_size, num_discrete_actions]

        # Distillation loss: match one-step policy to BC flow target distribution
        # Use MSE between action probabilities for stability
        distill_loss = F.mse_loss(predicted_action_probs, target_action_probs)

        # Q loss: maximize expected Q under the predicted discrete action distribution
        # Use straight-through Gumbel-Softmax for differentiable sampling
        tau = getattr(self.config, "gumbel_tau", 1.0)
        predicted_actions_one_hot = F.gumbel_softmax(
            predicted_logits, tau=tau, hard=True, dim=-1
        )  # [batch_size, chunk_size, num_discrete_actions]

        # Zero-out padded timesteps before flattening
        if actions_is_pad is not None:
            predicted_actions_one_hot = predicted_actions_one_hot * (~actions_is_pad).unsqueeze(-1).float()

        # Flatten for critic input to match expected format
        actions_flat = predicted_actions_one_hot.view(
            batch_size, -1
        )  # [batch_size, chunk_size * num_discrete_actions]

        q_preds = self.discrete_critic_forward(
            observations=observations,
            actions=actions_flat,
            use_target=False,
            observation_features=observation_features,
        )
        # Aggregate critics (use mean similar to continuous onestep flow)
        min_q_preds = q_preds.mean(dim=0, keepdim=True)
        q_loss = -min_q_preds.mean()

        if self.config.normalize_q_loss:
            lam = 1.0 / q_preds.abs().mean().detach()
            q_loss = lam * q_loss

        loss = self.config.alpha * distill_loss + q_loss

        info = {
            "q_loss": q_loss,
            "predicted_qs": torch.mean(q_preds),
            "distill_loss": distill_loss,
            "q": torch.mean(min_q_preds),
        }

        return loss, info

    def _init_normalization(self, dataset_stats):
        """Initialize input/output normalization modules."""
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        if self.config.dataset_stats is not None:
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
            self.normalize_inputs = NormalizeBuffer(
                self.config.input_features, self.config.normalization_mapping, params
            )
            stats = dataset_stats or params
            self.normalize_targets = NormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )
            self.unnormalize_targets = UnnormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
        self.encoder_discrete_critic = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )
        self.encoder_actor_bc_flow = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )
        self.encoder_actor_onestep_flow = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )
        self.encoder_discrete_actor_bc_flow = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )
        self.encoder_discrete_actor_onestep_flow = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + (continuous_action_dim) * self.config.chunk_size,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + (continuous_action_dim) * self.config.chunk_size,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        heads = [
            DiscreteCriticHead(
                input_dim=self.encoder_discrete_critic.output_dim
                + self.config.num_discrete_actions * self.config.chunk_size,  # Valid actions only
                **asdict(self.config.discrete_critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.discrete_critic = DiscreteCriticEnsemble(encoder=self.encoder_discrete_critic, ensemble=heads)
        target_heads = [
            DiscreteCriticHead(
                input_dim=self.encoder_discrete_critic.output_dim
                + self.config.num_discrete_actions * self.config.chunk_size,  # Valid actions only
                **asdict(self.config.discrete_critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.discrete_critic_target = DiscreteCriticEnsemble(
            encoder=self.encoder_discrete_critic, ensemble=target_heads
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor_bc_flow(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        params = asdict(self.config.actor_network_kwargs)
        self.actor_bc_flow = ActorVectorFieldPolicy(
            encoder=self.encoder_actor_bc_flow,
            network=MLP(
                input_dim=self.encoder_actor_bc_flow.output_dim
                + (continuous_action_dim) * self.config.chunk_size
                + 1,
                **params,
            ),
            action_dim=(continuous_action_dim) * self.config.chunk_size,
            # num_discrete_actions=self.config.num_discrete_actions* self.config.chunk_size,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = (
                self.config.num_discrete_actions * self.config.chunk_size
                if self.config.num_discrete_actions is not None
                else 0
            )
            self.target_entropy = -np.prod(dim) / 2

        if self.config.num_discrete_actions is not None:
            self._init_discrete_bc_flow()
            self._init_discrete_onestep_flow()

    def _init_actor_onestep_flow(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        params = asdict(self.config.actor_network_kwargs)
        self.actor_onestep_flow = ActorVectorFieldPolicy(
            encoder=self.encoder_actor_onestep_flow,
            network=MLP(
                input_dim=self.encoder_actor_onestep_flow.output_dim
                + (continuous_action_dim) * self.config.chunk_size,
                **params,
            ),
            action_dim=(continuous_action_dim) * self.config.chunk_size,
            # num_discrete_actions=self.config.num_discrete_actions* self.config.chunk_size,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

    def _init_discrete_bc_flow(self):
        """Initialize discrete BC flow matching actor."""
        embedding_dim = 16  # Default embedding dimension following reference
        # Calculate input dimension for the base network
        # base_input_dim = self.encoder_discrete_actor_bc_flow.output_dim

        self.discrete_actor_bc_flow = DiscreteActorVectorFieldPolicy(
            encoder=self.encoder_discrete_actor_bc_flow,
            network=MLP(
                input_dim=self.encoder_discrete_actor_bc_flow.output_dim
                + (embedding_dim) * self.config.chunk_size
                + 1,
                **asdict(self.config.discrete_actor_network_kwargs),
            ),
            action_dim=self.config.num_discrete_actions * self.config.chunk_size,  # Valid actions only
            num_discrete_actions=self.config.num_discrete_actions,
            encoder_is_shared=self.shared_encoder,
            embedding_dim=embedding_dim,
            **asdict(self.config.discrete_policy_kwargs),
        )

    def _init_discrete_onestep_flow(self):
        """Initialize discrete one-step flow matching actor."""
        embedding_dim = 16  # Default embedding dimension following reference
        # Calculate input dimension for the base network
        # base_input_dim = self.encoder_discrete_actor_onestep_flow.output_dim

        self.discrete_actor = DiscreteActorVectorFieldPolicy(
            encoder=self.encoder_discrete_actor_onestep_flow,
            network=MLP(
                input_dim=self.encoder_discrete_actor_bc_flow.output_dim
                + (embedding_dim) * self.config.chunk_size,
                **asdict(self.config.discrete_actor_network_kwargs),
            ),
            action_dim=self.config.num_discrete_actions * self.config.chunk_size,  # Valid actions only
            num_discrete_actions=self.config.num_discrete_actions,
            encoder_is_shared=self.shared_encoder,
            embedding_dim=embedding_dim,
            use_time=False,  # No time input for one-step flow
            **asdict(self.config.discrete_policy_kwargs),
        )

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: FQLConfig, input_normalizer: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.input_normalization = input_normalizer
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return

        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = "observation.environment_state" in self.config.input_features
        self.has_state = "observation.state" in self.config.input_features
        if self.has_env:
            dim = self.config.input_features["observation.environment_state"].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features["observation.state"].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        obs = self.input_normalization(obs)
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs, normalize=False)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs["observation.environment_state"]))
        if self.has_state:
            parts.append(self.state_encoder(obs["observation.state"]))
        if parts:
            return torch.cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor], normalize: bool = False) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.

        This function processes image observations through the vision encoder once and returns
        the resulting features.
        When the image encoder is shared between actor and critics AND frozen, these features can be safely cached and
        reused across policy components (actor, critic, discrete_critic), avoiding redundant forward passes.

        Performance impact:
        - The vision encoder forward pass is typically the main computational bottleneck during training and inference
        - Caching these features can provide 2-4x speedup in training and inference

        Normalization behavior:
        - When called from inside forward(): set normalize=False since inputs are already normalized
        - When called from outside forward(): set normalize=True to ensure proper input normalization

        Usage patterns:
        - Called in select_action() with normalize=True
        - Called in learner.py's get_observation_features() to pre-compute features for all policy components
        - Called internally by forward() with normalize=False

        Args:
            obs: Dictionary of observation tensors containing image keys
            normalize: Whether to normalize observations before encoding
                      Set to True when calling directly from outside the encoder's forward method
                      Set to False when calling from within forward() where inputs are already normalized

        Returns:
            Dictionary mapping image keys to their corresponding encoded features
        """
        if normalize:
            obs = self.input_normalization(obs)
        batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        return dict(zip(self.image_keys, chunks, strict=False))

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.

        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.

        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: The encoded image features.
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.GELU(),
        activate_final: bool = False,
        # dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        layer_norm: bool = False,
        default_init: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            if default_init is not None:
                nn.init.uniform_(layers[-1].weight, -default_init, default_init)
                nn.init.uniform_(layers[-1].bias, -default_init, default_init)
            else:
                orthogonal_init()(layers[-1].weight)
                # nn.init.zeros_(layers[-1].bias)

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                # if dropout_rate and dropout_rate > 0:
                #     layers.append(nn.Dropout(p=dropout_rate))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.GELU(),
        activate_final: bool = False,
        # dropout_rate: float | None = None,
        default_init: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            # dropout_rate=dropout_rate,
            final_activation=final_activation,
            layer_norm=layer_norm,
            default_init=default_init,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)
            # nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        output_normalization (nn.Module): normalization layer for actions.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        output_normalization: nn.Module | None = None,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.output_normalization = output_normalization
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
        do_output_normalization: bool = True,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        # NOTE: We normalize actions it helps for sample efficiency
        actions: dict[str, torch.tensor] = {"action": actions}
        # NOTE: Normalization layer took dict in input and outputs a dict that why
        if do_output_normalization and self.output_normalization is not None:
            actions = self.output_normalization(actions)["action"]
        else:
            actions = actions["action"]
        actions = actions.to(device)

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class DiscreteCriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.GELU(),
        activate_final: bool = False,
        # dropout_rate: float | None = None,
        default_init: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            # dropout_rate=dropout_rate,
            final_activation=final_activation,
            layer_norm=layer_norm,
            default_init=default_init,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)
            # nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class DiscreteCriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        output_normalization (nn.Module): normalization layer for actions.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        # output_normalization: nn.Module,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        # self.output_normalization = output_normalization
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        actions = actions.to(device)

        obs_enc = self.encoder(observations, cache=observation_features)

        # Concatenate observation encoding with discrete actions
        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


# class Policy(nn.Module):
#     def __init__(
#         self,
#         encoder: SACObservationEncoder,
#         network: nn.Module,
#         action_dim: int,
#         std_min: float = -5,
#         std_max: float = 2,
#         fixed_std: torch.Tensor | None = None,
#         init_final: float | None = None,
#         use_tanh_squash: bool = False,
#         encoder_is_shared: bool = False,
#     ):
#         super().__init__()
#         self.encoder: SACObservationEncoder = encoder
#         self.network = network
#         self.action_dim = action_dim
#         self.std_min = std_min
#         self.std_max = std_max
#         self.fixed_std = fixed_std
#         self.use_tanh_squash = use_tanh_squash
#         self.encoder_is_shared = encoder_is_shared

#         # Find the last Linear layer's output dimension
#         for layer in reversed(network.net):
#             if isinstance(layer, nn.Linear):
#                 out_features = layer.out_features
#                 break
#         # Mean layer
#         self.mean_layer = nn.Linear(out_features, action_dim)
#         if init_final is not None:
#             nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
#             nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
#         else:
#             orthogonal_init()(self.mean_layer.weight)

#         # Standard deviation layer or parameter
#         if fixed_std is None:
#             self.std_layer = nn.Linear(out_features, action_dim)
#             if init_final is not None:
#                 nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
#                 nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
#             else:
#                 orthogonal_init()(self.std_layer.weight)

#     def forward(
#         self,
#         observations: torch.Tensor,
#         observation_features: torch.Tensor | None = None,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # We detach the encoder if it is shared to avoid backprop through it
#         # This is important to avoid the encoder to be updated through the policy
#         obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

#         # Get network outputs
#         outputs = self.network(obs_enc)
#         means = self.mean_layer(outputs)

#         # Compute standard deviations
#         if self.fixed_std is None:
#             log_std = self.std_layer(outputs)
#             std = torch.exp(log_std)  # Match JAX "exp"
#             std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
#         else:
#             std = self.fixed_std.expand_as(means)

#         # Build transformed distribution
#         dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

#         # Sample actions (reparameterized)
#         actions = dist.rsample()

#         # Compute log_probs
#         log_probs = dist.log_prob(actions)

#         return actions, log_probs, means

#     def get_features(self, observations: torch.Tensor) -> torch.Tensor:
#         """Get encoded features from observations"""
#         device = get_device_from_parameters(self)
#         observations = observations.to(device)
#         if self.encoder is not None:
#             with torch.inference_mode():
#                 return self.encoder(observations)
#         return observations


class ActorVectorFieldPolicy(nn.Module):
    """
    Actor vector field network for flow matching.

    Args:
        hidden_dims (list[int]): Hidden layer dimensions.
        action_dim (int): Action dimension.
        layer_norm (bool): Whether to apply layer normalization.
        encoder (nn.Module, optional): Optional encoder module to encode the inputs.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break

        self.output_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)
            # nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None,
        actions: torch.Tensor,
        times: torch.Tensor = None,
        # is_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Return the vectors at the given states, actions, and times (optional).

        Args:
            observations (Tensor): Observations.
            actions (Tensor): Actions.
            times (Tensor, optional): Times.
            is_encoded (bool): Whether the observations are already encoded.
        """
        # if not is_encoded and self.encoder is not None:
        #     observations = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        inputs = [obs_enc, actions]
        if times is not None:
            inputs.append(times)
        x = torch.cat(inputs, dim=-1)

        # Get network outputs
        velocity = self.output_layer(self.network(x))
        return (
            velocity,
            None,
            None,
        )  # Return None for log_probs and means as they are not used in this context


# class ActorVectorFieldWithDiscretePolicy(nn.Module):
#     """
#     Actor vector field network for flow matching.

#     Args:
#         hidden_dims (list[int]): Hidden layer dimensions.
#         action_dim (int): Action dimension.
#         layer_norm (bool): Whether to apply layer normalization.
#         encoder (nn.Module, optional): Optional encoder module to encode the inputs.
#     """

#     def __init__(
#         self,
#         encoder: SACObservationEncoder,
#         network: nn.Module,
#         action_dim: int,
#         num_discrete_actions: int,
#         init_final: float | None = None,
#         encoder_is_shared: bool = False,
#     ):
#         super().__init__()
#         self.encoder: SACObservationEncoder = encoder
#         self.network = network
#         self.action_dim = action_dim
#         self.num_discrete_actions = num_discrete_actions
#         self.encoder_is_shared = encoder_is_shared
#         self.embed = nn.Embedding(3, 3)

#         # Find the last Linear layer's output dimension
#         for layer in reversed(network.net):
#             if isinstance(layer, nn.Linear):
#                 out_features = layer.out_features
#                 break

#         self.output_layer = nn.Linear(out_features, action_dim)
#         if init_final is not None:
#             nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
#             nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
#         else:
#             orthogonal_init()(self.output_layer.weight)
#             # nn.init.zeros_(self.output_layer.bias)

#     def forward(
#         self,
#         observations: torch.Tensor,
#         observation_features: torch.Tensor | None,
#         actions: torch.Tensor,
#         times: torch.Tensor = None,
#         # is_encoded: bool = False,
#     ) -> torch.Tensor:
#         """
#         Return the vectors at the given states, actions, and times (optional).

#         Args:
#             observations (Tensor): Observations.
#             actions (Tensor): Actions.
#             times (Tensor, optional): Times.
#             is_encoded (bool): Whether the observations are already encoded.
#         """
#         # if not is_encoded and self.encoder is not None:
#         #     observations = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
#         obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
#         inputs = [obs_enc, actions]
#         # inputs = [obs_enc, actions[:,:30], self.embed(actions[:, 30:].long()).flatten(1,2)]
#         if times is not None:
#             inputs.append(times)
#         x = torch.cat(inputs, dim=-1)

#         # Get network outputs
#         outputs = self.output_layer(self.network(x))

#         continuous_outputs = outputs[:, :-self.num_discrete_actions]

#         discrete_outputs = outputs[:, -self.num_discrete_actions:]

#         discrete_outputs = discrete_outputs.reshape(discrete_outputs.shape[0], -1, 3)

#         policy_dist = Categorical(logits=discrete_outputs)
#         discrete_action = policy_dist.sample()
#         # Action probabilities for calculating the adapted soft-Q loss
#         # discrete_action_probs = policy_dist.probs
#         discrete_action_probs = F.softmax(discrete_outputs, dim=1)
#         # discrete_action_probs = torch.clamp(discrete_action_probs, min=1e-6, max=1.0)
#         discrete_log_prob = F.log_softmax(discrete_outputs, dim=1)
#         # discrete_log_prob = torch.clamp(discrete_log_prob, min=-10, max=0)

#         discrete_action = discrete_action.reshape(discrete_outputs.shape[0], -1)
#         discrete_log_prob = discrete_log_prob.reshape(discrete_outputs.shape[0], -1)
#         discrete_action_probs = discrete_action_probs.reshape(discrete_outputs.shape[0], -1)
#         discrete_outputs = discrete_outputs.reshape(discrete_outputs.shape[0], -1)

#         return continuous_outputs, None, None, discrete_action, discrete_log_prob, discrete_action_probs, discrete_outputs


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: FQLConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


class DiscreteActorVectorFieldPolicy(nn.Module):
    """
    Discrete actor vector field network for flow matching.

    This implements discrete flow matching using the exact pattern from the reference implementation
    with embedding-based token representations and categorical cross-entropy loss.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        num_discrete_actions: int,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
        embedding_dim: int = 16,
        use_time: bool = True,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim  # This is num_discrete_actions * chunk_size
        self.num_discrete_actions = num_discrete_actions
        self.encoder_is_shared = encoder_is_shared
        self.chunk_size = action_dim // num_discrete_actions
        self.embedding_dim = embedding_dim

        # Embedding for discrete tokens (S+1 tokens including special token)
        # Following the reference pattern: S+1 embedding size
        self.token_embedding = nn.Embedding(num_discrete_actions + 1, embedding_dim)

        # Find the last Linear layer's output dimension from the base network
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break

        self.output_layer = nn.Linear(out_features, self.chunk_size * num_discrete_actions)

        # Additional network layers following the reference pattern
        # Input: observation encoding + flattened token embeddings with time
        # Output: logits for each position and discrete action (S*D pattern)
        # self.flow_net = nn.Sequential(
        #     nn.Linear(out_features + self.chunk_size * (self.embedding_dim + int(use_time)), 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.chunk_size * num_discrete_actions)
        # )

        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None,
        discrete_tokens: torch.Tensor,
        times: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for discrete flow matching following the reference pattern.

        Args:
            observations: State observations
            observation_features: Pre-computed observation features
            discrete_tokens: Discrete token sequence [batch_size, chunk_size] with values in [0, num_discrete_actions]
                           (num_discrete_actions can be used for special tokens if needed)
            times: Time steps for flow matching [batch_size, 1]

        Returns:
            logits: Raw logits [batch_size, chunk_size, num_discrete_actions]
            log_probs: Log probabilities [batch_size, chunk_size, num_discrete_actions]
            action_probs: Action probabilities [batch_size, chunk_size, num_discrete_actions]
        """
        batch_size = discrete_tokens.shape[0]

        # Get observation encoding
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # # Process through base network first
        # x = self.network(obs_enc)

        # Embed discrete tokens - following reference pattern (B, D, 16)
        token_emb = self.token_embedding(discrete_tokens)  # [batch_size, chunk_size, embedding_dim]

        inputs = [
            obs_enc,
            token_emb.flatten(1, 2),
        ]  # Flatten token embeddings to [batch_size, chunk_size * embedding_dim]
        if times is not None:
            inputs.append(times)

        x = torch.cat(inputs, dim=-1)

        # Add time embedding if provided - following reference pattern
        # if times is not None:
        #     # Expand time to match token dimensions: t[:, None, None].repeat(1, D, 1)
        #     time_emb = times.unsqueeze(1).repeat(1, self.chunk_size, 1)  # [batch_size, chunk_size, 1]
        #     # Concatenate time with token embeddings
        #     net_input = torch.cat([token_emb, time_emb], dim=-1)  # [batch_size, chunk_size, embedding_dim + 1]
        # else:
        #     net_input = token_emb

        # Flatten for network input following reference: (B, D * 17)
        # net_input_flat = net_input.reshape(batch_size, -1)

        # Combine with observation features
        # combined_input = torch.cat([x, net_input_flat], dim=-1)

        # Pass through flow network
        # logits_flat = self.flow_net(combined_input)
        logits_flat = self.output_layer(self.network(x))

        # Reshape to (B, D, S) following reference pattern
        logits = logits_flat.view(batch_size, self.chunk_size, self.num_discrete_actions)

        # Compute probabilities
        action_probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        return logits, log_probs, action_probs

    # def sample_action(
    #     self,
    #     observations: torch.Tensor,
    #     observation_features: torch.Tensor | None = None,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Sample discrete actions using the learned flow."""
    #     batch_size = observations["observation.state"].shape[0]
    #     device = observations["observation.state"].device

    #     # Sample initial noise in latent space
    #     discrete_latents = torch.randn(batch_size, self.action_dim, device=device)

    #     # Integrate the flow (simplified one-step integration for sampling)
    #     velocity, log_probs, action_probs = self.forward(
    #         observations, observation_features, discrete_latents, times=None
    #     )

    #     # Sample actions from the final probabilities
    #     chunk_size = self.action_dim // self.num_discrete_actions
    #     actions = []
    #     for i in range(chunk_size):
    #         chunk_probs = action_probs[:, i, :]  # [batch_size, num_discrete_actions]
    #         chunk_dist = Categorical(probs=chunk_probs)
    #         chunk_action = chunk_dist.sample()  # [batch_size]
    #         actions.append(chunk_action)

    #     # Stack to get [batch_size, chunk_size]
    #     sampled_actions = torch.stack(actions, dim=1)

    #     return sampled_actions, log_probs, action_probs

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class DiscretePolicy(nn.Module):
    """Legacy discrete policy for backward compatibility."""

    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # logits layer
        self.output_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        logits = self.output_layer(outputs)

        logits = logits.view(logits.shape[0], -1, 3)  # Reshape to [batch_size, num_actions, action_dim]
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=2)

        return action, log_prob, action_probs

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: FQLConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: FQLConfig):
        """Set up CNN encoder"""
        from transformers import AutoModel

        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # Mode is mean of base distribution, passed through transforms

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x


def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params
