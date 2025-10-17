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

from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.acfql.configuration_acfql import ACFQLConfig, is_image_feature
from lerobot.policies.normalize import NormalizeBuffer, UnnormalizeBuffer
from lerobot.policies.octo.modeling_octo import OctoPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters


class ACFQLPolicy(
    PreTrainedPolicy,
):
    config_class = ACFQLConfig
    name = "acfql"

    def __init__(
        self,
        config: ACFQLConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # # queues are populated during rollout of the policy, they contain the n latest observations and actions
        # self._queues = None

        # Determine action dimension and initialize all components
        action_dim = config.output_features["action"].shape[0]
        self._init_normalization(dataset_stats)
        self._init_octo_policy()
        self._init_encoders()
        self._init_critics(action_dim)
        self._init_actor_bc_flow(action_dim)
        self._init_actor_onestep_flow(action_dim)

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
        }

        return optim_params

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def compute_flow_actions(
        self, observations, observations_features, noises: Tensor, action_embeddings
    ) -> Tensor:
        actions = noises
        flow_steps = self.config.flow_steps

        # Euler method.
        for i in range(flow_steps):
            t_val = float(i) / flow_steps
            t = torch.full((actions.shape[0], 1), t_val, device=noises.device)
            vels = self.actor_bc_flow(
                observations, observations_features, actions, t, action_embeddings=action_embeddings
            )
            actions = actions + vels / flow_steps

        actions = torch.clamp(actions, -1.0, 1.0)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""

        observations_features = batch["observation_feature"]
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            batch_shape = batch["observation.state"].shape[0]
            action_dim = self.actor_onestep_flow.action_dim
            device = batch["observation.state"].device

            # Generate actions using distill-ddpg approach
            noises = torch.randn(batch_shape, action_dim, device=device)
            actions = self.actor_onestep_flow(batch, observations_features, noises)
            actions = torch.clamp(actions, -1.0, 1.0)

            # Reshape actions for chunking: [batch_size, chunk_size, action_dim_per_step]
            action_dim_per_step = action_dim // self.config.chunk_size
            actions = actions.reshape(batch_shape, self.config.chunk_size, action_dim_per_step)

            # Unnormalize actions
            actions = self.unnormalize_targets({"action": actions})["action"]

            # Add actions to queue (transpose to get [chunk_size, batch_size, action_dim_per_step])
            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        actions = self._action_queue.popleft()

        return actions

    @torch.no_grad()
    def select_action_chunk(self, observations: dict[str, Tensor]) -> Tensor:
        """Select a full action chunk for QC-FQL open-loop execution.

        Returns:
            Tensor: Action chunk of shape [chunk_size, action_dim_per_step]
        """
        observations = observations
        observations_features = None

        batch_shape = observations["observation.state"].shape[0]
        action_dim = self.actor_onestep_flow.action_dim
        device = observations["observation.state"].device

        # Generate actions using one-step flow actor
        noises = torch.randn(batch_shape, action_dim, device=device)
        actions = self.actor_onestep_flow(observations, observations_features, noises)
        actions = torch.clamp(actions, -1.0, 1.0)

        # Reshape actions for chunking: [batch_size, chunk_size, action_dim_per_step]
        action_dim_per_step = action_dim // self.config.chunk_size
        actions = actions.reshape(batch_shape, self.config.chunk_size, action_dim_per_step)

        return actions[0]

    @torch.no_grad()
    def select_action_with_embedding(self, observations: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Select a full action chunk for QC-FQL open-loop execution and return embedding for recording

        Returns:
            Tensor: Action chunk of shape [chunk_size, action_dim_per_step]
        """
        self.eval()
        observations = observations
        observations_features = None

        batch_shape = observations["observation.state"].shape[0]
        action_dim = self.actor_onestep_flow.action_dim
        device = observations["observation.state"].device

        # Generate actions using one-step flow actor
        noises = torch.randn(batch_shape, action_dim, device=device)
        actions, embedding = self.actor_onestep_flow(
            observations, observations_features, noises, return_action_embedding=True
        )
        actions = torch.clamp(actions, -1.0, 1.0)

        # Reshape actions for chunking: [batch_size, chunk_size, action_dim_per_step]
        action_dim_per_step = action_dim // self.config.chunk_size
        actions = actions.reshape(batch_shape, self.config.chunk_size, action_dim_per_step)

        return actions[0], embedding

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
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
        q_values = critics(observations, actions, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor_bc_flow", "actor_onestep_flow", "critic", "total"] = "critic",
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
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")
        valid: Tensor = batch["valid"]
        action_embeddings = batch.get("action_embeddings")
        next_action_embeddings = batch.get("next_action_embeddings")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["mask"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic, info = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                valid=valid,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                next_action_embeddings=next_action_embeddings,
            )

            return {"loss_critic": loss_critic, "info": info}

        if model == "actor_bc_flow":
            loss_actor_bc_flow, info = self.compute_loss_actor_bc_flow(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                valid=valid,
                action_embeddings=action_embeddings,
            )
            return {"loss_actor_bc_flow": loss_actor_bc_flow, "info": info}
        if model == "actor_onestep_flow":
            loss_actor_onestep_flow, info = self.compute_loss_actor_onestep_flow(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                action_embeddings=action_embeddings,
            )
            return {"loss_actor_onestep_flow": loss_actor_onestep_flow, "info": info}

        if model == "total":
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["mask"]
            valid: Tensor = batch["valid"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_total, info = self.compute_total_loss(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                valid=valid,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )
            return {"loss_total": loss_total, "info": info}

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

    def compute_total_loss(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        valid,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ):
        # critic
        loss_c, info_c = self.compute_loss_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            valid=valid,
            observation_features=observation_features,
            next_observation_features=next_observation_features,
        )

        # actor = (BC-flow) + (one-step distill + Q)
        loss_bc, info_bc = self.compute_loss_actor_bc_flow(
            observations=observations,
            observation_features=observation_features,
            actions=actions,
            valid=valid,
        )
        loss_one, info_one = self.compute_loss_actor_onestep_flow(
            observations=observations,
            observation_features=observation_features,
            actions=actions,
        )

        loss = loss_c + loss_bc + loss_one

        info = {}
        info.update(
            {f"critic/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in info_c.items()}
        )
        info.update(
            {f"actor_bc/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in info_bc.items()}
        )
        info.update(
            {f"actor_one/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in info_one.items()}
        )
        info["total_loss"] = loss.item()

        return loss, info

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        valid,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
        next_action_embeddings: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            # Compute next actions
            next_actions = self._compute_next_actions(
                next_observations, next_observation_features, next_action_embeddings=next_action_embeddings
            )

            # Compute Q-values for these actions
            next_qs = self.critic_forward(
                observations=next_observations,
                actions=next_actions,
                use_target=True,
                observation_features=next_observation_features,
            )  # (critic_ensemble_size, batch_size)

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                raise NotImplementedError(
                    "Subsampling critics is not implemented yet. "
                    "Please set num_subsample_critics to None or implement the subsampling logic."
                )

            # critics ensemble aggregation (min or mean)
            if self.config.q_agg == "min":
                next_q, _ = next_qs.min(dim=0)  # Get values from min operation
            else:
                next_q = next_qs.mean(dim=0)

            h = self.config.chunk_size
            gamma_h = self.config.discount**h
            bootstrap_mask = done[:, -1].squeeze(-1)
            td_target = rewards[:, -1] + gamma_h * bootstrap_mask * next_q

        # 3- compute predicted qs
        actions = actions[:, :, :].reshape(actions.shape[0], -1)  # [32, 150]

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up

        # # TD loss
        td_loss = (((q_preds - td_target_duplicate) ** 2) * valid[:, -1]).mean(dim=1).sum()

        # Total critic loss
        critics_loss = td_loss

        info = {
            "critic_loss": critics_loss,
            "td_loss": td_loss,
            "predicted_qs": torch.mean(q_preds),
            "target_qs": torch.mean(td_target_duplicate),
            "rewards": rewards.mean(),
        }

        return critics_loss, info

    def compute_loss_actor_bc_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions: Tensor | None,
        valid: Tensor | None,
        action_embeddings: Tensor | None = None,
    ) -> Tensor:
        batch_size = actions.shape[0]
        action_dim = self.actor_bc_flow.action_dim

        # BC flow loss - action chunking version
        x_0 = torch.randn(batch_size, action_dim, device=observations["observation.state"].device)
        x_1 = actions.reshape(
            batch_size, -1
        )  # Flatten the action dimension [batch_size, chunk_size * action_dim]
        t = torch.rand(batch_size, 1, device=observations["observation.state"].device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        vel_pred = self.actor_bc_flow(
            observations, observation_features, x_t, t, action_embeddings=action_embeddings
        )

        # Reshape to match action chunking structure
        vel_pred = vel_pred.reshape(batch_size, self.config.chunk_size, -1)
        vel = vel.reshape(batch_size, self.config.chunk_size, -1)

        bc_flow_loss = (((vel_pred - vel) ** 2) * valid[..., None]).mean()

        info = {
            "bc_flow_loss": bc_flow_loss,
        }

        return bc_flow_loss, info

    def compute_loss_actor_onestep_flow(
        self,
        observations,
        observation_features: Tensor | None,
        actions: Tensor | None,
        action_embeddings: Tensor | None = None,
    ) -> Tensor:
        import time

        batch_size = actions.shape[0]
        action_dim = self.actor_onestep_flow.action_dim

        # Distillation loss
        noises = torch.randn(batch_size, action_dim, device=observations["observation.state"].device)

        start_time = time.time()
        target_flow_actions = self.compute_flow_actions(
            observations, observation_features, noises, action_embeddings
        )
        print(f"[PROFILING] compute_flow_actions: {(time.time() - start_time) * 1000:.2f}ms")

        start_time = time.time()
        actor_actions = self.actor_onestep_flow(
            observations, observation_features, noises, action_embeddings=action_embeddings
        )
        print(f"[PROFILING] actor_onestep_flow forward: {(time.time() - start_time) * 1000:.2f}ms")

        distill_loss = F.mse_loss(input=actor_actions, target=target_flow_actions)

        # Q loss
        actor_actions = torch.clamp(actor_actions, -1.0, 1.0)

        start_time = time.time()
        q_preds = self.critic_forward(
            observations=observations,
            actions=actor_actions,
            use_target=False,
            observation_features=observation_features,
        )
        print(f"[PROFILING] critic_forward: {(time.time() - start_time) * 1000:.2f}ms")

        q_vals = q_preds.mean(dim=0)
        q_loss = -q_vals.mean()

        # # TODO (jpizarrom): make this configurable
        # lam = 1.0 / q_preds.abs().mean().detach()
        # q_loss = lam * q_loss

        # Total loss: alpha * distillation + q_loss
        actor_onestep_flow_loss = self.config.alpha * distill_loss + q_loss

        info = {
            "q_loss": q_loss,
            "predicted_qs": torch.mean(q_preds),
            "distill_loss": distill_loss,
            "q": torch.mean(q_vals),
        }

        return actor_onestep_flow_loss, info

    def _compute_next_actions(
        self,
        next_observations: dict[str, Tensor],
        next_observation_features: Tensor | None = None,
        next_action_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute next actions for target Q-value calculation.

        Similar to JAX _compute_next_actions but adapted for flow-based policies.
        """
        batch_size = next_observations["observation.state"].shape[0]
        action_dim = self.actor_onestep_flow.action_dim
        device = next_observations["observation.state"].device

        all_noises = torch.randn(batch_size, action_dim, device=device)

        next_actions = self.actor_onestep_flow(
            next_observations, next_observation_features, all_noises, action_embeddings=next_action_embeddings
        )
        next_actions = torch.clamp(next_actions, -1.0, 1.0)

        return next_actions

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

    def _init_octo_policy(self):
        """Initialize Octo VLA policy."""
        # ConRFT always requires an Octo model
        self.octo_policy = OctoPolicy.from_pretrained(self.config.base_vla_model_path)
        if self.config.freeze_base_vla:
            for param in self.octo_policy.parameters():
                param.requires_grad = False

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
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

        self.encoder_actor_bc_flow = OctoEncodingWrapper(
            self.octo_policy,
            use_proprio=self.config.use_proprio,
            state_dim=self.config.state_dim,
            proprio_latent_dim=self.config.proprio_latent_dim,
        )

        self.encoder_actor_onestep_flow = OctoEncodingWrapper(
            self.octo_policy,
            use_proprio=self.config.use_proprio,
            state_dim=self.config.state_dim,
            proprio_latent_dim=self.config.proprio_latent_dim,
        )

    def _init_critics(self, action_dim):
        """Build critic ensemble and targets"""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + action_dim * self.config.chunk_size,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + action_dim * self.config.chunk_size,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # if self.config.use_torch_compile:
        #     self.critic_ensemble = torch.compile(self.critic_ensemble)
        #     self.critic_target = torch.compile(self.critic_target)

    def _init_actor_bc_flow(self, action_dim):
        """Initialize policy actor network and default target entropy."""
        self.actor_bc_flow = ActorVectorFieldPolicy(
            encoder=self.encoder_actor_bc_flow,
            network=MLP(
                input_dim=self.encoder_actor_bc_flow.output_dim
                + action_dim * self.config.chunk_size
                + 1,  # add one for time
                **asdict(self.config.actor_network_kwargs),
            ),
            action_dim=action_dim * self.config.chunk_size,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        # if self.config.use_torch_compile:
        #     self.actor_bc_flow= torch.compile(self.actor_bc_flow)

    def _init_actor_onestep_flow(self, action_dim):
        """Initialize policy actor network and default target entropy."""
        self.actor_onestep_flow = ActorVectorFieldPolicy(
            encoder=self.encoder_actor_onestep_flow,
            network=MLP(
                input_dim=self.encoder_actor_onestep_flow.output_dim + action_dim * self.config.chunk_size,
                **asdict(self.config.actor_network_kwargs),
            ),
            action_dim=action_dim * self.config.chunk_size,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        # if self.config.use_torch_compile:
        #     self.actor_onestep_flow = torch.compile(self.actor_onestep_flow)


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: ACFQLConfig, input_normalizer: nn.Module) -> None:
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

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        # Initialize spatial embeddings for each image key separately to handle different sizes
        for key in self.image_keys:
            name = key.replace(".", "_")
            # Create dummy input with the specific shape for this image key
            dummy = torch.zeros(1, *self.config.input_features[key].shape)
            with torch.no_grad():
                encoded_dummy = self.image_encoder(dummy)
                _, channels, height, width = encoded_dummy.shape

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

        # Process each image separately to handle different sizes
        cached_features = {}
        for key in self.image_keys:
            image_tensor = obs[key]
            encoded_features = self.image_encoder(image_tensor)
            cached_features[key] = encoded_features

        return cached_features

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
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) optionally add normalization, and activation
            if not is_last or activate_final:
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
            final_activation=final_activation,
            layer_norm=layer_norm,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x)).squeeze(-1)


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
        output_normalization: nn.Module | None,
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
    ) -> torch.Tensor:
        import time

        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        # # NOTE: We normalize actions it helps for sample efficiency
        # actions: dict[str, torch.tensor] = {"action": actions}
        # # NOTE: Normalization layer took dict in input and outputs a dict that why
        # actions = self.output_normalization(actions)["action"]
        actions = actions.to(device)

        start_time = time.time()
        obs_enc = self.encoder(observations, cache=observation_features)
        print(f"[PROFILING] CriticEnsemble encoder: {(time.time() - start_time) * 1000:.2f}ms")

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        start_time = time.time()
        for critic in self.critics:
            q_values.append(critic(inputs))
        print(f"[PROFILING] CriticEnsemble critic heads: {(time.time() - start_time) * 1000:.2f}ms")

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


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
        encoder,
        network: nn.Module,
        action_dim: int,
        init_final: float | None = None,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: OctoEncodingWrapper = encoder
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
            nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None,
        actions: torch.Tensor,
        times: torch.Tensor = None,
        tasks: dict[str, Tensor] | None = None,
        action_embeddings: Tensor | None = None,
        return_action_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Return the vectors at the given states, actions, and times (optional).

        Args:
            observations (Tensor): Observations.
            actions (Tensor): Actions.
            times (Tensor, optional): Times.
            is_encoded (bool): Whether the observations are already encoded.
        """
        # obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)
        obs_enc, action_embeddings = self.encoder(
            observations, tasks=tasks, action_embeddings=action_embeddings
        )
        inputs = [obs_enc, actions]
        if times is not None:
            inputs.append(times)
        x = torch.cat(inputs, dim=-1)

        outputs = self.output_layer(
            self.network(x)
        )  # TODO(lilkm): there is no layer norm here, matching JAX implementation

        if return_action_embedding:
            return outputs, action_embeddings

        return outputs


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: ACFQLConfig):
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


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class OctoEncodingWrapper(nn.Module):
    """Wrapper around Octo transformer to extract action embeddings for ConRFT."""

    def __init__(
        self,
        octo_policy: OctoPolicy,
        use_proprio: bool = True,
        state_dim: int = 18,
        proprio_latent_dim: int = 64,
    ):
        super().__init__()
        self.octo_policy = octo_policy
        self.octo_transformer = self.octo_policy.model.octo_transformer
        self.text_processor = self.octo_policy.text_processor
        self.use_proprio = use_proprio
        self.state_dim = state_dim
        self.proprio_latent_dim = proprio_latent_dim
        self._compute_output_dim()

        # Create proprioception encoder if needed
        self.proprio_encoder = None
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(state_dim, self.proprio_latent_dim),
                nn.LayerNorm(self.proprio_latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        """Compute output dimension based on action embeddings and proprioception."""
        # Get the embedding dimension from the Octo model's config
        out = 0
        embedding_dim = self.octo_policy.config.token_embedding_size
        out = embedding_dim
        if self.use_proprio:
            out += self.proprio_latent_dim
        self._out_dim = out

    def get_cached_action_embeddings(
        self, observations: dict[str, Tensor], tasks: dict[str, Tensor] | None = None
    ) -> dict[str, Tensor]:
        """Extract and cache action embeddings from Octo transformer.
        This function processes observations through the Octo transformer once and returns
        the resulting action embeddings. When the Octo model is frozen, these embeddings can be safely cached and
        reused across policy components, avoiding redundant forward passes.
        Args:
            observations: Dictionary of observation tensors
            normalize: Whether to normalize observations before encoding (currently unused for Octo)
        Returns:
            Tensor containing the cached action embeddings
        """

        # Get batch size from observations
        batch_size = next(iter(observations.values())).shape[0]
        # Create empty tasks for the entire batch
        # raw_tasks = tasks["language_instruction"]
        raw_tasks = ["Pick the pink cube up."] * batch_size

        # Prepare batch in Octo format with proper batch size
        prepared_batch = self.octo_policy._prepare_batch(observations, raw_tasks=raw_tasks)
        obs, task_dict, _, _, timestep_pad_mask = prepared_batch

        # Get transformer outputs
        transformer_outputs = self.octo_transformer(obs, task_dict, timestep_pad_mask)

        # Extract action embeddings (readout_action tokens)
        action_embeddings = transformer_outputs["readout_action"]  # TimestepGroup object

        # Extract the actual tensor from TimestepGroup
        # TimestepGroup has .tokens attribute containing the tensor
        if hasattr(action_embeddings, "tokens"):
            action_embeddings = action_embeddings.tokens

        # Mean over tokens and take last timestep
        action_embeddings = action_embeddings.mean(dim=-2)  # Mean over tokens
        action_embeddings = action_embeddings[:, -1, :]  # Take last timestep

        return action_embeddings

    def forward(
        self,
        observations: dict[str, Tensor],
        tasks: dict[str, Tensor] | None = None,
        action_embeddings: Tensor | None = None,
        stop_gradient: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Extract action embeddings from Octo transformer and concatenate with proprioception"""
        if action_embeddings is None:
            # Get batch size from observations
            batch_size = next(iter(observations.values())).shape[0]

            # Prepare batch in Octo format with proper batch size
            if tasks and "language_instruction" in tasks:
                raw_tasks = tasks["language_instruction"]
            else:
                # Create empty tasks for the entire batch
                raw_tasks = ["Pick the pink cube up."] * batch_size

            prepared_batch = self.octo_policy._prepare_batch(observations, raw_tasks=raw_tasks)
            obs, task_dict, _, _, timestep_pad_mask = prepared_batch

            # TODO(lilkm): add masking when training
            # # Apply masking to wrist image when not stopping gradient (like JAX)
            # if not stop_gradient:
            #     # Create mask with 20% probability of masking wrist image
            #     mask_prob = 0.2
            #     mask = torch.rand(batch_size, device=device) < mask_prob
            #     # Expand mask to match image_wrist dimensions
            #     mask_expanded = mask.view(batch_size, 1, 1, 1, 1)
            #     image_wrist = torch.where(mask_expanded, torch.zeros_like(image_wrist), image_wrist)

            # Get transformer outputs
            transformer_outputs = self.octo_transformer(obs, task_dict, timestep_pad_mask)

            # Extract action embeddings (readout_action tokens)
            action_embeddings = transformer_outputs["readout_action"]  # TimestepGroup object

            # Extract the actual tensor from TimestepGroup
            # TimestepGroup has .tokens attribute containing the tensor
            if hasattr(action_embeddings, "tokens"):
                action_embeddings = action_embeddings.tokens

            # TODO(lilkm): check this
            # Mean over tokens and take last timestep like JAX
            action_embeddings = action_embeddings.mean(dim=-2)  # Mean over tokens
            action_embeddings = action_embeddings[:, -1, :]  # Take last timestep

            # # Flatten to [batch_size, embedding_dim] for consistency policy
            # # action_embeddings shape: [batch_size, horizon, n_tokens, embedding_dim]
            # # We want [batch_size, embedding_dim], so take first timestep and first token
            # if action_embeddings.dim() == 4:
            #     action_embeddings = action_embeddings[:, 0, 0, :]  # Take first timestep, first token
            # elif action_embeddings.dim() == 3:
            #     action_embeddings = action_embeddings.squeeze(1)  # Remove window dimension

        encoded = action_embeddings

        if stop_gradient:
            action_embeddings = action_embeddings.detach()

        # Add proprioception
        if self.use_proprio and "observation.state" in observations:
            state = observations["observation.state"]

            # TODO(lilkm): implement state stacking
            # # Handle state stacking like JAX
            # if self.enable_stacking:
            #     import einops
            #     # Combine stacking and channels into a single dimension
            #     if len(state.shape) == 2:
            #         state = einops.rearrange(state, "T C -> (T C)")
            #         # If encoded is 1D, we need to handle it
            #         if len(encoded.shape) == 1:
            #             encoded = encoded.unsqueeze(0)
            #     elif len(state.shape) == 3:
            #         state = einops.rearrange(state, "B T C -> B (T C)")

            state_encoded = self.proprio_encoder(state)
            encoded = torch.cat([encoded, state_encoded], dim=-1)

        return encoded, action_embeddings

    @property
    def output_dim(self) -> int:
        return self._out_dim


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: ACFQLConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: ACFQLConfig):
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


def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params
