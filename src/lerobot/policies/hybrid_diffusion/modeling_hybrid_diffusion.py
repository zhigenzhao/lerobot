#!/usr/bin/env python

# Copyright 2025 Zhigen Zhao (zhaozhigen@gmail.com)
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
"""
Hybrid Diffusion Policy for mixed continuous/discrete action spaces.

This implementation uses a two-stage training process:
1. Stage 1: Train VAE for discrete actions (first n_vae_training_steps)
2. Stage 2: Train diffusion model on concatenated continuous + latent actions
"""

import copy
import warnings
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.hybrid_diffusion.diffusion_components import HybridDiffusionModel
from lerobot.policies.hybrid_diffusion.configuration_hybrid_diffusion import HybridDiffusionConfig
from lerobot.policies.hybrid_diffusion.vae_action_encoder import DiscreteActionVAE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)


class HybridDiffusionPolicy(PreTrainedPolicy):
    """
    Hybrid Diffusion Policy that handles mixed continuous and discrete action spaces.

    This policy automatically handles datasets containing:
    - action.continuous: Continuous actions (e.g., joint positions, velocities)
    - action.discrete: Discrete actions (e.g., contact states, mode switches)

    The training process has two stages:
    1. VAE Training: Train a VAE to encode discrete actions into a continuous latent space
    2. Diffusion Training: Train a diffusion model on concatenated continuous actions + VAE latents

    During inference, continuous actions are used directly while discrete actions are
    decoded from the VAE latent space and binarized.
    """

    config_class = HybridDiffusionConfig
    name = "hybrid_diffusion"

    def __init__(
        self,
        config: HybridDiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration instance.
            dataset_stats: Dataset statistics for normalization. If not provided,
                they should be loaded via load_state_dict before using the policy.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Use concatenated action format with configured dimensions
        self.has_continuous = config.continuous_action_dim > 0
        self.has_discrete = config.discrete_action_dim > 0

        if not self.has_continuous and not self.has_discrete:
            raise ValueError("At least one of continuous_action_dim or discrete_action_dim must be > 0")

        # Get action dimensions from config
        self.continuous_dim = config.continuous_action_dim
        self.discrete_dim = config.discrete_action_dim
        self.total_action_dim = self.continuous_dim + self.discrete_dim

        # Initialize normalizers
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Initialize VAE for discrete actions (if present)
        self.vae = None
        self.vae_trained = False
        if self.has_discrete:
            self.vae = DiscreteActionVAE(
                discrete_dim=self.discrete_dim,
                latent_dim=config.vae_latent_dim,
                hidden_dims=config.vae_hidden_dims,
                horizon=config.horizon,
                eps=1e-8,
            )

        # Calculate total dimension for diffusion model
        self.diffusion_action_dim = self.continuous_dim
        if self.has_discrete:
            self.diffusion_action_dim += config.vae_latent_dim

        # Create modified config for diffusion model with updated action dimension
        self.diffusion_config = copy.deepcopy(config)
        self.diffusion_config.output_features = {
            ACTION: PolicyFeature(FeatureType.ACTION, shape=(self.diffusion_action_dim,))
        }

        # Initialize diffusion model
        self.diffusion = HybridDiffusionModel(self.diffusion_config)

        # Training state
        self.training_step = 0

        # Action queues for inference
        self._queues = None
        self.reset()

    def reset(self):
        """Clear observation and action queues. Should be called on env.reset()"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def _split_action(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """
        Split concatenated action into continuous and discrete parts.

        Args:
            action: Concatenated action tensor [..., total_action_dim]

        Returns:
            continuous_action: [..., continuous_dim]
            discrete_action: [..., discrete_dim]
        """
        continuous_action = action[..., :self.continuous_dim] if self.has_continuous else None
        discrete_action = action[..., self.continuous_dim:] if self.has_discrete else None
        return continuous_action, discrete_action

    def _combine_action(self, continuous_action: Tensor | None, discrete_action: Tensor | None) -> Tensor:
        """
        Combine continuous and discrete actions into concatenated format.

        Args:
            continuous_action: [..., continuous_dim] or None
            discrete_action: [..., discrete_dim] or None

        Returns:
            action: Concatenated action tensor [..., total_action_dim]
        """
        actions = []
        if self.has_continuous and continuous_action is not None:
            actions.append(continuous_action)
        if self.has_discrete and discrete_action is not None:
            actions.append(discrete_action)
        return torch.cat(actions, dim=-1)

    def _denormalize_discrete_actions(self, discrete_action: Tensor) -> Tensor:
        """
        Denormalize discrete actions from [-1,1] back to [0,1] for BCE loss.

        The dataset uses MIN_MAX normalization which maps [0,1] → [-1,1].
        We need to reverse this for the discrete part: [-1,1] → [0,1]

        Args:
            discrete_action: Normalized discrete actions in [-1,1] range

        Returns:
            Denormalized discrete actions in [0,1] range
        """
        # Reverse MIN_MAX normalization: x_norm = 2 * (x - min) / (max - min) - 1
        # Therefore: x = (x_norm + 1) * (max - min) / 2 + min
        # For binary data: min=0, max=1, so: x = (x_norm + 1) / 2
        return (discrete_action + 1.0) / 2.0

    def _renormalize_discrete_actions(self, discrete_action: Tensor) -> Tensor:
        """
        Renormalize discrete actions from [0,1] back to [-1,1] to match dataset format.

        This is the reverse of _denormalize_discrete_actions, applied after VAE decoding
        to ensure discrete actions match the expected dataset normalization.

        Args:
            discrete_action: Discrete actions in [0,1] range (after VAE decoding)

        Returns:
            Renormalized discrete actions in [-1,1] range
        """
        # Reverse of denormalization: x = (x_norm + 1) / 2
        # Therefore: x_norm = 2 * x - 1
        return 2.0 * discrete_action - 1.0

    def get_optim_params(self) -> list[dict]:
        """Get optimizer parameter groups for two-stage training.

        Following VQBet's approach: include all parameters from the start with different
        learning rates. The scheduler will handle the transition between stages.
        """
        param_groups = []

        # VAE parameters (if present) - separate group with different LR
        if self.has_discrete:
            param_groups.append({
                "params": list(self.vae.parameters()),
                "lr": self.config.vae_lr,
                "weight_decay": 0.0,  # No weight decay for VAE
            })

        # Diffusion model parameters - main group
        param_groups.append({
            "params": list(self.diffusion.parameters()),
            # Uses default LR from optimizer config
        })

        return param_groups

    def _prepare_diffusion_input(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Prepare input for diffusion model by concatenating continuous actions and VAE latents.

        Args:
            batch: Batch containing concatenated action

        Returns:
            combined_actions: Concatenated tensor for diffusion training (continuous + VAE latents)
        """
        # Split concatenated action
        action = batch["action"]
        continuous_action, discrete_action = self._split_action(action)

        diffusion_input = []

        if self.has_continuous:
            diffusion_input.append(continuous_action)

        if self.has_discrete:
            # Denormalize discrete actions before VAE encoding
            discrete_action_denorm = self._denormalize_discrete_actions(discrete_action)

            if self.training_step >= self.config.n_vae_training_steps:
                # Use frozen VAE encoder during diffusion training
                with torch.no_grad():
                    latent, _, _ = self.vae.encode(discrete_action_denorm)
            else:
                # During VAE training, use current encoder (with gradients)
                latent, _, _ = self.vae.encode(discrete_action_denorm)

            diffusion_input.append(latent)

        return torch.cat(diffusion_input, dim=-1)

    def _split_diffusion_output(self, combined_actions: Tensor) -> dict[str, Tensor]:
        """
        Split diffusion output back into continuous and discrete actions, then recombine.

        Args:
            combined_actions: Output from diffusion model [continuous + VAE latents]

        Returns:
            actions: Dictionary with single concatenated action key
        """
        idx = 0
        continuous_actions = None
        discrete_actions = None

        if self.has_continuous:
            continuous_actions = combined_actions[..., idx:idx + self.continuous_dim]
            idx += self.continuous_dim

        if self.has_discrete:
            latent = combined_actions[..., idx:idx + self.config.vae_latent_dim]
            # Decode latent to discrete actions
            discrete_actions = self.vae.decode(latent)
            # Apply sigmoid to get probabilities in [0,1] range
            discrete_actions = torch.sigmoid(discrete_actions)
            # Renormalize back to [-1,1] to match dataset format
            discrete_actions = self._renormalize_discrete_actions(discrete_actions)

        # Recombine into single action format to match dataset
        action = self._combine_action(continuous_actions, discrete_actions)
        return {"action": action}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""
        batch = self.normalize_inputs(batch)
        batch = dict(batch)  # Shallow copy

        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        batch = self.normalize_targets(batch)

        # Stage 1: Train VAE for discrete actions
        if self.has_discrete and self.training_step < self.config.n_vae_training_steps:
            # Split concatenated action to get discrete part
            action = batch["action"]
            _, discrete_action = self._split_action(action)

            # Denormalize discrete actions from [-1,1] back to [0,1] for BCE loss
            discrete_action = self._denormalize_discrete_actions(discrete_action)

            loss, recon_loss, kl_loss = self.vae.compute_loss(discrete_action, self.config.vae_beta)

            self.training_step += 1

            return loss, {
                "vae_total_loss": loss.item(),
                "vae_recon_loss": recon_loss.item(),
                "vae_kl_loss": kl_loss.item(),
            }

        # Print transition message (parameters are controlled by scheduler now)
        if self.has_discrete and not self.vae_trained and self.training_step >= self.config.n_vae_training_steps:
            self.vae_trained = True
            print("VAE training completed. Switching to diffusion training.")

        # Stage 2: Train diffusion model
        # Prepare combined actions for diffusion
        combined_actions = self._prepare_diffusion_input(batch)
        batch[ACTION] = combined_actions

        # Standard diffusion training
        loss = self.diffusion.compute_loss(batch)
        self.training_step += 1

        return loss, {"diffusion_loss": loss.item()}

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Generate action chunk using diffusion model."""
        # Prepare batch for diffusion
        prepared_batch = dict(batch)

        # Generate combined actions via diffusion
        combined_actions = self.diffusion.generate_actions(prepared_batch)

        # Split back into separate action types
        actions = self._split_diffusion_output(combined_actions)

        # Unnormalize actions
        actions = self.unnormalize_outputs(actions)

        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict a chunk of actions given environment observations."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Populate queues with current batch
        self._queues = populate_queues(self._queues, batch)

        # Stack observations from queues
        prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        return self._get_action_chunk(prepared_batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Select a single action given environment observations.

        This method manages action queues and generates new actions when needed,
        similar to the standard diffusion policy but handling mixed action types.
        """
        # Remove action from batch if present (for offline evaluation)
        if ACTION in batch:
            batch.pop(ACTION)

        # Check for each action type and remove if present
        for action_key in ["action.continuous", "action.discrete"]:
            if action_key in batch:
                batch.pop(action_key)

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Populate observation queues
        obs_keys = [k for k in batch.keys() if not k.startswith("action")]
        for key in obs_keys:
            if key in self._queues:
                self._queues[key].append(batch[key])

        # Generate new action chunk if queue is empty
        if len(self._queues[ACTION]) == 0:
            # Create prepared batch for action generation
            prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in obs_keys if k in self._queues}

            actions_dict = self._get_action_chunk(prepared_batch)

            # Convert to single action tensor for queue (concatenate all action types)
            action_parts = []
            if self.has_continuous:
                action_parts.append(actions_dict["action.continuous"])
            if self.has_discrete:
                action_parts.append(actions_dict["action.discrete"])

            actions = torch.cat(action_parts, dim=-1)

            # Add actions to queue (transpose to get (time, batch, action_dim))
            self._queues[ACTION].extend(actions.transpose(0, 1))

        # Pop next action and split back into components
        action = self._queues[ACTION].popleft()

        # Split action back into components
        result = {}
        idx = 0

        if self.has_continuous:
            result["action.continuous"] = action[..., idx:idx + self.continuous_dim]
            idx += self.continuous_dim

        if self.has_discrete:
            result["action.discrete"] = action[..., idx:idx + self.discrete_dim]

        return result

    @property
    def device(self):
        return get_device_from_parameters(self)

    @property
    def dtype(self):
        return get_dtype_from_parameters(self)
