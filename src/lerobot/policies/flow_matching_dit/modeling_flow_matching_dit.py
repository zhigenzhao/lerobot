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
"""Flow Matching DiT Policy using DiT (Diffusion Transformer) architecture with flow matching
for generative modeling instead of denoising diffusion."""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.flow_matching_dit.configuration_flow_matching_dit import FlowMatchingDiTConfig
from lerobot.policies.flow_matching_dit.dit_blocks import DiTBlock, FlowMatchingTimestepEmbedding, PositionalEncoding
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)

# Flow matching library imports
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


class FlowMatchingDiTWrapper(ModelWrapper):
    """Wrapper for DiT transformer to be compatible with flow_matching library's ModelWrapper interface."""

    def __init__(self, transformer, config):
        super().__init__(transformer)
        self.config = config

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        """
        Forward pass compatible with flow_matching library.

        Args:
            x: Input tensor (B, horizon, action_dim)
            t: Time tensor (B,) in [0,1]
            **extras: Additional arguments like global_cond
        """
        global_cond = extras.get("global_cond", None)
        return self.model.velocity_field(x, t, global_cond)


class FlowMatchingDiTPolicy(PreTrainedPolicy):
    """
    Flow Matching Policy using DiT (Diffusion Transformer) architecture with flow matching
    for generative modeling instead of denoising diffusion. Combines AdaLN timestep conditioning
    with cross-attention for observations and optimal transport flow matching.
    """

    config_class = FlowMatchingDiTConfig
    name = "flow_matching_dit"

    def __init__(
        self,
        config: FlowMatchingDiTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.flow_matching = FlowMatchingDiTModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.flow_matching.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Stateless method to generate actions from prepared observations."""
        actions = self.flow_matching.generate_actions(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # Normalize and prepare batch
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Populate queues with current batch
        self._queues = populate_queues(self._queues, batch)

        # Stack observations from queues
        prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        return self._get_action_chunk(prepared_batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        action_chunk = self.predict_action_chunk(batch)

        # The first `n_action_steps` actions are used for execution in the environment
        action = action_chunk[:, : self.config.n_action_steps]
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Forward pass for training."""
        batch = self.normalize_inputs(batch)

        # Prepare inputs
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Normalize target actions
        batch = self.normalize_targets(batch)

        # Forward pass through flow matching model
        loss = self.flow_matching.compute_loss(batch)

        # no output_dict so returning None
        return loss, None


class FlowMatchingDiTModel(nn.Module):
    """DiT-based flow matching model for action generation."""

    def __init__(self, config: FlowMatchingDiTConfig):
        super().__init__()
        self.config = config

        # Flow matching components
        self.flow_scheduler = CondOTProbPath()

        # Vision encoder (shared with diffusion DiT)
        if len(config.image_features) > 0:
            self.rgb_encoder = self._make_rgb_encoder(config)
        else:
            self.rgb_encoder = None

        # Compute observation encoding dimensions
        obs_encoding_dim = 0
        if config.robot_state_feature:
            obs_encoding_dim += config.robot_state_feature.shape[0]
        if config.env_state_feature:
            obs_encoding_dim += config.env_state_feature.shape[0]
        if len(config.image_features) > 0:
            obs_encoding_dim += self._get_rgb_encoding_size(config)

        # Observation encoder for cross-attention
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoding_dim, config.cross_attention_dim),
            nn.ReLU(),
            nn.Linear(config.cross_attention_dim, config.cross_attention_dim),
        )

        # Action embedding
        action_dim = config.output_features[ACTION].shape[0]
        self.action_embed = nn.Linear(action_dim, config.hidden_size)

        # Positional encoding for actions
        self.pos_encoding = PositionalEncoding(config.hidden_size, max_len=config.horizon)

        # Flow matching timestep embedding
        self.timestep_embed = FlowMatchingTimestepEmbedding(
            embed_dim=config.fm_time_embed_dim,
            hidden_dim=config.fm_time_embed_dim,
            min_period=config.fm_min_period,
            max_period=config.fm_max_period,
        )

        # DiT transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                timestep_embed_dim=config.fm_time_embed_dim,
                attention_dropout=config.attention_dropout,
                cross_attention_dim=config.cross_attention_dim if config.add_cross_attention else None,
                use_adaln_zero=config.use_adaln_zero,
            )
            for _ in range(config.num_layers)
        ])

        # Output projection (predicts velocity field)
        self.output_proj = nn.Linear(config.hidden_size, action_dim)

    def _make_rgb_encoder(self, config: FlowMatchingDiTConfig):
        """Create RGB encoder following diffusion transformer pattern."""
        # Use the same approach as diffusion transformer for consistency
        if config.use_separate_rgb_encoder_per_camera:
            num_images = len(config.image_features)
            encoders = [self._make_single_rgb_encoder(config) for _ in range(num_images)]
            return nn.ModuleList(encoders)
        else:
            return self._make_single_rgb_encoder(config)

    def _make_single_rgb_encoder(self, config: FlowMatchingDiTConfig):
        """Create a single RGB encoder."""
        # Use ResNet backbone
        model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )

        # Replace BatchNorm with GroupNorm if specified
        if config.use_group_norm:
            model = self._replace_bn_with_gn(model)

        # Remove final layers and add spatial softmax
        model = nn.Sequential(
            *list(model.children())[:-2],  # Remove avgpool and fc
            nn.AdaptiveAvgPool2d((config.spatial_softmax_num_keypoints // 4, config.spatial_softmax_num_keypoints // 4)),
            nn.Flatten(),
            nn.Linear(512 * (config.spatial_softmax_num_keypoints // 4) ** 2, config.cross_attention_dim),
        )
        return model

    def _replace_bn_with_gn(self, model):
        """Replace BatchNorm with GroupNorm."""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_groups = max(1, module.num_features // 16)
                gn = nn.GroupNorm(num_groups, module.num_features)
                setattr(model, name, gn)
            elif len(list(module.children())) > 0:
                setattr(model, name, self._replace_bn_with_gn(module))
        return model

    def _get_rgb_encoding_size(self, config: FlowMatchingDiTConfig):
        """Get the size of RGB encoding."""
        if config.use_separate_rgb_encoder_per_camera:
            return len(config.image_features) * config.cross_attention_dim
        else:
            return len(config.image_features) * config.cross_attention_dim

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations for cross-attention conditioning."""
        encodings = []

        # Encode state
        if self.config.robot_state_feature and OBS_STATE in batch:
            state = batch[OBS_STATE]  # (B, n_obs_steps, state_dim)
            encodings.append(state)

        # Encode environment state
        if self.config.env_state_feature and OBS_ENV_STATE in batch:
            env_state = batch[OBS_ENV_STATE]  # (B, n_obs_steps, env_state_dim)
            encodings.append(env_state)

        # Encode images
        if len(self.config.image_features) > 0 and OBS_IMAGES in batch:
            images = batch[OBS_IMAGES]  # (B, n_obs_steps, n_cameras, C, H, W)
            B, n_obs_steps, n_cameras = images.shape[:3]
            
            if self.config.crop_shape is not None:
                # Apply cropping
                images = self._crop_images(images)

            # Reshape for encoding
            images = images.view(B * n_obs_steps * n_cameras, *images.shape[3:])

            if self.config.use_separate_rgb_encoder_per_camera:
                # Use separate encoders - following diffusion transformer pattern
                # Combine batch and sequence dims while rearranging to make the camera index dimension first
                images_per_camera = einops.rearrange(images.view(B, n_obs_steps, n_cameras, *images.shape[1:]), 
                                                   "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(camera_images) for encoder, camera_images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the feature dim
                image_encodings = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    n=n_cameras, b=B, s=n_obs_steps
                )
            else:
                # Use shared encoder
                image_encodings = self.rgb_encoder(images)
                image_encodings = image_encodings.view(B, n_obs_steps, n_cameras, -1)
                image_encodings = image_encodings.flatten(start_dim=-2)

            encodings.append(image_encodings)

        # Concatenate all encodings
        obs_encoding = torch.cat(encodings, dim=-1)  # (B, n_obs_steps, total_dim)
        
        # Project to cross-attention dimension
        obs_encoding = self.obs_encoder(obs_encoding)  # (B, n_obs_steps, cross_attention_dim)
        
        return obs_encoding

    def _crop_images(self, images: Tensor) -> Tensor:
        """Apply cropping to images."""
        if self.training and self.config.crop_is_random:
            # Random crop during training
            return torchvision.transforms.functional.resized_crop(
                images,
                top=torch.randint(0, images.shape[-2] - self.config.crop_shape[0] + 1, (1,)).item(),
                left=torch.randint(0, images.shape[-1] - self.config.crop_shape[1] + 1, (1,)).item(),
                height=self.config.crop_shape[0],
                width=self.config.crop_shape[1],
                size=self.config.crop_shape,
            )
        else:
            # Center crop during evaluation
            return torchvision.transforms.functional.center_crop(images, self.config.crop_shape)

    def velocity_field(self, x: Tensor, t: Tensor, global_cond: Tensor) -> Tensor:
        """Compute velocity field for flow matching.
        
        Args:
            x: (B, horizon, action_dim) current sample
            t: (B,) continuous time in [0,1]
            global_cond: (B, n_obs_steps, cross_attention_dim) observation conditioning
            
        Returns:
            (B, horizon, action_dim) predicted velocity field
        """
        # Embed actions and add positional encoding
        action_embeddings = self.action_embed(x)  # (B, horizon, hidden_size)
        action_embeddings = self.pos_encoding(action_embeddings)

        # Embed timesteps
        timestep_emb = self.timestep_embed(t)  # (B, fm_time_embed_dim)

        # Pass through DiT blocks
        hidden_states = action_embeddings
        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                timestep_emb=timestep_emb,
                encoder_hidden_states=global_cond,
            )

        # Output projection to velocity field
        velocity = self.output_proj(hidden_states)
        return velocity

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Flow matching loss computation."""
        batch_size = batch[ACTION].shape[0]

        # Sample continuous time in [0,1]
        t = torch.rand(batch_size, device=batch[ACTION].device)

        # Sample from optimal transport path
        target_actions = batch[ACTION]  # (B, horizon, action_dim)
        noise = torch.randn_like(target_actions)
        path_sample = self.flow_scheduler.sample(t=t, x_0=noise, x_1=target_actions)
        x_t = path_sample.x_t     # Point on path at time t
        dx_t = path_sample.dx_t   # True velocity at that point

        # Encode observations for cross-attention
        obs_encoding = self.encode_observations(batch) if self.config.add_cross_attention else None

        # Predict velocity field
        pred_velocity = self.velocity_field(x_t, t, obs_encoding)

        # Flow matching loss
        loss = F.mse_loss(pred_velocity, dx_t, reduction="mean")
        return loss

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions using flow matching ODE integration."""
        batch_size = next(iter(batch.values())).shape[0]
        device = next(iter(batch.values())).device

        # Encode observations
        obs_encoding = self.encode_observations(batch) if self.config.add_cross_attention else None

        # Start from noise
        x_0 = torch.randn(
            batch_size, self.config.horizon, self.config.output_features[ACTION].shape[0],
            device=device
        )

        # Create model wrapper for ODE solver
        model_wrapper = FlowMatchingDiTWrapper(self, self.config)
        
        # Create wrapped velocity function for ODESolver
        def wrapped_velocity_model(x, t):
            return model_wrapper(x, t, global_cond=obs_encoding)
        
        # Initialize ODESolver with velocity model
        ode_solver = ODESolver(velocity_model=wrapped_velocity_model)
        
        # Create time grid for integration
        time_grid = torch.linspace(0, 1, self.config.num_integration_steps + 1, device=device)

        # Solve ODE using the sample method with method parameter
        solution = ode_solver.sample(
            time_grid=time_grid,
            x_init=x_0, 
            method=self.config.ode_solver_method,
            step_size=None
        )
        
        # Return final sample at t=1
        return solution[-1]