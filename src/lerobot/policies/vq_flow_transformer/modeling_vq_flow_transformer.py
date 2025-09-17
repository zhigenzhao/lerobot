#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
VQ Flow Transformer Policy combining VQVAE action discretization with discrete flow matching.

Uses the proper discrete flow matching implementation from the flow_matching library:
- MixtureDiscreteProbPath with PolynomialConvexScheduler
- MixturePathGeneralizedKL for loss
- MixtureDiscreteEulerSolver for sampling
"""

import math
from collections import deque
from typing import Dict, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.policies.vq_flow_transformer.configuration_vq_flow_transformer import VQFlowTransformerConfig
from lerobot.policies.vq_flow_transformer.vq_flow_transformer_utils import (
    VQFlowVAE,
    flatten_indices,
    unflatten_indices,
    DiscreteFlowSinusoidalPosEmb,
    DiscreteFlowTransformerBlock,
)

# Flow matching imports
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.utils import ModelWrapper


class VQFlowTransformerPolicy(PreTrainedPolicy):
    """
    VQ Flow Transformer Policy implementing stable two-phase training:

    Phase 1: Train VQVAE to discretize action chunks into tokens
    Phase 2: Train transformer to generate token sequences via discrete flow matching
    """

    config_class = VQFlowTransformerConfig
    name = "vq_flow_transformer"

    def __init__(
        self,
        config: VQFlowTransformerConfig,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        """
        Args:
            config: Policy configuration
            dataset_stats: Dataset statistics for normalization
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Observation and action queues for inference
        self._queues = None

        # Main model components
        self.vq_flow_transformer = VQFlowTransformerModel(config)

        self.reset()

    def get_optim_params(self) -> list[dict]:
        """Get optimizer parameter groups for two-phase training."""
        # Phase 1: VQVAE parameters
        vqvae_params = list(self.vq_flow_transformer.vqvae.parameters())

        # Phase 2: Transformer parameters (excluding VQVAE)
        transformer_params = []
        for name, param in self.vq_flow_transformer.named_parameters():
            if not name.startswith('vqvae.'):
                transformer_params.append(param)

        return [
            {
                "params": vqvae_params,
                "lr": self.config.phase1_lr,
                "weight_decay": self.config.phase1_weight_decay,
            },
            {
                "params": transformer_params,
                "lr": self.config.phase2_lr,
                "weight_decay": self.config.phase2_weight_decay,
            }
        ]

    def reset(self):
        """Clear observation and action queues. Called on env.reset()."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given observations."""
        # Normalize inputs
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Populate queues
        self._queues = populate_queues(self._queues, batch)

        # Stack observations from queues
        prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        # Generate actions
        actions = self.vq_flow_transformer.generate_actions(prepared_batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Select a single action given observations (with action caching)."""
        # Remove action if present (for offline eval)
        if ACTION in batch:
            batch.pop(ACTION)

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        # Generate new action chunk if queue is empty
        if len(self._queues[ACTION]) == 0:
            prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.vq_flow_transformer.generate_actions(prepared_batch)
            actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
            # Fill queue with actions (transpose to make time first dimension)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        # Return next action
        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Optional[Dict]]:
        """Forward pass for training."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        batch = self.normalize_targets(batch)

        # Two-phase training
        if not self.vq_flow_transformer.vqvae.phase1_complete.item():
            # Phase 1: VQVAE training
            loss, loss_dict = self.vq_flow_transformer.compute_vqvae_loss(batch)
            return loss, loss_dict
        else:
            # Phase 2: Discrete flow matching
            loss, loss_dict = self.vq_flow_transformer.compute_flow_loss(batch)
            return loss, loss_dict


class VQFlowTransformerModel(nn.Module):
    """Core VQ Flow Transformer model."""

    def __init__(self, config: VQFlowTransformerConfig):
        super().__init__()
        self.config = config

        # VQVAE for action discretization
        self.vqvae = VQFlowVAE(config)

        # Observation encoders
        self.rgb_encoder = self._build_rgb_encoder(config) if config.image_features else None

        # Calculate observation encoding dimension
        obs_encoding_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                obs_encoding_dim += self.rgb_encoder[0].feature_dim * num_images
            else:
                obs_encoding_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            obs_encoding_dim += config.env_state_feature.shape[0]

        # Discrete Flow Transformer
        self.transformer = DiscreteFlowTransformer(config, obs_encoding_dim)

        # Discrete flow matching components (from flow_matching library)
        scheduler = PolynomialConvexScheduler(n=config.scheduler_power)
        self.flow_path = MixtureDiscreteProbPath(scheduler=scheduler)
        self.flow_loss_fn = MixturePathGeneralizedKL(path=self.flow_path)

        # Wrapped model for sampling
        self.wrapped_transformer = WrappedTransformerModel(self.transformer)
        self.flow_solver = MixtureDiscreteEulerSolver(
            model=self.wrapped_transformer,
            path=self.flow_path,
            vocabulary_size=config.vocab_size
        )

    def _build_rgb_encoder(self, config: VQFlowTransformerConfig):
        """Build RGB encoder(s) for image observations."""
        if config.use_separate_rgb_encoder_per_camera:
            num_images = len(config.image_features)
            return nn.ModuleList([VQFlowRgbEncoder(config) for _ in range(num_images)])
        else:
            return VQFlowRgbEncoder(config)

    def _encode_observations(self, batch: Dict[str, Tensor]) -> Tensor:
        """Encode observations into conditioning vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        obs_features = [batch[OBS_STATE]]

        # Encode images
        if self.config.image_features and self.rgb_encoder is not None:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Separate encoders per camera
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                # Shared encoder
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            obs_features.append(img_features)

        # Add environment state if present
        if self.config.env_state_feature:
            obs_features.append(batch[OBS_ENV_STATE])

        # Concatenate and flatten to (B, obs_encoding_dim)
        return torch.cat(obs_features, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: Dict[str, Tensor]) -> Tensor:
        """Generate actions using discrete flow sampling."""
        batch_size = batch[OBS_STATE].shape[0]

        # Encode observations
        obs_encoding = self._encode_observations(batch)

        # Calculate sequence length (number of chunks needed)
        seq_len = self.config.horizon - self.config.action_chunk_size + 1

        device = get_device_from_parameters(self)

        with torch.no_grad():
            # Initialize from uniform distribution over vocabulary
            x_init = torch.randint(
                size=(batch_size, seq_len),
                high=self.config.vocab_size,
                device=device
            )

            # Set observation context for the wrapped model
            self.wrapped_transformer.set_context(obs_encoding)

            # Sample using discrete flow solver
            step_size = 1.0 / self.config.num_integration_steps
            samples = self.flow_solver.sample(
                x_init=x_init,
                step_size=step_size,
                verbose=False
            )

            # Convert back to hierarchical indices and decode
            hierarchical_indices = unflatten_indices(
                samples, self.config.vqvae_num_layers, self.config.vqvae_n_embed
            )

            # Decode to continuous actions
            action_chunks = self.vqvae.decode_from_indices(hierarchical_indices)

            # Reshape to (B, horizon, action_dim) with proper overlapping
            actions = self._reconstruct_sequence(action_chunks)

        # Extract n_action_steps worth of actions
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def _reconstruct_sequence(self, action_chunks: Tensor) -> Tensor:
        """Reconstruct overlapping action sequence from chunks."""
        B, num_chunks, chunk_size, action_dim = action_chunks.shape
        horizon = num_chunks + chunk_size - 1

        # Simple reconstruction: take first action from each chunk, plus remainder of last chunk
        actions = torch.zeros(B, horizon, action_dim, device=action_chunks.device)

        for i in range(num_chunks):
            if i < num_chunks - 1:
                # Take only first action from chunk
                actions[:, i] = action_chunks[:, i, 0]
            else:
                # Take all actions from last chunk
                actions[:, i:] = action_chunks[:, i]

        return actions

    def compute_vqvae_loss(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict]:
        """Compute VQVAE training loss (Phase 1)."""
        actions = batch[ACTION]  # (B, horizon, action_dim)

        # Create action chunks using sliding window
        chunks = self._create_action_chunks(actions)  # (B * num_chunks, chunk_size, action_dim)

        # Forward through VQVAE
        actions_recon, indices, vq_loss, recon_loss = self.vqvae(chunks)

        # Total loss
        total_loss = recon_loss + self.config.vqvae_commitment_beta * vq_loss

        # Update step counter and check for phase transition
        self.vqvae.optimized_steps += 1
        if self.vqvae.optimized_steps >= self.config.n_vqvae_training_steps:
            print(f"Phase 1 complete! Switching to Phase 2 after {self.vqvae.optimized_steps} steps")
            self.vqvae.freeze()

        loss_dict = {
            "vqvae_recon_loss": recon_loss.item(),
            "vqvae_vq_loss": vq_loss.item(),
            "vqvae_total_loss": total_loss.item(),
            "phase": 1,
        }

        return total_loss, loss_dict

    def compute_flow_loss(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict]:
        """Compute discrete flow matching loss (Phase 2)."""
        actions = batch[ACTION]  # (B, horizon, action_dim)

        # Encode observations
        obs_encoding = self._encode_observations(batch)

        # Create action chunks and encode to discrete tokens
        chunks = self._create_action_chunks(actions)
        B_chunks = chunks.shape[0]

        with torch.no_grad():
            # Encode to hierarchical indices then flatten
            hierarchical_indices = self.vqvae.encode_to_indices(
                chunks.view(-1, self.config.action_chunk_size // 1,
                          self.config.action_chunk_size, self.config.action_feature.shape[0])
            )
            flat_indices = flatten_indices(hierarchical_indices.squeeze(1), self.config.vqvae_n_embed)

        # Reshape to sequence format
        batch_size = actions.shape[0]
        seq_len = B_chunks // batch_size
        x_1 = flat_indices.view(batch_size, seq_len)

        # Sample x_0 from uniform distribution (source distribution)
        x_0 = torch.randint_like(x_1, high=self.config.vocab_size)

        # Sample random timestep (avoid t=1 with epsilon)
        epsilon = self.config.flow_epsilon
        t = torch.rand(batch_size, device=actions.device) * (1 - epsilon)

        # Sample from discrete flow path
        path_sample = self.flow_path.sample(t=t, x_0=x_0, x_1=x_1)

        # Set context for transformer
        self.wrapped_transformer.set_context(obs_encoding)

        # Get model logits
        logits = self.transformer(path_sample.x_t, path_sample.t, obs_encoding)

        # Compute discrete flow matching loss using proper loss function
        loss = self.flow_loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

        # Apply padding mask if needed
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            # Implementation would go here for masking padded actions
            pass

        loss_dict = {
            "flow_loss": loss.item(),
            "phase": 2,
        }

        return loss, loss_dict

    def _create_action_chunks(self, actions: Tensor) -> Tensor:
        """Create overlapping action chunks for VQVAE training."""
        B, horizon, action_dim = actions.shape
        chunk_size = self.config.action_chunk_size

        # Create sliding window chunks
        chunks = []
        for i in range(horizon - chunk_size + 1):
            chunk = actions[:, i:i + chunk_size]
            chunks.append(chunk)

        # Stack and flatten batch dimension
        chunks = torch.stack(chunks, dim=1)  # (B, num_chunks, chunk_size, action_dim)
        return chunks.view(-1, chunk_size, action_dim)  # (B * num_chunks, chunk_size, action_dim)


class WrappedTransformerModel(ModelWrapper):
    """Wrapper for transformer to be compatible with flow_matching library."""

    def __init__(self, transformer):
        super().__init__(transformer)
        self.context = None

    def set_context(self, obs_encoding: Tensor):
        """Set observation context for the model."""
        self.context = obs_encoding

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """
        Args:
            x: (B, T) discrete token sequence
            t: (B,) flow matching timesteps in [0,1]
        Returns:
            probs: (B, T, vocab_size) probability distributions
        """
        logits = self.model(x, t, self.context)
        return torch.softmax(logits, dim=-1)


class DiscreteFlowTransformer(nn.Module):
    """Transformer for discrete flow matching."""

    def __init__(self, config: VQFlowTransformerConfig, obs_encoding_dim: int):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.attention_embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.chunk_size, config.attention_embed_dim))

        # Time embedding
        self.time_embedding = DiscreteFlowSinusoidalPosEmb(
            config.fm_time_embed_dim,
            min_period=config.fm_min_period,
            max_period=config.fm_max_period
        )

        # Observation projection to cross-attention dimension
        self.obs_projection = nn.Linear(obs_encoding_dim, config.cross_attention_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiscreteFlowTransformerBlock(config, config.cross_attention_dim)
            for _ in range(config.n_decoder_layers)
        ])

        # Output head
        self.norm_f = nn.LayerNorm(config.attention_embed_dim)
        self.output_projection = nn.Linear(config.attention_embed_dim, config.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(config.embedding_dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def forward(self, tokens: Tensor, timestep: Tensor, obs_encoding: Tensor) -> Tensor:
        """
        Args:
            tokens: (B, T) discrete token sequence
            timestep: (B,) flow matching timesteps in [0,1]
            obs_encoding: (B, obs_dim) observation encoding
        Returns:
            logits: (B, T, vocab_size) output logits
        """
        B, T = tokens.shape

        # Token embeddings
        x = self.token_embedding(tokens)  # (B, T, D)

        # Add positional embeddings
        x = x + self.pos_embedding[:, :T]

        # Apply dropout
        x = self.dropout(x)

        # Time embeddings
        time_emb = self.time_embedding(timestep)  # (B, time_embed_dim)

        # Project observations for cross-attention
        obs_context = self.obs_projection(obs_encoding).unsqueeze(1)  # (B, 1, cross_attn_dim)

        # Create causal attention mask if needed
        attn_mask = None
        if self.config.use_causal_attention:
            attn_mask = torch.tril(torch.ones(T, T, device=tokens.device)).bool()
            attn_mask = attn_mask.masked_fill(~attn_mask, float('-inf')).masked_fill(attn_mask, 0.0)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, obs_context, time_emb, attn_mask)

        # Output projection
        x = self.norm_f(x)
        logits = self.output_projection(x)

        return logits


class VQFlowRgbEncoder(nn.Module):
    """RGB encoder for visual observations."""

    def __init__(self, config: VQFlowTransformerConfig):
        super().__init__()

        # Set up optional preprocessing
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Can't replace BatchNorm in pretrained model!")
            self.backbone = self._replace_submodules(
                self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = VQFlowSpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1]
        Returns:
            (B, D) image feature
        """
        # Preprocess: maybe crop
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        # Extract backbone feature
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)

        # Final linear layer with non-linearity
        x = self.relu(self.out(x))
        return x

    def _replace_submodules(self, root_module, predicate, func):
        """Replace submodules matching predicate with func output."""
        if predicate(root_module):
            return func(root_module)

        replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
        for *parents, k in replace_list:
            parent_module = root_module
            if len(parents) > 0:
                parent_module = root_module.get_submodule(".".join(parents))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)

        return root_module


class VQFlowSpatialSoftmax(nn.Module):
    """Spatial Soft Argmax for extracting keypoints from feature maps."""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # Create position grid
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps
        Returns:
            (B, K, 2) image-space coordinates of keypoints
        """
        if self.nets is not None:
            features = self.nets(features)

        # Flatten and apply softmax
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)

        # Compute expected keypoint locations
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints