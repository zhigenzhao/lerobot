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

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("vq_flow_transformer")
@dataclass
class VQFlowTransformerConfig(PreTrainedConfig):
    """
    Configuration class for VQ Flow Transformer policy.

    Combines Vector Quantized VAE with Discrete Flow Matching using proper
    Transformer architecture for stable two-phase training.

    Phase 1: Train VQVAE to discretize actions into tokens
    Phase 2: Train transformer to generate discrete token sequences via flow matching
    """

    ###################
    # Inputs / Outputs
    ###################
    n_obs_steps: int = 1
    """Number of observation steps used as input to the policy."""

    chunk_size: int = 100
    """Maximum number of steps in an episode, used for attention masks."""

    n_action_steps: int = 32
    """Number of action steps generated for each observation."""

    horizon: int = 64
    """Total horizon length for action prediction."""

    action_chunk_size: int = 8
    """Size of action chunks for VQVAE encoding."""

    ###################
    # Architecture
    ###################
    vision_backbone: str = "resnet18"
    """Vision backbone for image encoding."""

    pretrained_backbone_weights: Optional[str] = None
    """Pretrained weights for vision backbone."""

    use_group_norm: bool = True
    """Whether to use GroupNorm instead of BatchNorm."""

    spatial_softmax_num_keypoints: int = 32
    """Number of keypoints for spatial softmax pooling."""

    crop_shape: Optional[Tuple[int, int]] = (224, 224)
    """Image crop shape for preprocessing."""

    crop_is_random: bool = True
    """Whether to use random cropping during training."""

    use_separate_rgb_encoder_per_camera: bool = True
    """Whether to use separate RGB encoders for each camera."""

    ###################
    # VQVAE Parameters (from VQ-BeT)
    ###################
    vqvae_n_embed: int = 32
    """Size of the VQ codebook."""

    vqvae_embedding_dim: int = 256
    """Dimensionality of VQ embeddings."""

    vqvae_num_layers: int = 2
    """Number of layers in Residual VQ."""

    vqvae_enc_hidden_dim: int = 128
    """Hidden dimension for VQVAE encoder/decoder."""

    n_vqvae_training_steps: int = 20000
    """Number of steps to train VQVAE before switching to phase 2."""

    vqvae_commitment_beta: float = 0.25
    """Commitment loss weight for VQ."""

    ###################
    # Transformer Parameters
    ###################
    attention_embed_dim: int = 768
    """Embedding dimension for transformer."""

    n_attention_heads: int = 12
    """Number of attention heads."""

    n_decoder_layers: int = 8
    """Number of decoder layers."""

    attention_dropout: float = 0.1
    """Dropout probability for attention weights."""

    embedding_dropout: float = 0.1
    """Dropout probability for embeddings."""

    use_causal_attention: bool = True
    """Whether to use causal attention masks."""

    ###################
    # Discrete Flow Matching Parameters
    ###################
    flow_matching_type: str = "discrete"
    """Type of flow matching - discrete for token sequences."""

    num_integration_steps: int = 50
    """Number of integration steps for discrete flow sampling."""

    scheduler_power: float = 1.0
    """Power for polynomial convex scheduler (1.0 = linear)."""

    flow_epsilon: float = 0.1
    """Small epsilon to avoid numerical instability in jump coefficients."""

    fm_time_embed_dim: int = 256
    """Dimension for flow matching time embeddings."""

    fm_min_period: float = 0.004
    """Minimum period for flow matching time embeddings."""

    fm_max_period: float = 4.0
    """Maximum period for flow matching time embeddings."""

    ###################
    # Cross Attention
    ###################
    cross_attention_dim: int = 512
    """Dimension for cross-attention to observations."""

    ###################
    # Training Parameters
    ###################
    phase1_lr: float = 1e-3
    """Learning rate for phase 1 (VQVAE training)."""

    phase1_weight_decay: float = 1e-4
    """Weight decay for phase 1."""

    phase2_lr: float = 1e-4
    """Learning rate for phase 2 (transformer training)."""

    phase2_weight_decay: float = 1e-6
    """Weight decay for phase 2."""

    do_mask_loss_for_padding: bool = True
    """Whether to mask loss for padded actions."""

    ###################
    # Vocabulary
    ###################
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size for discrete tokens."""
        # For RVQ with L layers and C codes per layer: vocab_size = C^L
        return self.vqvae_n_embed ** self.vqvae_num_layers

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate horizon and action steps relationship
        if self.horizon < self.n_action_steps:
            raise ValueError(f"horizon ({self.horizon}) must be >= n_action_steps ({self.n_action_steps})")

        # Validate action chunk size
        if self.action_chunk_size > self.horizon:
            raise ValueError(f"action_chunk_size ({self.action_chunk_size}) must be <= horizon ({self.horizon})")

        # Validate attention dimensions
        if self.attention_embed_dim % self.n_attention_heads != 0:
            raise ValueError(f"attention_embed_dim ({self.attention_embed_dim}) must be divisible by n_attention_heads ({self.n_attention_heads})")

        # Validate VQ parameters
        if self.vqvae_n_embed <= 0:
            raise ValueError(f"vqvae_n_embed ({self.vqvae_n_embed}) must be positive")

        if self.vqvae_num_layers <= 0:
            raise ValueError(f"vqvae_num_layers ({self.vqvae_num_layers}) must be positive")

        # Validate discrete flow parameters
        if self.flow_matching_type != "discrete":
            raise ValueError(f"Only discrete flow matching supported, got: {self.flow_matching_type}")

        if self.num_integration_steps <= 0:
            raise ValueError(f"num_integration_steps ({self.num_integration_steps}) must be positive")

        # Check vocabulary size is reasonable
        vocab_size = self.vocab_size
        if vocab_size > 10000:
            raise ValueError(f"Vocabulary size ({vocab_size}) may be too large. Consider reducing vqvae_n_embed or vqvae_num_layers.")

    def get_optimizer_preset(self) -> AdamWConfig:
        """Get optimizer configuration for two-phase training."""
        return AdamWConfig(
            lr=self.phase1_lr,  # Will be overridden by get_optim_params
            weight_decay=self.phase1_weight_decay,
        )

    def get_scheduler_preset(self):
        """Get scheduler configuration (none for now)."""
        return None

    def validate_features(self) -> None:
        """Validate input/output features configuration."""
        if self.image_features:
            # Check that all input images have the same shape
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

            # Validate crop shape
            if self.crop_shape is not None:
                for key, image_ft in self.image_features.items():
                    if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                            f"for `crop_shape` and {image_ft.shape} for `{key}`."
                        )

    @property
    def observation_delta_indices(self) -> list:
        """Observation time indices relative to current step."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        """Action time indices relative to current step."""
        return list(range(1 - self.n_obs_steps, self.n_action_steps))

    @property
    def reward_delta_indices(self) -> None:
        """Reward time indices (not used)."""
        return None