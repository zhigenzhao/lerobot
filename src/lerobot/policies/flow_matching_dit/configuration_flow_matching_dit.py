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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("flow_matching_dit")
@dataclass
class FlowMatchingDiTConfig(PreTrainedConfig):
    """Configuration class for FlowMatchingDiTPolicy.

    Flow Matching Policy using DiT (Diffusion Transformer) architecture with flow matching
    for generative modeling instead of denoising diffusion. Combines the efficiency of DiT's
    AdaLN timestep conditioning with flow matching's optimal transport approach.
    
    Inherits most DiT parameters from DiffusionDiTConfig but replaces diffusion-specific
    parameters with flow matching parameters.

    Args:
        flow_matching_type: Type of flow matching to use. Currently supports "CondOT" 
            (Conditional Optimal Transport).
        num_integration_steps: Number of integration steps for ODE solver during inference.
            Fewer steps than diffusion (typically 20-50 vs 50-1000).
        fm_time_embed_dim: Embedding dimension for flow matching continuous time encoding.
            This replaces diffusion_step_embed_dim.
        fm_min_period: Minimum period for flow matching sinusoidal positional embeddings.
            Controls the finest time resolution.
        fm_max_period: Maximum period for flow matching sinusoidal positional embeddings.
            Controls the coarsest time resolution.
        ode_solver_method: ODE solver method for inference. Options include "euler", "midpoint", "rk4".
    """

    # Inputs / output structure (inherited from DiffusionDiTConfig)
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling (inherited from DiffusionDiTConfig)
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # DiT-specific architecture parameters.
    hidden_size: int = 768              # DiT hidden dimension
    num_layers: int = 12                # Number of DiT blocks
    num_attention_heads: int = 12       # Number of attention heads
    attention_dropout: float = 0.1      # Attention dropout
    patch_size: int = 2                 # Patch size for patch embedding

    # AdaLN timestep conditioning.
    use_adaln_zero: bool = True         # Use AdaLN-Zero initialization

    # Cross-attention global conditioning.
    cross_attention_dim: int = 512      # Observation embedding dimension
    add_cross_attention: bool = True    # Enable cross-attention

    # Loss computation (inherited from DiffusionDiTConfig)
    do_mask_loss_for_padding: bool = False

    # Training presets (inherited from DiffusionDiTConfig)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # Flow matching specific parameters (replacing diffusion parameters)
    flow_matching_type: str = "CondOT"
    num_integration_steps: int = 50
    fm_time_embed_dim: int = 256  # Same as diffusion_step_embed_dim
    
    # Flow matching positional embedding parameters
    fm_min_period: float = 4e-3
    fm_max_period: float = 4.0
    
    # ODE solver configuration
    ode_solver_method: str = "midpoint"

    def __post_init__(self):
        super().__post_init__()

        """Input validation for DiT architecture."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        # Validate DiT transformer parameters
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_attention_heads`. "
                f"Got {self.hidden_size} and {self.num_attention_heads}."
            )

        if self.attention_dropout < 0.0 or self.attention_dropout >= 1.0:
            raise ValueError(f"`attention_dropout` must be in [0, 1). Got {self.attention_dropout}.")
        
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive. Got {self.num_layers}.")

        if self.fm_time_embed_dim <= 0:
            raise ValueError(f"`fm_time_embed_dim` must be positive. Got {self.fm_time_embed_dim}.")

        if self.add_cross_attention and self.cross_attention_dim <= 0:
            raise ValueError(f"`cross_attention_dim` must be positive when cross-attention is enabled. Got {self.cross_attention_dim}.")

        # Validate flow matching specific parameters
        if self.flow_matching_type not in ["CondOT"]:
            raise ValueError(f"`flow_matching_type` must be 'CondOT'. Got {self.flow_matching_type}.")
        
        if self.num_integration_steps <= 0:
            raise ValueError(f"`num_integration_steps` must be positive. Got {self.num_integration_steps}.")
        
        if self.fm_min_period <= 0 or self.fm_max_period <= 0:
            raise ValueError(f"Flow matching periods must be positive. Got min: {self.fm_min_period}, max: {self.fm_max_period}.")
        
        if self.fm_min_period >= self.fm_max_period:
            raise ValueError(f"`fm_min_period` must be less than `fm_max_period`. Got {self.fm_min_period} >= {self.fm_max_period}.")
        
        if self.ode_solver_method not in ["euler", "midpoint", "rk4"]:
            raise ValueError(f"`ode_solver_method` must be one of ['euler', 'midpoint', 'rk4']. Got {self.ode_solver_method}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None