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
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamConfig, AdamWConfig
from lerobot.optim.schedulers import HybridDiffusionSchedulerConfig


@PreTrainedConfig.register_subclass("hybrid_diffusion")
@dataclass
class HybridDiffusionConfig(PreTrainedConfig):
    """Configuration class for HybridDiffusionPolicy.

    This policy extends DiffusionPolicy to handle mixed continuous/discrete action spaces
    using a two-stage training process:
    1. Stage 1: Train VAE for discrete actions
    2. Stage 2: Train diffusion model on concatenated continuous actions + VAE latents

    The policy automatically detects action.continuous and action.discrete fields in the dataset
    and handles them appropriately. No manual configuration of action types is needed.

    Args:
        vae_latent_dim: Dimensionality of the VAE latent space for discrete actions.
        n_vae_training_steps: Number of steps to train the VAE before switching to diffusion training.
        vae_lr: Learning rate for VAE training (stage 1).
        vae_hidden_dims: Hidden layer dimensions for VAE encoder/decoder networks.
        vae_beta: Weight for KL divergence term in VAE loss.

    All other arguments are inherited from DiffusionConfig and work the same way.

    Example usage:
        ```python
        config = HybridDiffusionConfig(
            # VAE settings
            vae_latent_dim=16,
            n_vae_training_steps=20000,
            vae_lr=1e-3,
            vae_hidden_dims=[64, 128],

            # Standard diffusion settings
            horizon=64,
            n_action_steps=32,
            num_train_timesteps=100,
            # ... other diffusion parameters
        )
        ```
    """

    # Inputs / output structure.
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

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_type: str = "adam"  # "adam" or "adamw"
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # VAE settings for discrete actions
    vae_latent_dim: int = 16
    n_vae_training_steps: int = 20000
    vae_lr: float = 1e-3
    vae_hidden_dims: list[int] = field(default_factory=lambda: [64, 128])
    vae_beta: float = 0.001  # Weight for KL divergence term

    # Action dimension specification for concatenated action format
    continuous_action_dim: int = 21  # Dimensions 0-20: continuous actions
    discrete_action_dim: int = 2     # Dimensions 21-22: discrete actions

    def __post_init__(self):
        super().__post_init__()

        # Input validation for diffusion parameters
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if self.crop_shape is not None and len(self.crop_shape) != 2:
            raise ValueError(f"`crop_shape` must be None or a tuple of two ints. Got {self.crop_shape}.")

        if self.noise_scheduler_type not in ["DDPM", "DDIM"]:
            raise ValueError(
                f"`noise_scheduler_type` must be either 'DDPM' or 'DDIM'. Got {self.noise_scheduler_type}."
            )

        if self.prediction_type not in ["epsilon", "sample"]:
            raise ValueError(
                f"`prediction_type` must be either 'epsilon' or 'sample'. Got {self.prediction_type}."
            )

        # Validate VAE parameters
        if self.vae_latent_dim <= 0:
            raise ValueError(f"vae_latent_dim must be positive, got {self.vae_latent_dim}")

        if self.n_vae_training_steps <= 0:
            raise ValueError(f"n_vae_training_steps must be positive, got {self.n_vae_training_steps}")

        if self.vae_lr <= 0:
            raise ValueError(f"vae_lr must be positive, got {self.vae_lr}")

        if not self.vae_hidden_dims or not all(dim > 0 for dim in self.vae_hidden_dims):
            raise ValueError(f"vae_hidden_dims must be a non-empty list of positive integers, got {self.vae_hidden_dims}")

        if self.vae_beta < 0:
            raise ValueError(f"vae_beta must be non-negative, got {self.vae_beta}")

        # Validate optimizer type
        if self.optimizer_type not in ["adam", "adamw"]:
            raise ValueError(f"optimizer_type must be 'adam' or 'adamw', got {self.optimizer_type}")

        # Validate action dimensions
        if self.continuous_action_dim <= 0:
            raise ValueError(f"continuous_action_dim must be positive, got {self.continuous_action_dim}")

        if self.discrete_action_dim <= 0:
            raise ValueError(f"discrete_action_dim must be positive, got {self.discrete_action_dim}")

    def get_optimizer_preset(self) -> AdamConfig | AdamWConfig:
        if self.optimizer_type == "adam":
            return AdamConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )
        elif self.optimizer_type == "adamw":
            return AdamWConfig(
                lr=self.optimizer_lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def get_scheduler_preset(self) -> HybridDiffusionSchedulerConfig:
        return HybridDiffusionSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_vae_training_steps=self.n_vae_training_steps,
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

    def validate_features(self) -> None:
        """Validate that the dataset has the required action features."""
        # Check image and environment state features
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

        # Check for single action key with concatenated continuous+discrete format
        if "action" not in self.output_features:
            raise ValueError(
                "HybridDiffusionPolicy requires 'action' feature in the dataset output_features. "
                "Action should contain concatenated continuous and discrete actions."
            )

        # Validate action dimensions match expected structure
        action_feature = self.output_features["action"]
        if len(action_feature.shape) != 1:
            raise ValueError(
                f"action feature must be 1-dimensional, got shape {action_feature.shape}"
            )

        expected_total_dim = self.continuous_action_dim + self.discrete_action_dim
        actual_dim = action_feature.shape[0]
        if actual_dim != expected_total_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {expected_total_dim} "
                f"(continuous={self.continuous_action_dim} + discrete={self.discrete_action_dim}), "
                f"got {actual_dim}"
            )
