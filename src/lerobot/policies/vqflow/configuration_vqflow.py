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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("vqflow")
@dataclass
class VQFlowConfig(PreTrainedConfig):
    """Configuration class for VQFlow Policy.

    VQFlow combines Vector Quantized Variational Autoencoder (VQVAE) with Discrete Flow Matching 
    using a DiT (Diffusion Transformer) backbone. The policy operates in two phases:
    
    Phase 1: Train VQVAE to discretize continuous actions into discrete tokens
    Phase 2: Train discrete flow matching DiT to model token generation conditioned on observations

    Defaults are configured for training with dual-arm manipulation providing proprioceptive 
    and multi-camera observations.

    Args:
        # Input/Output Structure
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Model action prediction horizon (total sequence length).
        n_action_steps: Number of action steps to run in environment per policy invocation.
        action_chunk_size: Size of action chunks for VQVAE quantization.
        
        # VQVAE Parameters (Phase 1)
        vqvae_n_embed: Size of each RVQ codebook (number of discrete codes per layer).
        vqvae_embedding_dim: Dimensionality of quantized embeddings.
        vqvae_num_layers: Number of residual quantization layers (hierarchical codes).
        vqvae_enc_hidden_dim: Hidden layer size in VQVAE encoder/decoder MLPs.
        n_vqvae_training_steps: Number of steps to train VQVAE before freezing.
        vqvae_commitment_beta: Weight for commitment loss in VQVAE training.
        
        # DiT Architecture (Phase 2)
        hidden_size: Hidden dimension of DiT transformer blocks.
        num_layers: Number of DiT transformer layers.
        num_attention_heads: Number of attention heads in each DiT block.
        cross_attention_dim: Dimension of cross-attention conditioning from observations.
        attention_dropout: Dropout probability in attention layers.
        use_adaln_zero: Whether to use AdaLN-Zero initialization for better training stability.
        
        # Discrete Flow Matching Parameters
        source_distribution: Type of source distribution ("uniform" or "mask").
        scheduler_power: Power parameter for polynomial convex scheduler (σ_t = (1-t)^n).
        num_integration_steps: Number of ODE integration steps during inference.
        flow_epsilon: Small epsilon to avoid numerical issues at t=1.
        
        # Vision Processing (same as other policies)
        vision_backbone: ResNet backbone for image encoding.
        crop_shape: Image crop shape for preprocessing.
        crop_is_random: Whether to use random crops during training.
        pretrained_backbone_weights: Pretrained weights for vision backbone.
        use_group_norm: Replace BatchNorm with GroupNorm in backbone.
        spatial_softmax_num_keypoints: Number of keypoints for spatial softmax pooling.
        use_separate_rgb_encoder_per_camera: Whether to use separate encoders per camera.
        
        # Time Embedding
        fm_time_embed_dim: Embedding dimension for flow matching continuous time.
        fm_min_period: Minimum period for flow matching sinusoidal embeddings.
        fm_max_period: Maximum period for flow matching sinusoidal embeddings.
        
        # Training Configuration
        phase1_lr: Learning rate for Phase 1 (VQVAE training).
        phase1_weight_decay: Weight decay for Phase 1.
        phase2_lr: Learning rate for Phase 2 (DiT training).  
        phase2_weight_decay: Weight decay for Phase 2.
        scheduler_warmup_steps: Number of warmup steps for learning rate scheduler.
    """

    # Input / output structure
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    action_chunk_size: int = 8  # Size of chunks for VQVAE quantization
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    
    # Compatibility with existing datasets
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # VQVAE Parameters (Phase 1: Action Discretization)
    vqvae_n_embed: int = 32                    # Larger codebook than VQ-BeT for richer representation
    vqvae_embedding_dim: int = 256             # Embedding dimension for quantized vectors
    vqvae_num_layers: int = 2                  # Number of RVQ layers for hierarchical coding
    vqvae_enc_hidden_dim: int = 128            # Hidden dimension in encoder/decoder MLPs
    n_vqvae_training_steps: int = 20000        # Steps to train VQVAE before switching to phase 2
    vqvae_commitment_beta: float = 0.25        # Weight for commitment loss
    
    # DiT Architecture (Phase 2: Discrete Flow Matching)  
    hidden_size: int = 768                     # DiT hidden dimension
    num_layers: int = 12                       # Number of DiT transformer blocks
    num_attention_heads: int = 12              # Number of attention heads
    cross_attention_dim: int = 512             # Observation conditioning dimension
    attention_dropout: float = 0.1             # Attention dropout probability
    use_adaln_zero: bool = True                # Use AdaLN-Zero initialization
    
    # Discrete Flow Matching Configuration
    source_distribution: str = "uniform"       # Source distribution type: "uniform" or "mask"  
    scheduler_power: float = 2.0               # Power for polynomial scheduler: σ_t = (1-t)^n
    num_integration_steps: int = 50            # ODE integration steps for inference
    flow_epsilon: float = 1e-3                # Small epsilon to avoid t=1 singularity
    
    # Vision backbone (same as other policies)
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # Flow matching time embedding
    fm_time_embed_dim: int = 256
    fm_min_period: float = 4e-3
    fm_max_period: float = 4.0
    
    # Loss computation
    do_mask_loss_for_padding: bool = False
    
    # Training presets (two-phase training)
    phase1_lr: float = 1e-3                    # VQVAE learning rate
    phase1_weight_decay: float = 1e-4          # VQVAE weight decay
    phase2_lr: float = 1e-4                    # DiT learning rate  
    phase2_weight_decay: float = 1e-6          # DiT weight decay
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        
        # Validate VQVAE parameters
        if self.vqvae_n_embed <= 0:
            raise ValueError(f"vqvae_n_embed must be positive, got {self.vqvae_n_embed}")
        
        if self.vqvae_num_layers <= 0:
            raise ValueError(f"vqvae_num_layers must be positive, got {self.vqvae_num_layers}")
            
        if self.n_vqvae_training_steps <= 0:
            raise ValueError(f"n_vqvae_training_steps must be positive, got {self.n_vqvae_training_steps}")
        
        # Validate DiT parameters
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
            
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        
        # Validate flow matching parameters
        if self.source_distribution not in ["uniform", "mask"]:
            raise ValueError(f"source_distribution must be 'uniform' or 'mask', got {self.source_distribution}")
            
        if self.scheduler_power <= 0:
            raise ValueError(f"scheduler_power must be positive, got {self.scheduler_power}")
            
        if self.num_integration_steps <= 0:
            raise ValueError(f"num_integration_steps must be positive, got {self.num_integration_steps}")
        
        # Validate vision backbone
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"vision_backbone must be a ResNet variant, got {self.vision_backbone}")
        
        # Validate time embedding parameters
        if self.fm_time_embed_dim <= 0:
            raise ValueError(f"fm_time_embed_dim must be positive, got {self.fm_time_embed_dim}")
            
        if self.fm_min_period >= self.fm_max_period:
            raise ValueError(f"fm_min_period ({self.fm_min_period}) must be less than fm_max_period ({self.fm_max_period})")
    
    @property 
    def vocab_size(self) -> int:
        """Calculate vocabulary size for discrete flow matching.
        
        For RVQ with multiple layers, we flatten the hierarchical codes into a single vocabulary.
        vocab_size = codebook_size^num_layers + 1 (for optional mask token)
        """
        base_vocab = self.vqvae_n_embed ** self.vqvae_num_layers
        return base_vocab + 1 if self.source_distribution == "mask" else base_vocab
    
    @property
    def mask_token_id(self) -> int:
        """ID of the mask token (if using mask source distribution)."""
        if self.source_distribution != "mask":
            raise ValueError("mask_token_id only valid when source_distribution='mask'")
        return self.vocab_size - 1

    def get_optimizer_preset(self) -> AdamConfig:
        """Get optimizer configuration for current training phase."""
        # Note: The actual training will switch between phase1 and phase2 parameters
        # This returns a default that will be overridden during training
        return AdamConfig(
            lr=self.phase2_lr,  # Default to phase 2
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.phase2_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """Validate that required input features are present."""
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"crop_shape {self.crop_shape} should fit within image shape {image_ft.shape} "
                        f"for {key}."
                    )

        # Check that all input images have the same shape
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"Image {key} shape {image_ft.shape} does not match "
                        f"{first_image_key} shape {first_image_ft.shape}. All images must have the same shape."
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