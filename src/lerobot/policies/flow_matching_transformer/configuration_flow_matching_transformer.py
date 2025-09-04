#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion_transformer.configuration_diffusion_transformer import (
    DiffusionTransformerConfig,
)


@PreTrainedConfig.register_subclass("flow_matching_transformer")
@dataclass
class FlowMatchingTransformerConfig(DiffusionTransformerConfig):
    """Configuration class for FlowMatchingTransformerPolicy.

    Flow Matching Policy using Transformer architecture as backbone with flow matching
    for generative modeling instead of denoising diffusion.
    
    Inherits most parameters from DiffusionTransformerConfig but replaces diffusion-specific
    parameters with flow matching parameters.

    Args:
        flow_matching_type: Type of flow matching to use. Currently supports "CondOT" 
            (Conditional Optimal Transport).
        num_integration_steps: Number of integration steps for ODE solver during inference.
        fm_time_embed_dim: Embedding dimension for flow matching time encoding. This replaces
            diffusion_step_embed_dim.
        fm_min_period: Minimum period for flow matching positional embeddings.
        fm_max_period: Maximum period for flow matching positional embeddings.
    """

    # Flow matching specific parameters (replacing diffusion parameters)
    flow_matching_type: str = "CondOT"
    num_integration_steps: int = 50
    fm_time_embed_dim: int = 128
    
    # Flow matching positional embedding parameters
    fm_min_period: float = 4e-3
    fm_max_period: float = 4.0

    def __post_init__(self):
        # Call parent post_init but skip diffusion-specific validations
        super(DiffusionTransformerConfig, self).__post_init__()

        """Input validation for transformer architecture."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        # Validate transformer parameters
        if self.attention_embed_dim % self.n_attention_heads != 0:
            raise ValueError(
                f"`attention_embed_dim` must be divisible by `n_attention_heads`. "
                f"Got {self.attention_embed_dim} and {self.n_attention_heads}."
            )

        if self.attention_dropout < 0.0 or self.attention_dropout >= 1.0:
            raise ValueError(f"`attention_dropout` must be in [0, 1). Got {self.attention_dropout}.")
        
        if self.embedding_dropout < 0.0 or self.embedding_dropout >= 1.0:
            raise ValueError(f"`embedding_dropout` must be in [0, 1). Got {self.embedding_dropout}.")
        
        if self.n_conditioning_layers < 0:
            raise ValueError(f"`n_conditioning_layers` must be non-negative. Got {self.n_conditioning_layers}.")
        
        if self.n_decoder_layers <= 0:
            raise ValueError(f"`n_decoder_layers` must be positive. Got {self.n_decoder_layers}.")

        # Validate flow matching parameters
        supported_flow_matching_types = ["CondOT"]
        if self.flow_matching_type not in supported_flow_matching_types:
            raise ValueError(
                f"`flow_matching_type` must be one of {supported_flow_matching_types}. "
                f"Got {self.flow_matching_type}."
            )

        if self.num_integration_steps <= 0:
            raise ValueError(f"`num_integration_steps` must be positive. Got {self.num_integration_steps}.")

        if self.fm_time_embed_dim <= 0:
            raise ValueError(f"`fm_time_embed_dim` must be positive. Got {self.fm_time_embed_dim}.")