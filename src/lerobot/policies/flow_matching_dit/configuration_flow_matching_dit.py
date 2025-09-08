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
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion_dit.configuration_diffusion_dit import DiffusionDiTConfig


@PreTrainedConfig.register_subclass("flow_matching_dit")
@dataclass
class FlowMatchingDiTConfig(DiffusionDiTConfig):
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
        # Call parent's parent to skip diffusion-specific validations
        super(DiffusionDiTConfig, self).__post_init__()

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