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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FlowMatchingSinusoidalPosEmb(nn.Module):
    """Flow matching sinusoidal positional embeddings for continuous time in [0,1].
    
    Based on flow matching transformer implementation with configurable periods.
    """

    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, pos: Tensor) -> Tensor:
        """
        Args:
            pos: (B,) tensor of positions in [0,1] (designed for flow matching time)
                 Can also handle 0D scalar tensors from ODE solver
        Returns:
            (B, dim) positional embeddings
        """
        # Handle 0D scalar input from ODE solver by adding batch dimension
        if pos.dim() == 0:
            pos = pos.unsqueeze(0)  # Convert () -> (1,)
        
        device = pos.device
        half_dim = self.dim // 2
        
        # Create geometric progression of frequencies
        # From flow matching transformer: natural log progression between periods
        log_min = math.log(self.min_period)
        log_max = math.log(self.max_period)
        freqs = torch.exp(torch.linspace(log_min, log_max, half_dim, device=device))
        
        # Apply frequencies to positions
        args = pos.unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return emb


# Import all other components from diffusion_dit that remain unchanged
from lerobot.policies.diffusion_dit.dit_blocks import (
    AdaLayerNorm,
    AdaLayerNormZero,
    FeedForward,
    PositionalEncoding,
)


class DiTBlock(nn.Module):
    """DiT Transformer Block with hybrid conditioning for flow matching.
    
    Identical to diffusion DiT block - only the time embedding input changes.
    Architecture follows the DiT paper with AdaLN-Zero gating.
    Uses PyTorch's built-in MultiheadAttention for efficiency.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        timestep_embed_dim: int,
        attention_dropout: float = 0.1,
        cross_attention_dim: Optional[int] = None,
        use_adaln_zero: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_zero = use_adaln_zero
        self.has_cross_attention = cross_attention_dim is not None
        
        # AdaLN normalization
        if use_adaln_zero:
            self.norm1 = AdaLayerNormZero(hidden_size, timestep_embed_dim, norm_eps=norm_eps)
            if self.has_cross_attention:
                self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
            self.norm3 = AdaLayerNormZero(hidden_size, timestep_embed_dim, norm_eps=norm_eps)
        else:
            self.norm1 = AdaLayerNorm(hidden_size, timestep_embed_dim, norm_eps=norm_eps)
            if self.has_cross_attention:
                self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
            self.norm3 = AdaLayerNorm(hidden_size, timestep_embed_dim, norm_eps=norm_eps)
        
        # Self-attention using PyTorch's MultiheadAttention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        # Cross-attention for global conditioning
        if self.has_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=attention_dropout,
                kdim=cross_attention_dim,
                vdim=cross_attention_dim,
                batch_first=True,
            )
        
        # Feed-forward network
        self.ff = FeedForward(hidden_size, dropout=attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if self.use_adaln_zero:
            # Self-attention with AdaLN-Zero
            norm_hidden_states, (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
                self.norm1(hidden_states, timestep_emb)
            
            attn_output, _ = self.self_attn(norm_hidden_states, norm_hidden_states, norm_hidden_states,
                                          attn_mask=attention_mask)
            hidden_states = hidden_states + gate_msa[:, None, :] * attn_output
            
            # Cross-attention (no AdaLN modulation)
            if self.has_cross_attention and encoder_hidden_states is not None:
                norm_hidden_states = self.norm2(hidden_states)
                attn_output, _ = self.cross_attn(norm_hidden_states, encoder_hidden_states, encoder_hidden_states)
                hidden_states = hidden_states + attn_output
            
            # Feed-forward with AdaLN-Zero
            # For MLP, we need to get fresh modulation parameters
            norm_hidden_states, (shift_mlp_new, scale_mlp_new, gate_mlp_new, _, _, _) = \
                self.norm3(hidden_states, timestep_emb)
            
            ff_output = self.ff(norm_hidden_states)
            hidden_states = hidden_states + gate_mlp_new[:, None, :] * ff_output
        
        else:
            # Standard AdaLN without gating
            # Self-attention
            norm_hidden_states = self.norm1(hidden_states, timestep_emb)
            attn_output, _ = self.self_attn(norm_hidden_states, norm_hidden_states, norm_hidden_states,
                                          attn_mask=attention_mask)
            hidden_states = hidden_states + attn_output
            
            # Cross-attention
            if self.has_cross_attention and encoder_hidden_states is not None:
                norm_hidden_states = self.norm2(hidden_states)
                attn_output, _ = self.cross_attn(norm_hidden_states, encoder_hidden_states, encoder_hidden_states)
                hidden_states = hidden_states + attn_output
            
            # Feed-forward
            norm_hidden_states = self.norm3(hidden_states, timestep_emb)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = hidden_states + ff_output
        
        return hidden_states


class FlowMatchingTimestepEmbedding(nn.Module):
    """Timestep embedding network for flow matching continuous time conditioning.
    
    Uses FlowMatchingSinusoidalPosEmb for continuous [0,1] time encoding.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        self.time_emb = FlowMatchingSinusoidalPosEmb(embed_dim, min_period, max_period)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(timesteps)
        return self.mlp(emb)