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


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization for timestep conditioning.
    
    Based on the diffusers implementation: applies scale and shift modulation
    to layer normalization based on timestep embeddings.
    """
    
    def __init__(self, hidden_size: int, timestep_embed_dim: int, norm_elementwise_affine: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.silu = nn.SiLU()
        self.linear = nn.Linear(timestep_embed_dim, 2 * hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
    
    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            timestep_emb: Timestep embedding of shape (batch, embedding_dim)
            
        Returns:
            Modulated tensor with adaptive normalization applied
        """
        emb = self.linear(self.silu(timestep_emb))
        scale, shift = emb.chunk(2, dim=1)
        
        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class AdaLayerNormZero(nn.Module):
    """AdaLayerNorm with zero-initialization and gating.
    
    Based on the diffusers implementation: generates 6 parameters for
    gated residual connections in attention and MLP layers.
    """
    
    def __init__(self, hidden_size: int, timestep_embed_dim: int, norm_elementwise_affine: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.silu = nn.SiLU()
        self.linear = nn.Linear(timestep_embed_dim, 6 * hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        
        # Zero initialization for better training stability
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            timestep_emb: Timestep embedding of shape (batch, embedding_dim)
            
        Returns:
            Tuple of (normalized_x, (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp))
        """
        emb = self.linear(self.silu(timestep_emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        
        x = self.norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        
        return x, (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size
        
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """DiT Transformer Block with hybrid conditioning.
    
    Combines AdaLayerNorm for timestep conditioning with cross-attention for global conditioning.
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


class TimestepEmbedding(nn.Module):
    """Timestep embedding network for diffusion conditioning.
    
    Uses the same sinusoidal embedding as the diffusion transformer for consistency.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.time_emb = DiffusionSinusoidalPosEmb(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.time_emb(timesteps)
        return self.mlp(emb)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence modeling."""
    
    def __init__(self, hidden_size: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]