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

"""DiT blocks adapted for discrete flow matching in VQFlow."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiscreteFlowTimestepEmbedding(nn.Module):
    """
    Time embedding for discrete flow matching.
    
    Maps continuous time t ∈ [0,1] to embedding space using sinusoidal encoding
    with configurable frequency range optimized for flow matching.
    """

    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding_dim ({dim}) must be divisible by 2")
        
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        
        # Precompute geometric progression of periods
        half_dim = dim // 2
        log_min = math.log(min_period)
        log_max = math.log(max_period)
        freqs = torch.exp(torch.linspace(log_min, log_max, half_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) tensor of time values in [0,1] for discrete flow matching
        Returns:
            (B, dim) time embeddings
        """
        # Handle scalar input from ODE solver
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Apply frequencies: t * freqs * 2π
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        
        # Sinusoidal encoding
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return emb


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization for timestep conditioning in DiT blocks."""
    
    def __init__(self, hidden_size: int, timestep_embed_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.silu = nn.SiLU()
        self.linear = nn.Linear(timestep_embed_dim, 2 * hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
    
    def forward(self, x: Tensor, timestep_emb: Tensor) -> Tensor:
        """
        Args:
            x: (B, seq_len, hidden_size) input tensor
            timestep_emb: (B, timestep_embed_dim) timestep embedding
        Returns:
            (B, seq_len, hidden_size) modulated tensor
        """
        emb = self.linear(self.silu(timestep_emb))
        scale, shift = emb.chunk(2, dim=1)
        
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class AdaLayerNormZero(nn.Module):
    """
    AdaLayerNorm with zero-initialization for stable training.
    
    Generates 6 parameters for gated residual connections:
    - 2 for self-attention (scale, gate) 
    - 2 for cross-attention (scale, gate)
    - 2 for MLP (scale, gate)
    """
    
    def __init__(self, hidden_size: int, timestep_embed_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.silu = nn.SiLU()
        # 6 parameters: 3 scales + 3 gates
        self.linear = nn.Linear(timestep_embed_dim, 6 * hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        
        # Zero initialization for stability
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: Tensor, timestep_emb: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, seq_len, hidden_size) input tensor
            timestep_emb: (B, timestep_embed_dim) timestep embedding
        Returns:
            Tuple of (norm_x, self_attn_scale, self_attn_gate, cross_attn_scale, cross_attn_gate, mlp_gate)
        """
        emb = self.linear(self.silu(timestep_emb))  # (B, 6 * hidden_size)
        
        # Split into 6 components
        (
            self_attn_scale, 
            self_attn_gate,
            cross_attn_scale,
            cross_attn_gate, 
            mlp_scale,
            mlp_gate
        ) = emb.chunk(6, dim=1)
        
        # Normalize input
        norm_x = self.norm(x)
        
        return (
            norm_x,
            self_attn_scale.unsqueeze(1),   # (B, 1, hidden_size)
            self_attn_gate.unsqueeze(1),
            cross_attn_scale.unsqueeze(1),
            cross_attn_gate.unsqueeze(1),
            mlp_scale.unsqueeze(1),
            mlp_gate.unsqueeze(1)
        )


class DiscreteFlowDiTBlock(nn.Module):
    """
    DiT transformer block for discrete flow matching.
    
    Features:
    - Self-attention on token sequences
    - Cross-attention to observation encodings
    - AdaLN-Zero conditioning on timestep
    - Gated residual connections for stable training
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_attention_heads: int,
        cross_attention_dim: int,
        attention_dropout: float = 0.0,
        use_adaln_zero: bool = True,
        timestep_embed_dim: int = 256
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = hidden_size // num_attention_heads
        self.use_adaln_zero = use_adaln_zero
        
        # Adaptive normalization
        if use_adaln_zero:
            self.norm1 = AdaLayerNormZero(hidden_size, timestep_embed_dim)
        else:
            self.norm1 = AdaLayerNorm(hidden_size, timestep_embed_dim)
            self.norm2 = AdaLayerNorm(hidden_size, timestep_embed_dim)
            self.norm3 = AdaLayerNorm(hidden_size, timestep_embed_dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Cross-attention to observations
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            kdim=cross_attention_dim,
            vdim=cross_attention_dim,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(attention_dropout)
        )
    
    def forward(
        self, 
        hidden_states: Tensor, 
        timestep_emb: Tensor, 
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            hidden_states: (B, seq_len, hidden_size) token embeddings
            timestep_emb: (B, timestep_embed_dim) timestep conditioning
            encoder_hidden_states: (B, obs_seq_len, cross_attention_dim) observation encodings
            attention_mask: (B, seq_len) mask for self-attention
        Returns:
            (B, seq_len, hidden_size) updated hidden states
        """
        if self.use_adaln_zero:
            # AdaLN-Zero with gated residual connections
            (
                norm_hidden_states,
                self_attn_scale,
                self_attn_gate, 
                cross_attn_scale,
                cross_attn_gate,
                mlp_scale,
                mlp_gate
            ) = self.norm1(hidden_states, timestep_emb)
            
            # Self-attention with gating
            attn_hidden_states, _ = self.self_attn(
                query=norm_hidden_states * (1 + self_attn_scale),
                key=norm_hidden_states * (1 + self_attn_scale),
                value=norm_hidden_states * (1 + self_attn_scale),
                key_padding_mask=attention_mask,
                need_weights=False
            )
            hidden_states = hidden_states + self_attn_gate * attn_hidden_states
            
            # Cross-attention with gating (if observations provided)
            if encoder_hidden_states is not None:
                cross_hidden_states, _ = self.cross_attn(
                    query=norm_hidden_states * (1 + cross_attn_scale),
                    key=encoder_hidden_states,
                    value=encoder_hidden_states,
                    need_weights=False
                )
                hidden_states = hidden_states + cross_attn_gate * cross_hidden_states
            
            # MLP with gating
            mlp_hidden_states = self.mlp(norm_hidden_states * (1 + mlp_scale))
            hidden_states = hidden_states + mlp_gate * mlp_hidden_states
            
        else:
            # Standard AdaLN
            # Self-attention
            norm_hidden_states = self.norm1(hidden_states, timestep_emb)
            attn_hidden_states, _ = self.self_attn(
                query=norm_hidden_states,
                key=norm_hidden_states, 
                value=norm_hidden_states,
                key_padding_mask=attention_mask,
                need_weights=False
            )
            hidden_states = hidden_states + attn_hidden_states
            
            # Cross-attention
            if encoder_hidden_states is not None:
                norm_hidden_states = self.norm2(hidden_states, timestep_emb)
                cross_hidden_states, _ = self.cross_attn(
                    query=norm_hidden_states,
                    key=encoder_hidden_states,
                    value=encoder_hidden_states,
                    need_weights=False
                )
                hidden_states = hidden_states + cross_hidden_states
            
            # MLP
            norm_hidden_states = self.norm3(hidden_states, timestep_emb)
            mlp_hidden_states = self.mlp(norm_hidden_states)
            hidden_states = hidden_states + mlp_hidden_states
        
        return hidden_states


class DiscreteFlowDiTEmbeddings(nn.Module):
    """Embeddings for discrete tokens and positions in DiT."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int, 
        max_position_embeddings: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Token embeddings for discrete vocabulary
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Learnable position embeddings for action sequence
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with appropriate scaling."""
        # Token embeddings: standard normal scaled by sqrt(hidden_size)
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        
        # Position embeddings: standard normal 
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, seq_len) discrete token indices
        Returns:
            (B, seq_len, hidden_size) embedded tokens with positions
        """
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)  # (B, seq_len, hidden_size)
        
        # Position embeddings  
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.position_embeddings(positions)  # (B, seq_len, hidden_size)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        return embeddings


class VQFlowRgbEncoder(nn.Module):
    """
    RGB encoder for VQFlow observations.
    
    Reuses the same ResNet + spatial softmax architecture as other LeRobot policies
    for consistency and proven effectiveness.
    """
    
    def __init__(self, config):
        super().__init__()
        import torchvision
        from lerobot.policies.utils import get_output_shape
        
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

        # Set up ResNet backbone
        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in pretrained model without ruining weights!")
            self.backbone = self._replace_batch_norm_with_group_norm(self.backbone)

        # Spatial softmax pooling
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        
        # Output projection to cross-attention dimension
        self.out = nn.Linear(self.feature_dim, config.cross_attention_dim)
        self.relu = nn.ReLU()

    def _replace_batch_norm_with_group_norm(self, module: nn.Module) -> nn.Module:
        """Replace BatchNorm2d with GroupNorm in the module tree."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                # Replace with GroupNorm (16 groups)
                num_groups = max(1, child.num_features // 16)
                group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features)
                setattr(module, name, group_norm)
            else:
                self._replace_batch_norm_with_group_norm(child)
        return module

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1]
        Returns:
            (B, cross_attention_dim) image features
        """
        # Optional cropping
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        
        # Extract backbone features and pool
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        
        # Project to cross-attention dimension
        x = self.relu(self.out(x))
        
        return x


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation for converting 2D feature maps to keypoints.
    
    Taken from robomimic implementation - converts feature maps to spatial coordinates
    representing centers of mass of activations.
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape: (C, H, W) input feature map shape
            num_kp: number of keypoints in output. If None, uses input channels.
        """
        super().__init__()
        
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # Coordinate grid for spatial expectations
        import numpy as np
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), 
            np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps
        Returns:
            (B, K, 2) spatial coordinates of keypoints
        """
        if self.nets is not None:
            features = self.nets(features)

        # Flatten spatial dimensions: (B, K, H, W) -> (B * K, H * W)
        features = features.reshape(-1, self._in_h * self._in_w)
        
        # Softmax normalization 
        attention = F.softmax(features, dim=-1)
        
        # Compute spatial expectation: (B * K, H * W) x (H * W, 2) -> (B * K, 2)
        expected_xy = attention @ self.pos_grid
        
        # Reshape to (B, K, 2)
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints