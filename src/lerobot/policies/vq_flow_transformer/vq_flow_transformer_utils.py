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

"""Utility classes and functions for VQ Flow Transformer policy implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.vq_flow_transformer.configuration_vq_flow_transformer import VQFlowTransformerConfig


class VQFlowMLP(nn.Module):
    """Multi-layer perceptron with ReLU activations for VQVAE encoder/decoder."""

    def __init__(self, in_channels: int, hidden_channels: list[int]):
        super().__init__()
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_channels[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class VectorQuantize(nn.Module):
    """Single layer vector quantization."""

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        embed = torch.randn(codebook_size, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("freeze_codebook", torch.tensor(False))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: Input tensor (..., dim)
        Returns:
            quantized: Quantized tensor (..., dim)
            indices: Nearest codebook indices (...)
            vq_loss: VQ loss scalar
        """
        flatten = x.reshape(-1, self.dim)  # (N, dim)

        # Compute distances to codebook
        dist = torch.sum(flatten**2, dim=1, keepdim=True) + \
               torch.sum(self.embed**2, dim=1) - \
               2 * torch.matmul(flatten, self.embed.t())

        # Find nearest codebook entries
        indices = torch.argmin(dist, dim=1)  # (N,)
        encodings = F.one_hot(indices, self.codebook_size).float()  # (N, codebook_size)
        indices = indices.view(*x.shape[:-1])  # Restore original shape except last dim

        # Quantize
        quantized_flatten = torch.matmul(encodings, self.embed)  # (N, dim)
        quantized = quantized_flatten.view_as(x)  # Restore original shape

        # VQ Loss: commitment loss + codebook loss
        commitment_loss = F.mse_loss(quantized.detach(), x)
        embedding_loss = F.mse_loss(quantized, x.detach())
        vq_loss = commitment_loss + embedding_loss

        # EMA update of codebook (only during training and when not frozen)
        if self.training and not self.freeze_codebook:
            # Update cluster sizes
            cluster_size = encodings.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # Update embeddings
            embed_sum = torch.matmul(encodings.t(), flatten)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Normalize embeddings
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, indices, vq_loss

    def get_codebook_vector(self, indices: Tensor) -> Tensor:
        """Get codebook vectors for given indices."""
        return F.embedding(indices, self.embed)


class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization for VQ Flow Transformer.

    Uses multiple layers of vector quantization with residual connections.
    """

    def __init__(self, dim: int, num_quantizers: int, codebook_size: int):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        # VQ layers
        self.vq_layers = nn.ModuleList([
            VectorQuantize(dim=dim, codebook_size=codebook_size)
            for _ in range(num_quantizers)
        ])

        # Frozen flag for phase switching
        self.register_buffer("frozen", torch.tensor(False))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: Input tensor (B, ..., dim)
        Returns:
            quantized: Quantized tensor (B, ..., dim)
            indices: Quantization indices (B, ..., num_quantizers)
            vq_loss: Vector quantization loss
        """
        quantized_out = 0.0
        residual = x
        all_indices = []
        all_losses = []

        for layer in self.vq_layers:
            quantized, indices, vq_loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(vq_loss)

        # Stack indices: (B, ..., num_quantizers)
        all_indices = torch.stack(all_indices, dim=-1)

        # Sum losses
        total_vq_loss = torch.stack(all_losses).sum()

        return quantized_out, all_indices, total_vq_loss

    def get_codebook_vectors(self, indices: Tensor) -> Tensor:
        """Get codebook vectors from indices.

        Args:
            indices: (B, ..., num_quantizers) indices for each layer
        Returns:
            vectors: (B, ..., num_quantizers, dim) codebook vectors
        """
        vectors = []
        for i, layer in enumerate(self.vq_layers):
            layer_indices = indices[..., i]
            layer_vectors = layer.get_codebook_vector(layer_indices)
            vectors.append(layer_vectors)

        return torch.stack(vectors, dim=-2)  # (B, ..., num_quantizers, dim)

    def freeze(self):
        """Freeze VQ layers for phase 2 training."""
        self.frozen = torch.tensor(True)
        for layer in self.vq_layers:
            layer.freeze_codebook = torch.tensor(True)
            for param in layer.parameters():
                param.requires_grad = False


class VQFlowVAE(nn.Module):
    """VQVAE for action discretization in VQ Flow Transformer."""

    def __init__(self, config: VQFlowTransformerConfig):
        super().__init__()
        self.config = config

        # Calculate input/output dimensions
        action_dim = config.action_feature.shape[0]
        input_dim = action_dim * config.action_chunk_size

        # Encoder: continuous actions -> embedding space
        self.encoder = VQFlowMLP(
            in_channels=input_dim,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                config.vqvae_embedding_dim
            ]
        )

        # Vector quantization
        self.vq_layer = ResidualVQ(
            dim=config.vqvae_embedding_dim,
            num_quantizers=config.vqvae_num_layers,
            codebook_size=config.vqvae_n_embed
        )

        # Decoder: embedding space -> continuous actions
        self.decoder = VQFlowMLP(
            in_channels=config.vqvae_embedding_dim,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                input_dim
            ]
        )

        # Training phase tracking
        self.register_buffer("optimized_steps", torch.tensor(0))
        self.register_buffer("phase1_complete", torch.tensor(False))

    def encode(self, actions: Tensor) -> Tensor:
        """Encode actions to continuous embeddings."""
        # Flatten action chunks: (B, chunk_size, action_dim) -> (B, chunk_size * action_dim)
        B, chunk_size, action_dim = actions.shape
        actions_flat = actions.reshape(B, chunk_size * action_dim)

        # Encode to embedding space
        z = self.encoder(actions_flat)
        return z

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize embeddings using ResidualVQ."""
        return self.vq_layer(z)

    def decode(self, z_q: Tensor) -> Tensor:
        """Decode quantized embeddings back to actions."""
        # Decode to flattened actions
        actions_flat = self.decoder(z_q)

        # Reshape back to action chunks
        B = z_q.shape[0]
        actions = actions_flat.reshape(B, self.config.action_chunk_size, -1)
        return actions

    def encode_to_indices(self, actions: Tensor) -> Tensor:
        """Encode actions directly to discrete indices.

        Args:
            actions: (B, num_chunks, chunk_size, action_dim) action sequences
        Returns:
            indices: (B, num_chunks, num_layers) discrete indices
        """
        B, num_chunks, chunk_size, action_dim = actions.shape
        all_indices = []

        for i in range(num_chunks):
            chunk = actions[:, i]  # (B, chunk_size, action_dim)
            z = self.encode(chunk)
            _, indices, _ = self.quantize(z)
            all_indices.append(indices)

        return torch.stack(all_indices, dim=1)  # (B, num_chunks, num_layers)

    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """Decode from discrete indices to continuous actions.

        Args:
            indices: (B, num_chunks, num_layers) discrete indices
        Returns:
            actions: (B, num_chunks, chunk_size, action_dim) continuous actions
        """
        B, num_chunks, num_layers = indices.shape
        all_actions = []

        for i in range(num_chunks):
            chunk_indices = indices[:, i]  # (B, num_layers)

            # Get codebook vectors and sum across layers
            vectors = self.vq_layer.get_codebook_vectors(chunk_indices)  # (B, num_layers, dim)
            z_q = vectors.sum(dim=1)  # (B, dim)

            # Decode chunk
            actions = self.decode(z_q)  # (B, chunk_size, action_dim)
            all_actions.append(actions)

        # Stack chunks along time dimension
        return torch.stack(all_actions, dim=1)  # (B, num_chunks, chunk_size, action_dim)

    def forward(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass for training.

        Args:
            actions: (B, chunk_size, action_dim) action chunks
        Returns:
            actions_recon: Reconstructed actions
            indices: Quantization indices
            vq_loss: Vector quantization loss
            recon_loss: Reconstruction loss
        """
        # Encode
        z = self.encode(actions)

        # Quantize
        z_q, indices, vq_loss = self.quantize(z)

        # Decode
        actions_recon = self.decode(z_q)

        # Reconstruction loss
        recon_loss = F.l1_loss(actions, actions_recon)

        return actions_recon, indices, vq_loss, recon_loss

    def freeze(self):
        """Freeze VQVAE for phase 2 training."""
        self.phase1_complete = torch.tensor(True)
        self.vq_layer.freeze()
        for param in self.parameters():
            param.requires_grad = False
        # Set to eval mode like VQ-BeT does - critical for LayerNorm and EMA behavior
        self.eval()


def flatten_indices(indices: Tensor, codebook_size: int) -> Tensor:
    """Convert hierarchical RVQ indices to flat vocabulary indices.

    For RVQ with L layers and codebook size C, convert L-dimensional indices
    to single integer in range [0, C^L).

    Args:
        indices: (B, ..., num_layers) hierarchical indices
        codebook_size: Size of each layer's codebook
    Returns:
        flat_indices: (B, ...) flattened indices
    """
    num_layers = indices.shape[-1]

    # Convert to flat indices using base-C arithmetic
    flat_indices = torch.zeros(indices.shape[:-1], dtype=indices.dtype, device=indices.device)

    for i in range(num_layers):
        flat_indices = flat_indices * codebook_size + indices[..., i]

    return flat_indices


def unflatten_indices(flat_indices: Tensor, num_layers: int, codebook_size: int) -> Tensor:
    """Convert flat vocabulary indices back to hierarchical RVQ indices.

    Args:
        flat_indices: (B, ...) flattened indices
        num_layers: Number of RVQ layers
        codebook_size: Size of each layer's codebook
    Returns:
        indices: (B, ..., num_layers) hierarchical indices
    """
    # Convert flat indices to hierarchical using base-C arithmetic
    indices = []
    remainder = flat_indices.clone()

    for _ in range(num_layers):
        layer_idx = remainder % codebook_size
        indices.append(layer_idx)
        remainder = remainder // codebook_size

    # Reverse order (since we computed from least to most significant)
    indices.reverse()

    return torch.stack(indices, dim=-1)


class DiscreteFlowSinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for discrete flow matching time."""

    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding_dim ({dim}) must be divisible by 2")

        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period

        # Precompute fraction for geometric progression of periods
        self.register_buffer("fraction", torch.linspace(0.0, 1.0, dim // 2))

    def forward(self, pos: Tensor) -> Tensor:
        """
        Args:
            pos: (B,) tensor of positions (designed for [0,1] flow matching time)
        Returns:
            (B, dim) positional embeddings
        """
        # Handle 0D scalar input from ODE solver by adding batch dimension
        if pos.dim() == 0:
            pos = pos.unsqueeze(0)  # Convert () -> (1,)

        # Create geometric progression of periods from min to max
        period = self.min_period * (self.max_period / self.min_period) ** self.fraction

        # Compute sinusoidal inputs: pos * (2π / period)
        # Shape: (B, 1) * (1, dim//2) -> (B, dim//2)
        sinusoid_input = pos.unsqueeze(-1) * (1.0 / period * 2 * math.pi)

        # Concatenate sin and cos components
        return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization (from DiT)."""

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(cond_dim, 6 * hidden_size, bias=True)

        # Zero-out adaLN modulation layers
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, T, D) input tensor
            c: (B, cond_dim) conditioning tensor
        Returns:
            modulated_x: x after adaptive layer norm
            gate_msa: gate for multi-head attention
            gate_mlp: gate for MLP
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(c).chunk(6, dim=-1)

        # Normalize
        x = self.norm(x)

        # Apply modulation
        x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        return x, gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)


class DiscreteFlowTransformerBlock(nn.Module):
    """Transformer block for discrete flow matching with cross-attention."""

    def __init__(self, config: VQFlowTransformerConfig, cross_attention_dim: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.attention_embed_dim

        # Self-attention
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.n_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # Cross-attention to observations
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.n_attention_heads,
            dropout=config.attention_dropout,
            kdim=cross_attention_dim,
            vdim=cross_attention_dim,
            batch_first=True
        )

        # MLP
        self.norm3 = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.GELU(),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.Dropout(config.attention_dropout)
        )

        # Using regular transformer approach without AdaLN-Zero

    def forward(self, x: Tensor, context: Tensor, timestep_emb: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (B, T, D) input sequence
            context: (B, T_ctx, D_ctx) context from observations
            timestep_emb: (B, time_embed_dim) timestep embeddings
            attn_mask: attention mask for self-attention
        Returns:
            (B, T, D) output sequence
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out

        # Cross-attention to context
        residual = x
        x = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(x, context, context)
        x = residual + cross_attn_out

        # MLP
        residual = x
        x = self.norm3(x)
        x = residual + self.mlp(x)

        return x


class PolynomialConvexScheduler:
    """Polynomial convex scheduler for discrete flow matching."""

    def __init__(self, power: float = 1.0, epsilon: float = 0.1):
        self.power = power
        self.epsilon = epsilon

    def get_alpha(self, t: Tensor) -> Tensor:
        """Get alpha coefficient for time t."""
        return t ** self.power

    def get_jump_coefficient(self, t: Tensor) -> Tensor:
        """Get jump coefficient with numerical stability."""
        alpha_t = self.get_alpha(t)
        # Add epsilon to avoid division by zero
        return alpha_t / (1 - alpha_t + self.epsilon)