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

"""Utility classes and functions for VQFlow policy implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.utils import ModelWrapper
from torch import Tensor

from lerobot.policies.vqflow.configuration_vqflow import VQFlowConfig
from lerobot.policies.vqflow.vqflow_conv_modules import TemporalEncoder, TemporalDecoder


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


class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization for VQFlow.
    
    Adapted from VQ-BeT's implementation but simplified for VQFlow use case.
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


class VQFlowVAE(nn.Module):
    """VQVAE for action discretization in VQFlow."""
    
    def __init__(self, config: VQFlowConfig):
        super().__init__()
        self.config = config

        # Calculate dimensions
        action_dim = config.action_feature.shape[0]

        # Encoder: continuous actions -> token embeddings
        self.encoder = TemporalEncoder(
            action_dim=action_dim,
            embedding_dim=config.vqvae_embedding_dim,
            stage_channels=config.vqvae_encoder_channels,
            num_groups=config.vqvae_num_groups
        )

        # Calculate number of output tokens dynamically
        self.num_tokens = config.vqvae_target_tokens

        # Vector quantization
        self.vq_layer = ResidualVQ(
            dim=config.vqvae_embedding_dim,
            num_quantizers=config.vqvae_num_layers,
            codebook_size=config.vqvae_n_embed
        )

        # Decoder: token embeddings -> continuous actions
        self.decoder = TemporalDecoder(
            embedding_dim=config.vqvae_embedding_dim,
            action_dim=action_dim,
            stage_channels=config.vqvae_encoder_channels,
            num_groups=config.vqvae_num_groups
        )

        # Training phase tracking
        self.register_buffer("optimized_steps", torch.tensor(0))
        self.register_buffer("phase1_complete", torch.tensor(False))
    
    def encode(self, actions: Tensor) -> Tensor:
        """Encode actions to continuous embeddings."""
        # actions: (B, action_horizon, action_dim) -> (B, target_tokens, embedding_dim)
        z = self.encoder(actions)
        return z
    
    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize embeddings using ResidualVQ."""
        return self.vq_layer(z)
    
    def decode(self, z_q: Tensor) -> Tensor:
        """Decode quantized embeddings back to actions."""
        # z_q: (B, target_tokens, embedding_dim) -> (B, action_horizon, action_dim)
        actions = self.decoder(z_q)
        return actions
    
    def encode_to_indices(self, actions: Tensor) -> Tensor:
        """Encode actions directly to discrete indices.

        Args:
            actions: (B, action_horizon, action_dim) full action sequences
        Returns:
            indices: (B, target_tokens, num_layers) discrete indices for each token
        """
        # Encode full sequence to token embeddings
        z = self.encode(actions)  # (B, target_tokens, embedding_dim)

        # Quantize each token independently
        B, target_tokens, embedding_dim = z.shape
        z_flat = z.reshape(B * target_tokens, embedding_dim)

        _, indices_flat, _ = self.quantize(z_flat)  # (B * target_tokens, num_layers)
        indices = indices_flat.reshape(B, target_tokens, -1)  # (B, target_tokens, num_layers)

        return indices
    
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """Decode from discrete indices to continuous actions.

        Args:
            indices: (B, target_tokens, num_layers) discrete indices
        Returns:
            actions: (B, action_horizon, action_dim) continuous actions
        """
        B, target_tokens, num_layers = indices.shape

        # Reconstruct token embeddings from indices
        indices_flat = indices.reshape(B * target_tokens, num_layers)
        vectors = self.vq_layer.get_codebook_vectors(indices_flat)  # (B * target_tokens, num_layers, dim)
        z_q_flat = vectors.sum(dim=1)  # (B * target_tokens, dim)
        z_q = z_q_flat.reshape(B, target_tokens, -1)  # (B, target_tokens, embedding_dim)

        # Decode to full action sequence
        actions = self.decode(z_q)  # (B, action_horizon, action_dim)

        return actions
    
    def forward(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass for training.

        Args:
            actions: (B, action_horizon, action_dim) full action sequences
        Returns:
            actions_recon: Reconstructed actions
            indices: Quantization indices
            vq_loss: Vector quantization loss
            recon_loss: Reconstruction loss
        """
        # Encode to token embeddings
        z = self.encode(actions)  # (B, target_tokens, embedding_dim)

        # Quantize each token independently
        B, target_tokens, embedding_dim = z.shape
        z_flat = z.reshape(B * target_tokens, embedding_dim)

        z_q_flat, indices_flat, vq_loss = self.quantize(z_flat)

        # Reshape back
        z_q = z_q_flat.reshape(B, target_tokens, embedding_dim)
        indices = indices_flat.reshape(B, target_tokens, -1)

        # Decode to full sequence
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
        # Set to eval mode like VQ-BeT does - critical for GroupNorm and EMA behavior
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
    B = indices.shape[0]
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


class DiscreteModelWrapper(ModelWrapper):
    """Wrapper to make discrete DiT compatible with flow_matching library."""
    
    def __init__(self, model, config):
        super().__init__(model)
        self.config = config
    
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """
        Args:
            x: (B, horizon) discrete tokens
            t: (B,) continuous time in [0,1]
            **extras: Additional arguments like obs_encoding
        Returns:
            probs: (B, horizon, vocab_size) probability distributions
        """
        obs_encoding = extras.get("obs_encoding", None)
        logits = self.model(x, t, obs_encoding)
        return F.softmax(logits, dim=-1)