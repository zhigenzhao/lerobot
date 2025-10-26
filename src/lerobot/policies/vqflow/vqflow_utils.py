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

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.utils import ModelWrapper
from torch import Tensor
from vector_quantize_pytorch import VectorQuantize

from lerobot.policies.vqflow.configuration_vqflow import VQFlowConfig
from lerobot.policies.vqflow.vqflow_conv_modules import TemporalEncoder, TemporalDecoder


class VQ(nn.Module):
    """Wrapper for vector-quantize-pytorch VectorQuantize to match VQFlow interface."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        use_cosine_sim: bool = False,
        threshold_ema_dead_code: int = 2,
        kmeans_init: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size

        # Use advanced VQ from vector-quantize-pytorch
        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            decay=0.99,  # EMA decay
            commitment_weight=1.0,
            use_cosine_sim=use_cosine_sim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
            kmeans_iters=10
        )

        self.register_buffer("frozen", torch.tensor(False))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: Input tensor (B, ..., dim)
        Returns:
            quantized: Quantized tensor (B, ..., dim)
            indices: Quantization indices (B, ...) - flat indices for single layer
            vq_loss: Vector quantization loss
        """
        # VectorQuantize returns (quantized, indices, loss)
        quantized, indices, vq_loss = self.vq(x)
        # Return flat indices directly (no extra dimension needed)
        return quantized, indices, vq_loss

    def get_codebook_vectors(self, indices: Tensor) -> Tensor:
        """Get codebook vectors from indices.

        Args:
            indices: (B, ...) flat indices for single layer
        Returns:
            vectors: (B, ..., dim) codebook vectors
        """
        # Use get_codes_from_indices method directly
        vectors = self.vq.get_codes_from_indices(indices)
        return vectors

    def freeze(self):
        """Freeze VQ layers for phase 2 training."""
        self.frozen = torch.tensor(True)
        self.vq.freeze_codebook = True
        for param in self.vq.parameters():
            param.requires_grad = False


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
        self.vq_layer = VQ(
            dim=config.vqvae_embedding_dim,
            codebook_size=config.vqvae_n_embed,
            use_cosine_sim=config.vqvae_use_cosine_sim,
            threshold_ema_dead_code=config.vqvae_threshold_ema_dead_code,
            kmeans_init=config.vqvae_kmeans_init
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
        """Quantize embeddings using VQ."""
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
            indices: (B, target_tokens) discrete indices for each token
        """
        # Encode to embeddings and quantize directly
        z = self.encode(actions)  # (B, target_tokens, embedding_dim)
        _, indices, _ = self.quantize(z)  # VQ can handle batched input directly
        return indices
    
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """Decode from discrete indices to continuous actions.

        Args:
            indices: (B, target_tokens) discrete indices for single layer VQ
        Returns:
            actions: (B, action_horizon, action_dim) continuous actions
        """
        # Get embeddings from indices and decode directly
        z_q = self.vq_layer.get_codebook_vectors(indices)  # (B, target_tokens, embedding_dim)
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
        # Encode, quantize, and decode
        z = self.encode(actions)  # (B, target_tokens, embedding_dim)
        z_q, indices, vq_loss = self.quantize(z)  # VQ handles batched input directly
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