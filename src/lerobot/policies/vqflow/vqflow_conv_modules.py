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

"""Hierarchical Conv1d modules for VQFlow temporal encoding and decoding."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VQFlowConv1dBlock(nn.Module):
    """Basic building block: Conv1d -> GroupNorm -> SiLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Ensure num_groups is valid for GroupNorm
        num_groups = min(num_groups, out_channels // 4) if out_channels >= 4 else 1
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class VQFlowDecoderBlock(nn.Module):
    """Decoder block: Conv1d -> GroupNorm -> SiLU -> (Optional) Upsample"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, upsample: bool = False,
                 upsample_factor: int = 2, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Ensure num_groups is valid for GroupNorm
        num_groups = min(num_groups, out_channels // 4) if out_channels >= 4 else 1
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU()

        self.upsample = upsample
        if upsample:
            # Use ConvTranspose1d for learnable upsampling
            self.upsample_layer = nn.ConvTranspose1d(
                out_channels, out_channels,
                kernel_size=4, stride=upsample_factor, padding=1
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        if self.upsample:
            x = self.upsample_layer(x)

        return x


class TemporalEncoder(nn.Module):
    """
    Temporal encoder using Conv1d blocks.

    Uses fixed 2x downsampling at each stage with configurable channel dimensions.
    """

    def __init__(self, action_dim: int, embedding_dim: int, stage_channels: list[int],
                 num_groups: int = 8):
        super().__init__()

        # Store configuration
        self.stage_channels = stage_channels
        self.num_stages = len(stage_channels)
        self.downsample_factor = 2 ** self.num_stages

        # Initial projection from action_dim to first stage channel
        self.input_proj = nn.Conv1d(action_dim, stage_channels[0], kernel_size=1)

        # Build stages dynamically
        self.stages = nn.ModuleList()
        in_channels = stage_channels[0]  # Start with first stage channels

        for stage_idx in range(self.num_stages):
            out_channels = stage_channels[stage_idx]

            # Create stage: Conv -> Conv -> Downsample by 2
            stage = nn.Sequential(
                VQFlowConv1dBlock(in_channels, out_channels, kernel_size=3, stride=1,
                                 padding=1, num_groups=num_groups),
                VQFlowConv1dBlock(out_channels, out_channels, kernel_size=3, stride=1,
                                 padding=1, num_groups=num_groups),
                VQFlowConv1dBlock(out_channels, out_channels, kernel_size=4, stride=2,
                                 padding=1, num_groups=num_groups),  # Always downsample by 2
            )

            self.stages.append(stage)
            in_channels = out_channels  # Next stage takes this stage's output

        # Final projection from last stage channels to embedding_dim
        self.output_proj = nn.Conv1d(stage_channels[-1], embedding_dim, kernel_size=1)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length after downsampling."""
        return input_length // self.downsample_factor

    def forward(self, actions: Tensor) -> Tensor:
        """
        Args:
            actions: (B, seq_len, action_dim) action sequences
        Returns:
            embeddings: (B, output_tokens, embedding_dim) temporal embeddings
        """
        batch_size, seq_len, action_dim = actions.shape

        # Calculate expected output length
        output_tokens = self.get_output_length(seq_len)

        # Transpose for conv1d: (B, seq_len, action_dim) -> (B, action_dim, seq_len)
        x = actions.transpose(1, 2)

        # Initial projection
        x = self.input_proj(x)  # (B, stage_channels[0], seq_len)

        # Progressive downsampling through stages
        for stage in self.stages:
            x = stage(x)  # Each stage reduces temporal dimension by 2

        # Final projection
        x = self.output_proj(x)  # (B, embedding_dim, output_tokens)

        # Transpose back: (B, embedding_dim, output_tokens) -> (B, output_tokens, embedding_dim)
        return x.transpose(1, 2)


class TemporalDecoder(nn.Module):
    """
    Temporal decoder using Conv1d blocks with upsampling.

    Mirrors the encoder architecture with fixed 2x upsampling at each stage.
    """

    def __init__(self, embedding_dim: int, action_dim: int, stage_channels: list[int],
                 num_groups: int = 8):
        super().__init__()

        # Store configuration (reverse channels for decoder)
        self.stage_channels = list(reversed(stage_channels))
        self.num_stages = len(stage_channels)
        self.upsample_factor = 2 ** self.num_stages

        # Initial projection from embedding_dim to first decoder channel
        self.input_proj = nn.Conv1d(embedding_dim, self.stage_channels[0], kernel_size=1)

        # Build stages dynamically
        self.stages = nn.ModuleList()
        in_channels = self.stage_channels[0]  # Start with first decoder channel

        for stage_idx in range(self.num_stages):
            out_channels = self.stage_channels[stage_idx]

            # Create stage: Refine -> Upsample by 2
            stage = nn.Sequential(
                VQFlowDecoderBlock(in_channels, out_channels, kernel_size=3, padding=1,
                                  upsample=False, num_groups=num_groups),
                VQFlowDecoderBlock(out_channels, out_channels, kernel_size=3, padding=1,
                                  upsample=True, upsample_factor=2, num_groups=num_groups),  # Always upsample by 2
            )

            self.stages.append(stage)
            in_channels = out_channels  # Next stage takes this stage's output

        # Final refinement layers (no upsampling)
        final_channels = min(64, self.stage_channels[-1] // 2)  # Reduce channels for final layers
        self.final_blocks = nn.Sequential(
            VQFlowDecoderBlock(self.stage_channels[-1], final_channels, kernel_size=3, padding=1,
                              upsample=False, num_groups=num_groups),
            VQFlowDecoderBlock(final_channels, final_channels, kernel_size=3, padding=1,
                              upsample=False, num_groups=num_groups),
        )

        # Output projection (no activation)
        self.output_proj = nn.Conv1d(final_channels, action_dim, kernel_size=1)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length after upsampling."""
        return input_length * self.upsample_factor

    def forward(self, token_embeddings: Tensor) -> Tensor:
        """
        Args:
            token_embeddings: (B, input_tokens, embedding_dim) quantized embeddings
        Returns:
            actions: (B, output_length, action_dim) reconstructed action sequences
        """
        batch_size, seq_len, emb_dim = token_embeddings.shape

        # Calculate expected output length
        output_length = self.get_output_length(seq_len)

        # Transpose for conv1d: (B, input_tokens, embedding_dim) -> (B, embedding_dim, input_tokens)
        x = token_embeddings.transpose(1, 2)

        # Initial projection
        x = self.input_proj(x)  # (B, stage_channels[0], input_tokens)

        # Progressive upsampling through stages
        for stage in self.stages:
            x = stage(x)  # Each stage increases temporal dimension by 2

        # Final refinement
        x = self.final_blocks(x)  # (B, final_channels, output_length)

        # Output projection
        x = self.output_proj(x)  # (B, action_dim, output_length)

        # Transpose back: (B, action_dim, output_length) -> (B, output_length, action_dim)
        return x.transpose(1, 2)


def create_vqflow_encoder_decoder(action_dim: int, embedding_dim: int, stage_channels: list[int],
                                  num_groups: int = 8):
    """
    Convenience function to create matching encoder and decoder.

    Args:
        action_dim: Dimension of action vectors
        embedding_dim: Dimension of VQ embeddings
        stage_channels: Channel dimensions for each stage
        num_groups: Number of groups for GroupNorm

    Returns:
        encoder, decoder: Matching encoder and decoder instances
    """
    encoder = TemporalEncoder(
        action_dim=action_dim,
        embedding_dim=embedding_dim,
        stage_channels=stage_channels,
        num_groups=num_groups
    )

    decoder = TemporalDecoder(
        embedding_dim=embedding_dim,
        action_dim=action_dim,
        stage_channels=stage_channels,  # Decoder will reverse these internally
        num_groups=num_groups
    )

    return encoder, decoder