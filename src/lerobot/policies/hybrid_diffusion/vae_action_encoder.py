#!/usr/bin/env python

# Copyright 2025 Zhigen Zhao (zhaozhigen@gmail.com)
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
"""VAE Action Encoder for discrete actions in Hybrid Diffusion Policy."""

import torch
import torch.nn.functional as F
from torch import nn


class VaeEncoder(nn.Module):
    """
    Variational Autoencoder encoder for discrete actions.
    Maps discrete action sequences to continuous latent space with mean and std.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: list[int] = None,
        output_dim: int = 16,
        horizon: int = 1,
        eps: float = 1e-8,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps

        # Build encoder network
        self.encoder = nn.Sequential()
        in_dim = input_dim * horizon

        for i, hidden_dim in enumerate(hidden_dims):
            self.encoder.add_module(
                f"linear{i}", nn.Linear(in_dim, hidden_dim * horizon)
            )
            self.encoder.add_module(f"act{i}", nn.Mish())
            in_dim = hidden_dim * horizon

        # Output layer that produces both mu and logvar
        self.encoder.add_module(
            "final_linear",
            nn.Linear(in_dim, 2 * output_dim * horizon),
        )

    def encode(self, x: torch.Tensor):
        """
        Encode input to latent parameters.

        Args:
            x: Input tensor of shape (batch_size, horizon, input_dim)

        Returns:
            mu: Mean of latent distribution
            std: Standard deviation of latent distribution
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.horizon * self.input_dim)

        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=-1)

        # Convert logvar to std with numerical stability
        std = torch.exp(0.5 * logvar) + self.eps
        return mu, std

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor):
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution
            std: Standard deviation of latent distribution

        Returns:
            z: Sampled latent variable
        """
        epsilon = torch.randn_like(mu)
        z = mu + epsilon * std
        return z

    def forward(self, x: torch.Tensor):
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch_size, horizon, input_dim)

        Returns:
            z: Sampled latent variable of shape (batch_size, horizon, output_dim)
        """
        batch_size = x.shape[0]
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        z = z.reshape(batch_size, self.horizon, self.output_dim)
        return z


class BinaryDecoder(nn.Module):
    """
    Binary decoder for reconstructing discrete actions from latent codes.
    Uses sigmoid activation to output binary probabilities.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: list[int] = None,
        output_dim: int = 4,
        horizon: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon

        # Build decoder network
        self.decoder = nn.Sequential()
        in_dim = input_dim * horizon

        for i, hidden_dim in enumerate(hidden_dims):
            self.decoder.add_module(
                f"linear{i}", nn.Linear(in_dim, hidden_dim * horizon)
            )
            self.decoder.add_module(f"act{i}", nn.Mish())
            in_dim = hidden_dim * horizon

        # Output layer with sigmoid activation
        self.decoder.add_module(
            "final_linear", nn.Linear(in_dim, output_dim * horizon)
        )
        self.decoder.add_module("final_act", nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        """
        Forward pass through decoder.

        Args:
            x: Latent tensor of shape (batch_size, horizon, input_dim)

        Returns:
            output: Reconstructed actions of shape (batch_size, horizon, output_dim)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.horizon * self.input_dim)
        x = self.decoder(x)
        x = x.reshape(batch_size, self.horizon, self.output_dim)
        return x


class DiscreteActionVAE(nn.Module):
    """
    Complete VAE for discrete actions combining encoder and decoder.
    """

    def __init__(
        self,
        discrete_dim: int,
        latent_dim: int = 16,
        hidden_dims: list[int] = None,
        horizon: int = 1,
        eps: float = 1e-8,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.discrete_dim = discrete_dim
        self.latent_dim = latent_dim
        self.horizon = horizon

        self.encoder = VaeEncoder(
            input_dim=discrete_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            horizon=horizon,
            eps=eps,
        )

        self.decoder = BinaryDecoder(
            input_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_dim=discrete_dim,
            horizon=horizon,
        )

    def encode(self, x: torch.Tensor):
        """
        Encode discrete actions to latent space.

        Args:
            x: Discrete actions of shape (batch_size, horizon, discrete_dim)

        Returns:
            z: Sampled latent codes
            mu: Mean of latent distribution
            std: Standard deviation of latent distribution
        """
        mu, std = self.encoder.encode(x)
        z = self.encoder.reparameterize(mu, std)

        # Reshape to match expected output format
        batch_size = x.shape[0]
        z = z.reshape(batch_size, self.horizon, self.latent_dim)
        mu = mu.reshape(batch_size, self.horizon, self.latent_dim)
        std = std.reshape(batch_size, self.horizon, self.latent_dim)

        return z, mu, std

    def decode(self, z: torch.Tensor):
        """
        Decode latent codes to discrete actions.

        Args:
            z: Latent codes of shape (batch_size, horizon, latent_dim)

        Returns:
            recon: Reconstructed actions of shape (batch_size, horizon, discrete_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Complete forward pass through VAE.

        Args:
            x: Discrete actions of shape (batch_size, horizon, discrete_dim)

        Returns:
            recon: Reconstructed actions
            mu: Mean of latent distribution
            std: Standard deviation of latent distribution
        """
        z, mu, std = self.encode(x)
        recon = self.decode(z)
        return recon, mu, std

    def compute_loss(self, x: torch.Tensor, beta: float = 0.001):
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Input discrete actions
            beta: Weight for KL divergence term

        Returns:
            loss: Total VAE loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        recon, mu, std = self.forward(x)

        # Binary cross-entropy reconstruction loss
        recon_loss = F.binary_cross_entropy(recon, x, reduction='mean')

        # KL divergence loss (assuming standard normal prior)
        kl_loss = -0.5 * torch.mean(1 + torch.log(std**2) - mu**2 - std**2)

        # Total loss
        loss = recon_loss + beta * kl_loss

        return loss, recon_loss, kl_loss