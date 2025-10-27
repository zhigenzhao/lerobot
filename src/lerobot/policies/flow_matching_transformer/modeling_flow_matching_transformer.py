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
"""Flow Matching Transformer Policy using the same transformer architecture as Diffusion Transformer 
but with flow matching instead of diffusion for generative modeling."""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.flow_matching_transformer.configuration_flow_matching_transformer import (
    FlowMatchingTransformerConfig,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)

# Flow matching library imports
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


class FlowMatchingTransformerWrapper(ModelWrapper):
    """Wrapper for transformer to be compatible with flow_matching library's ModelWrapper interface."""

    def __init__(self, transformer, config):
        super().__init__(transformer)
        self.config = config

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        """
        Forward pass compatible with flow_matching library.

        Args:
            x: Input tensor (B, horizon, action_dim)
            t: Time tensor (B,) in [0,1]
            **extras: Additional arguments like global_cond
        """
        global_cond = extras.get("global_cond", None)
        return self.model(x, t, global_cond=global_cond)


class FlowMatchingTransformerPolicy(PreTrainedPolicy):
    """
    Flow Matching Policy using Transformer architecture as backbone with flow matching
    for generative modeling instead of denoising diffusion.
    """

    config_class = FlowMatchingTransformerConfig
    name = "flow_matching_transformer"

    def __init__(
        self,
        config: FlowMatchingTransformerConfig):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            
        Note: Normalization is handled by external preprocessor/postprocessor pipelines."""
        super().__init__(config)
        config.validate_features()
        self.config = config        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.flow_matching = FlowMatchingTransformerModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.flow_matching.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Stateless method to generate actions from prepared observations."""
        actions = self.flow_matching.generate_actions(batch)
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # Normalize and prepare batch
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Populate queues with current batch
        self._queues = populate_queues(self._queues, batch)

        # Stack observations from queues
        prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        return self._get_action_chunk(prepared_batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow matching model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The flow matching model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            # Create prepared batch for action generation
            prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self._get_action_chunk(prepared_batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.flow_matching.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class FlowMatchingTransformerModel(nn.Module):
    """Flow matching model using transformer architecture."""

    def __init__(self, config: FlowMatchingTransformerConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [FlowMatchingRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = FlowMatchingRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.transformer = FlowMatchingTransformer1d(config, global_cond_dim=global_cond_dim)

        # Flow matching scheduler (replaces diffusion scheduler)
        self.flow_scheduler = self._make_flow_scheduler(config)

    def _make_flow_scheduler(self, config: FlowMatchingTransformerConfig):
        """Create flow matching scheduler."""
        if config.flow_matching_type == "CondOT":
            return CondOTProbPath()
        else:
            raise ValueError(f"Unsupported flow matching type: {config.flow_matching_type}")

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior (x_0 - noise)
        x_0 = torch.randn(
            size=(
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
            ),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Create wrapped model for ODESolver
        wrapped_model = FlowMatchingTransformerWrapper(self.transformer, self.config)

        # Create time grid from 0 to 1
        time_grid = torch.linspace(0, 1, self.config.num_integration_steps + 1, device=device)

        # Create ODESolver and sample
        solver = ODESolver(velocity_model=wrapped_model)
        extra_kwargs = {"global_cond": global_cond} if global_cond is not None else {}

        solution = solver.sample(time_grid=time_grid, x_init=x_0, method="midpoint", step_size=None, **extra_kwargs)

        return solution

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Flow matching loss computation.

        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Sample target actions, noise, and time
        target_actions = batch["action"]  # (B, horizon, action_dim)
        noise = torch.randn_like(target_actions)
        batch_size = target_actions.shape[0]
        t = torch.rand(batch_size, device=target_actions.device)

        # Sample from flow matching path
        path_sample = self.flow_scheduler.sample(t=t, x_0=noise, x_1=target_actions)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t

        # Predict velocity field
        pred_velocity = self.transformer(x_t, t, global_cond=global_cond)

        # Flow matching loss (MSE between predicted and true velocity)
        loss = F.mse_loss(pred_velocity, dx_t, reduction="none")

        # Mask loss wherever the action is padded with copies (same as diffusion).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when " f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class FlowMatchingSinusoidalPosEmb(nn.Module):
    """Flow matching positional embeddings with parameterized periods.
    
    Based on the flow matching positional encoding approach that works naturally
    with continuous [0,1] time values using configurable min/max periods.
    """

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
                 Can also handle 0D scalar tensors from ODE solver
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


class FlowMatchingConditioningEncoder(nn.Module):
    """Flow matching conditioning encoder for timestep and observation conditioning.
    
    Adapted from diffusion conditioning encoder for flow matching by using continuous time [0,1]
    instead of discrete diffusion timesteps.
    """
    
    def __init__(self, config: FlowMatchingTransformerConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        
        # Time embedding for flow matching time in [0,1]
        self.time_emb = FlowMatchingSinusoidalPosEmb(
            config.attention_embed_dim, 
            min_period=config.fm_min_period,
            max_period=config.fm_max_period
        )
        
        # Observation embedding
        self.cond_obs_emb = nn.Linear(global_cond_dim, config.attention_embed_dim)
        
        # Conditioning positional embedding
        # T_cond = 1 (time) + n_obs_steps (observations)
        T_cond = 1 + config.n_obs_steps
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, config.attention_embed_dim))
        
        # Dropout
        self.drop = nn.Dropout(config.embedding_dropout)
        
        # Encoder
        if config.n_conditioning_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.attention_embed_dim,
                nhead=config.n_attention_heads,
                dim_feedforward=4 * config.attention_embed_dim,
                dropout=config.attention_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.n_conditioning_layers
            )
        else:
            # Fallback when n_cond_layers == 0
            self.encoder = nn.Sequential(
                nn.Linear(config.attention_embed_dim, 4 * config.attention_embed_dim),
                nn.Mish(),
                nn.Linear(4 * config.attention_embed_dim, config.attention_embed_dim)
            )
    
    def forward(self, timestep: Tensor, global_cond: Tensor) -> Tensor:
        """
        Args:
            timestep: (B,) tensor of flow matching timesteps in [0,1]
            global_cond: (B, global_cond_dim) global conditioning from observations
        Returns:
            (B, T_cond, embed_dim) conditioning memory
        """
        # Use raw [0,1] timesteps directly - no scaling needed
        # Flow matching theory should work with continuous [0,1] time
        
        # Time embedding: (B,) -> (B, 1, embed_dim)
        time_emb = self.time_emb(timestep).unsqueeze(1)
        
        # Global conditioning: (B, global_cond_dim) -> (B, n_obs_steps, embed_dim)
        cond_obs_emb = self.cond_obs_emb(global_cond)
        cond_obs_emb = cond_obs_emb.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)
        
        # Concatenate: [time_emb, cond_obs_emb] -> (B, 1 + n_obs_steps, embed_dim)
        cond_embeddings = torch.cat([time_emb, cond_obs_emb], dim=1)
        
        # Add positional embeddings
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_embeddings + position_embeddings)
        
        # Apply encoder
        memory = self.encoder(x)
        return memory


class FlowMatchingActionDecoder(nn.Module):
    """Flow matching action decoder with cross-attention to conditioning memory.
    
    Copied from diffusion action decoder for flow matching use.
    """
    
    def __init__(self, config: FlowMatchingTransformerConfig, action_dim: int):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_emb = nn.Linear(action_dim, config.attention_embed_dim)
        
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.horizon, config.attention_embed_dim))
        
        # Dropout
        self.drop = nn.Dropout(config.embedding_dropout)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.attention_embed_dim,
            nhead=config.n_attention_heads,
            dim_feedforward=4 * config.attention_embed_dim,
            dropout=config.attention_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.n_decoder_layers
        )
        
        # Output layers
        self.ln_f = nn.LayerNorm(config.attention_embed_dim)
        self.head = nn.Linear(config.attention_embed_dim, action_dim)
    
    def forward(self, sample: Tensor, memory: Tensor, tgt_mask: Tensor = None, memory_mask: Tensor = None) -> Tensor:
        """
        Args:
            sample: (B, T, action_dim) noisy action sequence
            memory: (B, T_cond, embed_dim) conditioning memory from encoder
            tgt_mask: Causal mask for action sequence
            memory_mask: Cross-attention mask between actions and conditioning
        Returns:
            (B, T, action_dim) velocity field predictions
        """
        token_embeddings = self.input_emb(sample)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        
        # Apply decoder with cross-attention
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # Output projection
        x = self.ln_f(x)
        x = self.head(x)
        return x


def _create_flow_matching_attention_masks(config: FlowMatchingTransformerConfig, device: torch.device):
    """Create attention masks for flow matching transformer."""
    T = config.horizon
    T_cond = 1 + config.n_obs_steps
    
    # Causal mask for decoder
    mask = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    # Memory mask for cross-attention
    t, s = torch.meshgrid(torch.arange(T, device=device), torch.arange(T_cond, device=device), indexing='ij')
    memory_mask = t >= (s - 1)  # add one dimension since time is first token
    memory_mask = memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
    
    return mask, memory_mask


class FlowMatchingTransformer1d(nn.Module):
    """Flow matching transformer using encoder-decoder architecture.
    
    Uses separate encoder for conditioning (timestep + observations) and decoder for action generation
    with cross-attention, adapted from the diffusion transformer implementation.
    """

    def __init__(self, config: FlowMatchingTransformerConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        action_dim = config.action_feature.shape[0]
        
        # Flow matching conditioning encoder
        self.conditioning_encoder = FlowMatchingConditioningEncoder(config, global_cond_dim)
        
        # Flow matching action decoder
        self.action_decoder = FlowMatchingActionDecoder(config, action_dim)
        
        # Attention masks
        if config.use_causal_attention:
            # Note: masks will be created on the correct device in forward pass
            self.register_buffer("_mask_template", torch.zeros(1))  # placeholder
            self.register_buffer("_memory_mask_template", torch.zeros(1))  # placeholder
        else:
            self.mask = None
            self.memory_mask = None
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following transformer conventions."""
        ignore_types = (
            nn.Dropout, 
            FlowMatchingSinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential
        )
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, FlowMatchingConditioningEncoder):
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, FlowMatchingActionDecoder):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        # Don't raise error for unaccounted modules to maintain compatibility

    def forward(
        self,
        x: Tensor,
        timestep: Tensor | int,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        """Flow matching transformer forward pass.
        
        Args:
            x: (B, T, action_dim) tensor for noisy action sequences
            timestep: (B,) tensor of flow matching timesteps in [0,1]
            global_cond: (B, global_cond_dim) global conditioning from observations
        Returns:
            (B, T, action_dim) velocity field predictions
        """
        device = x.device
        
        # Create attention masks on correct device
        if self.config.use_causal_attention:
            mask, memory_mask = _create_flow_matching_attention_masks(self.config, device)
        else:
            mask = None
            memory_mask = None
        
        # Encode conditioning (timestep + observations) -> memory
        memory = self.conditioning_encoder(timestep, global_cond)
        
        # Decode actions with cross-attention to memory
        predictions = self.action_decoder(
            sample=x,
            memory=memory,
            tgt_mask=mask,
            memory_mask=memory_mask
        )
        
        return predictions


class FlowMatchingSpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
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

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class FlowMatchingRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    Copied from DiffusionRgbEncoder for flow matching use.
    """

    def __init__(self, config: FlowMatchingTransformerConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = _replace_flow_matching_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = FlowMatchingSpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_flow_matching_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module