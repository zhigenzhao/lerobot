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

"""VQFlow Policy: Vector Quantized Actions + Discrete Flow Matching with DiT backbone."""

from collections import deque
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.loss import MixturePathGeneralizedKL

# Flow matching library imports
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from torch import Tensor

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.policies.vqflow.configuration_vqflow import VQFlowConfig
from lerobot.policies.vqflow.discrete_dit_blocks import (
    DiscreteFlowDiTBlock,
    DiscreteFlowDiTEmbeddings,
    DiscreteFlowTimestepEmbedding,
    VQFlowRgbEncoder,
)
from lerobot.policies.vqflow.vqflow_utils import DiscreteModelWrapper, VQFlowVAE, flatten_indices, unflatten_indices


class VQFlowPolicy(PreTrainedPolicy):
    """
    VQFlow Policy combining VQVAE with Discrete Flow Matching using DiT backbone.
    
    Architecture:
    - Phase 1: Train VQVAE to discretize continuous actions into tokens
    - Phase 2: Train discrete flow matching DiT to model token generation
    
    The policy alternates between two training phases:
    1. VQVAE training (first n_vqvae_training_steps)
    2. Discrete flow matching training (remaining steps)
    """
    
    config_class = VQFlowConfig
    name = "vqflow"
    
    def __init__(
        self,
        config: VQFlowConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        # Normalization
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)
        
        # Observation queues for rollout
        self._queues = None
        
        # Core components
        self.vqflow = VQFlowModel(config)
        
        # Training phase tracking
        self.register_buffer("training_step", torch.tensor(0))
        self.register_buffer("current_phase", torch.tensor(1))  # Phase 1: VQVAE, Phase 2: DiT
        
        self.reset()
    
    def get_optim_params(self):
        """Get parameters for current training phase."""
        if self.current_phase == 1:
            # Phase 1: Only VQVAE parameters
            return self.vqflow.vqvae.parameters()
        else:
            # Phase 2: Only DiT parameters (VQVAE is frozen)
            return [p for p in self.vqflow.parameters() if p.requires_grad]
    
    def reset(self):
        """Clear observation and action queues. Should be called on env.reset()."""
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
        actions = self.vqflow.generate_actions(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # Normalize and prepare batch
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        
        # Populate queues with current batch
        self._queues = populate_queues(self._queues, batch)
        
        # Stack observations from queues
        prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        
        return self._get_action_chunk(prepared_batch)
    
    @torch.no_grad() 
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        # Remove action from batch if present (for offline evaluation)
        if ACTION in batch:
            batch.pop(ACTION)
        
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        
        self._queues = populate_queues(self._queues, batch)
        
        if len(self._queues[ACTION]) == 0:
            # Generate new action chunk
            prepared_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self._get_action_chunk(prepared_batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        
        action = self._queues[ACTION].popleft()
        return action
    
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Forward pass for training."""
        # Update training step
        self.training_step += 1
        
        # Check if we should switch phases
        if self.training_step >= self.config.n_vqvae_training_steps and self.current_phase == 1:
            self._switch_to_phase2()
        
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        if self.config.env_state_feature:
            pass  # Keep OBS_ENV_STATE as is
        batch = self.normalize_targets(batch)
        
        # Compute loss based on current phase
        if self.current_phase == 1:
            loss = self._compute_vqvae_loss(batch)
        else:
            loss = self._compute_discrete_flow_loss(batch)
        
        return loss, {}
    
    def _switch_to_phase2(self):
        """Switch from VQVAE training to discrete flow matching training."""
        print(f"Switching to Phase 2 at step {self.training_step}")
        self.current_phase = torch.tensor(2)
        self.vqflow.vqvae.freeze()
        
        # Update learning rate (this would be handled by the training script)
        # The actual optimizer switching happens in the training loop
    
    def _compute_vqvae_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute VQVAE loss for Phase 1 training."""
        target_actions = batch[ACTION]  # (B, horizon, action_dim)

        # Extract action sequences for training (use full horizon)
        start = self.config.n_obs_steps - 1
        end = start + self.config.horizon
        action_sequence = target_actions[:, start:end]  # (B, horizon, action_dim)

        # VQVAE forward pass on full sequence
        actions_recon, indices, vq_loss, recon_loss = self.vqflow.vqvae(action_sequence)

        # Total loss: reconstruction + VQ + commitment
        total_loss = recon_loss + vq_loss

        return total_loss
    
    def _compute_discrete_flow_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute discrete flow matching loss for Phase 2 training."""
        return self.vqflow.compute_discrete_flow_loss(batch)


class VQFlowModel(nn.Module):
    """Core VQFlow model combining VQVAE and discrete flow matching DiT."""
    
    def __init__(self, config: VQFlowConfig):
        super().__init__()
        self.config = config
        
        # VQVAE for action discretization
        self.vqvae = VQFlowVAE(config)
        
        # Observation encoder
        self._build_observation_encoder(config)
        
        # Discrete flow matching DiT
        self.dit = DiscreteFlowDiT(config, self.obs_encoding_dim)
        
        # Discrete flow matching components
        scheduler = PolynomialConvexScheduler(n=config.scheduler_power)
        self.discrete_path = MixtureDiscreteProbPath(scheduler=scheduler)
        self.discrete_loss_fn = MixturePathGeneralizedKL(path=self.discrete_path)
        
    def _build_observation_encoder(self, config: VQFlowConfig):
        """Build observation encoders for different modalities."""
        # Robot state is always present
        obs_dim = config.robot_state_feature.shape[0]

        # Add image encoders if present
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [VQFlowRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoders = nn.ModuleList(encoders)
                # FIX: Use cross_attention_dim since RGB encoders output cross_attention_dim features
                obs_dim += config.cross_attention_dim * num_images
            else:
                self.rgb_encoder = VQFlowRgbEncoder(config)
                obs_dim += config.cross_attention_dim * num_images

        # Add environment state if present
        if config.env_state_feature:
            obs_dim += config.env_state_feature.shape[0]

        # Final observation encoding dimension
        self.obs_encoding_dim = obs_dim * config.n_obs_steps
    
    def _encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations for cross-attention conditioning."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        obs_features = [batch[OBS_STATE]]
        
        # Encode images
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Use separate encoder per camera
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features = []
                for encoder, images in zip(self.rgb_encoders, images_per_camera):
                    features = encoder(images)
                    img_features.append(features)
                img_features = torch.cat(img_features, dim=-1)
                img_features = einops.rearrange(
                    img_features, "(b s) d -> b s d", b=batch_size, s=n_obs_steps
                )
            else:
                # Use shared encoder
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) d -> b s (n d)", 
                    b=batch_size, s=n_obs_steps, n=len(self.config.image_features)
                )
            obs_features.append(img_features)
        
        # Add environment state
        if self.config.env_state_feature:
            obs_features.append(batch[OBS_ENV_STATE])
        
        # Concatenate and flatten: (B, n_obs_steps, obs_dim) -> (B, obs_encoding_dim)
        obs_encoding = torch.cat(obs_features, dim=-1).flatten(start_dim=1)
        return obs_encoding
    
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions using discrete flow matching."""
        batch_size = batch[OBS_STATE].shape[0]
        
        # Encode observations
        obs_encoding = self._encode_observations(batch)
        
        # Sample from discrete flow matching
        tokens = self._sample_discrete_flow(batch_size, obs_encoding)
        
        # Decode tokens to actions
        with torch.no_grad():
            actions = self._decode_tokens_to_actions(tokens)
        
        # Extract action steps for execution
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]
    
    def _sample_discrete_flow(self, batch_size: int, obs_encoding: Tensor) -> Tensor:
        """Sample discrete tokens using flow matching ODE solver."""
        device = get_device_from_parameters(self)

        # Use dynamically calculated target_tokens
        seq_len = self.config.vqvae_target_tokens

        # Initialize with source distribution
        if self.config.source_distribution == "uniform":
            x_init = torch.randint(
                0, self.config.vocab_size,
                (batch_size, seq_len),
                device=device
            )
        else:  # mask
            x_init = torch.full(
                (batch_size, seq_len),
                self.config.mask_token_id,
                device=device
            )
        
        # Create model wrapper
        wrapped_model = DiscreteModelWrapper(self.dit, self.config)
        
        # Create solver
        solver = MixtureDiscreteEulerSolver(
            model=wrapped_model,
            path=self.discrete_path, 
            vocabulary_size=self.config.vocab_size
        )
        
        # Sample with discrete flow ODE
        step_size = 1.0 / self.config.num_integration_steps
        tokens = solver.sample(
            x_init=x_init,
            step_size=step_size,
            obs_encoding=obs_encoding
        )
        
        return tokens
    
    def _decode_tokens_to_actions(self, tokens: Tensor) -> Tensor:
        """Decode discrete tokens to continuous actions via VQVAE."""
        # Convert flat tokens to hierarchical indices
        indices = unflatten_indices(
            tokens, 
            num_layers=self.config.vqvae_num_layers,
            codebook_size=self.config.vqvae_n_embed
        )
        
        # Decode through VQVAE
        actions = self.vqvae.decode_from_indices(indices)
        
        return actions
    
    def compute_discrete_flow_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute discrete flow matching loss."""
        # Encode target actions to discrete tokens
        with torch.no_grad():
            target_actions = batch[ACTION]  # (B, horizon, action_dim)

            # Extract action sequence for encoding (use full horizon)
            start = self.config.n_obs_steps - 1
            end = start + self.config.horizon
            action_sequence = target_actions[:, start:end]  # (B, horizon, action_dim)

            hierarchical_indices = self.vqvae.encode_to_indices(action_sequence)  # (B, target_tokens, num_layers)

            # Flatten hierarchical indices to single vocabulary for each token
            batch_size, target_tokens, num_layers = hierarchical_indices.shape
            x_1 = []
            for i in range(target_tokens):
                token_indices = hierarchical_indices[:, i]  # (B, num_layers)
                flat_indices = flatten_indices(token_indices, self.config.vqvae_n_embed)
                x_1.append(flat_indices)
            x_1 = torch.stack(x_1, dim=1)  # (B, target_tokens)

        # Sample source distribution for all tokens
        batch_size, seq_len = x_1.shape
        if self.config.source_distribution == "uniform":
            x_0 = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=x_1.device)
        else:  # mask
            x_0 = torch.full((batch_size, seq_len), self.config.mask_token_id, device=x_1.device)

        # Sample time
        t = torch.rand(batch_size, device=x_1.device) * (1 - self.config.flow_epsilon)

        # Discrete path sampling
        path_sample = self.discrete_path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t  # (B, seq_len)

        # Encode observations for conditioning
        obs_encoding = self._encode_observations(batch)

        # DiT prediction for full sequence
        logits = self.dit(x_t, t, obs_encoding)  # (B, seq_len, vocab_size)
        
        # Compute generalized KL divergence loss
        loss = self.discrete_loss_fn(logits=logits, x_1=x_1, x_t=x_t, t=t)
        
        return loss


class DiscreteFlowDiT(nn.Module):
    """Discrete Flow DiT model for token sequence generation."""

    def __init__(self, config: VQFlowConfig, obs_encoding_dim: int):
        super().__init__()
        self.config = config
        self.obs_encoding_dim = obs_encoding_dim
        
        # Token and position embeddings
        self.embeddings = DiscreteFlowDiTEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.vqvae_target_tokens,
            dropout=config.attention_dropout
        )
        
        # Timestep embedding
        self.time_embed = DiscreteFlowTimestepEmbedding(
            dim=config.fm_time_embed_dim,
            min_period=config.fm_min_period,
            max_period=config.fm_max_period
        )
        
        # Time projection
        self.time_proj = nn.Sequential(
            nn.Linear(config.fm_time_embed_dim, config.fm_time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(config.fm_time_embed_dim * 4, config.fm_time_embed_dim)
        )

        # Observation projection for cross-attention
        self.obs_projection = nn.Linear(obs_encoding_dim, config.cross_attention_dim)

        # DiT transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DiscreteFlowDiTBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads, 
                cross_attention_dim=config.cross_attention_dim,
                attention_dropout=config.attention_dropout,
                use_adaln_zero=config.use_adaln_zero,
                timestep_embed_dim=config.fm_time_embed_dim
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm and output projection
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: Tensor, 
        timestep: Tensor, 
        obs_encoding: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            input_ids: (B, seq_len) discrete token indices
            timestep: (B,) continuous time values in [0,1]  
            obs_encoding: (B, obs_encoding_dim) observation encoding for cross-attention
        Returns:
            (B, seq_len, vocab_size) logits over vocabulary
        """
        # Ensure input_ids is always 2D: (B, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)  # (B,) → (B, 1)

        # Token embeddings
        hidden_states = self.embeddings(input_ids)  # (B, seq_len, hidden_size)
        
        # Timestep embedding
        time_emb = self.time_embed(timestep)  # (B, fm_time_embed_dim)
        time_emb = self.time_proj(time_emb)  # (B, fm_time_embed_dim)
        
        # Prepare observation encoding for cross-attention
        encoder_hidden_states = None
        if obs_encoding is not None:
            # Use initialized projection layer
            encoder_hidden_states = self.obs_projection(obs_encoding).unsqueeze(1)  # (B, 1, cross_attention_dim)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                timestep_emb=time_emb,
                encoder_hidden_states=encoder_hidden_states
            )
        
        # Final layer norm and output projection
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)  # (B, seq_len, vocab_size)

        return logits