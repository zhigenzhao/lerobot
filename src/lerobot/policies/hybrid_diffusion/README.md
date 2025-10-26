# Hybrid Diffusion Policy

A two-stage training policy that combines Variational Autoencoder (VAE) and diffusion models for handling mixed continuous-discrete action spaces, particularly suited for locomotion tasks with foot contacts.

## Overview

The Hybrid Diffusion Policy implements a two-stage training framework:

1. **Stage 1 (VAE Training)**: Train a VAE to encode/decode discrete actions (e.g., foot contacts)
2. **Stage 2 (Diffusion Training)**: Train a diffusion model on combined continuous + VAE-encoded discrete actions

This approach is particularly effective for tasks that have both continuous control (joint positions) and discrete decisions (contact states).

## Architecture

### Core Components

- **VAE Action Encoder**: Encodes discrete actions into continuous latent space
- **Diffusion Model**: Standard diffusion policy for continuous action prediction
- **Two-Stage Training**: Sequential training with proper parameter group management

### Key Features

- **Mixed Action Spaces**: Handles continuous + discrete actions seamlessly
- **VQBet-Style Training**: All parameters included from start with scheduler-controlled learning rates
- **Flexible Optimizers**: Support for both Adam and AdamW optimizers
- **Custom Scheduler**: `HybridDiffusionSchedulerConfig` for stage-specific learning rate control

## Configuration

### Key Parameters

```json
{
  "policy": {
    "_target_": "lerobot.policies.hybrid_diffusion.modeling_hybrid_diffusion.HybridDiffusionPolicy",
    "continuous_action_dim": 21,
    "discrete_action_dim": 2,
    "n_vae_training_steps": 5000,
    "vae_latent_dim": 16,
    "vae_beta": 1e-4,
    "vae_lr": 3e-4,
    "optimizer_type": "adamw"
  }
}
```

### Training Configuration Example

```json
{
  "training": {
    "lr_scheduler": {
      "_target_": "lerobot.optim.schedulers.get_scheduler",
      "scheduler_type": "hybrid_diffusion",
      "num_vae_training_steps": 5000,
      "warmup_steps": 500,
      "lr": 1e-4
    }
  }
}
```

## Usage

### Training

```bash
lerobot-train --config_path=config/your_robot/hybrid_diffusion/
```

### Stage Progression

1. **Steps 0-4999**: VAE training only (diffusion frozen)
2. **Step 5000**: Automatic transition message printed
3. **Steps 5000+**: Diffusion training only (VAE frozen)

### Action Format

Actions should be concatenated as: `[continuous_actions, discrete_actions]`

- Continuous actions: Raw joint positions/velocities
- Discrete actions: Binary contact states (0 or 1)

## Recent Updates

### Two-Stage Training Fix
- Implemented VQBet-style parameter groups to avoid optimizer issues
- All parameters included from start with different learning rates
- Custom scheduler handles stage transitions without parameter group changes

### Dimension Handling
- Fixed zero-range normalization issues with constant discrete dimensions
- Proper action space validation and dimension checking

### Optimizer Support
- Added AdamW optimizer option alongside Adam
- Configurable through `optimizer_type` parameter

### WandB Logging
- Fixed tensor logging warnings by converting to scalar values
- Proper loss tracking for both VAE and diffusion stages

## Technical Details

### VAE Loss Components
- **Reconstruction Loss**: BCE loss for discrete action reconstruction
- **KL Divergence**: Regularization term weighted by `vae_beta`
- **Total Loss**: `recon_loss + vae_beta * kl_loss`

### Action Processing
1. Split concatenated actions into continuous and discrete parts
2. Normalize discrete actions from [0,1] to [-1,1] for training
3. VAE encodes discrete actions to latent space
4. Combine continuous + latent for diffusion training

### Memory Efficiency
- Only forward pass through active stage during training
- Proper gradient computation for current training phase
- Minimal memory overhead from inactive components

## Troubleshooting

### Common Issues

1. **NaN Losses**: Check for zero-range discrete dimensions causing normalization issues
2. **High Gradient Norms**: Ensure proper learning rate scheduling between stages
3. **WandB Warnings**: Loss values should be Python floats, not tensors

### Validation

- Monitor VAE reconstruction quality during stage 1
- Check smooth transition at step boundary
- Verify combined action predictions make sense