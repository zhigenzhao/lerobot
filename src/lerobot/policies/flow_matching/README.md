# Flow Matching Policy

Flow Matching Policy using the same UNet architecture as Diffusion Policy but with flow matching for generative modeling instead of denoising diffusion.

## Overview

This implementation uses **flow matching** (specifically Conditional Optimal Transport) to generate action sequences, while maintaining the exact same UNet backbone as the Diffusion Policy for fair comparison. The key differences are:

- **Training**: Predicts velocity field `dx/dt` instead of denoising
- **Sampling**: Uses ODE integration instead of multi-step denoising
- **Architecture**: Same UNet, vision encoders, and conditioning as Diffusion Policy

## Architecture

```
Flow Matching Policy Architecture:
┌─────────────────────────────────────────────────────────┐
│                   SAME AS DIFFUSION                    │
│  Observations → Vision Encoder → Global Conditioning   │  
└─────────────────────┬───────────────────────────────────┘
                      ├─ Flow Matching Specific ─┐
                      ▼                          ▼
            ┌─────────────────┐        ┌─────────────────┐
            │   UNet Model    │        │  Flow Matching  │
            │   (SAME UNet)   │  ◄───► │   Scheduler     │
            │                 │        │ (CondOT Path)   │  
            └─────────────────┘        └─────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │   Velocity      │
            │  Predictions    │
            └─────────────────┘
```

## Key Components

### Flow Matching Training
- **Path**: `x_t = (1-t) * x_0 + t * x_1` where `x_0` is noise, `x_1` is target actions
- **Velocity**: `dx/dt = x_1 - x_0` (constant velocity for linear interpolation)
- **Loss**: MSE between predicted and true velocity fields

### ODE Integration Sampling
- Start from noise `x_0`
- Integrate velocity field using Euler method: `x_{t+dt} = x_t + dt * v(x_t, t)`
- Final sample at `t=1` gives clean actions

### UNet Architecture
- Identical to Diffusion Policy UNet
- 1D convolutional encoder-decoder with skip connections
- FiLM conditioning on timestep and global observations
- Same vision backbone (ResNet + SpatialSoftmax)

## Usage

### Training
```bash
lerobot-train --config_path=<flow_matching_config.yaml>
```

### Configuration
```yaml
policy:
  type: flow_matching
  flow_matching_type: "CondOT"  # Conditional Optimal Transport
  num_integration_steps: 50     # ODE integration steps
  flow_solver: "euler"          # ODE solver type
  # All other parameters same as diffusion policy
```

## Dependencies

Requires the `flow_matching` library:
```bash
pip install flow-matching
```

## Comparison with Diffusion Policy

| Aspect | Diffusion Policy | Flow Matching Policy |
|--------|------------------|---------------------|
| **UNet Architecture** | ✅ Same | ✅ Same |
| **Global Conditioning** | ✅ Same | ✅ Same |
| **Training Objective** | Denoise x_t → x_{t-1} | Predict velocity dx/dt |
| **Sampling Process** | Multi-step denoising | Single ODE integration |
| **Time Parameterization** | Discrete steps [0, T] | Continuous time [0, 1] |
| **Training Stability** | Can be noisy | Often more stable |
| **Inference Speed** | ~50-100 steps | ~50 steps (configurable) |

## Paper References

- **Flow Matching**: "Flow Matching for Generative Modeling" (https://arxiv.org/abs/2210.02747)
- **Conditional Flow Matching**: "Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport" (https://arxiv.org/abs/2302.00482)
- **Diffusion Policy**: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (https://arxiv.org/abs/2303.04137)