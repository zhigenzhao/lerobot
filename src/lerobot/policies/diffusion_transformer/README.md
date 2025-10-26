## Diffusion Transformer Policy

Implementation of Diffusion Policy using Transformer architecture as the backbone instead of UNet.

## Key Differences from Standard Diffusion Policy

- **Backbone**: Uses multi-head self-attention transformer instead of 1D convolutional UNet
- **Sequence Modeling**: Better handling of temporal dependencies in action sequences  
- **Attention Mechanism**: Causal self-attention for autoregressive action generation
- **Conditioning**: Global conditioning integrated as prefix tokens with cross-attention
- **Scalability**: More efficient scaling to longer action horizons

## Architecture

1. **Vision Encoder**: Reuses ResNet + SpatialSoftmax from standard diffusion policy
2. **Transformer Backbone**: Multi-layer transformer encoder with:
   - Multi-head causal self-attention
   - Layer normalization and residual connections
   - Positional embeddings for sequence positions
   - Sinusoidal timestep embeddings
3. **Conditioning Strategy**: 
   - Global observations encoded as prefix tokens
   - Cross-attention between action sequence and observation features
   - Timestep embedding added to all tokens

## References

- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [Original Implementation](https://github.com/real-stanford/diffusion_policy)
- [Transformer for Diffusion](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py)