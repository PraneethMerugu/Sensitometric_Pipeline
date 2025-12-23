"""
Equinox wrapper for GrainNet Flax model
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from .core import GrainNetFlax


class GrainNet(eqx.Module):
    """
    GrainNet film grain synthesis model wrapped as an Equinox module.

    This wraps the Flax implementation for easy integration into Equinox-based projects.
    Supports arbitrary image dimensions.

    Attributes:
        _flax_model: Internal Flax model
        _params: Model parameters
        activation: Output activation type
        block_nb: Number of residual blocks
    """

    _flax_model: GrainNetFlax = eqx.field(static=True)
    _params: dict
    activation: str = eqx.field(static=True)
    block_nb: int = eqx.field(static=True)

    def __init__(
        self,
        params: dict,
        activation: str = 'tanh',
        block_nb: int = 2,
    ):
        """
        Initialize GrainNet module

        Args:
            params: Pre-trained model parameters (from .pkl file)
            activation: Output activation type ('tanh' or 'sigmoid')
            block_nb: Number of residual blocks (1, 2, or 3)
        """
        self._flax_model = GrainNetFlax(activation=activation, block_nb=block_nb)
        self._params = params
        self.activation = activation
        self.block_nb = block_nb

    def __call__(
        self,
        image: jax.Array,
        grain_radius: float = 0.5,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jax.Array:
        """
        Apply film grain to input image

        Args:
            image: Input image, shape (H, W) or (H, W, 1) or (B, H, W, 1)
                   Values should be in range [0, 1]
            grain_radius: Grain strength parameter (0.0 to 1.0)
            key: JAX random key for noise generation. If None, a key is generated.

        Returns:
            Output image with grain applied, same shape as input

        Example:
            >>> model = load_grainnet("grainnet_flax.pkl")
            >>> key = jax.random.PRNGKey(0)
            >>> image = jnp.ones((512, 512, 1)) * 0.5
            >>> output = model(image, grain_radius=0.6, key=key)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Handle different input shapes
        original_shape = image.shape
        squeeze_batch = False
        squeeze_channel = False

        # Ensure NHWC format: (batch, height, width, channels)
        if image.ndim == 2:  # (H, W)
            image = image[None, :, :, None]  # (1, H, W, 1)
            squeeze_batch = True
            squeeze_channel = True
        elif image.ndim == 3:  # (H, W, 1) or (H, W, C)
            image = image[None, :, :, :]  # (1, H, W, C)
            squeeze_batch = True
        elif image.ndim == 4:  # (B, H, W, C)
            pass
        else:
            raise ValueError(f"Image must be 2D, 3D, or 4D, got shape {original_shape}")

        # Prepare grain_radius parameter
        batch_size = image.shape[0]
        if isinstance(grain_radius, (int, float)):
            grain_param = jnp.full((batch_size, 1), grain_radius, dtype=jnp.float32)
        else:
            grain_param = jnp.asarray(grain_radius)
            if grain_param.ndim == 0:
                grain_param = jnp.full((batch_size, 1), grain_param, dtype=jnp.float32)
            elif grain_param.ndim == 1:
                grain_param = grain_param[:, None]

        # Define single channel application function
        def apply_single_channel(img_c, g_param, k):
            # img_c: (Batch, H, W, 1) or (Batch, H, W)
            # Ensure (Batch, H, W, 1)
            if img_c.ndim == 3:
                img_c = img_c[..., None]
            return self._flax_model.apply(
                {'params': self._params},
                img_c,
                g_param,
                k
            )

        # Apply model
        # If single channel, apply directly
        if image.shape[-1] == 1:
            output = apply_single_channel(image, grain_param, key)
        else:
            # Multi-channel: vmap over channels
            # current shape: (B, H, W, C)
            # Move C to front for (C, B, H, W) to map over
            # Actually easier to map over the last dimension if we structure it right,
            # but vmap usually maps over leading dims.
            # Let's simple loop or use vmap on transposed.
            
            C = image.shape[-1]
            outputs = []
            # Split keys for each channel to ensure independent noise
            keys = jax.random.split(key, C)
            
            for i in range(C):
                # Extract channel i: (B, H, W, 1)
                img_c = image[..., i:i+1]
                # Apply
                out_c = apply_single_channel(img_c, grain_param, keys[i])
                outputs.append(out_c)
            
            # Stack back: (B, H, W, C)
            output = jnp.concatenate(outputs, axis=-1)

        # Restore original shape
        if squeeze_batch and squeeze_channel:
             # If we started with (H,W), output (H,W)
             # But if we had (H,W,C), output (H,W,C)
             if output.shape[-1] == 1:
                 output = output[0, :, :, 0]
             else:
                 # This branch shouldn't happen if squeeze_channel was True (input was 2D)
                 output = output[0]
        elif squeeze_batch:
            output = output[0]  # (H, W, C) or (H, W, 1)

        return output

    @property
    def params(self) -> dict:
        """Get model parameters"""
        return self._params
