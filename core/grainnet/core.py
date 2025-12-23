"""
Core Flax model definitions for GrainNet
"""

import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with two conv layers, LeakyReLU, and InstanceNorm"""
    channel: int
    k_size: Tuple[int, int] = (3, 3)

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(features=self.channel, kernel_size=self.k_size, padding='SAME', name='conv1')(x)
        y = nn.leaky_relu(y, negative_slope=0.01)
        y = nn.GroupNorm(num_groups=self.channel, name='norm1')(y)
        y = nn.Conv(features=self.channel, kernel_size=self.k_size, padding='SAME', name='conv2')(y)
        return nn.leaky_relu(y + x, negative_slope=0.01)


class MyNorm(nn.Module):
    """Custom Adaptive Instance Normalization layer"""
    channel_size: int
    insize: int = 1

    @nn.compact
    def __call__(self, x, grain_type):
        std = nn.Dense(features=self.channel_size, name='std_weight')(grain_type)
        mean = nn.Dense(features=self.channel_size, name='mean_weight')(grain_type)

        std = std[:, None, None, :]
        mean = mean[:, None, None, :]

        return x * std + mean


class GrainNetFlax(nn.Module):
    """
    Flax implementation of GrainNet for film grain synthesis

    Attributes:
        activation: Output activation type ('tanh' or 'sigmoid')
        block_nb: Number of residual blocks (1, 2, or 3)
    """
    activation: str = 'tanh'
    block_nb: int = 2

    def setup(self):
        if self.block_nb not in [1, 2, 3]:
            raise ValueError('block_nb must be 1, 2 or 3')

    @nn.compact
    def __call__(self, img, grain_radius, rng_key):
        """
        Apply grain to input image

        Args:
            img: Input image (batch, height, width, channels) - NHWC format
            grain_radius: Grain strength parameter (batch, 1)
            rng_key: JAX random key for noise generation

        Returns:
            Output image with grain applied (batch, height, width, channels)
        """
        import jax

        # Generate noise
        noise = jax.random.normal(rng_key, img.shape)

        # Concatenate noise and image
        x = jnp.concatenate([noise, img], axis=-1)

        # Entry convolution
        x0 = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME', name='entry_conv')(x)
        x0 = nn.leaky_relu(x0, negative_slope=0.01)

        # Block 1
        x1 = ResidualBlock(channel=16, name='block1_resblock')(x0)
        x1 = nn.GroupNorm(num_groups=16, name='block1_norm')(x1)
        x1 = MyNorm(channel_size=16, name='mn1')(x1, grain_radius)

        if self.block_nb == 3:
            # Augment channels
            x2 = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', name='augment')(x1)
            x2 = nn.leaky_relu(x2, negative_slope=0.01)

            # Block 2
            x3 = ResidualBlock(channel=32, name='block2_resblock')(x2)
            x3 = nn.GroupNorm(num_groups=32, name='block2_norm')(x3)
            x3 = MyNorm(channel_size=32, name='mn2')(x3, grain_radius)

            # Reduce channels
            x4 = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME', name='reduce')(x3)
            x4 = nn.leaky_relu(x4, negative_slope=0.01)

            # Block 3 with skip connection
            x5 = ResidualBlock(channel=16, name='block3_resblock')(x4 + x1)
            x5 = nn.GroupNorm(num_groups=16, name='block3_norm')(x5)
            x5 = MyNorm(channel_size=16, name='mn3')(x5, grain_radius)

            x6 = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME', name='out_conv')(x5 + x0)

        elif self.block_nb == 2:
            # Block 3
            x5 = ResidualBlock(channel=16, name='block3_resblock')(x1)
            x5 = nn.GroupNorm(num_groups=16, name='block3_norm')(x5)
            x5 = MyNorm(channel_size=16, name='mn3')(x5, grain_radius)

            x6 = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME', name='out_conv')(x5 + x0)

        else:  # block_nb == 1
            x6 = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME', name='out_conv')(x1)

        # Apply output activation
        if self.activation == 'tanh':
            x6 = jnp.tanh(x6)
            x6 = jnp.clip(0.5 * x6 + 0.5, 0.0, 1.0)
        elif self.activation == 'sigmoid':
            x6 = nn.sigmoid(x6)

        return x6
