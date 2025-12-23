"""
Utilities for loading pre-trained GrainNet models
"""

import pickle
from pathlib import Path
from typing import Union
from .model import GrainNet


def load_grainnet(
    weights_path: Union[str, Path],
    activation: str = 'tanh',
    block_nb: int = 2,
) -> GrainNet:
    """
    Load a pre-trained GrainNet model from a pickle file

    Args:
        weights_path: Path to the .pkl file containing model parameters
        activation: Output activation type ('tanh' or 'sigmoid')
                   Must match the activation used during training
        block_nb: Number of residual blocks (1, 2, or 3)
                 Must match the architecture used during training

    Returns:
        GrainNet model ready for inference

    Example:
        >>> model = load_grainnet("grainnet_flax.pkl")
        >>> # Use the model
        >>> import jax
        >>> output = model(image, grain_radius=0.5, key=jax.random.PRNGKey(0))

    Raises:
        FileNotFoundError: If weights_path doesn't exist
        ValueError: If activation or block_nb are invalid
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if activation not in ['tanh', 'sigmoid']:
        raise ValueError(f"activation must be 'tanh' or 'sigmoid', got '{activation}'")

    if block_nb not in [1, 2, 3]:
        raise ValueError(f"block_nb must be 1, 2, or 3, got {block_nb}")

    # Load parameters
    with open(weights_path, 'rb') as f:
        params = pickle.load(f)

    # Create and return model
    return GrainNet(
        params=params,
        activation=activation,
        block_nb=block_nb,
    )
