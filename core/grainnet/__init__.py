"""
GrainNet: Film grain synthesis network

A JAX/Flax implementation wrapped as an Equinox module for easy integration.
"""

from .model import GrainNet
from .loader import load_grainnet

__version__ = "0.1.0"
__all__ = ["GrainNet", "load_grainnet"]
