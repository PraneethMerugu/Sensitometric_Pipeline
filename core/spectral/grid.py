
import jax.numpy as jnp

# The "Source of Truth" for the entire pipeline
WL_START = 360.0
WL_END = 830.0
WL_BINS = 64

def get_wavelengths():
    """Returns the (64,) wavelength grid in nm."""
    return jnp.linspace(WL_START, WL_END, WL_BINS)
