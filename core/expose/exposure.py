"""
Spectral Exposure Module

This module calculates the 'Actinic Exposure' by integrating the incident
spectral radiance against the film's spectral sensitivity curves.

Reference:
Module I: Spectral Exposure Calculation
"Final Film Emulation Pipeline", Section 2.1
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
from pathlib import Path
from typing import Tuple

# ==============================================================================
# CONSTANTS (Must match generate_lut.py to ensure grid alignment)
# ==============================================================================

WAVELENGTHS = jnp.linspace(360.0, 830.0, 64)

# ==============================================================================
# EQUINOX MODULE
# ==============================================================================

class SpectralExposure(eqx.Module):
    """
    Computes the Actinic Exposure (Latent Image formation) from Spectral Radiance.

    Attributes:
        sensitivities: Linear spectral sensitivity curves for R, G, B.
                       Shape: (3, 64)
        wavelengths: The wavelength grid these sensitivities are sampled on.
                       Shape: (64,)
    """
    sensitivities: jax.Array
    wavelengths: jax.Array

    def __init__(self, sensitivities: jax.Array, wavelengths: jax.Array):
        """
        Args:
            sensitivities: Linear sensitivity array of shape (3, N_wavelengths).
            wavelengths: Array of wavelengths of shape (N_wavelengths,).
        """
        self.sensitivities = sensitivities
        self.wavelengths = wavelengths

    @classmethod
    def from_csvs(
        cls,
        red_path: str,
        green_path: str,
        blue_path: str,
        target_wavelengths: jax.Array = WAVELENGTHS
    ) -> "SpectralExposure":
        """
        Load sensitivities from Kodak 'log sensitivity' CSV files, interpolate
        them to the simulation grid, and convert to linear space.

        Args:
            red_path: Path to Red channel log_ssf.csv
            green_path: Path to Green channel log_ssf.csv
            blue_path: Path to Blue channel log_ssf.csv
            target_wavelengths: Grid to interpolate data onto.

        Returns:
            Initialized SpectralExposure module.
        """
        def process_channel(file_path: str) -> jax.Array:
            # 1. Load Data
            # Expected CSV format: "wavelengths, sensitivities"
            df = pd.read_csv(file_path)
            
            # Extract columns (assumes standard Kodak headers or positions 0,1)
            # Using iloc is safer if headers slightly vary in casing
            w_raw = df.iloc[:, 0].values.astype(float)
            s_log_raw = df.iloc[:, 1].values.astype(float)

            # 2. Interpolate to Target Grid
            # jnp.interp requires sorted x-coordinates (Kodak data is sorted)
            s_log_interp = jnp.interp(
                target_wavelengths, 
                jnp.array(w_raw), 
                jnp.array(s_log_raw),
                left=-10.0, # If out of bounds, assume effectively zero sensitivity
                right=-10.0
            )

            # 3. Linearize
            # The data is Log Sensitivity. S_linear = 10^(S_log)
            # We clip the lower bound to avoid numerical underflow issues
            return jnp.power(10.0, s_log_interp)

        # Process all three channels
        print(f"Loading Red:   {red_path}")
        r_curve = process_channel(red_path)
        
        print(f"Loading Green: {green_path}")
        g_curve = process_channel(green_path)
        
        print(f"Loading Blue:  {blue_path}")
        b_curve = process_channel(blue_path)

        # Stack into (3, 64) array
        sensitivities = jnp.stack([r_curve, g_curve, b_curve], axis=0)

        return cls(sensitivities=sensitivities, wavelengths=target_wavelengths)

    @eqx.filter_jit
    def __call__(self, spectral_image: jax.Array) -> jax.Array:
        """
        Perform the spectral integration.

        Equation: H_c = Integral( L(lambda) * S_c(lambda) d_lambda )
        Discrete: H_c = dot( L, S_c )

        Args:
            spectral_image: Input spectral radiance image.
                            Shape: (H, W, N_wavelengths)

        Returns:
            Actinic RGB image (Linear Exposure).
            Shape: (H, W, 3)
        """
        # Contract over the last dimension (wavelengths)
        # spectral_image: (..., 64)
        # sensitivities:  (3, 64)
        # result:         (..., 3)
        actinic_exposure = jnp.dot(spectral_image, self.sensitivities.T)
        
        return actinic_exposure


# ==============================================================================
# MAIN TEST SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("--- Spectral Exposure Module Test ---")

    # Paths to the uploaded data
    base_dir = Path("data/Kodak_Vision3_250d")
    r_path = base_dir / "Red_log_ssf.csv"
    g_path = base_dir / "Green_log_ssf.csv"
    b_path = base_dir / "Blue_log_ssf.csv"

    # 1. Initialize Module
    try:
        exposure_mod = SpectralExposure.from_csvs(
            str(r_path), str(g_path), str(b_path)
        )
        print("Successfully loaded sensitivity curves.")
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        print("Please ensure the CSV files are in the data directory.")
        exit(1)

    # 2. Verify Data Shapes
    print(f"Sensitivity Shape: {exposure_mod.sensitivities.shape} (Expected: 3, 64)")
    
    # 3. Create a Dummy Spectral Image (Flat White Spectrum)
    # Shape (10, 10, 64)
    H, W = 10, 10
    flat_spectrum_value = 1.0
    dummy_spectral_img = jnp.ones((H, W, 64)) * flat_spectrum_value
    
    # 4. Compute Exposure
    actinic_rgb = exposure_mod(dummy_spectral_img)
    
    print(f"Output Image Shape: {actinic_rgb.shape} (Expected: {H}, {W}, 3)")
    print("Mean Actinic Exposure (Integration of 1.0 radiance):")
    print(jnp.mean(actinic_rgb, axis=(0, 1)))
    
    # 5. Physics Sanity Check
    # Blue layer typically has higher raw sensitivity integrals because it lacks 
    # the masking filters present in upper layers (though digital values depend heavily on normalization).
    # We just check that values are positive and non-zero.
    if jnp.all(actinic_rgb > 0):
        print(">> SUCCESS: Calculated non-zero positive exposures.")
    else:
        print(">> WARNING: Zeros detected in exposure output.")