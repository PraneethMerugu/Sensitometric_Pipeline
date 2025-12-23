"""
Colorimetric Transformation Module

[cite_start]This module simulates the subtractive mixing of dyes in the emulsion[cite: 22].
It maps Analytical Dye Densities (C, M, Y) to Integral Spectral Densities (R, G, B)
using a coupling matrix derived from the film's actual spectral dye density curves.

The transformation follows Beer's Law:
    D_integral = Matrix @ D_analytical + D_base

Reference:
Module V: Colorimetric Transformation
[cite_start]"Final Film Emulation Pipeline", Section 6 [cite: 143]
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

class ColorimetricTransform(eqx.Module):
    """
    Applies Beer's Law to convert dye concentrations to optical density.
    
    Attributes:
        mixing_matrix: 3x3 matrix defining dye absorption profiles.
                       Rows = Scanner Channels [Red, Green, Blue]
                       Cols = Film Dyes [Cyan, Magenta, Yellow]
                       Example: matrix[0,1] is the Red density of the Magenta dye.
        [cite_start]base_density: 3x1 vector defining the minimum density (Orange Mask)[cite: 153].
        scan_wavelengths: The specific wavelengths (R, G, B) used for the simulation.
    """
    mixing_matrix: jax.Array
    base_density: jax.Array
    scan_wavelengths: jax.Array

    def __init__(
        self, 
        mixing_matrix: jax.Array, 
        base_density: jax.Array,
        scan_wavelengths: jax.Array
    ):
        self.mixing_matrix = mixing_matrix
        self.base_density = base_density
        self.scan_wavelengths = scan_wavelengths

    @classmethod
    def from_csvs(
        cls,
        cyan_path: str,
        magenta_path: str,
        yellow_path: str,
        min_density_path: str,
        scan_wavelengths: jax.Array = jnp.array([650.0, 550.0, 450.0])
    ) -> "ColorimetricTransform":
        """
        Derive the mixing matrix by sampling spectral density CSVs at specific
        scanner wavelengths.

        Args:
            cyan_path: Path to 'cyan_density.csv' (Cyan dye spectral absorption).
            magenta_path: Path to 'magenta_density.csv'.
            yellow_path: Path to 'yellow_density.csv'.
            min_density_path: Path to 'minimum_density.csv' (Base/Fog/Mask).
            scan_wavelengths: Array of 3 wavelengths [Red_nm, Green_nm, Blue_nm].
                              Defaults to [650, 550, 450] (Typical Status M centers).

        Returns:
            Initialized ColorimetricTransform module.
        """
        def load_and_interp(path: str) -> jax.Array:
            # 1. Load CSV
            # Handles variable headers (e.g., "x, y" or "wavelength, density")
            try:
                df = pd.read_csv(path)
                # Ensure sorted unique index for interpolation
                # (Fixes potential duplicates in data)
                if 'x' in df.columns and 'y' in df.columns:
                    df = df.sort_values('x').drop_duplicates('x')
                    w = df['x'].values.astype(float)
                    d = df['y'].values.astype(float)
                else:
                    # Fallback to column index
                    df = df.sort_values(df.columns[0]).drop_duplicates(df.columns[0])
                    w = df.iloc[:, 0].values.astype(float)
                    d = df.iloc[:, 1].values.astype(float)
            except Exception as e:
                raise ValueError(f"Failed to load {path}: {e}")

            # 2. Interpolate to Scan Wavelengths
            # jnp.interp requires sorted x-coordinates (w)
            return jnp.interp(scan_wavelengths, jnp.array(w), jnp.array(d))

        print(f"Loading Colorimetric Data...")
        
        # Load Dye Columns (Analytical Densities)
        # Each vector is [Density_Red, Density_Green, Density_Blue] for that dye
        c_vec = load_and_interp(cyan_path)
        m_vec = load_and_interp(magenta_path)
        y_vec = load_and_interp(yellow_path)
        
        # Load Base Density (Orange Mask)
        base_vec = load_and_interp(min_density_path)

        # Construct Matrix: Columns are C, M, Y
        # shape: (3, 3)
        # [[Cr, Mr, Yr],
        #  [Cg, Mg, Yg],
        #  [Cb, Mb, Yb]]
        mixing_matrix = jnp.stack([c_vec, m_vec, y_vec], axis=1)

        print("--- Derived Mixing Matrix ---")
        print(mixing_matrix)
        print("--- Derived Base Density ---")
        print(base_vec)

        return cls(mixing_matrix, base_vec, scan_wavelengths)

    @eqx.filter_jit
    def __call__(self, analytical_dyes: jax.Array) -> jax.Array:
        """
        Convert Analytical Dyes (C, M, Y) to Integral Densities (R, G, B).

        Args:
            analytical_dyes: (H, W, 3) Image of pure dye amounts.
                             Order: Cyan, Magenta, Yellow.

        Returns:
            (H, W, 3) Integral Density Image (The Virtual Negative).
            Order: Red, Green, Blue.
        """
        # Reshape for matrix multiplication: (H, W, 3) -> (H, W, 3, 1)
        dyes_reshaped = analytical_dyes[..., None]
        
        # Apply Beer's Law: D = A @ C + b
        # (3,3) @ (3,1) -> (3,1)
        integral_density = self.mixing_matrix @ dyes_reshaped
        
        # Remove singleton dim and add base mask
        integral_density = integral_density.squeeze(-1) + self.base_density
        
        return integral_density

    def invert(self, integral_density: jax.Array) -> jax.Array:
        """
        Invert a negative scan back to analytical dye amounts.
        Equation: C = A_inv @ (D - b)
        """
        # Subtract mask
        dyes_centered = integral_density - self.base_density
        
        # Invert Matrix
        inv_matrix = jnp.linalg.inv(self.mixing_matrix)
        
        # Solve
        dyes_reshaped = dyes_centered[..., None]
        analytical = inv_matrix @ dyes_reshaped
        return analytical.squeeze(-1)

# ==============================================================================
# MAIN TEST SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("--- Colorimetric Transformation Test ---")
    
    # Define Paths
    base_dir = Path("data/Kodak_Vision3_250d")
    c_path = base_dir / "cyan_density.csv"
    m_path = base_dir / "magenta_density.csv"
    y_path = base_dir / "yellow_density.csv"
    min_path = base_dir / "minimum_density.csv"
    
    try:
        # 1. Initialize from CSVs
        color_mod = ColorimetricTransform.from_csvs(
            str(c_path), str(m_path), str(y_path), str(min_path),
            scan_wavelengths=jnp.array([650.0, 550.0, 450.0])
        )
        
        # 2. Verify Physics (Crosstalk)
        # The Magenta dye (Column 1) should have a 'bump' in the Blue channel (Row 2).
        # Matrix Index [2, 1] -> Blue Density of Magenta Dye.
        
        m_blue_density = color_mod.mixing_matrix[2, 1]
        m_green_density = color_mod.mixing_matrix[1, 1]
        
        print(f"\nChecking Magenta Crosstalk:")
        print(f"  Magenta Principal Density (Green channel): {m_green_density:.4f}")
        print(f"  Magenta Unwanted Absorption (Blue channel): {m_blue_density:.4f}")
        
        if m_blue_density > 0.1:
            print("  >> SUCCESS: Detected significant unwanted blue absorption (physics-accurate).")
        else:
            print("  >> WARNING: Unwanted absorption seems low. Check CSV data.")

        # 3. Simulate a 'White' Exposure
        # White Light -> High Exposure -> High Dye Density (Negative)
        # Input: 1.0 unit of each dye
        pixel_in = jnp.ones((1, 1, 3)) 
        pixel_out = color_mod(pixel_in)
        
        print(f"\nSimulated Negative Density (Input C=M=Y=1.0):")
        print(f"  R: {pixel_out[0,0,0]:.3f}")
        print(f"  G: {pixel_out[0,0,1]:.3f}")
        print(f"  B: {pixel_out[0,0,2]:.3f}")
        
        # The Blue channel density should be highest due to the yellow dye + orange mask
        if pixel_out[0,0,2] > pixel_out[0,0,0]:
             print("  >> SUCCESS: Blue channel is densest (Orange Mask effect).")

    except FileNotFoundError as e:
        print(f"\nError: Could not find data files.\n{e}")
        print("Please ensure the CSV files are in 'Sensitometric_Pipeline/data/Kodak_Vision3_250d/'")