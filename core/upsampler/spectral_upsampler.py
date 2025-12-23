"""
Spectral Upsampler - Fast Image Processing with Precomputed LUT

This module provides an Equinox-based spectral upsampler that converts RGB images
to spectral images using a precomputed lookup table.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.ndimage import map_coordinates
import equinox as eqx
from pathlib import Path
from typing import Optional

# ==============================================================================
# EQUINOX MODULE FOR SPECTRAL UPSAMPLER
# ==============================================================================

class SpectralUpsampler(eqx.Module):
    """
    Spectral upsampler with immutable state management via Equinox.

    Attributes:
        lut: Lookup table of polynomial coefficients (size, size, size, 3)
        wavelengths: Wavelength array (64,)
        w_norm_min: Minimum wavelength for normalization
        w_norm_range: Wavelength range for normalization
        lut_size: Size of the LUT grid
        effective_cmf: Effective color matching functions for verification
        xyz_to_rgb: XYZ to RGB conversion matrix
    """
    lut: jax.Array
    wavelengths: jax.Array
    w_norm_min: float
    w_norm_range: float
    lut_size: int
    effective_cmf: jax.Array
    xyz_to_rgb: jax.Array

    def __init__(
        self,
        lut: jax.Array,
        wavelengths: jax.Array,
        w_norm_min: float,
        w_norm_range: float,
        effective_cmf: jax.Array,
        xyz_to_rgb: jax.Array,
    ):
        """
        Initialize the spectral upsampler.

        Args:
            lut: Precomputed lookup table
            wavelengths: Wavelength array
            w_norm_min: Minimum wavelength for normalization
            w_norm_range: Wavelength range for normalization
            effective_cmf: Effective color matching functions
            xyz_to_rgb: XYZ to RGB conversion matrix
        """
        self.lut = lut
        self.wavelengths = wavelengths
        self.w_norm_min = w_norm_min
        self.w_norm_range = w_norm_range
        self.lut_size = lut.shape[0]
        self.effective_cmf = effective_cmf
        self.xyz_to_rgb = xyz_to_rgb

    @classmethod
    def from_file(cls, lut_path: str) -> "SpectralUpsampler":
        """
        Load a spectral upsampler from a saved LUT file.

        Args:
            lut_path: Path to the LUT NPZ file

        Returns:
            SpectralUpsampler instance
        """
        # Load NPZ file
        lut_data = jnp.load(lut_path)

        return cls(
            lut=lut_data['lut'],
            wavelengths=lut_data['wavelengths'],
            w_norm_min=float(lut_data['w_norm_min']),
            w_norm_range=float(lut_data['w_norm_range']),
            effective_cmf=lut_data['effective_cmf'],
            xyz_to_rgb=lut_data['xyz_to_rgb'],
        )

    @classmethod
    def from_lut_size(cls, size: int = 32, data_dir: str = "data/luts") -> "SpectralUpsampler":
        """
        Load a spectral upsampler from a LUT file based on size.

        Args:
            size: LUT grid size
            data_dir: Directory containing LUT files

        Returns:
            SpectralUpsampler instance
        """
        lut_path = Path(data_dir) / f"spectral_lut_{size}.npz"
        if not lut_path.exists():
            raise FileNotFoundError(
                f"LUT file not found: {lut_path}\n"
                f"Please run generate_lut.py first to create the LUT."
            )
        return cls.from_file(str(lut_path))

    def _algebraic_sigmoid(self, x: jax.Array) -> jax.Array:
        """Jakob 2019 Sigmoid: Maps R -> (0, 1)"""
        return 0.5 * (x / jnp.sqrt(1.0 + x**2) + 1.0)

    def _eval_spectrum_at_pixel(self, coeffs: jax.Array) -> jax.Array:
        """
        Evaluate the spectral reflectance from polynomial coefficients.

        Args:
            coeffs: Polynomial coefficients [A, B, C]

        Returns:
            Spectral reflectance array of length 64
        """
        w_norm = 2.0 * (self.wavelengths - self.w_norm_min) / self.w_norm_range - 1.0
        poly = coeffs[0] * w_norm**2 + coeffs[1] * w_norm + coeffs[2]
        return self._algebraic_sigmoid(poly)

    @eqx.filter_jit
    def upsample(self, rgb_image: jax.Array) -> jax.Array:
        """
        Convert RGB image to spectral image.

        Args:
            rgb_image: Input RGB image of shape (H, W, 3) with values in [0, 1]

        Returns:
            Spectral image of shape (H, W, 64) representing reflectance spectra
        """
        # 1. Scale RGB to LUT Index Space
        coords = jnp.clip(rgb_image, 0.0, 1.0) * (self.lut_size - 1)

        # 2. Prepare for interpolation (Dimensions: 3 x N_pixels)
        flat_coords = coords.reshape(-1, 3).transpose()

        # 3. Trilinear Interpolate Coefficients (A, B, C)
        # We must interpolate each channel separately
        coeffs_list = [
            map_coordinates(self.lut[..., i], flat_coords, order=1, mode='nearest')
            for i in range(3)
        ]
        coeffs = jnp.stack(coeffs_list, axis=-1)

        # 4. Evaluate Spectrum (Vectorized)
        spd_flat = vmap(self._eval_spectrum_at_pixel)(coeffs)

        # 5. Reshape to Image
        h, w, _ = rgb_image.shape
        return spd_flat.reshape(h, w, len(self.wavelengths))

    @eqx.filter_jit
    def verify_reconstruction(self, rgb_image: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Verify the quality of spectral reconstruction by converting back to RGB.

        Args:
            rgb_image: Original RGB image of shape (H, W, 3)

        Returns:
            Tuple of (reconstructed_rgb, max_error, mean_error)
        """
        # Upsample to spectral
        spd_img = self.upsample(rgb_image)

        # Convert back to RGB
        h, w, _ = rgb_image.shape
        flat_spd = spd_img.reshape(-1, len(self.wavelengths))
        xyz_recon = jnp.dot(flat_spd, self.effective_cmf)
        rgb_recon = jnp.dot(xyz_recon, self.xyz_to_rgb.T).reshape(h, w, 3)

        # Compute errors
        diff = jnp.abs(rgb_image - rgb_recon)
        max_err = jnp.max(diff)
        mean_err = jnp.mean(diff)

        return rgb_recon, max_err, mean_err


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_upsampler(lut_size: int = 32, data_dir: str = "data/luts") -> SpectralUpsampler:
    """
    Create a spectral upsampler from a saved LUT.

    Args:
        lut_size: Size of the LUT grid
        data_dir: Directory containing LUT files

    Returns:
        SpectralUpsampler instance
    """
    return SpectralUpsampler.from_lut_size(lut_size, data_dir)


@jit
def upsample_rgb_to_spectral(
    rgb_image: jax.Array,
    upsampler: SpectralUpsampler
) -> jax.Array:
    """
    Convenience function to upsample RGB image to spectral.

    Args:
        rgb_image: RGB image of shape (H, W, 3)
        upsampler: SpectralUpsampler instance

    Returns:
        Spectral image of shape (H, W, 64)
    """
    return upsampler.upsample(rgb_image)


# ==============================================================================
# MAIN DEMO
# ==============================================================================

if __name__ == "__main__":
    import time

    print("--- Spectral Upsampler Demo ---")

    # 1. Load the upsampler
    print("Loading spectral upsampler...")
    try:
        upsampler = create_upsampler(lut_size=32, data_dir="data/luts")
        print(f"Loaded LUT with size {upsampler.lut_size}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run generate_lut.py first.")
        exit(1)

    # 2. Create Test Image (Gradient)
    H, W = 256, 256
    print(f"Creating test image ({H}x{W})...")
    xx, yy = jnp.meshgrid(jnp.linspace(0, 1, W), jnp.linspace(0, 1, H))
    test_img = jnp.stack([xx, yy, jnp.ones_like(xx)*0.5], axis=-1)

    # 3. Upsample
    print("Upsampling image to spectral...")
    start_t = time.time()
    spd_img = upsampler.upsample(test_img)
    spd_img.block_until_ready()
    elapsed = time.time() - start_t
    print(f"Upsampled in {elapsed:.4f}s")
    print(f"Output shape: {spd_img.shape}")

    # 4. Verify Accuracy
    print("\nVerifying reconstruction quality...")
    rgb_recon, max_err, mean_err = upsampler.verify_reconstruction(test_img)

    print(f"Max Reconstruction Error:  {max_err:.5f}")
    print(f"Mean Reconstruction Error: {mean_err:.5f}")

    if max_err < 0.02:
        print(">> SUCCESS: Reconstruction is accurate.")
    else:
        print(">> WARNING: High reconstruction error detected.")

    # 5. Test with pure colors
    print("\nTesting with pure RGB colors...")
    color_names = ['Red', 'Green', 'Blue', 'White', 'Black']
    color_values = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 1.0],  # White
        [0.0, 0.0, 0.0],  # Black
    ]

    for name, color_val in zip(color_names, color_values):
        # Create 1x1x3 image
        color = jnp.array([[color_val]])
        spd = upsampler.upsample(color)
        rgb_recon, max_e, mean_e = upsampler.verify_reconstruction(color)
        print(f"  {name}: max_err={max_e:.5f}, mean_err={mean_e:.5f}")
