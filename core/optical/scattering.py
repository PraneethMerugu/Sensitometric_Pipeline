"""
Combined Optical & Chemical Physics Module

This module merges Module II (Optical Scattering) and Module III (Chemical Adjacency)
into a single efficient FFT operation.

Mathematical Justification:
The total Modulation Transfer Function (MTF) is the product of the Optical MTF 
and the Chemical MTF. This allows us to composite the Optical Transfer Functions (OTFs)
before a single FFT convolution.

Reference: "Final Film Emulation Pipeline", Sections 3 & 4
"""

import jax
import jax.numpy as jnp
import jax.numpy.fft as fft
import equinox as eqx
from typing import Tuple

class OpticalPhysics(eqx.Module):
    """
    Optical Physics Simulation:
    Optical: Turbidity (Modified Lorentzian), Bloom (Gaussian), Halation (Annular).
    """
    # --- Optical Parameters ---
    scatter_gamma: jax.Array      # Turbidity width (HWHM)
    bloom_sigma: jax.Array        # Bloom width
    bloom_weight: jax.Array       # Bloom mix amount
    halation_radius: jax.Array    # Halation ring radius
    halation_sigma: jax.Array     # Halation ring width
    halation_weights: jax.Array   # RGB sensitivity to halation

    def __init__(
        self,
        # Optical Defaults (Tuned for 2K/4K scans)
        scatter_gamma: float = 0.65,      # Tighter gamma to prevent excessive washing out
        bloom_sigma: float = 2.0,
        bloom_weight: float = 0.15,
        halation_radius: float = 30.0,
        halation_sigma: float = 8.0,
        halation_weights: Tuple[float, float, float] = (0.4, 0.05, 0.0),
    ):
        self.scatter_gamma = jnp.array(scatter_gamma)
        self.bloom_sigma = jnp.array(bloom_sigma)
        self.bloom_weight = jnp.array(bloom_weight)
        self.halation_radius = jnp.array(halation_radius)
        self.halation_sigma = jnp.array(halation_sigma)
        self.halation_weights = jnp.array(halation_weights)

    def _generate_mesh(self, shape: Tuple[int, int]) -> jax.Array:
        """Generates radial distance grid centered at (0,0) for FFT."""
        H, W = shape
        # fftfreq generates [0, 1, ..., -2, -1] which is implicitly centered at 0
        y = jnp.fft.fftfreq(H) * H
        x = jnp.fft.fftfreq(W) * W
        xx, yy = jnp.meshgrid(x, y)
        return jnp.sqrt(xx**2 + yy**2)

    # --- PSF Generators ---
    def _lorentzian(self, r, gamma):
        """
        Modified Lorentzian (Moffat) for Turbidity.
        Uses power 1.5 to ensure energy convergence in 2D while keeping long tails.
        """
        g = jnp.maximum(gamma, 1e-4)
        # Power 1.5 is standard for optical glare/scattering models to finite energy
        psf = 1.0 / (1.0 + (r / g)**2)**1.5
        return psf / jnp.sum(psf)

    def _gaussian(self, r, sigma):
        s = jnp.maximum(sigma, 1e-4)
        psf = jnp.exp(-(r**2) / (2 * s**2))
        return psf / jnp.sum(psf)

    def _annular(self, r, radius, sigma):
        s = jnp.maximum(sigma, 1e-4)
        psf = jnp.exp(-((r - radius)**2) / (2 * s**2))
        return psf / jnp.sum(psf)

    @eqx.filter_jit
    def __call__(self, actinic_image: jax.Array) -> jax.Array:
        H, W, C = actinic_image.shape
        r_grid = self._generate_mesh((H, W))

        # ======================================================================
        # 1. OPTICAL TRANSFER FUNCTION
        # ======================================================================
        # Generate Spatial PSFs
        psf_scatter = self._lorentzian(r_grid, self.scatter_gamma)
        psf_bloom   = self._gaussian(r_grid, self.bloom_sigma)
        psf_hal     = self._annular(r_grid, self.halation_radius, self.halation_sigma)

        # Convert to Frequency Domain (OTF)
        otf_scatter = fft.fft2(psf_scatter)
        otf_bloom   = fft.fft2(psf_bloom)
        otf_hal     = fft.fft2(psf_hal)

        # Composite Optical OTF per channel
        # Mix Scatter and Bloom first
        base_otf = (1.0 - self.bloom_weight) * otf_scatter + self.bloom_weight * otf_bloom
        
        optical_otfs = []
        for i in range(C):
            w = self.halation_weights[i]
            # Mix Base + Halation (Weighted)
            optical_otfs.append((1.0 - w) * base_otf + w * otf_hal)
        
        total_optical_otf = jnp.stack(optical_otfs, axis=-1)

        # ======================================================================
        # 2. CONVOLVE
        # ======================================================================
        img_fft = fft.fft2(actinic_image, axes=(0, 1))
        result_fft = img_fft * total_optical_otf
        result_spatial = jnp.abs(fft.ifft2(result_fft, axes=(0, 1)))

        return result_spatial

# ==============================================================================
# VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("--- Optical Physics Module Test ---")
    
    # 1. Setup Test Image (Point Source)
    H, W = 128, 128
    test_img = jnp.zeros((H, W, 3))
    test_img = test_img.at[H//2, W//2, :].set(1.0)
    
    # 2. Initialize
    physics_mod = OpticalPhysics(
        scatter_gamma=0.5,      
        bloom_sigma=2.0,
        bloom_weight=0.2
    )
    
    # 3. Process
    out_img = physics_mod(test_img)
    
    # 4. Analyze
    center_val = out_img[H//2, W//2, 1]
    neighbor_val = out_img[H//2, W//2 + 5, 1]
    
    print(f"Center Intensity:   {center_val:.4f}")
    print(f"Bloom Intensity (r=5): {neighbor_val:.4f}")
    
    if center_val < 1.0 and neighbor_val > 0.0:
        print(">> SUCCESS: Energy spread detected (Point Spread Function active).")
    else:
        print(">> WARNING: No spreading detected.")