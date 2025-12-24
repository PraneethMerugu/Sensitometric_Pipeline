# Master Implementation Plan: High-Fidelity Spectral Halation

This plan details the precision upgrade of the `OpticalPhysics` module to support **Spectral Radiative Transfer**, replacing static weights with physical light transport simulations.

## Phase 1: The Physics Upgrade (`core/optical/scattering.py`)

**Action:** Replace the entire content of `core/optical/scattering.py` with the code below.

**Technical Justification:**
* **`transmission_spectrum`**: Defines the "Yellow Filter" physics. It blocks wavelengths < 500nm (Blue) from reaching the base, preventing incorrect blue halation.
* **`compute_halation_mask`**: Performs the dot-product integration of the upsampled spectrum against the transmission curve.
* **`_diffused_annulus`**: Replaces the simple ring with a physically accurate "turbid reflection" (Gaussian offset).

```python
"""
core/optical/scattering.py

Combined Optical & Chemical Physics Module (High Fidelity Spectral Edition)
Implements Module II: Optical Scattering with Spectral Radiative Transfer.
"""

import jax
import jax.numpy as jnp
import jax.numpy.fft as fft
import equinox as eqx
from typing import Tuple, Optional
from core.upsampler.spectral_upsampler import SpectralUpsampler

class OpticalPhysics(eqx.Module):
    """
    Simulates physical light transport in the emulsion using Spectral Radiative Transfer.
    
    Mechanism:
    1. Spectral Upsampling: Reconstructs SPD from RGB input.
    2. Transmission Integration: Calculates light reaching the base (filtering out Blue).
    3. Compound Convolution: Applies Lorentzian Bloom (Forward) and Diffused Halation (Backward).
    """
    # --- Optical Parameters ---
    scatter_gamma: jax.Array      # Turbidity width (Lorentzian HWHM)
    bloom_mix: jax.Array          # Amount of forward scatter (Bloom)
    
    halation_radius: jax.Array    # Radius of reflection (Base thickness)
    halation_sigma: jax.Array     # Diffusion of the reflection (Rem-jet scattering)
    halation_gain: jax.Array      # Intensity of the halation effect
    
    # --- Physics Data ---
    # Shape (64,): Represents the combined transmission of the emulsion stack
    # T_total = T_Blue_layer * T_Yellow_Filter * T_Green_layer * T_Red_layer
    transmission_spectrum: jax.Array 
    
    # --- Dependencies ---
    upsampler: SpectralUpsampler

    def __init__(
        self,
        upsampler: SpectralUpsampler,
        scatter_gamma: float = 0.65,
        bloom_mix: float = 0.20,
        halation_radius: float = 30.0,
        halation_sigma: float = 8.0,
        halation_gain: float = 1.0,
        transmission_spectrum: Optional[jax.Array] = None
    ):
        self.upsampler = upsampler
        self.scatter_gamma = jnp.array(scatter_gamma)
        self.bloom_mix = jnp.array(bloom_mix)
        self.halation_radius = jnp.array(halation_radius)
        self.halation_sigma = jnp.array(halation_sigma)
        self.halation_gain = jnp.array(halation_gain)

        # PHYSICS KERNEL: The "Yellow Filter" Simulation
        # If no real data is provided, we generate a synthetic curve that mimics Vision3.
        # Vision3 blocks blue light from hitting the red layer/base.
        if transmission_spectrum is None:
            # Wavelengths match the upsampler (360-830nm, 64 bins)
            wl = jnp.linspace(360.0, 830.0, 64)
            
            # Sigmoid Cutoff at 540nm (Yellow Filter)
            # Low transmission for < 540nm (Blue/Green), High for > 540nm (Red)
            # This ensures Blue light does NOT cause halation.
            k = 0.1 # Steepness
            wl0 = 540.0 # Cutoff wavelength
            t_curve = 1.0 / (1.0 + jnp.exp(-k * (wl - wl0)))
            
            self.transmission_spectrum = t_curve
        else:
            self.transmission_spectrum = transmission_spectrum

    def _generate_mesh(self, shape: Tuple[int, int]) -> jax.Array:
        H, W = shape
        y = jnp.fft.fftfreq(H) * H
        x = jnp.fft.fftfreq(W) * W
        xx, yy = jnp.meshgrid(x, y)
        return jnp.sqrt(xx**2 + yy**2)

    # --- PSF Generators ---
    def _lorentzian(self, r, gamma):
        """Modified Lorentzian for Forward Scattering (Bloom)."""
        g = jnp.maximum(gamma, 1e-4)
        # Power 1.5 ensures finite energy in 2D integration
        psf = 1.0 / (1.0 + (r / g)**2)**1.5
        return psf / jnp.sum(psf)

    def _diffused_annulus(self, r, radius, sigma):
        """Gaussian Ring for Backward Scattering (Halation)."""
        s = jnp.maximum(sigma, 1e-4)
        # Gaussian offset by radius = Diffused Ring
        psf = jnp.exp(-((r - radius)**2) / (2 * s**2))
        return psf / jnp.sum(psf)

    def compute_halation_mask(self, rgb_image: jax.Array) -> jax.Array:
        """
        Phase I: Spectral Light Transport
        Integrates Scene Radiance against Stack Transmission.
        """
        # 1. Upsample RGB -> Spectral (H, W, 64)
        # Note: This uses the pre-computed LUT in the upsampler
        spectral_image = self.upsampler.upsample(rgb_image)
        
        # 2. Integrate (Dot Product)
        # Sum(L_lambda * T_lambda) over wavelength axis
        # Result: (H, W) intensity map of light acting on the base
        halation_intensity = jnp.dot(spectral_image, self.transmission_spectrum)
        
        return halation_intensity

    @eqx.filter_jit
    def __call__(self, actinic_image: jax.Array) -> jax.Array:
        H, W, C = actinic_image.shape
        r_grid = self._generate_mesh((H, W))

        # 1. Compute OTFs (Frequency Domain Kernels)
        psf_bloom = self._lorentzian(r_grid, self.scatter_gamma)
        psf_hal = self._diffused_annulus(r_grid, self.halation_radius, self.halation_sigma)

        otf_bloom = fft.fft2(psf_bloom)
        otf_hal = fft.fft2(psf_hal)

        # 2. Phase I: Calculate Halation Source (The Physics Upgrade)
        # Use spectral physics to find WHICH pixels cause halation
        halation_source = self.compute_halation_mask(actinic_image)
        
        # 3. Phase II: Convolution (FFT)
        img_fft = fft.fft2(actinic_image, axes=(0, 1))
        hal_fft = fft.fft2(halation_source, axes=(0, 1))

        # Apply Bloom: Mix clean image with bloomed image
        # Bloom affects all channels equally (structural scattering)
        bloomed_fft = img_fft * ((1.0 - self.bloom_mix) + self.bloom_mix * otf_bloom[..., None])
        
        # Apply Halation: Convolve the Spectral Mask with the Ring Kernel
        halated_layer_fft = hal_fft * otf_hal * self.halation_gain
        
        # 4. Reconstruction
        bloomed_spatial = jnp.abs(fft.ifft2(bloomed_fft, axes=(0, 1)))
        halated_spatial = jnp.abs(fft.ifft2(halated_layer_fft, axes=(0, 1)))

        # 5. Composite
        # Halation is primarily Red (back-scatter into red layer).
        # We add it to the Red channel.
        # (Optional: Add small amount to Green for cross-talk fidelity)
        halation_color_vector = jnp.array([1.0, 0.05, 0.0]) 
        
        final_image = bloomed_spatial + (halated_spatial[..., None] * halation_color_vector)

        return final_image

Phase 2: The Integration Upgrade (pipeline.py)

Action: Update pipeline.py to initialize the SpectralUpsampler and pass it to the new OpticalPhysics module.

Specific Code Changes:

    Imports: Add the upsampler factory.
    Python

from core.upsampler.spectral_upsampler import create_upsampler

__init__ Modification: Inside FilmPipeline.__init__, initialize the upsampler before OpticalPhysics and pass it in.
Python

    # ... inside __init__ ...

    # 0. Initialize Shared Resources
    # Load the 32x32x32 LUT (Fast & Accurate enough for halation)
    self.upsampler = create_upsampler(lut_size=32, data_dir="data/luts")

    # 1. Optical
    self.optical = OpticalPhysics(
        upsampler=self.upsampler,  # <--- INJECT DEPENDENCY HERE
        scatter_gamma=params.get('scatter_gamma', 0.65),
        bloom_mix=params.get('bloom_weight', 0.15), # Renamed param for clarity
        halation_radius=params.get('halation_radius', 30.0),
        halation_sigma=params.get('halation_sigma', 8.0),
        halation_gain=params.get('halation_gain', 1.5)
    )
    # ... rest of init ...

    Dependency Handling: Ensure FilmPipeline class definition includes the upsampler type hint if strict typing is used, or just let Equinox handle it (it automates the PyTree registration).

Phase 3: Verification Strategy

Test Case: "The Blue Light Test" Create a test script tests/test_spectral_physics.py:

    Input: Create an image with two dots:

        Dot A: Pure Red (1.0, 0.0, 0.0)

        Dot B: Pure Blue (0.0, 0.0, 1.0)

    Execution: Run pipeline.optical(input).

    Assertion (The Robustness Check):

        Dot A (Red): Should have a strong halo (values > 0 in neighborhood).

        Dot B (Blue): Should have near-zero halo.

        Why? Because the transmission_spectrum sigmoid we defined blocks Blue light.

    Failure Condition: If Dot B has a red halo, the Spectral Integration is failing (or the transmission curve is wrong).