"""
core/optical/scattering.py

Combined Optical & Chemical Physics Module (High Fidelity Spectral Edition)
Implements Module II: Optical Scattering with Spectral Radiative Transfer.
"""

import jax
import jax.numpy as jnp
import jax.numpy.fft as fft
import equinox as eqx
from typing import Tuple, Optional, Dict
from core.upsampler.spectral_upsampler import SpectralUpsampler

class OpticalPhysics(eqx.Module):
    """
    Simulates physical light transport in the emulsion using Spectral Radiative Transfer.
    New "Sandwich Model" implementation:
    1. RGB -> Spectral Upsampling
    2. Spectral Integration -> Blue, Green, Red Exposure Maps
    3. Multi-Pass Scattering:
       - Top Bloom (Blue)
       - Yellow Filter Attenuation
       - Bottom Bloom (Green+Red+FilteredBlue)
       - Halation (Reflection from Bottom)
    """
    # --- Optical Parameters ---
    scatter_gamma: jax.Array      # Turbidity width (Lorentzian HWHM)
    bloom_mix: jax.Array          # Amount of forward scatter (Bloom)
    
    halation_radius: jax.Array    # Radius of reflection (Base thickness)
    halation_sigma: jax.Array     # Diffusion of the reflection (Rem-jet scattering)
    halation_gain: jax.Array      # Intensity of the halation effect
    
    # --- Physics Data ---
    # Shape (64,): Spectral Sensitivity Functions for Blue, Green, Red layers
    ssf_blue: jax.Array
    ssf_green: jax.Array
    ssf_red: jax.Array

    # Shape (64,): Transmission of the Yellow Filter
    transmission_yellow: jax.Array
    
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
        sensitivity_data: Optional[Dict[str, jax.Array]] = None
    ):
        self.upsampler = upsampler
        self.scatter_gamma = jnp.array(scatter_gamma)
        self.bloom_mix = jnp.array(bloom_mix)
        self.halation_radius = jnp.array(halation_radius)
        self.halation_sigma = jnp.array(halation_sigma)
        self.halation_gain = jnp.array(halation_gain)

        # PHYSICS KERNEL: Spectral Sensitivity & Filters
        # Default: Load standard Vision3 250D curves if not provided
        if sensitivity_data is None:
             # Load default data from disk (using helper or synthetics)
             # For robustness, we will generate synthetic Gaussian sensitivities if files missing
             # But ideally we load the CSVs. Since we are inside the module, we can use numpy to load.
             # However, we are in an equinox module, init happens on CPU usually.
             
             # Fallback Synthetic Data (Gaussian approximations of Vision3)
             wl = jnp.linspace(360.0, 830.0, 64)
             
             def gaussian(x, mu, sig):
                 return jnp.exp(-0.5 * ((x - mu) / sig)**2)
             
             self.ssf_blue = gaussian(wl, 450.0, 40.0)
             self.ssf_green = gaussian(wl, 540.0, 40.0)
             self.ssf_red = gaussian(wl, 640.0, 40.0) # Red has a long tail, but gaussian for now

             # Synthetic Yellow Filter (Sigmoid cutoff at 520nm)
             # Blocks Blue (<520), Passes Green/Red (>520)
             k = 0.15
             wl0 = 510.0
             self.transmission_yellow = 1.0 / (1.0 + jnp.exp(-k * (wl - wl0)))
             
        else:
             self.ssf_blue = sensitivity_data['blue']
             self.ssf_green = sensitivity_data['green']
             self.ssf_red = sensitivity_data['red']
             self.transmission_yellow = sensitivity_data.get('yellow', jnp.ones(64))

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

    def _physical_halation_kernel(self, r_grid):
        """
        Physically derived Halation Kernel (Fresnel Reflection).
        Based on reflection off the base-air/remjet interface.
        """
        # r_grid is distance in pixels.
        # r_norm = r_grid / self.halation_radius # Radius normalized to critical ring
        
        # Avoid division by zero
        r_radius = jnp.maximum(self.halation_radius, 1e-4)
        r_norm = r_grid / r_radius
        
        # Empirical fit to Fresnel reflection curve:
        # Sharp peak at r_norm = 1.0 (Critical Angle), rapid falloff after.
        intensity = jnp.exp(4.0 * (r_norm - 1.0)) * (r_norm <= 1.0)
        
        # Add diffusion (scattering by the rem-jet backing)
        # Convolve this sharp ring with a small Gaussian (Approximated)
        # We use halation_sigma to control the width of this peak effectively? 
        # The snippet had fixed 0.2. Let's make it relative?
        # User snippet: kernel = intensity * jnp.exp(-0.5 * ((r_norm - 1.0)/0.2)**2)
        # I will stick to the snippet's constants as requested, 
        # but maybe scaling 0.2 by some factor if needed? 
        # Snippet hardcoded 0.2. I'll use it.
        
        kernel = intensity * jnp.exp(-0.5 * ((r_norm - 1.0)/0.2)**2)
        
        return kernel / (jnp.sum(kernel) + 1e-8)

    def integrate_exposure(self, spectral_cube: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Integrates the spectral cube against the film sensitivity functions.
        Returns: (Blue_Exp, Green_Exp, Red_Exp) maps.
        """
        # spectral_cube: (H, W, 64)
        # ssf: (64,)
        b_exp = jnp.dot(spectral_cube, self.ssf_blue)
        g_exp = jnp.dot(spectral_cube, self.ssf_green)
        r_exp = jnp.dot(spectral_cube, self.ssf_red)
        return b_exp, g_exp, r_exp
        
    @eqx.filter_jit
    def __call__(self, actinic_image: jax.Array) -> jax.Array:
        H, W, C = actinic_image.shape
        r_grid = self._generate_mesh((H, W))

        # 1. Compute Kernels (OTFs)
        psf_bloom = self._lorentzian(r_grid, self.scatter_gamma)
        # psf_hal = self._diffused_annulus(r_grid, self.halation_radius, self.halation_sigma)
        psf_hal = self._physical_halation_kernel(r_grid)

        otf_bloom = fft.fft2(psf_bloom)
        otf_hal = fft.fft2(psf_hal)

        # 2. Spectral Upsampling & Integration (The Physics Core)
        # Transform RGB -> Spectral -> Layer Exposure Maps
        spectral_image = self.upsampler.upsample(actinic_image)
        b_exp, g_exp, r_exp = self.integrate_exposure(spectral_image)
        
        # 3. The "Sandwich" Scattering Model
        
        # --- PASS 1: Top Layer (Blue) ---
        # Blue light scatters immediately (Bloom)
        b_fft = fft.fft2(b_exp)
        # Bloom equation: (1-mix)*img + mix*convolved
        b_bloomed_fft = b_fft * ((1.0 - self.bloom_mix) + self.bloom_mix * otf_bloom)
        b_bloomed = jnp.abs(fft.ifft2(b_bloomed_fft))
        
        # --- ABSORPTION: Yellow Filter ---
        # Calculate transmission factor for the broadband Blue channel
        blue_pass_factor = jnp.dot(self.ssf_blue * self.transmission_yellow, jnp.ones(64)) / jnp.sum(self.ssf_blue)
        
        # --- PASS 2: Deep Layers (Green + Red + Filtered Blue) ---
        # The filter absorbs Blue light *before* it reaches the scattering volume of G/R.
        # Deep Source = G_exp + R_exp + (B_exp * transmission)
        # We use the unbloomed b_exp because the filter is sharp and geometry implies 
        # the blue halo also passes through or is formed by the deep scattering?
        # User fix: "Add the unbloomed b_exp to the deep source and then blooming it"
        
        blue_filtered = b_exp * blue_pass_factor
        deep_source = g_exp + r_exp + blue_filtered
        
        deep_fft = fft.fft2(deep_source)
        # Apply Bloom to this combined deep light (Simulating scattering in the emulsion bulk)
        deep_bloomed_fft = deep_fft * ((1.0 - self.bloom_mix) + self.bloom_mix * otf_bloom)
        
        # --- PASS 3: Halation (Reflection) ---
        # Halation is driven by the light reaching the base (Deep Bloomed)
        # Convolve with PHYSICAL Halation Kernel
        halation_fft = deep_bloomed_fft * otf_hal * self.halation_gain
        halation_signal = jnp.abs(fft.ifft2(halation_fft))
        
        # 4. Re-Assemble Latent Image (RGB)
        # We need individual channels. 
        # Standard Channel Blooms:
        # B_final = B_bloomed
        # G_final = Bloom(G) + Halation crosstalk
        # R_final = Bloom(R) + Halation crosstalk
        
        # Note: We calculated deep_bloomed_fft effectively as Sum(Bloom(G), Bloom(R), Bloom(B_filt)).
        # But we haven't computed Bloom(G) and Bloom(R) individually for the final image yet.
        # We'll compute them now.
        
        g_fft = fft.fft2(g_exp)
        r_fft = fft.fft2(r_exp)
        
        g_bloomed_fft = g_fft * ((1.0 - self.bloom_mix) + self.bloom_mix * otf_bloom)
        r_bloomed_fft = r_fft * ((1.0 - self.bloom_mix) + self.bloom_mix * otf_bloom)
        
        g_bloomed = jnp.abs(fft.ifft2(g_bloomed_fft))
        r_bloomed = jnp.abs(fft.ifft2(r_bloomed_fft))
        
        # Composite Halation
        # Halation is primarily Red/Orange, but we map it based on halation_vector
        # Standard Halation Color: Reddish
        hal_vector = jnp.array([0.0, 0.05, 1.0]) # B, G, R weights
        
        b_final = b_bloomed + (halation_signal * hal_vector[0])
        g_final = g_bloomed + (halation_signal * hal_vector[1])
        r_final = r_bloomed + (halation_signal * hal_vector[2])
        
        final_latent = jnp.stack([b_final, g_final, r_final], axis=-1)

        return final_latent