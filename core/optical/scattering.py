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

    def _diffused_annulus(self, r, radius, sigma):
        """Gaussian Ring for Backward Scattering (Halation)."""
        s = jnp.maximum(sigma, 1e-4)
        # Gaussian offset by radius = Diffused Ring
        psf = jnp.exp(-((r - radius)**2) / (2 * s**2))
        return psf / jnp.sum(psf)

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
        psf_hal = self._diffused_annulus(r_grid, self.halation_radius, self.halation_sigma)

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
        # The bloomed blue light passes through the yellow filter.
        # We need a scalar attenuation factor for the *already integrated* blue light.
        # Approximation: We assume the Blue SSF is fully "behind" the filter relative to the base?
        # No, physically: Blue Layer -> Filter -> Green -> Red -> Base.
        # So "Blue Bloomed" is what forms the Blue Image.
        # But "Blue Light" that continues down is filtered.
        # We calculate "Filtered Blue" by attenuating the Blue Exposure.
        # Since we don't have the full spectrum at this point (we collapsed it), 
        # we need an effective transmission coefficient.
        # T_eff = dot(Blue_SSF * Yellow_T, Spectrum) / dot(Blue_SSF, Spectrum) ?
        # Simpler: Just integrate the Yellow T against the Blue Sensitivity to get a scalar loss.
        # scalar_T = dot(SSF_Blue * T_Yellow) / sum(SSF_Blue)
        # This represents "How much of the light that excites Blue also passes through Yellow?"
        
        blue_pass_factor = jnp.dot(self.ssf_blue * self.transmission_yellow, jnp.ones(64)) / jnp.sum(self.ssf_blue)
        # Ideally we'd filter the spectrum itself, but for performance we operate on the exposure map.
        # We assume the spectrum of the light matches the SSF (simplification).
        
        blue_continuing = b_bloomed * blue_pass_factor

        # --- PASS 2: Bottom Layers (Green + Red + Filtered Blue) ---
        # Light reaching here includes the scene's Green/Red light + the filtered Blue scatter.
        # We combine them into a "Deep Scatter Source".
        deep_source = g_exp + r_exp + blue_continuing
        
        deep_fft = fft.fft2(deep_source)
        # Apply Bloom to this combined deep light (Simulating scattering in the emulsion bulk)
        deep_bloomed_fft = deep_fft * ((1.0 - self.bloom_mix) + self.bloom_mix * otf_bloom)
        deep_bloomed = jnp.abs(fft.ifft2(deep_bloomed_fft))
        
        # --- PASS 3: Halation (Reflection) ---
        # Halation happens when light hits the base (bottom of the stack).
        # The light at the bottom is the "Deep Bloomed" light.
        # We verify: Does Green/Red bloom before hitting the base? Yes.
        # Ideally, we'd add Halation, then Bloom? No, light scatters -> hits base -> reflects -> scatters back.
        # Conventional approximation: Halation is a diffuse convolution of the light hitting the base.
        
        # We take the Deep Bloomed light as the source for Halation
        # Convolve with Halation Kernel (The Ring)
        halation_fft = deep_bloomed_fft * otf_hal * self.halation_gain
        halation_signal = jnp.abs(fft.ifft2(halation_fft))
        
        # 4. Re-Assemble Latent Image (RGB)
        # The "Latent Image" is what the developer sees.
        # Blue Channel = The Top Layer Bloom
        # Green Channel = Green Exposure + (Optional Halation Crosstalk)
        # Red Channel = Red Exposure + Halation (Strongest here)
        
        # Note: We need to output the *Exposure* that drives the density.
        # We also need to add the "Bloom" effect to Green/Red exposure maps themselves.
        # Wait, pure G/R exposure maps haven't been bloomed yet in isolation?
        # In the "Deep" pass we bloomed the sum. 
        # Let's approximate: 
        #   Blue_Final = b_bloomed
        #   Green_Final = Bloom(g_exp) + Small_Halation
        #   Red_Final   = Bloom(r_exp) + Strong_Halation
        
        # To save FFTs, we can just bloom G and R individually? 
        # Or just subtract Blue from the Deep Bloom? 
        # If Bloom is linear: Bloom(G+R+B_filt) = Bloom(G) + Bloom(R) + Bloom(B_filt).
        # So Bloom(G+R) = Deep_Bloom - Bloom(B_filt).
        # But we need Bloom(G) and Bloom(R) separate for the channels?
        # Actually, let's just run 2 more standard blooms or vectorise the FFT.
        
        # Vectorized Approach:
        # Stack [B, G, R] -> FFT -> Apply Bloom -> IFFT.
        stack_exp = jnp.stack([b_exp, g_exp, r_exp], axis=-1)
        stack_fft = fft.fft2(stack_exp, axes=(0, 1))
        
        # Apply Bloom to all channels (Structural Scattering is roughly isotropic)
        stack_bloomed_fft = stack_fft * ((1.0 - self.bloom_mix) + self.bloom_mix * otf_bloom[..., None])
        stack_bloomed = jnp.abs(fft.ifft2(stack_bloomed_fft, axes=(0, 1)))
        
        # Now add Halation
        # Halation source was determined by the "Deep Path" logic:
        # Source = Bloom(G) + Bloom(R) + Bloom(Filter * B)
        # This is roughly: stack_bloomed[..., 1] + stack_bloomed[..., 2] + blue_pass_factor * stack_bloomed[..., 0]
        
        hal_source_approx = stack_bloomed[..., 1] + stack_bloomed[..., 2] + (stack_bloomed[..., 0] * blue_pass_factor)
        hal_source_fft = fft.fft2(hal_source_approx)
        
        # Apply Halation Kernel
        hal_signal_fft = hal_source_fft * otf_hal * self.halation_gain
        hal_signal = jnp.abs(fft.ifft2(hal_signal_fft))
        
        # Composite Halation into layers
        # Primarily Red, some Green
        hal_vector = jnp.array([0.0, 0.05, 1.0]) # B, G, R weights
        
        final_latent = stack_bloomed + (hal_signal[..., None] * hal_vector)

        return final_latent