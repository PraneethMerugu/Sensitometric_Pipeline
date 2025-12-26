"""
Spatially Variant Matrix Model for Chemical Development

This module replaces the iterative Reaction-Diffusion solver with a single-pass, 
convolution-based approximation that models "organic" film characteristics 
(adjacency effects, gelatin tanning) using a dynamic inhibitor field.

Reference:
"Spatially Variant Linear-Spatial (Matrix) Model"
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

class ChemicalDevelopment(eqx.Module):
    """
    Simulates chemical development using a Spatially Variant Matrix Model.
    
    Transforms "Macro Density" (Theoretical) -> "Micro Density" (Developed)
    by simulating the release and diffusion of inhibitor clouds (DIR couplers),
    modulated by local gelatin hardness (Tanning).
    
    Upgrades:
    - Sigmoidal Exhaustion: Non-linear limiter for inhibition.
    - Anisotropic Diffusion: Directional kernels (Bromide Drag).
    """
    sigma_soft: jax.Array
    sigma_hard: jax.Array
    drag_ratio: jax.Array # New: Vertical Stretch Factor (sigma_y / sigma_x)
    
    exhaustion_alpha: jax.Array # New: Sigmoid Gain
    exhaustion_beta: jax.Array  # New: Sigmoid Threshold
    
    coupling_matrix: jax.Array
    d_min: jax.Array
    d_max: jax.Array
    
    def __init__(
        self, 
        sigma_soft: float = 2.0, 
        sigma_hard: float = 0.5, 
        drag_ratio: float = 1.0, # 1.0 = Isotropic, >1.0 = Vertical Drag
        coupling_matrix: jax.Array = None,
        d_min: float = 0.0,
        d_max: float = 3.0,
        exhaustion_alpha: float = 2.0,
        exhaustion_beta: float = 0.5
    ):
        self.sigma_soft = jnp.array(sigma_soft)
        self.sigma_hard = jnp.array(sigma_hard)
        self.drag_ratio = jnp.array(drag_ratio)
        self.d_min = jnp.array(d_min)
        self.d_max = jnp.array(d_max)
        self.exhaustion_alpha = jnp.array(exhaustion_alpha)
        self.exhaustion_beta = jnp.array(exhaustion_beta)
        
        if coupling_matrix is None:
            # Default: Mild self-inhibition, no cross-talk
            self.coupling_matrix = jnp.eye(3) * 0.5
        else:
            self.coupling_matrix = jnp.array(coupling_matrix)

    def __call__(self, D_macro: jax.Array) -> jax.Array:
        return self.apply(D_macro)

    def apply(self, D_macro: jax.Array) -> jax.Array:
        # 1. Generate Basis Clouds (Convolution)
        
        # Helper for Anisotropic Gaussian Blur
        def gaussian_blur(img, sigma_base, drag_ratio, max_window_size):
             # sigma_x = sigma_base
             # sigma_y = sigma_base * drag_ratio
             sx = sigma_base
             sy = sigma_base * drag_ratio
             
             h, w = img.shape
             
             # Calculate Window Size (based on the larger sigma)
             limit = jnp.minimum(h, w)
             window_size = jnp.minimum(max_window_size, limit)
             window_size = window_size - (1 - window_size % 2) # Ensure odd
             radius = window_size // 2
             
             x = jnp.arange(-radius, radius + 1)
             
             # Kernel X
             kx = jnp.exp(-0.5 * (x / sx)**2)
             kx = kx / jnp.sum(kx)
             
             # Kernel Y
             ky = jnp.exp(-0.5 * (x / sy)**2)
             ky = ky / jnp.sum(ky)
             
             # Separable convolution
             # 1. Vertical (Col) using ky
             k_col = ky.reshape(-1, 1)
             vert = jax.scipy.signal.correlate(img, k_col, mode='same')
             
             # 2. Horizontal (Row) using kx
             k_row = kx.reshape(1, -1)
             return jax.scipy.signal.correlate(vert, k_row, mode='same')

        # Vectorize over channels
        blur = jax.vmap(gaussian_blur, in_axes=(2, None, None, None), out_axes=2)
        
        # --- Soft Cloud ---
        # Fixed window size for "Soft" effect? 
        w_soft = 61 # Increased window for drag
        cloud_soft = blur(D_macro, self.sigma_soft, self.drag_ratio, w_soft)
        
        # --- Hard Cloud ---
        w_hard = 25
        cloud_hard = blur(D_macro, self.sigma_hard, self.drag_ratio, w_hard)
        
        # 2. Compute Tanning Mask (Per Channel)
        tanning_mask = (D_macro - self.d_min) / (self.d_max - self.d_min + 1e-6)
        tanning_mask = jnp.clip(tanning_mask, 0.0, 1.0)
        
        # 3. Basis Interpolation
        inhibitor_field_linear = (cloud_hard * tanning_mask) + (cloud_soft * (1.0 - tanning_mask))
        
        # 4. Sigmoidal Exhaustion (Non-Linearity)
        # I_nonlinear = 1.0 / (1.0 + exp(-alpha * (I - beta)))
        # Applying exhaustion curve to the linear inhibitor field
        inhibitor_field = 1.0 / (1.0 + jnp.exp(-self.exhaustion_alpha * (inhibitor_field_linear - self.exhaustion_beta)))
        
        # 5. Apply Chemical Coupling Matrix
        inhibition_term = jnp.einsum('ij, hwi -> hwj', self.coupling_matrix, inhibitor_field)
        
        # 6. Final Subtraction
        D_micro = D_macro - inhibition_term
        
        return D_micro

