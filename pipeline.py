"""
Film Emulation Pipeline (Main Entry Point)

This module integrates the Optical, Chemical, and Colorimetric simulations into a 
unified execution graph.

Flow:
1. Actinic Exposure (Input Linear RGB)
2. Optical Physics (MTF, Halation, Bloom) -> Latent Image
3. Chemical Diffusion (Development Tank) -> Developed Analytical Density (CMY)
4. Colorimetric Transform (Dyes + Mask) -> Integral Density (Visual Negative)
5. GrainNet (Texture Synthesis) -> Final Film Scan
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any

from core.optical import OpticalPhysics
from core.chemical import ChemicalDiffusion
from core.sensitometry import SensitometricCurve
from core.color import ColorimetricTransform
from core.grainnet import GrainNet

class FilmPipeline(eqx.Module):
    optical: OpticalPhysics
    chemistry: ChemicalDiffusion
    color_transform: ColorimetricTransform
    grain_net: GrainNet
    
    def __init__(
        self,
        params: Dict[str, Any],
        grain_model_params: Dict,
        color_transform: ColorimetricTransform
    ):
        # 1. Optical
        self.optical = OpticalPhysics(
            scatter_gamma=params.get('scatter_gamma', 0.65),
            bloom_sigma=params.get('bloom_sigma', 2.0),
            bloom_weight=params.get('bloom_weight', 0.15),
            halation_radius=params.get('halation_radius', 30.0),
            halation_sigma=params.get('halation_sigma', 8.0)
        )
        
        # 2. Physics / Sensitometry
        # We assume curve_params is passed in params
        curve = SensitometricCurve(params=params['curve_params'])
        
        # Resolve Matrix
        c_mat = params.get('coupling_matrix', None)
        if c_mat is None:
            # Fallback to legacy gamma
            g = params.get('gamma', 2.0)
            c_mat = jnp.eye(3) * g
            
        self.chemistry = ChemicalDiffusion(
            tone_curve=curve,
            diff_coeff=params.get('diff_coeff', 1.0),
            k_ads=params.get('k_ads', 0.5),
            k_des=params.get('k_des', 0.1),
            coupling_matrix=c_mat
        )
        
        # 3. Color
        self.color_transform = color_transform
        
        # 4. Grain
        self.grain_net = GrainNet(params=grain_model_params)

    def __call__(self, actinic_exposure: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        """
        Run the full pipeline.
        
        Args:
            actinic_exposure: (H, W, 3) Linear RGB input (0-1 typically).
            key: PRNGKey for GrainNet.
            
        Returns:
            (H, W, 3) Final Image (with grain).
        """
        # 1. OPTICAL TRANSPORT (Linear)
        # Simulates light spreading in the emulsion *before* development
        # Output: Latent Image Distribution (E)
        latent_image = self.optical(actinic_exposure)
        
        # 2. CHEMICAL DEVELOPMENT (Non-Linear)
        # Simulates the development tank. 
        # Applies Tone Mapping AND Edge Effects (Friedman-Ross).
        # Output: Analytical Dye Densities (C, M, Y) 
        # Note: The solver computes Developed Silver, which we map 1:1 to Dye amount here.
        # (params shape (3,5) implies 3 channels handled independently)
        developed_density = self.chemistry.simulate(latent_image, t_end=5.0)
        
        # 3. COLORIMETRY
        # Convert dye densities (CMY) to Visual Integral Densities (RGB)
        # "The Virtual Inverse"
        # Input: Developed Density (CMY) -> Output: Integral Density (RGB Negative)
        # We need to ensure the order is correct. Our chemistry solves for R, G, B 'layers'.
        # R layer -> Cyan Dye
        # G layer -> Magenta Dye
        # B layer -> Yellow Dye
        # So developed_density (R,G,B order) IS (C,M,Y order) for the color module.
        virtual_negative = self.color_transform(developed_density)
        
        # Positive Print (Optional but usually desired for viewing)
        # Since GrainNet expects image-like stats, do we feed it the Density or the Transmittance?
        # Standard: Grain is applied to DENSITY.
        
        # 4. GRAIN SYNTHESIS
        # Apply grain to the uniform density image.
        # GrainNet (in this codebase) usually expects [0,1] image-like range?
        # Let's check GrainNet signature: it takes `image` and `grain_radius`.
        # Taking "Virtual Negative" as the base.
        # NOTE: GrainNet's `grain_radius` is a scalar strength.
        final_with_grain = self.grain_net(virtual_negative, grain_radius=0.5, key=key)
        
        # 5. INVERSION (To Positive)
        # D -> T = 10^-D
        # This gives us the linear view of the negative.
        # Usually we want a positive print.
        # For this pipeline stage, let's return the "Scan" of the negative.
        # i.e. Transmittance. 
        final_scan = jnp.power(10.0, -final_with_grain)
        
        return final_scan

# ==============================================================================
# PIPELINE VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("--- Pipeline Integration Test ---")
    
    # 1. Dummy Data / Params
    h, w = 64, 64
    dummy_input = jnp.zeros((h, w, 3))
    dummy_input = dummy_input.at[h//4:3*h//4, w//4:3*w//4, :].set(0.5)
    
    key = jax.random.PRNGKey(42)
    
    # Sensitometry Params
    curve_p = jnp.array([[-0.2, 2.0, 0.6, -1.0, 1.0]] * 3) # Simple Sigmoid
    
    pipeline_params = {
        'curve_params': curve_p,
        'scatter_gamma': 0.5,
        'gamma': 3.0 # High chemical adjacency
    }
    
    # Mock GrainNet Params (Empty)
    # GrainNetFlax needs valid params structure even if dummy
    # We'll just mock the class call if needed, but better to instantiate if compatible.
    # Since we can't easily load a pkl here, we might fail unless we mock GrainNet.
    # Let's trust the imports.
    
    try:
        # Mock Color Transform
        # Identity matrix for testing
        color_params = jnp.eye(3)
        base_dens = jnp.zeros((3,))
        color_sys = ColorimetricTransform(color_params, base_dens, jnp.array([650., 550., 450.]))
        
        # Mock GrainNet for this test 
        # (avoiding loading large pkl or random init of flax model which might be complex)
        # We'll update the class to allow empty init or similar? 
        # Actually, let's just initialize GrainNet with dummy params.
        from core.grainnet.core import GrainNetFlax
        flax_model = GrainNetFlax()
        dummy_init_key = jax.random.PRNGKey(0)
        dummy_params = flax_model.init(dummy_init_key, jnp.zeros((1, 32, 32, 1)), jnp.array([[0.5]]), dummy_init_key)['params']
        
        pipeline = FilmPipeline(
            params=pipeline_params,
            grain_model_params=dummy_params,
            color_transform=color_sys
        )
        
        print("Running Pipeline...")
        out_img = pipeline(dummy_input, key)
        
        print(f"Output Shape: {out_img.shape}")
        print(f"Output Mean:  {jnp.mean(out_img):.4f}")
        
        if not jnp.isnan(out_img).any():
             print(">> SUCCESS: Pipeline ran end-to-end without NaNs.")
    
    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
