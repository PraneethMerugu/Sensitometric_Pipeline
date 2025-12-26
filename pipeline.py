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
from typing import Dict, Any, Union, Optional
from pathlib import Path

from core.optical import OpticalPhysics
from core.chemical import ChemicalDevelopment
from core.sensitometry import SensitometricCurve
from core.color.spectral_mixer import SpectralDyeMixer
from core.color.scanner import VirtualScanner
from core.grainnet import GrainNet, load_grainnet
from core.upsampler.spectral_upsampler import create_upsampler, SpectralUpsampler
from core.config import FilmConfig, load_config, SensitometryConfig, ChemicalConfig, GrainConfig

class FilmPipeline(eqx.Module):
    optical: OpticalPhysics
    upsampler: SpectralUpsampler
    chemistry: ChemicalDevelopment
    tone_curve: SensitometricCurve
    dye_mixer: SpectralDyeMixer
    scanner: VirtualScanner
    grain_net: GrainNet
    
    # --- Parameters ---
    config: FilmConfig

    def __init__(
        self,
        config: FilmConfig
    ):
        self.config = config
        
        # 1. Shared Resources
        self.upsampler = create_upsampler(lut_size=32, data_dir=config.paths.lut_dir)

        # 2. Optical
        # Pass the optical config directly
        self.optical = OpticalPhysics(
            config=config.optical,
            upsampler=self.upsampler
        )
        
        # 2. Physics / Sensitometry
        # Ideally config.sensitometry.curve_params should be populated by the builder if not present.
        if config.sensitometry.curve_params is not None:
             self.tone_curve = SensitometricCurve(params=jnp.array(config.sensitometry.curve_params))
        else:
             if config.sensitometry.red_curve_path:
                 print("Fitting Sensitometric Curves from Config Paths...")
                 self.tone_curve = SensitometricCurve.fit_from_csvs(
                     config.sensitometry.red_curve_path,
                     config.sensitometry.green_curve_path,
                     config.sensitometry.blue_curve_path
                 )
             else:
                 raise ValueError("FilmConfig must have either `curve_params` or valid curve paths.")
        
        # 3. Chemical
        # Pass the chemical config directly
        self.chemistry = ChemicalDevelopment(
            config=config.chemical
        )
        
        # 4. Color (Spectral)
        self.dye_mixer = SpectralDyeMixer.from_csvs(
            config.paths.cyan_density,
            config.paths.magenta_density,
            config.paths.yellow_density,
            config.paths.min_density
        )
        
        self.scanner = VirtualScanner.from_status_m(
            config.paths.status_m_blue,
            config.paths.status_m_green,
            config.paths.status_m_red
        )
        
        # 4. Grain
        if config.grain.enabled:
            # Try to load if path exists
            p = Path(config.grain.model_path)
            if p.exists():
                self.grain_net = load_grainnet(p)
            else:
                 # Fallback for now or raise warning
                 print(f"Warning: Grain model not found at {p}. Using uninitialized dummy.")
                 from core.grainnet.core import GrainNetFlax
                 flax_model = GrainNetFlax()
                 dummy_init_key = jax.random.PRNGKey(0)
                 dummy_params = flax_model.init(dummy_init_key, jnp.zeros((1, 32, 32, 1)), jnp.array([[0.5]]), dummy_init_key)['params']
                 self.grain_net = GrainNet(params=dummy_params)
        else:
             # Create a pass-through or dummy? 
             # For now, just load a dummy to keep type safety, but we might want a flag to skip in __call__
             # Or we can just set grain radius to 0 effectively or handling it in __call__
             pass
             # We will assume it exists for now to satisfy Equinox structure, 
             # but maybe we should make it Optional in the class definition.
             # For now, let's just make it a dummy if disabled.
             from core.grainnet.core import GrainNetFlax
             flax_model = GrainNetFlax()
             dummy_params = flax_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 32, 32, 1)), jnp.array([[0.5]]), jax.random.PRNGKey(0))['params']
             self.grain_net = GrainNet(params=dummy_params)

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
        latent_image = self.optical(actinic_exposure)
        
        # 2. CHEMICAL DEVELOPMENT (Non-Linear)
        d_macro = self.tone_curve(latent_image)
        developed_density = self.chemistry(d_macro)
        
        # 3. SPECTRAL SYNTHESIS
        film_strip_spectral = self.dye_mixer(developed_density)
        
        # 4. VIRTUAL SCANNING
        scan_rgb = self.scanner(film_strip_spectral)
        
        # 5. GRAIN SYNTHESIS
        final_with_grain = self.grain_net(scan_rgb, grain_radius=0.5, key=key)
        
        return final_with_grain

def build_pipeline(config_path: str) -> FilmPipeline:
    """
    Factory function to build a pipeline from a JSON config file.
    """
    config = load_config(config_path)
    return FilmPipeline(config)

def run_one_off(config_path: str, image: jnp.ndarray) -> jnp.ndarray:
    """
    Run a single image through the pipeline defined by the config.
    """
    pipeline = build_pipeline(config_path)
    key = jax.random.PRNGKey(0) # Deterministic for one-off
    return pipeline(image, key)

# ==============================================================================
# PIPELINE VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("--- Pipeline Integration Test ---")
    
    # 1. Dummy Data
    h, w = 64, 64
    dummy_input = jnp.zeros((h, w, 3))
    dummy_input = dummy_input.at[h//4:3*h//4, w//4:3*w//4, :].set(0.5)
    
    key = jax.random.PRNGKey(42)
    
    # 2. Create Config Programmatically
    # Equinox modules are immutable, so we must instantiate with desired values.
    
    # Sensitometry with custom params
    sensitometry = SensitometryConfig(
        curve_params=[[-0.2, 2.0, 0.6, -1.0, 1.0]] * 3
    )
    
    # Chemical with custom params
    chemical = ChemicalConfig(
        gamma=3.0,
        d_max=3.0,
        drag_ratio=1.0, 
        exhaustion_alpha=2.0
    )
    
    # Grain (disabled)
    grain = GrainConfig(enabled=False)
    
    config = FilmConfig(
        name="Test Configuration",
        sensitometry=sensitometry,
        chemical=chemical,
        grain=grain
    )

    # Mock GrainNet Params if needed (logic handled in __init__ fallback now if enabled, but we disabled it)
    
    try:
        print("Initializing Pipeline with Config...")
        pipeline = FilmPipeline(config=config)
        
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
