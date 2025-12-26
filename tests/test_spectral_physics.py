"""
tests/test_spectral_physics.py

Verification for the High-Fidelity Spectral Halation upgrade.
Tests the "Blue Light" condition: Blue light should be blocked by the yellow filter 
impeding it from reaching the base, thus causing ZERO halation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import sys
import os

# Add workspace root to path
sys.path.append(os.getcwd())

from core.optical.scattering import OpticalPhysics
from core.upsampler.spectral_upsampler import create_upsampler
from core.config import OpticalConfig

def test_blue_light_physics():
    print("--- Spectral Halation Verification (The Blue Light Test) ---")

    # 1. Initialize Resources
    # We need the real upsampler for this to work physically
    try:
        upsampler = create_upsampler(lut_size=32, data_dir="data/luts")
        print("Loaded Spectral Upsampler.")
    except Exception as e:
        print(f"FAILED to load upsampler: {e}")
        return False

    # 2. Configure Physics
    # High halation gain to make it obvious
    # 2. Configure Physics
    # High halation gain to make it obvious
    config = OpticalConfig(
        scatter_gamma=0.5,
        bloom_weight=0.0,     # Disable Bloom to isolate Halation
        halation_radius=10.0, # Small radius
        halation_sigma=2.0,
        halation_gain=10.0    # Exaggerate effect
    )
    
    optical = OpticalPhysics(
        config=config,
        upsampler=upsampler
    )

    # 3. Create Test Image (128x128)
    # Dot A: RED   (64, 32)
    # Dot B: BLUE  (64, 96)
    H, W = 128, 128
    img = jnp.zeros((H, W, 3))
    
    # Red Dot
    img = img.at[64, 32, :].set(jnp.array([10.0, 0.0, 0.0])) 
    # Blue Dot
    img = img.at[64, 96, :].set(jnp.array([0.0, 0.0, 10.0]))

    print("Created Actinic Image with Red (left) and Blue (right) sources.")

    # 4. Run Simulation
    # JIT compile for speed
    process = eqx.filter_jit(optical)
    result = process(img)
    
    # 5. Inspect Results
    # We look at the halo *around* the dots, not the dots themselves.
    # Offset by +10 pixels in X
    red_halo_sample = result[64, 32 + 10, :]   # Should be RED
    blue_halo_sample = result[64, 96 + 10, :]  # Should be BLACK (or very dark)

    print(f"\n--- Measurement (Offset +10px) ---")
    print(f"Red Source Halo Intensity:  {red_halo_sample}")
    print(f"Blue Source Halo Intensity: {blue_halo_sample}")

    # 6. Assertions
    # Red halo should exist and be primarily red
    red_has_halo = red_halo_sample[0] > 0.1
    
    # Blue halo should be effectively non-existent (due to Yellow Filter transmission < 1%)
    # It might not be exactly zero due to slight spectral overlap in the Up-sampling 
    # (Blue LED might have some energy > 540nm in the reconstruction? unlikely for pure 0,0,1)
    # But it should be orders of magnitude lower than Red.
    blue_has_no_halo = blue_halo_sample[0] < (red_halo_sample[0] * 0.05) 

    success = True
    if not red_has_halo:
        print(">> FAILURE: Red light generated no halation!")
        success = False
    
    if not blue_has_no_halo:
        print(">> FAILURE: Blue light leaked through the Yellow Filter!")
        print(f"   (Blue Halo Strength / Red Halo Strength = {blue_halo_sample[0]/red_halo_sample[0]:.4f})")
        success = False

    if success:
        print("\n>> SUCCESS: Physics Verified.")
        print("   - Red light crossed the stack and halated.")
        print("   - Blue light was absorbed by the Yellow Filter.")
    
    return success

if __name__ == "__main__":
    if test_blue_light_physics():
        exit(0)
    else:
        exit(1)
