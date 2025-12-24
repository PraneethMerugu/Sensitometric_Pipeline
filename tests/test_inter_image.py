import jax
import jax.numpy as jnp
from core.chemical import ChemicalDiffusion
from core.sensitometry import SensitometricCurve

def test_inter_image_coupling():
    print("--- Test: Inter-Image Coupling (DIR Simulation) ---")
    
    # 1. Setup
    # Create a tone curve (Standard Sigmoid)
    # Params: [Dmin, Dmax, k, h0, nu]
    params = jnp.array([
        [-0.4, 1.0, 0.5, -1.5, 0.5], # R
        [-0.4, 1.0, 0.5, -1.5, 0.5], # G
        [-0.4, 1.0, 0.5, -1.5, 0.5]  # B
    ])
    curve = SensitometricCurve(params)
    
    # Define Coupling Matrix: Green inhibits Red strongly
    # Row 0 (Red): Self=2.0, Green=5.0 (Strong Inhibition), Blue=0
    # Row 1 (Green): Self=2.0
    # Row 2 (Blue): Self=2.0
    coupling_matrix = jnp.array([
        [2.0, 5.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
    ])
    
    chem = ChemicalDiffusion(
        tone_curve=curve,
        diff_coeff=1.0,
        coupling_matrix=coupling_matrix
    )
    
    # 2. Scenarios
    H, W = 32, 32
    
    # Scene A: Flat Red (Low Exposure), No Green
    # Red = 0.5 (Mid-grey latent), Green = 0.0, Blue = 0.0
    img_A = jnp.zeros((H, W, 3))
    img_A = img_A.at[:, :, 0].set(0.5)
    
    # Scene B: Flat Red + Bright Green Spot
    # Red = 0.5, Green = Spot, Blue = 0.0
    img_B = jnp.array(img_A)
    # Add Green spot in center
    img_B = img_B.at[10:22, 10:22, 1].set(1.0)
    
    print("Running Simulation for Scene A (Ref)...")
    res_A = chem.simulate(img_A, t_end=5.0)
    
    print("Running Simulation for Scene B (Interference)...")
    res_B = chem.simulate(img_B, t_end=5.0)
    
    # 3. Measurements
    # Measure Red density in the center
    center_y, center_x = 16, 16
    
    dens_A_red = res_A[center_y, center_x, 0]
    dens_B_red = res_B[center_y, center_x, 0]
    
    print(f"\nResults (Center Pixel Red Density):")
    print(f"Scenario A (Red Only):      {dens_A_red:.4f}")
    print(f"Scenario B (Red + Green):   {dens_B_red:.4f}")
    
    # 4. Assertions
    diff = dens_A_red - dens_B_red
    print(f"Inhibition Strength (Delta): {diff:.4f}")
    
    if dens_B_red < dens_A_red:
        print(">> SUCCESS: Red Density was suppressed by Green activity.")
        print("   (This confirms the Coupling Matrix is working as a Cross-Inhibitor)")
    else:
        print(">> FAILURE: Red Density was NOT suppressed.")
        
    assert dens_B_red < dens_A_red, "Cross-channel inhibition failed"

if __name__ == "__main__":
    test_inter_image_coupling()
