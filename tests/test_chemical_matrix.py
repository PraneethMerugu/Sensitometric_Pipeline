import jax
import jax.numpy as jnp
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.chemical import ChemicalDevelopment
from core.config import ChemicalConfig

class TestChemicalMatrix(unittest.TestCase):
    def test_shape_consistency(self):
        print("\n--- Test 1: Shape Consistency ---")
        H, W = 64, 64
        D_macro = jnp.ones((H, W, 3)) * 0.5
        
        D_macro = jnp.ones((H, W, 3)) * 0.5
        
        config = ChemicalConfig()
        chem = ChemicalDevelopment(config)
        D_micro = chem(D_macro)
        
        self.assertEqual(D_micro.shape, (H, W, 3))
        print(">> PASS: Shape preserved.")
        
    def test_adjacency_effect(self):
        print("\n--- Test 2: Adjacency Effect (Acutance) ---")
        # Create a step edge: 0.2 -> 0.8
        H, W = 64, 64
        D_macro = jnp.ones((H, W, 3)) * 0.2
        D_macro = D_macro.at[:, 32:, :].set(0.8)
        
        # Use strong Hard/Soft difference to exaggerate effect
        # Soft=Large Halo, Hard=Tiny Halo. 
        # Tanning should make the boundary sharp.
        # Soft=Large Halo, Hard=Tiny Halo. 
        # Tanning should make the boundary sharp.
        config = ChemicalConfig(sigma_soft=8.0, sigma_hard=1.0, coupling_matrix=jnp.eye(3)*2.0)
        chem = ChemicalDevelopment(config)
        
        D_micro = chem(D_macro)
        
        # Check Green Channel
        row = D_micro[32, :, 1]
        
        # Ideally: 
        # Low side near edge (x<32) should dip (Inhibition from neighbour high) -> Wait, high neighbour inhibits low side?
        # The equation for D_micro is: D_micro = D_macro - Inhibition.
        # Inhibition = Conv(D)
        # Near edge on LOW side (0.2): 
        #   It sees some High density from right side diffusing in.
        #   So Inhibition increases.
        #   So D_micro decreases (Dip).
        # Near edge on HIGH side (0.8):
        #   It sees some Low density from left side (not much inhibition).
        #   Relative to "flat high area" (which has self-inhibition), the edge has LESS inhibition coming from the left.
        #   So D_micro increases (Overshoot/Peak).
        
        flat_low = row[10]
        flat_high = row[55]
        edge_low = jnp.min(row[25:32])
        edge_high = jnp.max(row[32:40])
        
        print(f"Flat Low: {flat_low:.4f}, Edge Low: {edge_low:.4f}")
        print(f"Flat High: {flat_high:.4f}, Edge High: {edge_high:.4f}")
        
        # Verify Dip
        self.assertLess(edge_low, flat_low + 0.01, "Expected Dip (or at least no rise) on low side edge")
        
        # Verify Peak (High side edge > Flat High)
        # Note: If self-inhibition is strong in flat area, edge sees "less" inhibition from the dark neighbor.
        self.assertGreater(edge_high, flat_high, "Expected Peak (Overshoot) on high side edge")
        print(">> PASS: Edge effects (Mackie Lines) detected.")

    def test_inter_image_coupling(self):
        print("\n--- Test 3: Inter-Image Coupling ---")
        # Red inhibits Green
        # Matrix:
        # R -> R (self), R -> G (cross), ...
        # Chem Matrix usually defined as: Output = Matrix @ Input
        # But our implementation: einsum('ij, hwi -> hwj')
        # i=Source(Inhibitor), j=Target(Victim)
        # So K[0, 1] is Red inhibiting Green.
        
        K = jnp.zeros((3,3))
        K = K.at[0, 1].set(5.0) # Strong Red -> Green inhibition
        
        K = K.at[0, 1].set(5.0) # Strong Red -> Green inhibition
        
        config = ChemicalConfig(coupling_matrix=K)
        chem = ChemicalDevelopment(config)
        
        H, W = 32, 32
        
        # Case A: Just Green (No Red inhibitor)
        D_green_only = jnp.zeros((H,W,3))
        D_green_only = D_green_only.at[:,:,1].set(0.5)
        res_A = chem(D_green_only)
        
        # Case B: Green + Red (Red inhibits Green)
        D_mixed = jnp.zeros((H,W,3))
        D_mixed = D_mixed.at[:,:,1].set(0.5) # Same Green
        D_mixed = D_mixed.at[:,:,0].set(1.0) # Add Red
        res_B = chem(D_mixed)
        
        dens_A = res_A[16,16,1]
        dens_B = res_B[16,16,1]
        
        print(f"Green Density (Green Only): {dens_A:.4f}")
        print(f"Green Density (Green + Red): {dens_B:.4f}")
        
        # Old Expectation: B < A (Inhibition)
        # New Expectation (High-Pass Coupling):
        # In flat areas, D_micro == D_macro because high_pass_detail == 0.
        # So Global Colorimetry is PRESERVED.
        
        # 1. Verify Flat Field Preservation
        # NOTE: Step 6 (Exhaustion) applies a global tone curve: 
        # D_out = d_max * tanh(D_in / d_max)
        # We must account for this compression. The Matrix itself should contribute 0.
        
        # D_out = d_max * tanh(D_in / d_max)
        # We must account for this compression. The Matrix itself should contribute 0.
        
        d_max = chem.config.d_max
        expected_val = d_max * jnp.tanh(0.5 / d_max) # ~0.4954 for d_max=3.0
        
        diff_A = jnp.abs(dens_A - expected_val)
        diff_B = jnp.abs(dens_B - expected_val)
        
        self.assertLess(diff_A, 1e-3, f"Density changed beyond exhaustion curve! Got {dens_A:.4f}, expected {expected_val:.4f}")
        self.assertLess(diff_B, 1e-3, f"Density changed beyond exhaustion curve! Got {dens_B:.4f}, expected {expected_val:.4f}")
        print(f">> PASS: Global Colorimetry Preserved (Matches exhaustion curve: {expected_val:.4f}).")
        
        # 2. Verify Cross-Talk on Edges (Sharpening)
        # We need an EDGE to see the effect.
        # Create a Red Edge, measure effect on Green.
        
        # Red: Low (0.0) -> High (1.0) at x=16
        D_edge = jnp.zeros((H,W,3))
        D_edge = D_edge.at[:, :, 1].set(0.5) # Flat Green
        D_edge = D_edge.at[:, 16:, 0].set(1.0) # Red Step
        
        res_edge = chem(D_edge)
        
        # At the edge (x=16):
        # Red Density: 0.0 -> 1.0
        # High Pass (Red): Postive Spike (Edge)
        # Matrix: Green += 5.0 * Red_Detail
        # Expect Green to SPIKE at the edge
        
        g_flat = res_edge[5, 5, 1]  # 0.5
        g_edge = res_edge[16, 16, 1] # Should be > 0.5
        
        print(f"Green Flat: {g_flat:.4f}")
        print(f"Green at Red Edge: {g_edge:.4f}")
        
        self.assertGreater(g_edge, g_flat + 0.01, "Cross-talk sharpening failed!")

        print(">> PASS: Cross-channel inhibition verified.")

if __name__ == "__main__":
    unittest.main()
