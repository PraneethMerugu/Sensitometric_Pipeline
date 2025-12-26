
import jax
import jax.numpy as jnp
import unittest
from core.chemical import ChemicalDevelopment

class TestChemicalMatrix(unittest.TestCase):
    def test_shape_consistency(self):
        print("\n--- Test 1: Shape Consistency ---")
        H, W = 64, 64
        D_macro = jnp.ones((H, W, 3)) * 0.5
        
        chem = ChemicalDevelopment()
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
        chem = ChemicalDevelopment(sigma_soft=8.0, sigma_hard=1.0, coupling_matrix=jnp.eye(3)*2.0)
        
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
        
        chem = ChemicalDevelopment(coupling_matrix=K)
        
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
        
        # Expect B to be lower (inhibited)
        self.assertLess(dens_B, dens_A)
        print(">> PASS: Cross-channel inhibition verified.")

if __name__ == "__main__":
    unittest.main()
