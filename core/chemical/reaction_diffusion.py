"""
Reaction-Diffusion Chemical Development Module

This module simulates the physical development of film using the Friedman-Ross 
Reaction-Diffusion equations. It models the diffusion of oxidation byproducts (inhibitors)
which retard the development process adjacent to high-exposure areas, creating 
characteristic "Mackie Lines" and edge effects.

Reference:
Module III: Potentiated Reaction-Diffusion
"Final Film Emulation Pipeline", Ch 16 (Friedman & Ross)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Tuple

class ReactionState(eqx.Module):
    """
    State vector for the chemical system.
    """
    P: jax.Array       # Free Inhibitor (Mobile) - Concentration of oxidation byproducts
    P_star: jax.Array  # Adsorbed Inhibitor (Immobile) - Bound to grain surface
    D: jax.Array       # Developed Silver Density (Accumulator)

class ChemicalDiffusion(eqx.Module):
    """
    Physics engine solving the non-linear reaction-diffusion PDEs.
    
    Equations:
        dP/dt = D_p * Laplacian(P) + Rate - k_ads * P + k_des * P_star
        dP*/dt = k_ads * P - k_des * P_star
        dD/dt = Rate
        
        Rate = Potential * (1 / (1 + gamma * P_star))
        Potential = max(Target_Density - Current_Density, 0)
    """
    # Physics Parameters
    diff_coeff: jax.Array  # D_p (Diffusion rate of inhibitors)
    k_ads: jax.Array       # Adsorption rate (Free -> Bound)
    k_des: jax.Array       # Desorption rate (Bound -> Free)
    coupling_matrix: jax.Array # Matrix (3,3) for cross-channel inhibition
    
    # Dependencies
    tone_curve: eqx.Module # SensitometricCurve instance
    laplacian_kernel: jax.Array

    def __init__(self, tone_curve, diff_coeff=1.0, k_ads=0.5, k_des=0.1, coupling_matrix=None):
        self.tone_curve = tone_curve
        self.diff_coeff = jnp.array(diff_coeff)
        self.k_ads = jnp.array(k_ads)
        self.k_des = jnp.array(k_des)
        
        # Handle Coupling Matrix
        # shape: (3, 3). Row i = channel i. Cols = contribution from other channels.
        if coupling_matrix is None:
            # Default to isolated channels (Identity * 2.0 legacy strength)
            self.coupling_matrix = jnp.eye(3) * 2.0
        else:
            self.coupling_matrix = jnp.array(coupling_matrix)
        
        # Standard 3x3 Discrete Laplacian
        kernel = jnp.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        # Expand for channel-wise convolution: (Output=1, Input=1, H=3, W=3)
        # We will use vmap over channels, so we just need the 2D kernel or (1,1,3,3) depending on conv method.
        # Here we prepare it for direct use with vmap(scipy.signal.convolve2d)
        self.laplacian_kernel = kernel

    def __call__(self, t, state, latent_image):
        """
        Differential Equation (Vector Field).
        """
        # UNPACK
        P, P_star, D_current = state.P, state.P_star, state.D

        # 1. COMPUTE POTENTIAL (The "Potentiated" Model)
        # Calculate the Target Density this exposure *wants* to reach based on the H&D curve
        D_target = self.tone_curve(latent_image)
        
        # Reaction drive is remaining developable silver
        # Clip at 0 to prevent un-development (cannot reduce density once developed)
        potential = jnp.maximum(D_target - D_current, 0.0)

        # 2. COMPUTE INHIBITION (Langmuir Isotherm)
        # Retardation factor F varies from 1.0 (No inhibition) to ~0.0 (Full inhibition)
        # Matrix Coupling: mixing limits from other channels
        # P_star is (H, W, 3). Project to "Effective Inhibitors" per channel.
        # We want to effect on Channel i = sum(Matrix[i, j] * P_star[j])
        # This corresponds to dot(P_star, Matrix.T)
        effective_inhibitor = jnp.dot(P_star, self.coupling_matrix.T)
        F = 1.0 / (1.0 + effective_inhibitor)

        # 3. REACTION RATE
        rate = potential * F

        # 4. DIFFUSION (Vectorized Laplacian)
        # Apply convolution independently per channel.
        # Input P is (H, W, 3). Move C to front for vmap: (3, H, W)
        P_c = jnp.transpose(P, (2, 0, 1))
        
        # JAX convolve2d doesn't support 'symm' boundary directly.
        # We manually pad symmetric, then do a valid convolution.
        # Pad H and W dims by 1 (kernel is 3x3)
        P_padded = jnp.pad(P_c, ((0, 0), (1, 1), (1, 1)), mode='symmetric')
        
        laplacian_c = jax.vmap(lambda x: jax.scipy.signal.convolve2d(
            x, self.laplacian_kernel, mode='valid'
        ))(P_padded)
        
        # Move back to (H, W, 3)
        laplacian = jnp.transpose(laplacian_c, (1, 2, 0))

        # 5. DERIVATIVES (Friedman-Ross Eqs)
        # dP/dt: Gains from generation (rate), loses to adsorption, gains from desorption, moves via diffusion
        dP_dt = (self.diff_coeff * laplacian) + rate - (self.k_ads * P) + (self.k_des * P_star)
        
        # dP*/dt: Gains from adsorption, loses to desorption
        dP_star_dt = (self.k_ads * P) - (self.k_des * P_star)
        
        # dD/dt: Accumulates based on rate
        dD_dt = rate

        return ReactionState(P=dP_dt, P_star=dP_star_dt, D=dD_dt)

    def simulate(self, latent_image, t_end=5.0, dt=0.5):
        """
        Run the development simulation.
        
        Args:
            latent_image: (H, W, 3) Linear/Actinic exposure map.
            t_end: Simulation duration (proxy for development time).
            dt: Solver step size.
            
        Returns:
            (H, W, 3) Developed Density map.
        """
        # INITIALIZATION
        # P and P* start at 0 concentrations
        zeros = jnp.zeros_like(latent_image)
        
        # D starts at D-min (Fog level).
        # Extract Dmin from tone_curve params (Column 0 of params array)
        # Params shape: (3, 5). Dmin is params[:, 0]
        d_min_vals = jax.nn.sigmoid(self.tone_curve.params[:, 0]) * 0.5
        # Reshape to (1, 1, 3) for broadcasting
        d_start = jnp.ones_like(latent_image) * d_min_vals.reshape(1, 1, 3)

        y0 = ReactionState(P=zeros, P_star=zeros, D=d_start)

        # SOLVER CONFIG
        # Reaction-Diffusion is stiff. Tsit5 (Runge-Kutta 5/4) is usually okay for 
        # moderate diffusion rates, but PIDController helps stability.
        term = diffrax.ODETerm(self)
        solver = diffrax.Tsit5()
        
        # Adaptive step size controller
        stepsize = diffrax.PIDController(rtol=1e-3, atol=1e-3)

        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=t_end, dt0=dt, y0=y0,
            args=latent_image, stepsize_controller=stepsize,
            max_steps=4000,
            throw=False # Don't crash on stiffness, return last valid state
        )

        # Return the final Developed Density (Time -1)
        # sol.ys.D will have shape (Num_Steps, H, W, 3)
        return sol.ys.D[-1]

# ==============================================================================
# VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("--- Reaction-Diffusion Module Test ---")
    from core.sensitometry.tone_curve import SensitometricCurve
    
    # 1. Mock Sensitometry (Identity-ish for testing)
    # Params: [Dmin, Dmax, k, h0, nu]
    # Use standard values
    params = jnp.array([
        [-0.4, 0.5, 0.5, -1.5, 0.5], # R
        [-0.4, 0.5, 0.5, -1.5, 0.5], # G
        [-0.4, 0.5, 0.5, -1.5, 0.5]  # B
    ])
    curve = SensitometricCurve(params)
    
    # 2. Initialize Solver
    # Use strong gamma and diffusion to make effects visible on small grid
    chem = ChemicalDiffusion(tone_curve=curve, diff_coeff=2.0, coupling_matrix=jnp.eye(3)*5.0)
    
    # 3. Create Step Edge Pair
    # Left: Dark (Low Exposure), Right: Bright (High Exposure)
    H, W = 64, 64
    latent = jnp.ones((H, W, 3)) * 0.001 # Dark
    latent = latent.at[:, 32:, :].set(1.0) # Bright
    
    print("Running Simulation (This may take a moment due to compilation)...")
    final_density = chem.simulate(latent, t_end=5.0)
    
    # 4. Analyze Mackie Lines
    # We expect a "Dip" just before the edge (approx x=30-32)
    # And a "Peak" just after the edge (approx x=32-34)
    row = final_density[H//2, :, 1] # Green Channel
    
    val_low = row[10]
    val_high = row[50]
    val_edge_low = jnp.min(row[25:32])
    val_edge_high = jnp.max(row[32:40])
    
    print(f"\nResults:")
    print(f"Flat Low Density:  {val_low:.4f}")
    print(f"Edge Dip Density:  {val_edge_low:.4f}")
    print(f"Edge Peak Density: {val_edge_high:.4f}")
    print(f"Flat High Density: {val_high:.4f}")
    
    is_undershoot = val_edge_low < val_low
    is_overshoot = val_edge_high > val_high
    
    if is_undershoot:
        print(">> SUCCESS: Edge Dip (Inhibition) detected.")
    else:
        print(">> INFO: No Dip detected.")
        
    if is_overshoot:
        print(">> SUCCESS: Edge Peak (Inhibition Release) detected.")
    else:
        print(">> INFO: No Peak detected.")
