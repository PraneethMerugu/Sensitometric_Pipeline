"""
Sensitometric Tone Mapping Module

This module maps Log-Actinic Exposure (h) to Optical Density (D) using the
Generalized Logistic Function (Richards' Curve).

It uses Optimistix (Levenberg-Marquardt) to solve the non-linear least squares 
problem, ensuring a robust fit to the Kodak datasheet.

Reference:
Module IV: Sensitometric Tone Mapping
"Final Film Emulation Pipeline", Section 5
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import pandas as pd
from pathlib import Path
from typing import Tuple, List

class SensitometricCurve(eqx.Module):
    """
    Parametric Tone Curve using the Generalized Logistic Function.
    
    Equation:
        D(h) = Dmin + (Dmax - Dmin) / (1 + exp(-k * (h - h0)))**nu
    
    Attributes:
        params: Array of shape (3, 5) storing [Dmin, Dmax, k, h0, nu] for R, G, B.
    """
    params: jax.Array

    def __init__(self, params: jax.Array):
        self.params = params

    @staticmethod
    def _generalized_logistic(h, p):
        """
        The characteristic curve function.
        p = [Dmin, Dmax, k, h0, nu]
        """
        dmin, dmax, k, h0, nu = p
        # Safe logistic function
        denom = (1.0 + jnp.exp(-k * (h - h0)))**nu
        return dmin + (dmax - dmin) / denom

    @classmethod
    def fit_from_csvs(
        cls,
        red_path: str,
        green_path: str,
        blue_path: str
    ) -> "SensitometricCurve":
        """
        Fits parameters using Optimistix Levenberg-Marquardt solver.
        """
        
        def load_channel(file_path: str) -> Tuple[jax.Array, jax.Array]:
            df = pd.read_csv(file_path, header=None)
            try:
                h = df.iloc[:, 0].astype(float).values
                d = df.iloc[:, 1].astype(float).values
            except ValueError:
                h = df.iloc[1:, 0].astype(float).values
                d = df.iloc[1:, 1].astype(float).values
            return jnp.array(h), jnp.array(d)

        # 1. Define the Residual Function for Least Squares
        # Optimistix expects a function that returns the vector of residuals (y_pred - y_true)
        
        def residual_fn(params_unconstrained, args):
            h_data, d_data = args
            
            # constrain parameters to valid physical ranges
            # Dmin: 0.0 - 0.5
            dmin = jax.nn.sigmoid(params_unconstrained[0]) * 0.5 
            # Dmax: > 1.5
            dmax = 1.5 + jax.nn.softplus(params_unconstrained[1])
            # k: > 0 (Slope)
            k    = jax.nn.softplus(params_unconstrained[2])
            # h0: Unbounded (Offset)
            h0   = params_unconstrained[3]
            # nu: > 0 (Asymmetry)
            nu   = jax.nn.softplus(params_unconstrained[4])
            
            p_physical = jnp.stack([dmin, dmax, k, h0, nu])
            
            d_pred = cls._generalized_logistic(h_data, p_physical)
            
            # Return residual vector
            return d_pred - d_data

        # 2. Solver Setup (Levenberg-Marquardt is ideal for curve fitting)
        solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)

        # Initial Guesses (Unconstrained)
        # Matches typical film: Dmin=0.2, Dmax=2.5, k=0.6, h0=-1.5, nu=1.0
        init_guess = jnp.array([-0.4, 0.5, 0.5, -1.5, 0.5])

        final_params_list = []
        paths = [red_path, green_path, blue_path]
        names = ["Red", "Green", "Blue"]

        for name, path in zip(names, paths):
            print(f"Fitting {name} Channel (Optimistix)...")
            h_data, d_data = load_channel(path)
            
            # Run Solver
            sol = optx.least_squares(
                residual_fn,
                solver,
                init_guess,
                args=(h_data, d_data),
                max_steps=1000,
                throw=False # Don't crash on convergence warning, just use best result
            )
            
            # Check convergence
            if sol.result != optx.RESULTS.successful:
                print(f"  > Warning: Optimization status: {sol.result}")

            # Transform back to physical space
            p_final_unc = sol.value
            dmin = jax.nn.sigmoid(p_final_unc[0]) * 0.5
            dmax = 1.5 + jax.nn.softplus(p_final_unc[1])
            k    = jax.nn.softplus(p_final_unc[2])
            h0   = p_final_unc[3]
            nu   = jax.nn.softplus(p_final_unc[4])
            
            print(f"  > Params: Dmin={dmin:.2f}, Dmax={dmax:.2f}, k={k:.2f}, h0={h0:.2f}, nu={nu:.2f}")
            
            final_params_list.append(jnp.stack([dmin, dmax, k, h0, nu]))

        all_params = jnp.stack(final_params_list)
        return cls(params=all_params)

    @eqx.filter_jit
    def __call__(self, actinic_image_linear: jax.Array) -> jax.Array:
        # 1. Linear -> Log Exposure
        h_log = jnp.log10(jnp.maximum(actinic_image_linear, 1e-6))
        
        # 2. Apply Curve (Vectorized over channels)
        # Move channel dim to front: (H,W,3) -> (3,H,W)
        h_log_c = jnp.transpose(h_log, (2, 0, 1))
        
        # Map over channels (3,)
        output_c = jax.vmap(self._generalized_logistic)(h_log_c, self.params)
        
        # Move channel dim back: (3,H,W) -> (H,W,3)
        return jnp.transpose(output_c, (1, 2, 0))

# ==============================================================================
# MAIN TEST SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("--- Sensitometric Curve Module Test (Optimistix) ---")
    
    base_dir = Path("data/Kodak_Vision3_250d")
    r_path = base_dir / "Red_d_log_e.csv"
    g_path = base_dir / "Green_d_log_e.csv"
    b_path = base_dir / "Blue_d_log_e.csv"
    
    try:
        curve_mod = SensitometricCurve.fit_from_csvs(
            str(r_path), str(g_path), str(b_path)
        )
        print("\n>> Fitting Complete.")
        
        # Validation: Check monotonic behavior
        test_vals = jnp.array([0.0001, 0.18, 100.0]).reshape(1, 1, 3) # Low, Mid, High
        test_vals = jnp.tile(test_vals, (1, 1, 1)) # (1,1,3)
        out = curve_mod(test_vals)
        
        print(f"\nShadow Density (0.0001): {out[0,0,0]:.3f}")
        print(f"Mid Density    (0.18):   {out[0,0,1]:.3f}")
        print(f"High Density   (100.0):  {out[0,0,2]:.3f}")
        
    except (FileNotFoundError, ImportError) as e:
        print(f"Error: {e}")