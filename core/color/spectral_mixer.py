
import jax
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
from core.spectral.grid import get_wavelengths

class SpectralDyeMixer(eqx.Module):
    # Shape: (3, 64) -> [Cyan, Magenta, Yellow] spectral profiles
    dye_profiles: jax.Array 
    # Shape: (64,) -> Base density (Orange Mask)
    base_profile: jax.Array 

    def __call__(self, dye_concentrations: jax.Array) -> jax.Array:
        """
        Converts Analytical Density (CMY) to Spectral Transmittance.
        Args:
            dye_concentrations: (H, W, 3)
        Returns:
            (H, W, 64) The physical film strip
        """
        # 1. Sum densities (Broadband Beer's Law)
        # (H,W,3) @ (3,64) -> (H,W,64)
        spectral_density = jnp.dot(dye_concentrations, self.dye_profiles)
        
        # 2. Add Base Density
        total_density = spectral_density + self.base_profile
        
        # 3. Convert to Transmittance (T = 10^-D)
        return jnp.power(10.0, -total_density)

    @classmethod
    def from_csvs(cls, c_path, m_path, y_path, base_path):
        wl_grid = get_wavelengths()
        
        def load_interp(path):
            df = pd.read_csv(path)
            # Assuming columns are [wavelength, density]
            # Handle potential header issues by using iloc
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            return jnp.interp(wl_grid, x, y)

        c_curve = load_interp(c_path)
        m_curve = load_interp(m_path)
        y_curve = load_interp(y_path)
        base_curve = load_interp(base_path)

        # Stack rows: (3, 64)
        dye_stack = jnp.stack([c_curve, m_curve, y_curve], axis=0)
        
        return cls(dye_stack, base_curve)
