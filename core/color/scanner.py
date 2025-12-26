
import jax
import jax.numpy as jnp
import equinox as eqx
from core.spectral.grid import get_wavelengths

class VirtualScanner(eqx.Module):
    # Shape: (64,)
    illuminant_spd: jax.Array 
    # Shape: (3, 64) -> [R, G, B] or [X, Y, Z] sensitivities
    sensor_sensitivity: jax.Array
    
    def __call__(self, film_transmittance: jax.Array) -> jax.Array:
        """
        Scans the spectral film strip.
        Args:
            film_transmittance: (H, W, 64)
        Returns:
            (H, W, 3) RGB Image
        """
        # 1. Apply Illuminant
        # (H,W,64) * (64,) -> (H,W,64)
        light_reaching_sensor = film_transmittance * self.illuminant_spd
        
        # 2. Integrate against Sensor
        # (H,W,64) @ (64,3) -> (H,W,3)
        # Note the transpose on sensor_sensitivity (3,64) -> (64,3)
        rgb = jnp.dot(light_reaching_sensor, self.sensor_sensitivity.T)
        
        return rgb

    @classmethod
    def from_status_m(cls, blue_path, green_path, red_path):
        import pandas as pd
        wl_grid = get_wavelengths()
        
        def load_interp(path):
            df = pd.read_csv(path)
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            return jnp.interp(wl_grid, x, y)

        b_curve = load_interp(blue_path)
        g_curve = load_interp(green_path)
        r_curve = load_interp(red_path)
        
        # Stack: (3, 64) -> R, G, B order typically for output
        # Status M files are separate.
        # We want output RGB.
        sensor_stack = jnp.stack([r_curve, g_curve, b_curve], axis=0)
        
        # Uniform Illuminant for now (Equal Energy)
        illuminant = jnp.ones_like(wl_grid)
        
        return cls(illuminant, sensor_stack)
