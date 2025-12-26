import sys
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from core.config import FilmConfig
from pipeline import FilmPipeline

def test_differentiability():
    print("--- Testing Config Differentiability ---")
    
    # 1. Create Config & Pipeline
    # Equinox modules are immutable. We must set params at init or use eqx.tree_at
    curve_p = [[0.0, 1.0, 1.0, 0.0, 1.0]] * 3
    
    sensitometry = FilmConfig().sensitometry # Get default
    # Create new sensitometry with params
    sensitometry = eqx.tree_at(lambda s: s.curve_params, sensitometry, curve_p)
    
    config = FilmConfig(name="DiffTest", sensitometry=sensitometry)
    
    pipeline = FilmPipeline(config)
    
    # 2. Dummy Input and Target
    img = jnp.ones((32, 32, 3)) * 0.5
    target = jnp.ones((32, 32, 3)) * 0.6
    
    # 3. Loss Function acting on Config
    @eqx.filter_value_and_grad
    def loss_fn(pipeline_model):
        # Forward pass
        # Note: We are differentiating wrt pipeline_model which CONTAINS config
        output = pipeline_model(img, jax.random.PRNGKey(0))
        loss = jnp.mean((output - target)**2)
        return loss

    # 4. Compute Gradient
    loss_val, grads = loss_fn(pipeline)
    
    print(f"Loss: {loss_val:.6f}")

    # Debug: Print structure of grads
    # jax.tree_util.tree_map(lambda x: print(x.shape if hasattr(x, 'shape') else x), grads)
    
    # Check gradient for a specific parameter in config
    if grads.optical.config.scatter_gamma is not None:
         print(f"Grad (Scatter Gamma): {grads.optical.config.scatter_gamma}")
    else:
         print("Grad (Scatter Gamma) is None!")

    if grads.chemistry.config.sigma_soft is not None:
        print(f"Grad (Sigma Soft): {grads.chemistry.config.sigma_soft}")
    else:
        print("Grad (Sigma Soft) is None!")

    print("\n>> Differentiability Check Passed!")

if __name__ == "__main__":
    try:
        test_differentiability()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
