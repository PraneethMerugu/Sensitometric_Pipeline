"""
Spectral Upsampler LUT Generation Tool

This module generates and saves lookup tables (LUTs) for spectral upsampling.
The LUT maps RGB values to polynomial coefficients that reconstruct spectral data.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import time
from pathlib import Path

# ==============================================================================
# 1. PHYSICS CONSTANTS & ANALYTIC CMFs
# ==============================================================================

WAVELENGTHS = jnp.linspace(360.0, 830.0, 64)
W_NORM_MIN = 360.0
W_NORM_RANGE = 470.0

def wyman_2013_cmf(wavelengths):
    """
    Analytic approximation of CIE 1931 2-deg XYZ CMFs (Wyman et al. 2013).
    Provides infinite smoothness for the optimizer.
    """
    def lobe(x, amp, mu, sig1, sig2):
        sig = jnp.where(x < mu, sig1, sig2)
        return amp * jnp.exp(-0.5 * ((x - mu) / sig) ** 2)

    x  = lobe(wavelengths, 1.056, 599.8, 37.9, 31.0) + \
         lobe(wavelengths, 0.362, 442.0, 16.0, 26.7) + \
         lobe(wavelengths, -0.065, 501.1, 20.4, 26.2)

    y  = lobe(wavelengths, 0.821, 568.8, 46.9, 40.5) + \
         lobe(wavelengths, 0.286, 530.9, 16.3, 31.1)

    z  = lobe(wavelengths, 1.217, 437.0, 11.8, 36.0) + \
         lobe(wavelengths, 0.681, 459.0, 26.0, 13.8)

    return jnp.stack([x, y, z], axis=-1)

def standard_d65_interpolated(wavelengths):
    """Interpolated D65 Illuminant SPD."""
    base_wl = jnp.array([360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 830])
    base_p  = jnp.array([46.0, 82.8, 104.9, 115.9, 107.7, 100.0, 90.0, 83.0, 78.0, 62.0, 46.0, 66.0, 60.0])
    return jnp.interp(wavelengths, base_wl, base_p)

# ==============================================================================
# 2. WHITE POINT CALIBRATION
# ==============================================================================

# Standard Rec. 709 / sRGB Matrix (Linear D65)
XYZ_TO_RGB = jnp.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

def get_calibrated_cmf():
    """
    Computes an 'Effective CMF' that includes the D65 illuminant and
    is normalized such that a perfect 1.0 reflector yields exactly
    the D65 XYZ coordinates expected by the sRGB matrix.
    """
    raw_cmf = wyman_2013_cmf(WAVELENGTHS)
    raw_ill = standard_d65_interpolated(WAVELENGTHS)

    # Combined Physics: CMF * Illuminant
    effective_cmf_raw = raw_cmf * raw_ill[:, None]

    # 1. Measure current "White" (Sum of unscaled effective CMF)
    current_white_xyz = jnp.sum(effective_cmf_raw, axis=0)

    # 2. Target "White" (Standard D65 XYZ normalized to Y=1.0)
    # These are the specific constants the sRGB matrix is derived from.
    target_white_xyz = jnp.array([0.95047, 1.00000, 1.08883])

    # 3. Compute Correction Factors per channel
    correction = target_white_xyz / current_white_xyz

    # This CMF accounts for delta_lambda implicitly via the sum
    return effective_cmf_raw * correction

EFFECTIVE_CMF = get_calibrated_cmf()

# ==============================================================================
# 3. FORWARD MODEL & SOLVER
# ==============================================================================

def algebraic_sigmoid(x):
    """Jakob 2019 Sigmoid: Maps R -> (0, 1)"""
    return 0.5 * (x / jnp.sqrt(1.0 + x**2) + 1.0)

def coeffs_to_rgb(coeffs):
    """
    Converts coefficients [A, B, C] to RGB.
    """
    w_norm = 2.0 * (WAVELENGTHS - W_NORM_MIN) / W_NORM_RANGE - 1.0
    poly = coeffs[0] * w_norm**2 + coeffs[1] * w_norm + coeffs[2]
    reflectance = algebraic_sigmoid(poly)

    # Integration: XYZ = Sum(Reflectance * Effective_CMF)
    xyz = jnp.dot(reflectance, EFFECTIVE_CMF)
    rgb = jnp.dot(XYZ_TO_RGB, xyz)
    return rgb

def solve_pixel_robust(target_rgb):
    """
    Solves for coefficients using aggressive multi-start initialization
    to capture difficult Gamut Boundary colors (Pure Red/Green/Blue).
    """
    # EXPANDED GUESSES (The "Aggressive" Strategy)
    # We include extreme values (magnitude 20) to help find
    # the steep slopes required for saturated primaries.
    guesses = jnp.array([
        [0.0, 0.0, 0.0],       # 1. Grey (Safe)
        [0.0, 0.0, 0.8],       # 2. Bias Up (White-ish)
        [0.0, 20.0, -10.0],    # 3. Extreme Slope Up (Red)
        [0.0, -20.0, 10.0],    # 4. Extreme Slope Down (Blue)
        [-25.0, 0.0, 10.0],    # 5. Extreme Parabola Down (Green)
        [25.0, 0.0, -10.0],    # 6. Extreme Parabola Up (Magenta)
    ])

    def optimize_run(init_coeffs):
        def loss_fn(c):
            pred = coeffs_to_rgb(c)
            # Standard MSE
            err = jnp.sum((pred - target_rgb)**2)

            # --- CRITICAL CHANGE: TINY REGULARIZATION ---
            # We strictly minimize error. Smoothness is a distant secondary goal.
            # If we penalize large coeffs too much, we kill the saturated colors.
            reg = jnp.sum(c**2 * jnp.array([1e-7, 1e-8, 1e-9]))
            return err + reg

        def step(i, c):
            val, grad = jax.value_and_grad(loss_fn)(c)
            H = jax.hessian(loss_fn)(c)

            # Levenberg-Marquardt Damping
            H_inv = jnp.linalg.inv(H + jnp.eye(3) * 1e-3)
            update = jnp.dot(H_inv, grad)

            # --- CRITICAL CHANGE: RELAXED CLIPPING ---
            # Allow large jumps. Saturated colors dwell far from the origin.
            update = jnp.clip(update, -10.0, 10.0)

            return c - update

        # 35 iterations to ensure we climb the long shallow gradient of the sigmoid
        final_c = jax.lax.fori_loop(0, 35, step, init_coeffs)
        return final_c, loss_fn(final_c)

    # Run all guesses in parallel
    candidates_c, candidates_l = vmap(optimize_run)(guesses)

    # Return winner
    return candidates_c[jnp.argmin(candidates_l)]

# ==============================================================================
# 4. OFFLINE LUT GENERATOR
# ==============================================================================

# static_argnames=['size'] ensures JAX recompiles if grid size changes
@jit(static_argnames=['size'])
def generate_lut(size=32):
    """
    Generates a (size, size, size, 3) Lookup Table.
    """
    # Create normalized RGB grid
    ticks = jnp.linspace(0, 1, size)
    # indexing='ij' gives (R, G, B) order
    grid = jnp.stack(jnp.meshgrid(ticks, ticks, ticks, indexing='ij'), axis=-1)

    # Flatten to list of pixels (N^3, 3)
    flat_targets = grid.reshape(-1, 3)

    # Vectorized solve over the entire batch
    flat_coeffs = vmap(solve_pixel_robust)(flat_targets)

    # Reshape back to volume
    return flat_coeffs.reshape(size, size, size, 3)

def save_lut(lut, size, output_dir="data/luts"):
    """
    Save LUT to disk with metadata in NPZ format.

    Args:
        lut: JAX array of shape (size, size, size, 3)
        size: LUT grid size
        output_dir: Directory to save LUT
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"spectral_lut_{size}.npz"

    # Save using NPZ format
    jnp.savez(
        filename,
        lut=lut,
        wavelengths=WAVELENGTHS,
        effective_cmf=EFFECTIVE_CMF,
        xyz_to_rgb=XYZ_TO_RGB,
        # Scalars need to be wrapped in arrays for NPZ
        w_norm_min=jnp.array(W_NORM_MIN),
        w_norm_range=jnp.array(W_NORM_RANGE),
        size=jnp.array(size)
    )

    print(f"LUT saved to {filename}")
    return filename

# ==============================================================================
# 5. MAIN SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("--- Spectral Upsampler LUT Generator ---")

    # Generate LUT
    LUT_SIZE = 32
    print(f"Generating {LUT_SIZE}x{LUT_SIZE}x{LUT_SIZE} LUT...")
    start_t = time.time()
    LUT = generate_lut(size=LUT_SIZE)
    LUT.block_until_ready()
    elapsed = time.time() - start_t
    print(f"LUT generated in {elapsed:.2f}s")

    # Save LUT
    output_file = save_lut(LUT, LUT_SIZE)

    # Verify LUT with a test
    print("\nVerifying LUT quality...")
    test_colors = jnp.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 1.0],  # White
        [0.5, 0.5, 0.5],  # Grey
    ])

    # Get coefficients from solve
    coeffs_direct = vmap(solve_pixel_robust)(test_colors)
    rgb_reconstructed = vmap(coeffs_to_rgb)(coeffs_direct)

    max_err = jnp.max(jnp.abs(test_colors - rgb_reconstructed))
    mean_err = jnp.mean(jnp.abs(test_colors - rgb_reconstructed))

    print(f"Max Reconstruction Error:  {max_err:.5f}")
    print(f"Mean Reconstruction Error: {mean_err:.5f}")

    if max_err < 0.02:
        print(">> SUCCESS: LUT is Calibrated and Robust.")
    else:
        print(">> WARNING: Check D65 Curves.")
