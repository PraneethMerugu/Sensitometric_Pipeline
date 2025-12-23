Based on the analysis of your "Computational Microsensitometry" pipeline and the mathematical models from Friedman & Ross, here is the definitive implementation guide.

```markdown
# Engineering Guide: Potentiated Reaction-Diffusion Pipeline

**Target System:** `Film` Emulation Pipeline
**Frameworks:** `jax`, `equinox`, `diffrax`
**Objective:** Transition from linear filter-based adjacency to a physics-based **Virtual Development Tank** using non-linear PDEs.

---

## 1. Architectural Paradigm Shift

We are moving the Sensitometric Curve from a post-process filter to a **reaction potential driver**.

* **Old Flow:** `Exposure` -> `Optical Blur` -> `Reaction Filters` -> `Tone Curve` -> `Color`
* **New Flow:** `Exposure` -> `Optical Blur` -> **`Reaction-Diffusion Solver (Internal Curve)`** -> `Color`

The development rate of the silver is now physically driven by the *potential density* (from the H&D curve) and physically throttled by *inhibitor diffusion* (Friedman equations).

---

## 2. Implementation Steps

### Phase 1: Refactor Optical Physics
**File:** `Film/core/optical/scattering.py`

The current `OpticalChemicalPhysics` class mixes linear optical theory (MTF) with linear chemical theory. We must strip the chemical part to isolate the **Latent Image**.

1.  **Rename Class:** `OpticalChemicalPhysics` -> `OpticalPhysics`.
2.  **Clean Attributes:** Remove `chemical_sigma` and `chemical_strength` from `__init__`.
3.  **Clean Computation:**
    * In `__call__`, remove the "2. CHEMICAL TRANSFER FUNCTION" block.
    * Remove `total_chem_otf`.
    * **Output:** The method should return the image after `total_optical_otf` is applied. This is now the **Latent Image Map ($E$)**.

### Phase 2: The Reaction-Diffusion Solver (New Core)
**File:** Create `Film/core/chemical/reaction_diffusion.py`

This is the physics engine. It solves the Friedman-Ross equations (Ch 16) over time.

#### 2.1 State Definition
Define the chemical species involved in the reaction.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

class ReactionState(eqx.Module):
    P: jax.Array       # Free Inhibitor (Mobile)
    P_star: jax.Array  # Adsorbed Inhibitor (Immobile)
    D: jax.Array       # Developed Silver/Dye Density (Accumulator)

```

#### 2.2 The Physics Field

Implement the differential equations. Crucially, inject the `SensitometricCurve` module here to calculate reaction potential.

```python
class ChemicalDiffusion(eqx.Module):
    # Physics Parameters
    diff_coeff: jax.Array  # D_p (Diffusion rate)
    k_ads: jax.Array       # Adsorption rate
    k_des: jax.Array       # Desorption rate
    gamma: jax.Array       # Inhibition strength (Mackie line intensity)
    
    # Dependencies
    tone_curve: eqx.Module # SensitometricCurve instance
    laplacian_kernel: jax.Array

    def __init__(self, tone_curve, diff_coeff=1.0, k_ads=0.5, k_des=0.1, gamma=2.0):
        self.tone_curve = tone_curve
        self.diff_coeff = jnp.array(diff_coeff)
        self.k_ads = jnp.array(k_ads)
        self.k_des = jnp.array(k_des)
        self.gamma = jnp.array(gamma)
        
        # Standard 3x3 Discrete Laplacian
        kernel = jnp.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        # Expand for channel-wise convolution: (1, 1, 3, 3)
        self.laplacian_kernel = kernel[None, None, :, :]

    def __call__(self, t, state, latent_image):
        # UNPACK
        P, P_star, D_current = state.P, state.P_star, state.D

        # 1. COMPUTE POTENTIAL (The "Potentiated" Model)
        # Calculate the Target Density this exposure *wants* to reach
        D_target = self.tone_curve(latent_image)
        
        # Reaction drive is remaining developable silver
        # Clip at 0 to prevent un-development
        potential = jnp.maximum(D_target - D_current, 0.0)

        # 2. COMPUTE INHIBITION (Langmuir Isotherm)
        F = 1.0 / (1.0 + self.gamma * P_star)

        # 3. REACTION RATE
        rate = potential * F

        # 4. DIFFUSION (Vectorized Laplacian)
        # Apply convolution independently per channel using vmap
        # Input P is (C, H, W)
        laplacian = jax.vmap(lambda x: jax.scipy.signal.convolve2d(
            x, self.laplacian_kernel[0, 0], mode='same', boundary='symm'
        ))(P)

        # 5. DERIVATIVES (Friedman-Ross Eqs 16.24-16.27)
        # dP/dt = Diffusion + Generation - Adsorption + Desorption
        dP_dt = (self.diff_coeff * laplacian) + rate - (self.k_ads * P) + (self.k_des * P_star)
        
        # dP*/dt = Adsorption - Desorption
        dP_star_dt = (self.k_ads * P) - (self.k_des * P_star)
        
        # dD/dt = Generation Rate
        dD_dt = rate

        return ReactionState(P=dP_dt, P_star=dP_star_dt, D=dD_dt)

    def simulate(self, latent_image, t_end=5.0, dt=0.1):
        # INITIALIZATION
        # P and P* start at 0
        zeros = jnp.zeros_like(latent_image)
        
        # D starts at D-min (Fog). Extract from tone_curve parameters.
        # Assuming params shape is (3, 5) where col 0 is Dmin.
        # Reshape to (3, 1, 1) for broadcasting
        d_min = self.tone_curve.params[:, 0].reshape(-1, 1, 1)
        d_start = jnp.ones_like(latent_image) * d_min

        y0 = ReactionState(P=zeros, P_star=zeros, D=d_start)

        # SOLVER CONFIG
        # Tsit5 is standard. Use PIDController for stiff diffusion safety.
        term = diffrax.ODETerm(self)
        solver = diffrax.Tsit5()
        stepsize = diffrax.PIDController(rtol=1e-3, atol=1e-3)

        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=t_end, dt0=dt, y0=y0,
            args=latent_image, stepsize_controller=stepsize,
            max_steps=2000
        )

        # Return the final Developed Density (Time -1)
        return sol.ys.D[-1]

```

### Phase 3: Pipeline Integration

**File:** `Film/pipeline.py` (Main Entry Point)

Wire the new physics module into the execution graph.

```python
# ... Imports ...
from Film.core.optical.scattering import OpticalPhysics
from Film.core.chemical.reaction_diffusion import ChemicalDiffusion
from Film.core.sensitometry.tone_curve import SensitometricCurve
from Film.core.color.dye_densities import apply_color_matrix # Hypothetical

def run_simulation(actinic_exposure, params):
    # 1. SETUP MODULES
    # Initialize physics with learned/preset parameters
    optical = OpticalPhysics(
        scatter_gamma=params['scatter'], 
        halation_radius=params['hal_rad']
    )
    
    # Load Curve
    curve = SensitometricCurve(params=params['curve_params'])
    
    # Initialize Solver with the Curve
    chemistry = ChemicalDiffusion(
        tone_curve=curve,
        diff_coeff=params['chem_diff'],
        gamma=params['chem_gamma']
    )

    # 2. OPTICAL TRANSPORT (Linear)
    # Simulates light spreading in the emulsion *before* development
    latent_image = optical(actinic_exposure)

    # 3. CHEMICAL DEVELOPMENT (Non-Linear)
    # Simulates the development tank. 
    # Applies Tone Mapping AND Edge Effects simultaneously.
    developed_density = chemistry.simulate(latent_image, t_end=5.0)

    # 4. COLORIMETRY
    # Convert dye densities to visual color (Virtual Negative)
    # Note: Ensure apply_color_matrix works on density data, not exposure data
    virtual_negative = apply_color_matrix(developed_density)

    # 5. GRAIN SYNTHESIS
    # Critical: Condition grain on the DENSITY, not the EXPOSURE
    final_image = grain_net(virtual_negative, density_map=developed_density)

    return final_image

```

---

## 3. Critical Technical Constraints

1. **Boundary Conditions:**
* The Laplacian convolution **must** use `boundary='symm'`. If you use zero-padding, the inhibitor will "drain" out of the image edges, causing bright halos around the frame border.


2. **Stiffness Handling:**
* Reaction-Diffusion systems are mathematically "stiff."
* If `diff_coeff` is set too high (> 5.0), `Tsit5` may become unstable.
* **Contingency:** If instability occurs (NaNs), switch the solver in `simulate()` to `diffrax.Kvaerno5()` (an implicit solver).


3. **Positive Clipping:**
* The term `jnp.maximum(D_target - D_current, 0.0)` is essential. Without it, if `D` overshoots slightly due to numerical error, the reaction rate could become negative, effectively "undeveloping" the film, which is physically impossible.


4. **Verification (Mackie Lines):**
* To verify the code is working, feed it a **Step Edge** (Black/White split).
* **Correct Output:** You should see a "Dip" (darker line) on the low-density side and a "Peak" (brighter line) on the high-density side of the border.
* **Incorrect Output:** If the edge is purely blurred or purely sharp with no fringes, the diffusion/inhibition coupling (`gamma`) is too weak.



```

```