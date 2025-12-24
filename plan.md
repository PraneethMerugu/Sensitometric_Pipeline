# Implementation Plan: Cross-Channel Chemical Diffusion (Inter-Image Effects)

**Objective:** Upgrade the `ChemicalDiffusion` module from scalar inhibition (isolated channels) to matrix-based inhibition (coupled channels). This simulates **Development Inhibitor Releasing (DIR)** couplers, which are responsible for color separation and "punch" in modern negative films.

**Target Files:**
1.  `core/chemical/reaction_diffusion.py` (Core Logic)
2.  `pipeline.py` (Integration)

---

## Step 1: Update Physics Engine (`core/chemical/reaction_diffusion.py`)

**Context:**
Currently, `ChemicalDiffusion` uses a scalar `gamma` to calculate inhibition ($F$). We must replace this with a 3x3 `coupling_matrix` to allow inhibitors from one channel to affect others.

**Task 1.1: Modify `__init__`**
* **Change Signature:** Remove `gamma`. Add `coupling_matrix` (default `None`).
* **Logic:**
    * If `coupling_matrix` is provided, assert shape is `(3, 3)`.
    * If `None`, default to an Identity matrix multiplied by a standard gamma (e.g., 2.0) to preserve backward compatibility (isolated channels).
* **Code Reference:**
    ```python
    # OLD
    def __init__(self, tone_curve, diff_coeff=1.0, k_ads=0.5, k_des=0.1, gamma=2.0):
        self.gamma = jnp.array(gamma)

    # NEW
    def __init__(self, tone_curve, diff_coeff=1.0, k_ads=0.5, k_des=0.1, coupling_matrix=None):
        # ... validation ...
        if coupling_matrix is None:
             self.coupling_matrix = jnp.eye(3) * 2.0
        else:
             self.coupling_matrix = jnp.array(coupling_matrix)
    ```

**Task 1.2: Modify `__call__` (The Vector Field)**
* **Math:** Replace $F = 1 / (1 + \gamma P^*)$ with $F = 1 / (1 + \text{coupling\_matrix} @ P^*)$.
* **Implementation Detail:**
    * Input `state.P_star` has shape `(H, W, 3)`.
    * Use `jnp.dot(state.P_star, self.coupling_matrix.T)` to calculate the "effective coupled inhibitors".
    * Ensure the resulting shape remains `(H, W, 3)`.
* **Safety:**
    * Ensure `diffrax.PIDController` is used in `simulate()` (if not already) as this coupling makes the ODE system "stiff".

---

## Step 2: Update Pipeline Wiring (`pipeline.py`)

**Context:**
The `FilmPipeline` class currently passes a scalar `gamma` from the input dictionary to the chemistry module.

**Task 2.1: Update `FilmPipeline.__init__`**
* **Logic:**
    * Check `params` for a key `'coupling_matrix'`.
    * If missing, check for legacy `'gamma'` key and construct a diagonal matrix: `jnp.eye(3) * params['gamma']`.
    * Pass this matrix to the `ChemicalDiffusion` constructor.
* **Code Reference:**
    ```python
    # Inside FilmPipeline.__init__
    
    # 1. Resolve Matrix
    c_mat = params.get('coupling_matrix', None)
    if c_mat is None:
        # Fallback to legacy gamma
        g = params.get('gamma', 2.0)
        c_mat = jnp.eye(3) * g
    
    # 2. Instantiate
    self.chemistry = ChemicalDiffusion(
        tone_curve=curve,
        diff_coeff=params.get('diff_coeff', 1.0),
        k_ads=params.get('k_ads', 0.5),
        k_des=params.get('k_des', 0.1),
        coupling_matrix=c_mat  # <--- CHANGED
    )
    ```

---

## Step 3: Verification Protocol

**Context:**
We must verify that activity in one channel (e.g., Green) suppresses development in another (e.g., Red).

**Task 3.1: Create Verification Script (`tests/test_inter_image.py`)**
Create a script that performs the following "Color Punch" test:

1.  **Setup:**
    * Define a Coupling Matrix with strong cross-inhibition:
        ```python
        # Red is inhibited by Green
        matrix = jnp.array([
            [2.0, 4.0, 0.0], # Red Row: Self=2, Green_Source=4
            [0.0, 2.0, 0.0], # Green Row: Self=2
            [0.0, 0.0, 2.0]  # Blue Row: Self=2
        ])
        ```
2.  **Input:**
    * Create a "Flat Red" image (constant low exposure).
    * Create a "Red + Green Spot" image (same Red base, but with a bright Green spot in the center).
3.  **Run Simulation:**
    * Run both images through `ChemicalDiffusion`.
4.  **Assertion:**
    * Compare the **Red Density** in the center of both images.
    * **Pass Condition:** The Red density in the "Red + Green Spot" image must be **LOWER** than in the "Flat Red" image. (Because the Green activity released inhibitors that "bled" into the Red layer).

---

## Summary of Files to Change

| File | Nature of Change | Complexity |
| :--- | :--- | :--- |
| `core/chemical/reaction_diffusion.py` | Core Physics Logic | High |
| `pipeline.py` | Parameter Passing | Low |