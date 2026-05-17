# README for Codex — Skew-symmetric DG flux comparison and sphere extension

## 0. Project goal

This project builds a simplex-quadrature nodal DG / SDG solver for scalar advection, first on planar triangular meshes and then toward a spherical manifold DG solver.

The current personal report direction is:

> Fix the volume discretization as the skew-symmetric / split form, then compare central, upwind, and Lax–Friedrichs (LF) numerical fluxes. After the planar version is verified, extend the same idea to the sphere using manifold geometry.

Core mathematical decomposition:

```text
RHS(q) = volume term + surface term

volume term:
    fixed as skew-symmetric / split form

surface term:
    compare central, upwind, LF
```

For scalar advection,

```math
q_t + \nabla \cdot (V q) = 0.
```

The skew-symmetric volume form is

```math
R_vol = -1/2 div(V q) - 1/2 V · grad(q) - 1/2 (div V) q.
```

On a reference triangle with contravariant velocities `(u, v)`, the current split volume form is

```math
R_vol = -1/2 [D_r(u q) + D_s(v q)]
        -1/2 [u D_r q + v D_s q]
        -1/2 [D_r u + D_s v] q.
```

---

## 1. Current repository status

The uploaded files are currently flat in `/mnt/data`, but the Python imports assume a package named `core`, for example:

```python
from core.operators import build_local_operators
from core.operators_split import compute_split_rhs
```

When running locally or in Codex, make sure the files are placed as:

```text
project_root/
  core/
    __init__.py
    basis.py
    quadrature.py
    operators.py
    rhs.py
    operators_split.py
    time_integration.py
    cfl.py
    cases.py
    mesh.py
    mesh_octahedron.py
    topology.py
    connectivity.py
    geometry/
      __init__.py
      face_metrics.py
      connectivity.py        # if used as core.geometry.connectivity
      sphere_mapping.py
      sdg_sphere_mapping.py
      sphere_velocity.py
  tests or root:
    test_lsrk_stationary_xy.py
    test_lsrk_temporal_sinx.py
    test_error_evolution_sinxy.py
    test_exchange_h_refinement.py
```

Some current imports use `core.geometry.face_metrics`, `core.geometry.connectivity`, etc. If these files are flat, move them into `core/geometry/` or update the imports consistently.

---

## 2. Important existing files

### Reference element and quadrature

- `quadrature.py`
  - Table 1 / Table 2 triangle quadrature rules.
  - Boundary nodes are ordered edge0, edge1, edge2, then interior nodes.

- `basis.py`
  - Jacobi / simplex polynomial basis and gradients.

- `operators.py`
  - Defines `ReferenceElement`.
  - Builds Vandermonde matrices, mass matrix, differentiation matrices `Dr`, `Ds`.
  - Provides `edge_slices`, `boundary_values`, `boundary_weight_diag`, `lift_boundary_penalty`.

### RHS and flux

- `rhs.py`
  - Currently contains:
    - `compute_upwind_flux(...)`
    - `compute_boundary_penalty(..., tau=0.0)`
    - `compute_split_volume_rhs(...)`
    - `compute_volume_divergence(...)`
    - `compute_sdg_rhs_single_element(...)`
  - Current `compute_boundary_penalty` uses `tau`:
    - `tau=0.0`: upwind
    - `tau=1.0`: central
  - It does **not yet** support an explicit `flux_type="lf"` or `alpha_lf` parameter.

- `operators_split.py`
  - Contains `mapped_gradient_split_2d(...)` and `compute_split_rhs(...)`.
  - Current `compute_split_rhs(...)` reads `tau = kwargs.get("tau", 0.0)`.
  - It calls `compute_boundary_penalty(..., tau=tau)`.
  - It does **not yet** pass `flux_type` / `alpha_lf`.

### Time integration

- `time_integration.py`
  - Contains LSRK54 coefficients and `lsrk54_step(...)`.

### Mesh / geometry

- `mesh.py`
  - Structured planar triangular mesh and unit triangle mesh utilities.

- `face_metrics.py`
  - Planar affine triangle metrics:
    - `compute_volume_metrics(...)` gives `rx, sx, ry, sy, J`
    - `compute_face_metrics(...)` gives `nx, ny, edge_lengths, sJ`

- `mesh_octahedron.py`, `sphere_mapping.py`, `sdg_sphere_mapping.py`, `sphere_velocity.py`
  - Early sphere / octahedral patch mapping and solid-body velocity utilities.
  - These are not yet fully connected to a final manifold DG RHS.

### Existing tests

- `test_lsrk_stationary_xy.py`
  - Planar stationary test using `q=x-y`, velocity `(1,1)`.

- `test_lsrk_temporal_sinx.py`
  - Temporal convergence with LSRK54.

- `test_error_evolution_sinxy.py`
  - Periodic sine-wave advection error evolution.

- `test_exchange_h_refinement.py`
  - h-refinement / exchange test.

---

## 3. Immediate next task: add explicit central / upwind / LF flux switching

### 3.1 Required API

Modify `rhs.py` so `compute_boundary_penalty` supports:

```python
compute_boundary_penalty(
    q_minus,
    q_plus,
    nx,
    ny,
    u,
    v,
    tau=None,
    flux_type="upwind",
    alpha_lf=1.0,
)
```

Backward compatibility requirement:

- Existing tests that pass only `tau=0.0` or `tau=1.0` must still work.
- If `tau` is provided and `flux_type` is not explicitly provided, interpret:
  - `tau=0.0` as upwind
  - `tau=1.0` as central
  - intermediate `tau` as the old blended central/upwind formula.

Recommended behavior:

```text
flux_type = "central": C = 0
flux_type = "upwind" : C = abs(n dot V)
flux_type = "lf"     : C = alpha_lf * abs(n dot V)
```

Normal flux formula:

```math
n · f* = 1/2 a_n (q^- + q^+) + 1/2 C (q^- - q^+),
```

where

```math
a_n = n · V.
```

Boundary penalty:

```math
p = n · f(q^-) - n · f*.
```

Since

```math
n · f(q^-) = a_n q^-,
```

```math
p = a_n q^- - [1/2 a_n(q^-+q^+) + 1/2 C(q^- - q^+)].
```

### 3.2 Add helper function

Add this helper to `rhs.py`:

```python
def compute_normal_flux(q_minus, q_plus, nx, ny, u, v, flux_type="upwind", alpha_lf=1.0, tau=None):
    """
    Compute scalar normal numerical flux n · f* for advection.

    flux_type:
        "central" -> C = 0
        "upwind"  -> C = |n·V|
        "lf"      -> C = alpha_lf |n·V|

    tau:
        Backward-compatible old interface.
        If tau is not None, use C = (1 - tau) |n·V| unless flux_type is explicitly lf.
    """
```

Implementation logic:

```python
ndotV = nx * u + ny * v

if flux_type is None:
    flux_type = "upwind"

flux_type = flux_type.lower()

if flux_type in ("central", "centered"):
    C = 0.0
elif flux_type == "upwind":
    C = np.abs(ndotV)
elif flux_type in ("lf", "lax-friedrichs", "lax_friedrichs", "rusanov"):
    C = alpha_lf * np.abs(ndotV)
else:
    raise ValueError(...)

# Backward compatibility: old tau blend, only if tau is not None and flux_type was not lf.
# Option A: easiest is to keep tau behavior in compute_boundary_penalty when tau is provided.

flux_star_n = 0.5 * ndotV * (q_minus + q_plus) + 0.5 * C * (q_minus - q_plus)
return flux_star_n
```

### 3.3 Update `compute_boundary_penalty`

Pseudo-code:

```python
def compute_boundary_penalty(q_minus, q_plus, nx, ny, u, v,
                             tau=None, flux_type="upwind", alpha_lf=1.0):
    ndotV = nx * u + ny * v
    flux_internal_n = ndotV * q_minus

    if tau is not None and flux_type == "upwind":
        # Preserve old behavior exactly for old callers.
        flux_star_n = (
            0.5 * ndotV * (q_minus + q_plus)
            + 0.5 * (1.0 - tau) * np.abs(ndotV) * (q_minus - q_plus)
        )
    else:
        flux_star_n = compute_normal_flux(
            q_minus, q_plus, nx, ny, u, v,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )

    return flux_internal_n - flux_star_n
```

Important: choose a clean backward-compatibility policy. The current tests call `tau=tau`, so keep those working.

### 3.4 Update `operators_split.py`

Inside `compute_split_rhs`, add:

```python
flux_type = kwargs.get("flux_type", None)
alpha_lf = kwargs.get("alpha_lf", 1.0)
tau = kwargs.get("tau", None)

# Backward compatibility: old default was tau=0.0, i.e. upwind.
if flux_type is None and tau is None:
    flux_type = "upwind"
```

Then pass:

```python
p_face = compute_boundary_penalty(
    q_minus=qM,
    q_plus=qP,
    nx=nx_face,
    ny=ny_face,
    u=u_face,
    v=v_face,
    tau=tau,
    flux_type=flux_type,
    alpha_lf=alpha_lf,
)
```

---

## 4. Add a new planar flux comparison test

Create a new test file:

```text
test_flux_comparison_sinxy.py
```

Purpose:

- Fixed skew-symmetric volume term.
- Compare three surface fluxes:
  - central
  - upwind
  - LF with `alpha_lf = 1.0` and optionally `alpha_lf = 1.5`.

Use the same setup as `test_error_evolution_sinxy.py` as much as possible.

### 4.1 Recommended exact solution

Use periodic planar advection:

```math
q(x,y,t) = sin(2π(x + y - (c_x + c_y)t)).
```

with

```python
cx = 1.0
cy = 1.0
```

### 4.2 Metrics to print

For each flux type, print:

```text
flux_type
alpha_lf
L2_final
Linf_final
mass_initial
mass_final
mass_error
energy_initial
energy_final
energy_change
```

Weighted mass:

```python
mass = np.sum(J[:, None] * engine.w_s[None, :] * q)
```

Weighted energy:

```python
energy = np.sum(J[:, None] * engine.w_s[None, :] * q**2)
```

Weighted L2 error:

```python
l2 = np.sqrt(
    np.sum(J[:, None] * engine.w_s[None, :] * error**2)
    / np.sum(J[:, None] * engine.w_s[None, :])
)
```

### 4.3 Expected results

- `central` should be least dissipative but may be less stable for long runs.
- `upwind` should be more stable and dissipative.
- `lf, alpha_lf=1.0` should match upwind for scalar linear advection.
- `lf, alpha_lf>1.0` should be more dissipative.

Add an assertion/check:

```python
max_abs_difference_between_upwind_and_lf_alpha1 should be near roundoff
```

This is only true if LF uses the same pointwise `alpha_lf * abs(ndotV)` with `alpha_lf=1`.

---

## 5. Energy / well-posed analysis target for report

The report should include energy analysis. Code does not need to prove this, but numerical tests should record mass/energy.

Continuous skew-symmetric energy identity:

```math
\frac{d}{dt}\int_\Omega q^2 dx
= -\int_{\partial\Omega} (V\cdot n)q^2 dS
  -\int_\Omega (\nabla\cdot V)q^2 dx.
```

For divergence-free velocity and periodic/closed boundary, energy is conserved at the continuous level.

DG expectation:

- central flux: no jump dissipation; low dissipation / energy-conserving behavior.
- upwind flux: adds jump dissipation

```math
-\frac12\int_{\Gamma_h}|a_n|[q]^2 dS.
```

- LF flux: adds dissipation controlled by `alpha_lf`

```math
-\frac12\int_{\Gamma_h}\alpha [q]^2 dS.
```

---

## 6. Sphere / manifold DG extension target

This is not fully finished yet. Do this after planar flux switching and planar tests.

The manifold DG idea avoids longitude-latitude patch singularities and computes geometry directly in 3D.

Given nodal sphere coordinates

```math
X(r,s) = (X,Y,Z)^T,
```

compute covariant basis:

```math
a_1 = ∂X/∂r = D_r X,

a_2 = ∂X/∂s = D_s X.
```

Surface Jacobian and normal:

```math
N = a_1 × a_2,
J = ||N||,
n_surf = N / J.
```

Contravariant basis:

```math
a^1 = (a_2 × n_surf) / J,

a^2 = (n_surf × a_1) / J.
```

3D solid-body rotation velocity:

```math
V_3D = ω × X.
```

Project to reference-coordinate contravariant velocity:

```math
u_tilde = a^1 · V_3D,

v_tilde = a^2 · V_3D.
```

Manifold skew-symmetric volume RHS:

```math
R_vol = -1/(2J) [D_r(J u_tilde q) + D_s(J v_tilde q)]
        -1/2 [u_tilde D_r q + v_tilde D_s q]
        -q/(2J) [D_r(J u_tilde) + D_s(J v_tilde)].
```

Sphere surface normal speed used by flux:

```math
v_n^{sJ} = J_M (n_r u_tilde_M + n_s v_tilde_M).
```

Unified penalty form:

```math
P = 1/2 (v_n^{sJ} - C)(q_M - q_P),
```

where

```text
central: C = 0
upwind : C = |v_n^{sJ}|
LF     : C = alpha_lf |v_n^{sJ}|
```

Final surface contribution:

```math
R_surf = (1/J) * lift(P)
```

Exact implementation details depend on how face geometry and face weights are stored.

---

## 7. Suggested implementation order

Do not start with the sphere. Finish the planar flux comparison first.

### Step 1 — package/import cleanup

Make the file layout match imports:

```text
core/
core/geometry/
```

or update imports.

### Step 2 — add explicit flux switching

Modify:

- `rhs.py`
- `operators_split.py`

Add:

- `flux_type="central" | "upwind" | "lf"`
- `alpha_lf`

Keep old `tau` tests working.

### Step 3 — run old tests

Run:

```bash
python test_lsrk_stationary_xy.py
python test_lsrk_temporal_sinx.py
python test_error_evolution_sinxy.py
python test_exchange_h_refinement.py
```

If imports fail, fix package structure first.

### Step 4 — add flux comparison test

Create and run:

```bash
python test_flux_comparison_sinxy.py
```

Check:

- upwind and LF with alpha=1 match for scalar linear advection.
- central has lower dissipation.
- LF with alpha>1 is more dissipative.

### Step 5 — record mass and energy

Add helper functions for weighted mass and energy. Use them in the flux comparison test.

### Step 6 — only then start manifold sphere RHS

Create a new file, suggested:

```text
manifold_geometry.py
manifold_rhs.py
test_manifold_solid_body.py
```

Do not overwrite working planar tests.

---

## 8. Do not overclaim current status

Current safe statement:

> The planar skew-symmetric DG framework and LSRK time integration are partially implemented and tested. The immediate missing feature is explicit central/upwind/LF flux switching and flux-comparison diagnostics. The manifold sphere formulation is written mathematically but still needs a complete verified implementation and long-time solid-body rotation tests.

Avoid saying:

> The full spherical manifold DG solver with all flux comparisons is complete.

That is not true yet.

---

## 9. Useful formulas for Codex comments/docstrings

### Normal numerical flux

```math
f_n^* = 1/2 a_n(q^- + q^+) + 1/2 C(q^- - q^+)
```

### Penalty

```math
p = a_n q^- - f_n^*
```

### Flux dissipation coefficient

```text
central: C = 0
upwind : C = |a_n|
LF     : C = alpha_lf |a_n|
```

### Old tau compatibility

```text
tau = 0 -> upwind
tau = 1 -> central
old formula: C = (1 - tau) |a_n|
```

---

## 10. Recommended final deliverables before writing results

Minimum code deliverables:

1. `flux_type` and `alpha_lf` supported in `rhs.py` and `operators_split.py`.
2. Old tests still run.
3. New `test_flux_comparison_sinxy.py` runs.
4. Output table comparing central/upwind/LF.
5. Mass and energy diagnostics included.
6. Optional: plot error/energy vs time.

Then the report can honestly include:

- theory: skew-symmetric DG and energy estimate
- implementation: planar skew-symmetric solver
- numerical comparison: central/upwind/LF on planar tests
- extension: mathematical manifold DG sphere formulation
- future work: complete verified sphere solver and long-time solid-body rotation benchmark
## Current Status

### Phase 1 completed

- Planar skew-symmetric RHS now supports `flux_type="central"`, `"upwind"`, and `"lf"`.
- `alpha_lf` is supported.
- Legacy `tau=0/1` interface remains compatible.
- Added `test_flux_comparison_sinxy.py`.
- Verified `LF(alpha=1)` equals upwind for scalar linear advection.

Validation:
- `test_flux_comparison_sinxy.py` passed.
- `test_lsrk_stationary_xy.py` passed.
- `test_lsrk_temporal_sinx.py` passed.
- `compileall core test/test_flux_comparison_sinxy.py` passed.

Known environment limitations:
- `test_exchange_h_refinement.py` was not run because `matplotlib` is unavailable.
- `test_trace_exchange_h_convergence.py` exceeded the 180-second runtime limit.
## Next Tasks

### Phase 2 target

Build manifold sphere geometry diagnostics before implementing full sphere RHS.

Tasks:

1. Verify sphere nodes lie on radius `R`.
2. Compute covariant basis `a1`, `a2`.
3. Compute surface Jacobian `J` and surface normal `n`.
4. Compute contravariant basis `a^1`, `a^2`.
5. Verify biorthogonality.
6. Verify total surface area approximates `4πR²`.

Important constraints:

- Do not implement the full sphere RHS yet.
- Do not modify the working planar flux switching unless necessary.
- Do not remove the legacy `tau` interface.
- Do not perform large-scale refactoring.

### Phase 2 step 1 completed

Added `test/test_manifold_geometry_sphere.py` as a geometry-only diagnostic.

This diagnostic verifies:

1. Sphere DG nodes satisfy `||X|| = R`.
2. Covariant bases `a1 = Dr X` and `a2 = Ds X` are finite.
3. Surface Jacobian `J = ||a1 x a2||` is positive and finite.
4. Surface normals are finite and unit length.
5. Contravariant bases `a^1`, `a^2` satisfy `a^i dot a_j ~= delta^i_j`.
6. The global surface area integral approximates `4*pi*R^2`.

Validation:

- `test_manifold_geometry_sphere.py` passed.
- `compileall test/test_manifold_geometry_sphere.py` passed.
- For `nsub=4`, `N=4`, the computed sphere area relative error was about `5.306071e-05`.

Scope note:

- No sphere RHS was implemented.
- No planar flux switching files were modified for this phase.
- No large refactor was performed.
