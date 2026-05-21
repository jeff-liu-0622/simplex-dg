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

### Phase 2.2 completed

Added `test/test_manifold_velocity_sphere.py` as a velocity-only diagnostic on
the octahedral manifold sphere mesh.

This diagnostic verifies:

1. The octahedral sphere manifold mesh and reference element can be reused from
   the Phase 2.1 geometry diagnostic.
2. The solid-body rotation velocity is computed directly in 3D as
   `V3D = omega x X`.
3. `V3D` is tangent to the sphere, i.e. `V3D dot X ~= 0`.
4. Contravariant velocity components are computed as
   `u_tilde = a^1 dot V3D` and `v_tilde = a^2 dot V3D`.
5. `u_tilde` and `v_tilde` are finite.
6. Reconstructed velocity `V_rec = u_tilde a1 + v_tilde a2` approximates `V3D`.

Validation:

- `test_manifold_velocity_sphere.py` passed.
- `compileall test/test_manifold_velocity_sphere.py` passed.
- Diagnostic setup: `nsub=8`, `N=4`, `R=1`, `u0=1`, `alpha=pi/4`.
- `max_tangency_error = 1.387779e-16`.
- `max_reconstruction_error = 8.320434e-03`.
- `max_speed = 9.999999e-01`.

Scope note:

- No full sphere RHS was implemented.
- No time integration was added.
- No planar flux switching files were modified.
- No large refactor was performed.

### Next target: Phase 2.3

Build a manifold constant-state RHS diagnostic before moving to a full sphere
solver.

Target:

1. Use the manifold geometry and velocity diagnostics already verified.
2. Build a diagnostic RHS path for the constant state `q = 1`.
3. Check whether the computed RHS remains near zero for solid-body rotation.
4. Keep this as a diagnostic-only step until it passes.

Important status note:

- The full sphere RHS is **not complete**.
- The project should not move to full sphere advection, time integration, or
  long-time solid-body rotation benchmarks until the `q=1` RHS diagnostic
  passes.

### Phase 2.3 blocked: constant-state volume RHS needs investigation

Added `test/test_manifold_constant_rhs_sphere.py` as a volume-only diagnostic
for the manifold skew-symmetric RHS with `q = 1`.

This diagnostic computes:

1. The octahedral manifold sphere mesh and reference element.
2. Manifold geometry from the Phase 2.1 path:
   `X`, `a1`, `a2`, `J`, `n_surf`, `a^1`, `a^2`.
3. Solid-body velocity from the Phase 2.2 path:
   `V3D = omega x X`,
   `u_tilde = a^1 dot V3D`,
   `v_tilde = a^2 dot V3D`.
4. The volume-only skew-symmetric RHS:

```text
R_vol =
  -1/(2J) [Dr(J u_tilde q) + Ds(J v_tilde q)]
  -1/2 [u_tilde Dr(q) + v_tilde Ds(q)]
  -q/(2J) [Dr(J u_tilde) + Ds(J v_tilde)]
```

Validation:

- `test_manifold_constant_rhs_sphere.py` ran to completion.
- `compileall test/test_manifold_constant_rhs_sphere.py` passed.
- Diagnostic setup: `nsub=8`, `N=4`, `R=1`, `u0=1`, `alpha=pi/4`.
- `max_abs_rhs = 1.583918e-01`.
- `rms_rhs = 7.347184e-03`.
- `mean_rhs = -2.949912e-15`.
- `max_abs_divJv = 1.583918e-01`.

Status:

- The diagnostic values are finite.
- The mean RHS is near zero, but the max and RMS RHS are not near zero.
- Because `max_abs_rhs` matches `max_abs_divJv`, the issue appears tied to
  the discrete metric / divergence term `1/J [Dr(J u_tilde) + Ds(J v_tilde)]`.
- Do not tune tolerance to force this through.
- Mark this phase as **needs investigation** before any full sphere RHS,
  surface flux, time integration, or long-time solver work.

Scope note:

- No surface penalty was implemented.
- No central/upwind/LF sphere surface flux was implemented.
- No time integration was added.
- No Phase 1 planar flux switching files were modified.

### Phase 2.3a completed: divJv refinement diagnostic

Extended `test/test_manifold_constant_rhs_sphere.py` with a refinement
diagnostic for the discrete divergence term

```text
divJv = 1/J [Dr(J u_tilde) + Ds(J v_tilde)].
```

This diagnostic keeps the original single `q=1` volume RHS output and adds:

1. A refinement table for `nsub = [2, 4, 8, 16]`.
2. Observed order based on `max_abs_divJv`.
3. Observed order based on `rms_divJv`.
4. The element/node location of the largest `max_abs_divJv`.
5. Local diagnostic data at that point: `X`, `J`, `u_tilde`, `v_tilde`,
   `divJv`, patch-boundary distance, and original-octahedron-vertex distance.

Validation:

- `test_manifold_constant_rhs_sphere.py` ran to completion.
- `compileall test/test_manifold_constant_rhs_sphere.py` passed.

Refinement table:

```text
nsub    max_abs_divJv    rms_divJv       mean_divJv       order(max)   order(rms)
2       2.302358e-01     3.348427e-02    1.072690e-15     -            -
4       1.859073e-01     1.519299e-02    3.616307e-15     0.3085       1.1401
8       1.583918e-01     7.347184e-03    2.389099e-15     0.2311       1.0481
16      1.438451e-01     3.640459e-03    1.876180e-14     0.1390       1.0131
```

Largest-error location on the finest mesh:

```text
nsub = 16
element = 512
node = 10
patch_id = 3
xi = 0.000000000000e+00
eta = 5.956812018558e-02
X = (-1.546127850337e-17, -8.416727998165e-02, 9.964516390576e-01)
J = 2.941073401484e-03
u_tilde = -1.840670393809e+00
v_tilde = -1.444700723635e+01
divJv = 1.438450661613e-01
patch_boundary_distance = 0.000000000000e+00
near_patch_boundary = True
octahedron_vertex_distance = 8.424204345152e-02
near_octahedron_vertex = False
```

Current interpretation:

- `rms_divJv` improves at roughly first order.
- `max_abs_divJv` decreases only slowly.
- The largest error is on a patch boundary (`xi = 0`) and is not exactly at an
  original octahedron vertex.
- This suggests the blocked `q=1` volume RHS issue is localized around
  patch-boundary / element-boundary behavior in the discrete metric divergence,
  rather than a global mean-divergence error.

Next investigation target:

- Inspect whether the manifold geometry/velocity representation is continuous
  enough across octahedral patch boundaries for the volume-only strong
  derivative `Dr(J u_tilde) + Ds(J v_tilde)`.
- Do not implement sphere surface flux or time integration until this
  constant-state metric-divergence issue is understood.
Phase 2.3b: boundary vs interior divergence diagnostic

Goal:
Identify whether the large pointwise metric divergence error is localized at face nodes / patch boundaries.

Do not implement sphere surface flux or time integration yet.

### Phase 2.3b completed: boundary vs interior divJv diagnostic

Extended `test/test_manifold_constant_rhs_sphere.py` with boundary/interior
classification for the discrete metric divergence term.

Node categories:

1. `interior`: local reference-element interior nodes.
2. `face_boundary`: local reference-element face nodes.
3. `near_patch_boundary`: nodes on an octahedral parent patch boundary.
4. `near_octahedron_vertex`: nodes near an original octahedron vertex / pole
   region.

Validation:

- `test_manifold_constant_rhs_sphere.py` ran to completion.
- `compileall test/test_manifold_constant_rhs_sphere.py` passed.

Category table:

```text
nsub category                  count    max_abs_divJv    rms_divJv       mean_divJv
2    interior                    224     1.586261e-02     4.843792e-03    2.694242e-16
2    face_boundary               480     2.302358e-01     4.041623e-02    1.448725e-15
2    near_patch_boundary         240     2.302358e-01     4.750437e-02    2.057527e-15
2    near_octahedron_vertex      648     2.302358e-01     3.489708e-02    1.082296e-15
4    interior                    896     1.180285e-02     1.659218e-03    8.261775e-16
4    face_boundary              1920     1.859073e-01     1.836468e-02    4.918317e-15
4    near_patch_boundary         480     1.859073e-01     3.125489e-02    5.313516e-15
4    near_octahedron_vertex      632     1.859073e-01     3.184094e-02    2.479205e-15
8    interior                   3584     9.515720e-03     6.791528e-04    1.692171e-15
8    face_boundary              7680     1.583918e-01     8.885778e-03    2.714315e-15
8    near_patch_boundary         960     1.583918e-01     2.158269e-02    2.071721e-14
8    near_octahedron_vertex      600     1.583918e-01     3.155380e-02    9.134637e-15
16   interior                  14336     8.340087e-03     3.172431e-04    6.051400e-15
16   face_boundary             30720     1.438451e-01     4.403485e-03    2.469332e-14
16   near_patch_boundary        1920     1.438451e-01     1.516375e-02    3.594201e-14
16   near_octahedron_vertex      600     1.438451e-01     3.125155e-02    8.684696e-15
```

Top-10 location summary on `nsub=16`:

- All top 10 `|divJv|` locations are on octahedral patch boundaries.
- All top 10 are also near an original octahedron vertex / pole region under
  the current near-vertex classification.
- The largest location remains:

```text
element = 512
node = 10
patch_id = 3
xi = 0.000000000000e+00
eta = 5.956812018558e-02
X = (-1.546127850337e-17, -8.416727998165e-02, 9.964516390576e-01)
J = 2.941073401484e-03
u_tilde = -1.840670393809e+00
v_tilde = -1.444700723635e+01
divJv = 1.438450661613e-01
near_patch_boundary = True
near_octahedron_vertex = True
patch_boundary_distance = 0.000000000000e+00
octahedron_vertex_distance = 8.424204345152e-02
```

Current interpretation:

- The large pointwise error is boundary dominated.
- Interior-node max error is much smaller and decreases with refinement:
  from `1.586261e-02` at `nsub=2` to `8.340087e-03` at `nsub=16`.
- Patch-boundary / near-pole-region nodes control the global max.
- This supports the hypothesis that the blocked constant-state volume RHS is
  tied to patch-boundary metric/velocity representation, not a global
  divergence bias.

Next investigation target:

- Examine patch-boundary continuity and metric identity behavior for
  `J u_tilde` and `J v_tilde`.
- Still do not move to long-time sphere simulation, LSRK, or sphere surface
  flux implementation until the constant-state divergence issue is understood.

### Phase 2.3c completed: face metric-flux continuity diagnostic

Extended `test/test_manifold_constant_rhs_sphere.py` with a face-based metric
flux jump diagnostic.

For each shared face, the diagnostic computes

```text
Fr = J * u_tilde
Fs = J * v_tilde
Fn = nr * Fr + ns * Fs
jump_Fn = Fn_M + Fn_P
```

where `(nr, ns)` is the reference-triangle outward normal for each local face.
For a consistent shared face, the two outward normals should oppose each other,
so `jump_Fn` should be small.

Validation:

- `test_manifold_constant_rhs_sphere.py` ran to completion.
- `compileall test/test_manifold_constant_rhs_sphere.py` passed.

Face jump summary:

```text
nsub category                    count    max_abs_jump_Fn  rms_jump_Fn     mean_jump_Fn
2    all_shared_faces              200     4.600975e-01     1.610595e-01   -5.087954e-02
2    not_near_patch_boundary       120     1.590557e-01     7.158876e-02   -2.588031e-02
2    near_patch_boundary            80     4.600975e-01     2.390878e-01   -8.837839e-02
2    near_octahedron_vertex        200     4.600975e-01     1.610595e-01   -5.087954e-02
4    all_shared_faces              880     1.906304e-01     5.733367e-02   -1.803619e-02
4    not_near_patch_boundary       720     8.758010e-02     3.439285e-02   -1.222432e-02
4    near_patch_boundary           160     1.906304e-01     1.129443e-01   -4.418961e-02
4    near_octahedron_vertex        196     1.906304e-01     8.405963e-02   -2.507814e-02
8    all_shared_faces             3680     8.832209e-02     2.295264e-02   -7.338228e-03
8    not_near_patch_boundary      3360     4.520453e-02     1.683080e-02   -5.932692e-03
8    near_patch_boundary           320     8.832209e-02     5.553445e-02   -2.209635e-02
8    near_octahedron_vertex        180     8.832209e-02     4.397965e-02   -1.242306e-02
16   all_shared_faces            15040     4.417763e-02     9.940585e-03   -3.267255e-03
16   not_near_patch_boundary     14400     2.288430e-02     8.323583e-03   -2.921424e-03
16   near_patch_boundary           640     4.417763e-02     2.762810e-02   -1.104844e-02
16   near_octahedron_vertex        180     4.417763e-02     2.219960e-02   -6.136284e-03
```

Top-10 face jump summary on `nsub=16`:

- All top 10 `|jump_Fn|` locations are near patch boundaries.
- All top 10 are also near the original-octahedron-vertex / pole region under
  the current classification.
- Largest location:

```text
element_M = 1279
element_P = 1535
face_M = 1
face_P = 0
node_M = 5
node_P = 4
patch_id_M = 5
patch_id_P = 6
X_M = (9.999893952284e-01, 4.605369763389e-03, -0.000000000000e+00)
X_P = (6.123129033599e-17, 9.999828583820e-01, -5.855163709587e-03)
Fn_M = -1.130328014005e-04
Fn_P = -4.406459829416e-02
jump_Fn = -4.417763109556e-02
near_patch_boundary = True
near_octahedron_vertex = True
patch_boundary_distance = 0.000000000000e+00
octahedron_vertex_distance = 4.605381973162e-03
```

Current interpretation:

- Face metric-flux jumps are also patch-boundary / near-pole dominated.
- The not-near-patch-boundary jumps are smaller, but not zero.
- The largest face jump locations show `X_M` and `X_P` are not the same
  physical sphere point under the current layout connectivity plus patch
  mapping. This suggests the current octahedral layout adjacency and/or
  patch-local sphere mapping is not yet a physically continuous manifold
  connectivity.
- This strongly supports the earlier conclusion that the blocked constant-state
  RHS is tied to boundary / patch metric representation, not to time stepping
  or surface flux choice.

Next investigation target:

- Verify the octahedral sphere topology and face pairing in physical 3D, not
  only in unfolded layout coordinates.
- Check whether patch-local coordinates on both sides of a shared face map to
  the same 3D curve before using that face for metric identities or fluxes.
- Do not move to long-time sphere simulation, LSRK, or sphere surface flux
  implementation until the manifold connectivity / metric issue has a clear
  resolution.

### Phase 2.4 completed: 3D projected octahedron topology integrated

Added `core/geometry/sphere_manifold_topology.py` as the formal core interface
for the physically continuous projected octahedron sphere mesh.

New core interface:

- `create_projected_octahedron_sphere_mesh(nsub, R=1.0, ndigits=12)`
  returns `VX`, `VY`, `VZ`, `EToV`, `patch_ids`, and `nodes_xyz`.
- `map_reference_nodes_to_projected_sphere(nodes_xyz, EToV, r, s, R=1.0)`
  maps high-order reference nodes to the projected sphere.
- `projected_sphere_mesh_hmin(nodes_xyz, EToV)` returns the minimum chord
  length.

Important topology conclusion:

- The old unfolded layout connectivity is not suitable as the physical sphere
  face pairing.
- It can pair faces that are adjacent in the 2D layout but far apart on the
  3D sphere; prior diagnostics showed shared-face `xyz` error up to about
  `1.999`.
- The new projected octahedron mesh builds connectivity from shared 3D
  projected vertices, so physical shared faces coincide.

Added `test/test_sphere_manifold_topology.py`.

Projected mesh face-continuity diagnostic:

```text
nsub   Nv     K      shared   max_face_match_error   rms_face_match_error
2      18     32     48       0.000000e+00           0.000000e+00
4      66     128    192      0.000000e+00           0.000000e+00
8      258    512    768      0.000000e+00           0.000000e+00
16     1026   2048   3072     0.000000e+00           0.000000e+00
```

The existing manifold geometry / constant-state diagnostic can now choose the
new projected mesh path.  For `q = 1`, `nsub=16`, `N=4`, `R=1`,
`u0=1`, `alpha=pi/4`, using the projected topology:

```text
max_abs_divJv = 2.164720e-06
rms_divJv     = 3.913355e-07
mean_divJv    = 0.000000e+00
max_abs_rhs   = 2.164720e-06
rms_rhs       = 3.913355e-07
```

Validation:

- `test_sphere_manifold_topology.py` passed.
- `test_manifold_constant_rhs_sphere.py` ran to completion with the projected
  mesh diagnostic passing.
- `compileall core/geometry/sphere_manifold_topology.py
  test/test_sphere_manifold_topology.py
  test/test_manifold_geometry_sphere.py
  test/test_manifold_constant_rhs_sphere.py` passed.

Status:

- Phase 2.4 is completed.
- The topology issue identified in Phase 2.3c has a clear replacement path:
  use the projected 3D octahedron mesh for sphere manifold diagnostics and
  future sphere RHS work.
- The old layout-based diagnostics are still useful as failure/regression
  evidence, but should not be used as the physical sphere face topology.

Next target:

- Build the sphere surface flux / full RHS diagnostic on top of the projected
  3D topology.
- Do not start long-time sphere simulation or LSRK benchmarks until the
  projected-topology full RHS diagnostic passes.

### Phase 2.5 completed: projected sphere full RHS q=1 diagnostic

Added `test/test_sphere_full_rhs_constant.py`.

This diagnostic builds the full constant-state sphere RHS on the projected 3D
octahedron topology only:

- projected sphere mesh from `core/geometry/sphere_manifold_topology.py`
- high-order projected nodal geometry
- solid-body rotation velocity `V3D = omega x X`
- existing skew-symmetric manifold volume RHS
- minimal face surface penalty skeleton for `central`, `upwind`, and `lf`

The surface skeleton uses the projected topology face pairing and aligns
neighbor face nodes by matching their 3D coordinates.  The penalty form is

```text
P = 0.5 * (v_n_sJ - C) * (qM - qP)
```

with

```text
v_n_sJ = J_face * (nr * u_tilde + ns * v_tilde)
central: C = 0
upwind:  C = |v_n_sJ|
lf:      C = alpha_lf * |v_n_sJ|
```

The lifted surface term reuses `engine.lift_boundary_penalty` with unit edge
lengths so this diagnostic does not double-count edge-length scaling in the
face metric-flux skeleton.

For `q = 1`, `nsub=16`, `N=4`, `R=1`, `u0=1`, `alpha=pi/4`:

```text
flux       alpha_lf   max_abs_rhs    rms_rhs       mean_rhs      max_abs_surface  rms_surface    max_abs_volume
central    1.0000    2.164720e-06   3.913355e-07  0.000000e+00  0.000000e+00     0.000000e+00   2.164720e-06
upwind     1.0000    2.164720e-06   3.913355e-07  0.000000e+00  0.000000e+00     0.000000e+00   2.164720e-06
lf         1.0000    2.164720e-06   3.913355e-07  0.000000e+00  0.000000e+00     0.000000e+00   2.164720e-06
lf         1.5000    2.164720e-06   3.913355e-07  0.000000e+00  0.000000e+00     0.000000e+00   2.164720e-06
```

Additional checks:

```text
max_face_match_error = 0.000000e+00
max_abs_penalty      = 0.000000e+00
```

Validation:

- `test_sphere_full_rhs_constant.py` passed.
- `compileall test/test_sphere_full_rhs_constant.py` passed.
- Upwind and LF with `alpha_lf=1` are identical for `q=1`.
- The full RHS remains at the projected-topology volume-only metric-divergence
  level, about `2.16e-06`.

Status:

- Phase 2.5 is completed.
- The q=1 surface term is exactly zero in this diagnostic because
  `qM - qP = 0` on the projected 3D face pairing.
- This is still a diagnostic skeleton, not a long-time sphere solver.

Next target:

- Move from constant-state q=1 to a nonconstant sphere surface-flux / full RHS
  diagnostic before any LSRK or long-time sphere simulation.
- Keep using projected 3D topology; do not return to unfolded layout
  connectivity for physical sphere face pairing.

### Phase 2.6 completed with diagnostic note: smooth nonconstant snapshot

Added `test/test_sphere_full_rhs_smooth_snapshot.py`.

This diagnostic keeps the Phase 2.5 projected-sphere full RHS skeleton and
replaces the constant state with a smooth nonconstant field:

```text
q = X + 0.5Y - 0.25Z
```

Setup:

- projected 3D octahedron sphere topology
- solid-body rotation velocity `V3D = omega x X`
- skew-symmetric manifold volume RHS
- Phase 2.5 face penalty skeleton for `central`, `upwind`, and `lf`
- no time integration
- no long-time sphere simulation

For `nsub=16`, `N=4`, `R=1`, `u0=1`, `alpha=pi/4`:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      mean_rhs      max_abs_volume  rms_volume   max_abs_surface  rms_surface   mass_rate     energy_rate
central    1.0000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
upwind     1.0000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.0000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.5000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
```

Additional checks:

```text
max upwind/LF(alpha=1) RHS difference      = 0.000000e+00
max upwind/LF(alpha=1.5) surface difference = 0.000000e+00
max_face_match_error                        = 0.000000e+00
max_abs_penalty                             = 0.000000e+00
```

Validation:

- `test_sphere_full_rhs_smooth_snapshot.py` passed.
- `compileall test/test_sphere_full_rhs_smooth_snapshot.py` passed.
- All reported RHS, volume, surface, mass-rate, and energy-rate values are
  finite.
- Upwind and LF with `alpha_lf=1` match exactly for this snapshot.
- The discrete mass rate is near machine precision.

Diagnostic note:

- `LF(alpha_lf=1.5)` does not differ from upwind in this smooth snapshot.
- This is expected for a continuous nodal field evaluated from the same
  physical 3D coordinates on both sides of each projected shared face:
  `qM - qP = 0`, so the jump-based surface penalty is zero for all flux types.
- Therefore Phase 2.6 verifies the smooth full-RHS volume path and confirms
  continuous-trace consistency, but it does not exercise dissipative flux
  separation.

Status:

- Phase 2.6 is completed as a smooth nonconstant snapshot diagnostic.
- A short-time LSRK sphere test should wait until one additional diagnostic
  exercises nonzero interelement jumps, for example a deliberately trace-jumped
  snapshot or a one-step RHS/flux probe that can distinguish upwind from
  stronger LF dissipation without starting a long-time simulation.

### Phase 2.7 completed: sphere jump-flux diagnostic

Added `test/test_sphere_flux_jump_diagnostic.py`.

This diagnostic deliberately adds an element-wise trace jump to the smooth
snapshot field:

```text
q_base = X + 0.5Y - 0.25Z
q      = q_base + eps * sign_K
eps    = 1e-2
sign_K = +1 for even element index, -1 for odd element index
```

The purpose is not exact-solution accuracy.  It is a surface-flux probe that
forces `qM - qP` to be nonzero on shared faces while still using the projected
3D octahedron topology and solid-body rotation velocity.

For `nsub=16`, `N=4`, `R=1`, `u0=1`, `alpha=pi/4`:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      max_abs_volume  rms_volume   max_abs_surface  rms_surface   mass_rate     energy_rate    max_jump_q    rms_jump_q
central    1.0000    2.989002e+00  6.125412e-01 7.288674e-01    4.208125e-01 2.482967e+00     4.453583e-01  6.641606e-15  1.109175e-10  2.000000e-02 1.632993e-02
upwind     1.0000    4.853700e+00  7.816204e-01 7.288674e-01    4.208125e-01 4.875371e+00     6.592748e-01 -4.989931e-03 -1.115925e-02  2.000000e-02 1.632993e-02
lf         1.0000    4.853700e+00  7.816204e-01 7.288674e-01    4.208125e-01 4.875371e+00     6.592748e-01 -4.989931e-03 -1.115925e-02  2.000000e-02 1.632993e-02
lf         1.5000    5.941709e+00  9.518491e-01 7.288674e-01    4.208125e-01 6.114180e+00     8.544105e-01 -7.484896e-03 -1.673887e-02  2.000000e-02 1.632993e-02
```

Additional checks:

```text
max upwind/LF(alpha=1) RHS difference         = 0.000000e+00
max upwind/LF(alpha=1.5) RHS difference       = 1.335862e+00
max upwind/LF(alpha=1.5) surface difference   = 1.335862e+00
max_face_match_error                          = 0.000000e+00
max_abs_penalty                               = 1.911791e-03
```

Validation:

- `test_sphere_flux_jump_diagnostic.py` passed.
- `compileall test/test_sphere_flux_jump_diagnostic.py` passed.
- All diagnostics are finite.
- The face jump is active: `max_abs_jump_q = 2.000000e-02`.
- Upwind and LF with `alpha_lf=1` match exactly.
- LF with `alpha_lf=1.5` separates from upwind.
- LF with `alpha_lf=1.5` has larger surface RMS than upwind.

Status:

- Phase 2.7 is completed.
- The sphere projected-topology volume RHS, continuous-trace behavior, and
  jump-flux dissipation probe have all passed as diagnostics.

Next target:

- It is now reasonable to consider a short-time sphere LSRK diagnostic.
- Keep the next run short and diagnostic-only; do not start a long-time sphere
  simulation or convergence study until the short-time diagnostic behavior is
  understood.

### Phase 2.8 completed: short-time sphere LSRK sanity diagnostic

Added `test/test_sphere_lsrk_short_sanity.py`.

This is a very short-time sanity test, not a production sphere solver, not a
one-period solid-body rotation test, and not a convergence study.

Setup:

- projected 3D octahedron sphere topology
- `nsub=4`, `N=4`
- solid-body rotation velocity
- smooth continuous initial condition:

```text
q0 = X + 0.5Y - 0.25Z
```

- existing LSRK54 stepper
- `T_final = 1e-3`
- `dt = 2.5e-4`
- flux cases: `central`, `upwind`, `lf alpha=1`, `lf alpha=1.5`

Short-time diagnostics:

```text
flux       alpha_lf   q_min          q_max          mass_final    mass_change   energy_final  energy_change  max_abs_dq    nonfinite
central    1.0000   -1.144630e+00   1.144630e+00   4.716308e-11  4.716315e-11  2.748824e+00 -2.211240e-09  7.290438e-04 False
upwind     1.0000   -1.144630e+00   1.144630e+00   1.106633e-10  1.106634e-10  2.748824e+00 -2.187651e-09  7.290439e-04 False
lf         1.0000   -1.144630e+00   1.144630e+00   1.106633e-10  1.106634e-10  2.748824e+00 -2.187651e-09  7.290439e-04 False
lf         1.5000   -1.144630e+00   1.144630e+00   1.420149e-10  1.420150e-10  2.748824e+00 -2.175917e-09  7.290439e-04 False
```

Additional check:

```text
max upwind/LF(alpha=1) final q difference = 0.000000e+00
```

Validation:

- `test_sphere_lsrk_short_sanity.py` passed.
- `compileall test/test_sphere_lsrk_short_sanity.py` passed.
- All flux cases ran to completion.
- No final-state NaN/Inf was detected.
- Upwind and LF with `alpha_lf=1` produced identical final states.
- The test intentionally does not compare to an exact solution.

Status:

- Phase 2.8 is completed as a short-time LSRK sanity diagnostic.
- Long-time sphere simulation is still not complete.
- A one-period rotation benchmark and convergence study are still future work,
  not implied by this sanity test.

### Phase 2.9a completed with caution: extended-time sphere rotation diagnostic

Added `test/test_sphere_solid_body_rotation_extended.py`.

This is an extended-time diagnostic for solid-body rotation on the projected
3D octahedron sphere topology.  It is not a one-period test, not a final
production solver, and not a convergence study.

Setup:

- projected 3D octahedron topology
- `nsub=4`, `N=4`
- solid-body rotation velocity `V3D = omega x X` with `||omega|| = 1`
- Gaussian bell initial condition:

```text
q0 = exp(-20 * ||X - normalize([1,1,1])||^2)
```

- LSRK54
- `dt = 5e-3`
- final times: `T = 1e-2`, `1e-1`, `1.0`
- flux cases: `central`, `upwind`, `lf alpha=1`, `lf alpha=1.5`

Results for `T = 1e-2`:

```text
flux       alpha_lf  steps  q_min          q_max          mass_initial  mass_final    mass_change   energy_initial  energy_final   energy_change  max_abs_dq   nonfinite
central    1.0000       2 -2.765493e-03   9.983510e-01   1.572674e-01  1.572770e-01  9.595107e-06  3.968749e-02   3.968741e-02 -7.905753e-08  4.235040e-02 False
upwind     1.0000       2 -2.622198e-03   9.983320e-01   1.572674e-01  1.572765e-01  9.085585e-06  3.968749e-02   3.968743e-02 -6.456717e-08  4.201997e-02 False
lf         1.0000       2 -2.622198e-03   9.983320e-01   1.572674e-01  1.572765e-01  9.085585e-06  3.968749e-02   3.968743e-02 -6.456717e-08  4.201997e-02 False
lf         1.5000       2 -2.554490e-03   9.983240e-01   1.572674e-01  1.572763e-01  8.856733e-06  3.968749e-02   3.968744e-02 -5.549911e-08  4.187200e-02 False
```

For `T = 1e-2`:

```text
max upwind/LF(alpha=1) final q difference = 0.000000e+00
```

Results for `T = 1e-1`:

```text
flux       alpha_lf  steps  q_min          q_max          mass_initial  mass_final    mass_change   energy_initial  energy_final   energy_change  max_abs_dq   nonfinite
central    1.0000      20 -5.487870e-02   8.419823e-01   1.572674e-01  1.576373e-01  3.698437e-04  3.968749e-02   3.973635e-02  4.885591e-05  3.828167e-01 False
upwind     1.0000      20 -4.163683e-02   8.412712e-01   1.572674e-01  1.577089e-01  4.414722e-04  3.968749e-02   3.974172e-02  5.422875e-05  3.797203e-01 False
lf         1.0000      20 -4.163683e-02   8.412712e-01   1.572674e-01  1.577089e-01  4.414722e-04  3.968749e-02   3.974172e-02  5.422875e-05  3.797203e-01 False
lf         1.5000      20 -3.641024e-02   8.412433e-01   1.572674e-01  1.577087e-01  4.413002e-04  3.968749e-02   3.974565e-02  5.815659e-05  3.785028e-01 False
```

For `T = 1e-1`:

```text
max upwind/LF(alpha=1) final q difference = 0.000000e+00
```

Results for `T = 1.0`:

```text
flux       alpha_lf  steps  q_min          q_max          mass_initial  mass_final    mass_change    energy_initial  energy_final   energy_change   max_abs_dq   nonfinite
central    1.0000     200 -1.828837e+00   3.060440e+00   1.572674e-01  1.235437e-01 -3.372376e-02  3.968749e-02   1.970031e-01  1.573156e-01   3.060428e+00 False
upwind     1.0000     200 -2.787589e-01   1.030181e+00   1.572674e-01  1.463482e-01 -1.091925e-02  3.968749e-02   4.046505e-02  7.775569e-04   1.030181e+00 False
lf         1.0000     200 -2.787589e-01   1.030181e+00   1.572674e-01  1.463482e-01 -1.091925e-02  3.968749e-02   4.046505e-02  7.775569e-04   1.030181e+00 False
lf         1.5000     200 -2.162508e-01   1.016881e+00   1.572674e-01  1.475802e-01 -9.687267e-03  3.968749e-02   3.951593e-02 -1.715629e-04   1.016881e+00 False
```

For `T = 1.0`:

```text
max upwind/LF(alpha=1) final q difference = 0.000000e+00
```

Validation:

- `test_sphere_solid_body_rotation_extended.py` passed.
- `compileall test/test_sphere_solid_body_rotation_extended.py` passed.
- All flux cases ran to completion for `T = 1e-2`, `1e-1`, and `1.0`.
- No NaN/Inf was detected.
- Upwind and LF with `alpha_lf=1` produced identical final states at every
  tested final time.
- Mass change stayed below the diagnostic sanity bound.

Interpretation:

- The extended diagnostic is stable enough to reach `T = 1.0` on the small
  projected sphere mesh.
- The central flux case shows clear oscillation and energy growth by `T=1.0`
  (`q_min=-1.828837`, `q_max=3.060440`, energy change `1.573156e-01`).
- Upwind and LF remain much more controlled at `T=1.0`.
- LF with `alpha_lf=1.5` is slightly more dissipative than upwind/LF(1), with
  lower energy at `T=1.0`.

Status:

- Phase 2.9a is completed with caution.
- It is reasonable to proceed to Phase 2.9b one-period testing, but start with
  the dissipative fluxes (`upwind` and `lf`) and keep `nsub=4, N=4` before
  attempting larger meshes.
- Central flux should be treated carefully in one-period testing because the
  `T=1.0` diagnostic already shows significant oscillation and energy growth.

### Phase 2.9b needs investigation: one-period sphere return diagnostic

Extended `test/test_sphere_solid_body_rotation_extended.py` with a one-period
return diagnostic.

This is a baseline one-period test only.  It is not a convergence study and
does not claim final production solver status.

Setup:

- projected 3D octahedron topology
- `nsub=4`, `N=4`
- `||omega|| = 1`
- `T_period = 2*pi`
- Gaussian bell initial condition:

```text
q0 = exp(-20 * ||X - normalize([1,1,1])||^2)
```

- central flux skipped for the main one-period run because Phase 2.9a already
  showed large central-flux oscillation and energy growth by `T=1.0`
- primary flux cases: `upwind`, `lf alpha=1`, `lf alpha=1.5`

Baseline one-period result with `dt = 5e-3`:

```text
flux       alpha_lf   T_final       steps  q_min          q_max          mass_initial  mass_final    mass_change    rel_mass      energy_initial  energy_final   energy_change   rel_energy    L2_return    Linf_return  nonfinite
upwind     1.0000    6.283185e+00   1257 -1.917254e+02   2.582572e+02   1.572674e-01 -3.454135e-01 -5.026810e-01 -3.196345e+00  3.968749e-02   7.203384e+02  7.202987e+02  1.814926e+04  3.795967e+01 2.582572e+02 False
lf         1.0000    6.283185e+00   1257 -1.917254e+02   2.582572e+02   1.572674e-01 -3.454135e-01 -5.026810e-01 -3.196345e+00  3.968749e-02   7.203384e+02  7.202987e+02  1.814926e+04  3.795967e+01 2.582572e+02 False
lf         1.5000    6.283185e+00   1257 -2.939814e+01   2.818745e+01   1.572674e-01 -9.090573e-02 -2.481732e-01 -1.578033e+00  3.968749e-02   2.471037e+01  2.467068e+01  6.216236e+02  7.033284e+00 2.939814e+01 False
```

For `dt = 5e-3`:

```text
max upwind/LF(alpha=1) final q difference = 0.000000e+00
```

Because the one-period result was very large despite no NaN/Inf, the diagnostic
was retried with `dt = 2.5e-3`.

Retry one-period result with `dt = 2.5e-3`:

```text
flux       alpha_lf   T_final       steps  q_min          q_max          mass_initial  mass_final    mass_change    rel_mass      energy_initial  energy_final   energy_change   rel_energy    L2_return    Linf_return  nonfinite
upwind     1.0000    6.283185e+00   2514 -1.917291e+02   2.582591e+02   1.572674e-01 -3.454075e-01 -5.026749e-01 -3.196307e+00  3.968749e-02   7.203470e+02  7.203073e+02  1.814948e+04  3.795989e+01 2.582591e+02 False
lf         1.0000    6.283185e+00   2514 -1.917291e+02   2.582591e+02   1.572674e-01 -3.454075e-01 -5.026749e-01 -3.196307e+00  3.968749e-02   7.203470e+02  7.203073e+02  1.814948e+04  3.795989e+01 2.582591e+02 False
lf         1.5000    6.283185e+00   2514 -2.939903e+01   2.818727e+01   1.572674e-01 -9.091125e-02 -2.481787e-01 -1.578068e+00  3.968749e-02   2.471054e+01  2.467086e+01  6.216279e+02  7.033309e+00 2.939903e+01 False
```

For `dt = 2.5e-3`:

```text
max upwind/LF(alpha=1) final q difference = 0.000000e+00
```

Validation:

- The one-period diagnostic ran to completion for `upwind`, `lf alpha=1`, and
  `lf alpha=1.5`.
- No NaN/Inf was detected.
- Upwind and LF with `alpha_lf=1` produced identical final states.
- `compileall test/test_sphere_solid_body_rotation_extended.py` passed.

Interpretation:

- Phase 2.9b is **needs investigation**, not completed as an acceptable
  one-period return result.
- Reducing `dt` from `5e-3` to `2.5e-3` did not materially improve the
  one-period result.
- The large mass loss, energy growth, overshoot/undershoot, and return error
  therefore do not appear to be primarily a time-step issue.
- LF with `alpha_lf=1.5` is more controlled than upwind/LF(1), but still has
  large mass loss and return error.

Next investigation target:

- Examine mass conservation and the manifold surface scaling/lift convention.
- Recheck whether the current surface penalty uses the correct face metric
  (`sJ`) and whether `J_face * (nr*u_tilde + ns*v_tilde)` is sufficient for
  the lifted face term over long integrations.
- Do not proceed to h-refinement, dt-refinement, or larger one-period runs
  until the one-period mass/energy behavior is understood.

### Phase 2.10 needs investigation: global conservation and surface scaling

Added `test/test_sphere_global_conservation_diagnostic.py`.

This diagnostic does not run time integration.  It evaluates the semi-discrete
RHS and computes the global mass rate

```text
mass_rate = sum_K sum_i J_Ki * w_i * RHS_Ki
```

split into volume and surface contributions.

Setup:

- projected 3D octahedron topology
- `nsub=4`, `N=4`
- solid-body rotation velocity
- flux cases: `central`, `upwind`, `lf alpha=1`, `lf alpha=1.5`
- q cases:

```text
constant: q = 1
smooth:   q = X + 0.5Y - 0.25Z
jump:     q = X + 0.5Y - 0.25Z + 1e-2 * sign_K
```

Surface scaling note recorded by the diagnostic:

```text
Current surface skeleton uses v_n_sJ = J_face * (nr*u_tilde + ns*v_tilde).
It does not compute a separate physical face line Jacobian sJ.
engine.lift_boundary_penalty is called with edge_lengths=np.ones(3), so the
lift still applies boundary quadrature weights but not the reference/physical
edge lengths.
```

Results for `q = 1`:

```text
flux       alpha_lf   mass_total    mass_volume   mass_surface  max_vol       max_surf      rms_vol       rms_surf      max_jump_q   rms_jump_q
central    1.0000    1.643930e-24  1.643930e-24  0.000000e+00  1.459898e-03 0.000000e+00 2.526749e-04 0.000000e+00 0.000000e+00 0.000000e+00
upwind     1.0000    1.643930e-24  1.643930e-24  0.000000e+00  1.459898e-03 0.000000e+00 2.526749e-04 0.000000e+00 0.000000e+00 0.000000e+00
lf         1.0000    1.643930e-24  1.643930e-24  0.000000e+00  1.459898e-03 0.000000e+00 2.526749e-04 0.000000e+00 0.000000e+00 0.000000e+00
lf         1.5000    1.643930e-24  1.643930e-24  0.000000e+00  1.459898e-03 0.000000e+00 2.526749e-04 0.000000e+00 0.000000e+00 0.000000e+00
```

Results for smooth continuous trace:

```text
flux       alpha_lf   mass_total    mass_volume   mass_surface  max_vol       max_surf      rms_vol       rms_surf      max_jump_q   rms_jump_q
central    1.0000    2.189221e-15  2.189221e-15  0.000000e+00  7.290434e-01 0.000000e+00 4.207562e-01 0.000000e+00 0.000000e+00 0.000000e+00
upwind     1.0000    2.189221e-15  2.189221e-15  0.000000e+00  7.290434e-01 0.000000e+00 4.207562e-01 0.000000e+00 0.000000e+00 0.000000e+00
lf         1.0000    2.189221e-15  2.189221e-15  0.000000e+00  7.290434e-01 0.000000e+00 4.207562e-01 0.000000e+00 0.000000e+00 0.000000e+00
lf         1.5000    2.189221e-15  2.189221e-15  0.000000e+00  7.290434e-01 0.000000e+00 4.207562e-01 0.000000e+00 0.000000e+00 0.000000e+00
```

Results for jumped trace:

```text
flux       alpha_lf   mass_total     mass_volume   mass_surface   max_vol       max_surf      rms_vol       rms_surf      max_jump_q   rms_jump_q
central    1.0000    1.439820e-15   2.338407e-15 -5.041540e-18  7.290447e-01 4.388738e-01 4.207562e-01 1.143303e-01 2.000000e-02 1.632993e-02
upwind     1.0000   -4.791135e-03   2.338407e-15 -4.791135e-03  7.290447e-01 8.209082e-01 4.207562e-01 1.705355e-01 2.000000e-02 1.632993e-02
lf         1.0000   -4.791135e-03   2.338407e-15 -4.791135e-03  7.290447e-01 8.209082e-01 4.207562e-01 1.705355e-01 2.000000e-02 1.632993e-02
lf         1.5000   -7.186702e-03   2.338407e-15 -7.186702e-03  7.290447e-01 1.038746e+00 4.207562e-01 2.215763e-01 2.000000e-02 1.632993e-02
```

Additional checks:

```text
max_face_match_error = 0.000000e+00
max_abs_penalty      = 7.531948e-03
```

Validation:

- `test_sphere_global_conservation_diagnostic.py` passed.
- `compileall test/test_sphere_global_conservation_diagnostic.py` passed.
- All diagnostics are finite.
- Projected face matching remains exact.

Interpretation:

- For `q=1` and the smooth continuous-trace field, global mass rate is near
  machine precision and surface contribution is zero.
- For the jumped field, volume contribution remains near machine precision:
  `mass_rate_volume = 2.338407e-15`.
- The nonzero global mass rate is surface dominated:
  `mass_rate_surface = -4.791135e-03` for upwind/LF(1), and
  `-7.186702e-03` for LF(1.5).
- Central flux remains globally conservative for the jumped diagnostic to
  near roundoff, while upwind/LF do not.

Status:

- Phase 2.10 is **needs investigation**.
- The one-period mass problem is consistent with a surface-term conservation
  issue, not a volume RHS mass issue.

Next investigation target:

- Revisit the sphere surface penalty in conservative flux form.
- Check face-pair cancellation using a single shared numerical flux per
  physical face.
- Check whether the current use of `J_face` should be replaced or augmented by
  a true physical face line metric `sJ`.
- Check the interaction between `v_n_sJ`, boundary quadrature weights, and
  `engine.lift_boundary_penalty(edge_lengths=np.ones(3))`.
- Do not start additional one-period, h-refinement, or dt-refinement runs until
  the surface conservation issue is resolved.

### Phase 2.11 completed: conservative sphere surface flux diagnostic

Extended `test/test_sphere_global_conservation_diagnostic.py` with a new
conservative shared-face surface helper.  The old surface skeleton is kept for
comparison.

The new helper:

- visits each projected 3D shared face once;
- aligns the two high-order face node orderings using the physical 3D face;
- uses the physical line-metric conormal flux
  `V3D dot (tau x n_surf)` instead of `J_face * (nr*u_tilde + ns*v_tilde)`;
- assigns equal and opposite paired face penalties before lifting;
- calls `engine.lift_boundary_penalty(edge_lengths=np.ones(3))`, so the lift
  applies edge quadrature weights while the physical line metric is already
  included in the face flux.

Diagnostic setup:

- projected 3D octahedron topology
- `nsub=4`, `N=4`
- solid-body rotation velocity
- q cases: constant, smooth continuous trace, jumped trace
- flux cases: central, upwind, LF(1), LF(1.5)

Old surface skeleton, jumped trace:

```text
flux       alpha_lf   mass_total     mass_volume   mass_surface   max_surf      rms_surf
central    1.0000    1.439820e-15   2.338407e-15 -5.041540e-18  4.388738e-01 1.143303e-01
upwind     1.0000   -4.791135e-03   2.338407e-15 -4.791135e-03  8.209082e-01 1.705355e-01
lf         1.0000   -4.791135e-03   2.338407e-15 -4.791135e-03  8.209082e-01 1.705355e-01
lf         1.5000   -7.186702e-03   2.338407e-15 -7.186702e-03  1.038746e+00 2.215763e-01
```

New conservative surface helper, jumped trace:

```text
flux       alpha_lf   mass_total     mass_volume   mass_surface   max_surf      rms_surf
central    1.0000    1.696560e-15   2.338407e-15 -5.528509e-18  1.262315e+00 2.726122e-01
upwind     1.0000    1.686151e-15   2.338407e-15 -1.387779e-17  2.597243e+00 3.803772e-01
lf         1.0000    1.686151e-15   2.338407e-15 -1.387779e-17  2.597243e+00 3.803772e-01
lf         1.5000    1.675743e-15   2.338407e-15  3.469447e-18  3.264707e+00 4.923137e-01
```

Additional Phase 2.11 checks:

```text
q=1 conservative mass_rate_total       = 1.643930e-24
smooth conservative mass_rate_total    = 2.189221e-15
jump upwind/LF(1) mass_rate_total      = 1.686151e-15
jump LF(1.5) mass_rate_total           = 1.675743e-15
max_face_match_error                   = 0.000000e+00
max_abs_vn_orientation_error           = 6.900538e-04
```

Validation:

- `test_sphere_global_conservation_diagnostic.py` passed.
- `compileall test/test_sphere_global_conservation_diagnostic.py` passed.
- All diagnostics are finite.
- The old upwind/LF jumped trace remains non-conservative, as expected.
- The new conservative helper restores global mass conservation for constant,
  smooth, and jumped q cases to roundoff.
- Upwind and LF with `alpha_lf=1` remain identical.
- LF with `alpha_lf=1.5` differs in surface dissipation while remaining mass
  conservative.

Interpretation:

- Phase 2.10's mass leak was caused by the old non-conservative surface
  skeleton, not by the volume RHS.
- The nonzero `max_abs_vn_orientation_error = 6.900538e-04` shows the two
  independently computed side metrics are not exactly opposite at this
  resolution/order.  The conservative helper therefore uses a single paired
  face metric flux, formed from the two orientations, before assigning equal
  and opposite SAT penalties.

Status:

- Phase 2.11 is completed as a conservation diagnostic/fix.
- A one-period retry should not use the old surface skeleton.  The next step
  should first wire the conservative surface helper into the sphere full RHS
  time-integration diagnostic and rerun short/extended-time checks before
  repeating the one-period return test.

### Phase 2.12 needs investigation: conservative helper integrated into time RHS

Added `core/rhs_sphere.py` as the reusable sphere surface helper module.

It provides:

- `compute_sphere_surface_penalty_old(...)`
- `compute_sphere_surface_penalty_conservative(...)`
- `compute_sphere_surface_penalty(..., surface_mode="old"|"conservative")`

Updated sphere RHS/time diagnostics so they can select `surface_mode`.  The
new RHS/time tests use `surface_mode="conservative"` by default, while the old
surface helper remains available for comparison.

Updated files:

- `test/test_sphere_full_rhs_constant.py`
- `test/test_sphere_full_rhs_smooth_snapshot.py`
- `test/test_sphere_flux_jump_diagnostic.py`
- `test/test_sphere_lsrk_short_sanity.py`
- `test/test_sphere_solid_body_rotation_extended.py`

Validation commands:

- `test_sphere_full_rhs_constant.py` passed.
- `test_sphere_full_rhs_smooth_snapshot.py` passed.
- `test_sphere_flux_jump_diagnostic.py` passed.
- `test_sphere_lsrk_short_sanity.py` passed.
- `test_sphere_solid_body_rotation_extended.py` passed as a diagnostic.
- `test_sphere_global_conservation_diagnostic.py` still passed.
- `compileall core/rhs_sphere.py` and the updated sphere tests passed.

Constant-state full RHS with conservative surface mode:

```text
flux       alpha_lf   max_abs_rhs    rms_rhs       max_abs_surface  rms_surface    max_abs_volume
central    1.0000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
upwind     1.0000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
lf         1.0000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
lf         1.5000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
```

Smooth continuous q snapshot with conservative surface mode:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      max_abs_surface  rms_surface   mass_rate     energy_rate
central    1.0000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
upwind     1.0000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.0000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.5000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
```

Jump-flux snapshot with conservative surface mode:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      max_abs_surface  rms_surface   mass_rate     energy_rate    max_jump_q
central    1.0000    6.805272e+00  1.161742e+00 6.299533e+00     1.082847e+00  6.559423e-15  2.916898e-04  2.000000e-02
upwind     1.0000    1.343968e+01  1.598955e+00 1.293394e+01     1.542462e+00  6.144662e-15 -2.385151e-02  2.000000e-02
lf         1.0000    1.343968e+01  1.598955e+00 1.293394e+01     1.542462e+00  6.144662e-15 -2.385151e-02  2.000000e-02
lf         1.5000    1.675688e+01  2.024959e+00 1.625114e+01     1.980607e+00  6.079881e-15 -3.592311e-02  2.000000e-02
```

Short-time LSRK sanity with conservative surface mode:

```text
flux       alpha_lf   q_min          q_max          mass_change    rel_mass      energy_change  max_abs_dq    nonfinite
central    1.0000   -1.144630e+00   1.144630e+00  -1.606869e-11  2.437624e+05 -4.054229e-09  7.290438e-04 False
upwind     1.0000   -1.144630e+00   1.144630e+00  -1.566276e-11  2.376044e+05 -3.998674e-09  7.290438e-04 False
lf         1.0000   -1.144630e+00   1.144630e+00  -1.566276e-11  2.376044e+05 -3.998674e-09  7.290438e-04 False
lf         1.5000   -1.144630e+00   1.144630e+00  -1.546491e-11  2.346029e+05 -3.971380e-09  7.290439e-04 False
```

Note: the relative mass value in the short-time linear-field test is not very
meaningful because the initial mass is near zero; the absolute mass change is
the useful number.

Extended-time `T=1.0` with conservative surface mode:

```text
flux       alpha_lf   q_min          q_max          mass_change    rel_mass      energy_change  max_abs_dq    nonfinite
central    1.0000   -4.440264e+44   1.579108e+45  5.806715e+42  3.692255e+43 4.321447e+87  1.579108e+45 False
upwind     1.0000   -1.728190e+01   4.626756e+02  4.862797e+01  3.092056e+02 4.568466e+03  4.625408e+02 False
lf         1.0000   -1.728190e+01   4.626756e+02  4.862797e+01  3.092056e+02 4.568466e+03  4.625408e+02 False
lf         1.5000   -5.211151e+00   4.485010e+00 -2.593106e-02 -1.648851e-01 7.850901e-01  5.346022e+00 False
```

Comparison to the old helper at `T=1.0`:

```text
old upwind mass_change = -1.091925e-02
new upwind mass_change =  4.862797e+01
```

Interpretation:

- The conservative helper fixes the semi-discrete global mass-rate diagnostic
  for jumped q, but the integrated time RHS is not yet acceptable.
- `T=1e-2` and the smooth short-time linear-field test remain finite.
- By `T=1e-1`, central is already highly oscillatory; upwind/LF(1) remain
  finite and close to the old qualitative behavior.
- By `T=1.0`, central and upwind/LF(1) are unstable in the conservative path.
- LF(1.5) remains much more controlled than upwind/LF(1), but still shows
  large dissipation and nontrivial mass loss.
- Upwind and LF with `alpha_lf=1` remain identical in all time diagnostics.

Status:

- Phase 2.12 is **needs investigation**.
- Do not proceed directly to Phase 2.13 one-period testing.
- The next step should investigate the conservative SAT sign/scaling and the
  interaction between the skew-symmetric volume form, line-metric face flux,
  and `lift_boundary_penalty` before repeating one-period rotation.

### Phase 2.12a completed: face metric and energy-rate diagnostic

Added `test/test_sphere_face_metric_and_energy_diagnostic.py`.

This diagnostic does not run time integration and does not modify the sphere
RHS.  It compares two candidate face metric fluxes on the projected 3D
octahedron topology:

```text
A = V3D dot (tau x n_surf)
B = J * (nr*u_tilde + ns*v_tilde)
```

It also evaluates semi-discrete mass and energy rates for:

- q cases: `constant`, `smooth`, `jump`, `gaussian`
- flux cases: `central`, `upwind`, `lf alpha=1`, `lf alpha=1.5`
- surface modes: `old`, `conservative`

Face metric comparison, `nsub=4`, `N=4`:

```text
num_face_nodes       = 960
max_abs(A-B)         = 3.895263e-01
rms(A-B)             = 1.595215e-01
ratio_min            = 2.000000e+00
ratio_max            = 2.828427e+00
ratio_mean           = 2.483829e+00
max_abs_ratio        = 2.828427e+00
sign_mismatch_count  = 0
max_face_match_error = 0.000000e+00
```

Top worst metric comparison locations all show the same sign and a ratio
`A/B = 2.828427` on the hypotenuse-like face.  Example:

```text
element = 57
face    = 1
node    = 7
patch   = 4
X       = (6.3960e-01, -4.2640e-01, 6.3960e-01)
A       = 6.025654e-01
B       = 2.130390e-01
ratio   = 2.828427e+00
sign    = 1/1
```

Interpretation:

- The two metric fluxes have consistent signs on this diagnostic.
- They are not on the same scale.  The ratio range `2` to `2.828427` strongly
  indicates reference-face normal / edge-parameter scaling differences.
- This supports the otherpeople inspection: their manifold helpers use
  `J*(nr*u_tilde + ns*v_tilde)` as the reference metric flux, while the
  conservative helper introduced in Phase 2.11 uses a physical conormal with a
  different scaling convention.

Energy-rate diagnostic, jumped q:

```text
mode           flux       alpha   mass_total     energy_total   energy_volume  energy_surface  max_surface   rms_surface
old            central    1.0     1.439820e-15   2.383151e-07   2.023672e-08   2.180784e-07   4.388738e-01  1.143303e-01
old            upwind     1.0    -4.791135e-03  -3.435585e-03   2.023672e-08  -3.435606e-03   8.209082e-01  1.705355e-01
old            lf         1.0    -4.791135e-03  -3.435585e-03   2.023672e-08  -3.435606e-03   8.209082e-01  1.705355e-01
old            lf         1.5    -7.186702e-03  -5.153497e-03   2.023672e-08  -5.153518e-03   1.038746e+00  2.215763e-01
conservative   central    1.0     1.696560e-15   3.155135e-04   2.023672e-08   3.154933e-04   1.262315e+00  2.726122e-01
conservative   upwind     1.0     1.686151e-15  -5.704031e-03   2.023672e-08  -5.704051e-03   2.597243e+00  3.803772e-01
conservative   lf         1.0     1.686151e-15  -5.704031e-03   2.023672e-08  -5.704051e-03   2.597243e+00  3.803772e-01
conservative   lf         1.5     1.675743e-15  -8.713804e-03   2.023672e-08  -8.713824e-03   3.264707e+00  4.923137e-01
```

For `q=1`, `smooth q`, and `gaussian q`, both old and conservative surface
terms are zero because the trace is continuous.  The corresponding mass and
energy rates are at roundoff / volume-only levels.

Diagnostic conclusion:

- Conservative upwind/LF energy-rate surface terms are **not positive** in the
  jumped-field semi-discrete diagnostic.  No anti-dissipative sign flag was
  triggered.
- The old helper is energy-dissipative for upwind/LF on the jumped field but
  not globally conservative.
- The conservative helper is globally conservative and even more dissipative
  in this static jumped-field diagnostic, but it uses a face metric whose scale
  differs from the external manifold convention by factors of `2` to
  `2.828427`.
- The Phase 2.12 extended-time instability is therefore more consistent with a
  face metric / reference scaling or volume-surface compatibility issue than
  with a simple upwind/LF sign reversal.

Validation:

- `test_sphere_face_metric_and_energy_diagnostic.py` passed.
- `compileall test/test_sphere_face_metric_and_energy_diagnostic.py` passed.

Next investigation target:

- Re-express the conservative shared-face helper using the external manifold
  metric convention `J*(nr*u_tilde + ns*v_tilde)` or rescale the physical
  conormal metric to the same reference-face convention before testing time
  integration again.
- Continue avoiding one-period rotation until the face metric scaling is made
  consistent with the volume/reference operator convention.

### Phase 2.12b completed: reference-face-length scaling diagnostic

Extended `test/test_sphere_face_metric_and_energy_diagnostic.py`.

This diagnostic does not run time integration and does not modify the core
sphere RHS.  It checks whether the physical conormal face metric from Phase
2.12a differs from the reference contravariant metric flux by the local
reference-face length:

```text
A        = V3D dot (tau x n_surf)
B        = J * (nr*u_tilde + ns*v_tilde)
Lref     = [2, 2*sqrt(2), 2]
A_scaled = A / Lref[face_id]
```

Face metric scaling comparison, `nsub=4`, `N=4`:

```text
num_face_nodes        = 960
max_abs(A-B)          = 3.895263e-01
rms(A-B)              = 1.595215e-01
max_abs(A/Lref-B)     = 8.326673e-17
rms(A/Lref-B)         = 1.986589e-17
ratio_min             = 2.000000e+00
ratio_max             = 2.828427e+00
ratio_mean            = 2.483829e+00
scaled_ratio_min      = 1.000000e+00
scaled_ratio_max      = 1.000000e+00
scaled_ratio_mean     = 1.000000e+00
sign_mismatch_count   = 0
scaled_sign_mismatch  = 0
max_face_match_error  = 0.000000e+00
```

Per-face scaling table:

```text
face  Lref           count  max|A-B|      max|A/L-B|    rms(A/L-B)    ratio             scaled_ratio
0     2.000000e+00   320    3.012779e-01  5.551115e-17  2.031350e-17  2..2              1..1
1     2.828427e+00   560    3.895263e-01  8.326673e-17  1.939632e-17  2.82843..2.82843  1..1
2     2.000000e+00    80    2.311118e-01  5.551115e-17  2.125480e-17  2..2              1..1
```

Interpretation:

- `A/Lref` matches `B` to roundoff.
- The Phase 2.12a ratio range `2` to `2.828427` is exactly the
  reference-face-length convention difference.
- The metric issue is therefore a scaling convention mismatch, not a sign
  mismatch.

Added a diagnostic-only `conservative_scaled` surface variant inside
`test_sphere_face_metric_and_energy_diagnostic.py`.  It uses the scaled
physical conormal metric `A/Lref` as the shared face metric flux.  This variant
is intentionally not wired into the production/time RHS yet.

Semi-discrete diagnostic with `surface_mode = conservative_scaled`:

```text
q_case    flux       alpha   mass_total     energy_surface  max_surface   rms_surface
jump      central    1.0     2.334938e-15   1.346452e-04    5.580711e-01  1.186105e-01
jump      upwind     1.0     1.967176e-15  -2.475772e-03    1.147132e+00  1.658934e-01
jump      lf         1.0     1.967176e-15  -2.475772e-03    1.147132e+00  1.658934e-01
jump      lf         1.5     1.987993e-15  -3.780981e-03    1.441662e+00  2.147444e-01
gaussian  central    1.0     4.652341e-16   0.000000e+00    0.000000e+00  0.000000e+00
gaussian  upwind     1.0     4.652341e-16   0.000000e+00    0.000000e+00  0.000000e+00
gaussian  lf         1.0     4.652341e-16   0.000000e+00    0.000000e+00  0.000000e+00
gaussian  lf         1.5     4.652341e-16   0.000000e+00    0.000000e+00  0.000000e+00
```

Diagnostic conclusion:

- `conservative_scaled` is globally conservative in this semi-discrete
  diagnostic for both jumped and Gaussian q cases.
- For jumped q, upwind and LF(1) remain identical.
- For jumped q, upwind/LF surface energy rates are nonpositive:
  `-2.475772e-03` for upwind/LF(1), and `-3.780981e-03` for LF(1.5).
- LF(1.5) is more dissipative than upwind/LF(1), as expected.
- For Gaussian q, the trace is continuous and the surface term is zero.
- No anti-dissipative upwind/LF surface-energy flag was triggered.

Validation:

- `test_sphere_face_metric_and_energy_diagnostic.py` passed.
- `compileall test/test_sphere_face_metric_and_energy_diagnostic.py` passed.

Next investigation target:

- Promote the scaled conservative face metric convention from diagnostic-only
  into a reusable sphere surface helper, then rerun short and extended time
  diagnostics.
- Do not rerun one-period rotation until the scaled conservative helper is
  tested in short/extended time and remains finite with acceptable mass and
  energy behavior.

### Phase 2.12c needs investigation: conservative_scaled wired into sphere RHS

Updated `core/rhs_sphere.py` with a reusable
`surface_mode="conservative_scaled"` path.

The new core mode:

- keeps the old modes `surface_mode="old"` and `surface_mode="conservative"`
  available for comparison;
- visits each projected 3D shared face once;
- aligns paired high-order face nodes by physical coordinates;
- computes the shared metric velocity using
  `[V3D dot (tau x n_surf)] / Lref`, which matches
  `J*(nr*u_tilde + ns*v_tilde)` from Phase 2.12b;
- assigns equal and opposite conservative paired penalties before lifting;
- keeps `engine.lift_boundary_penalty(edge_lengths=np.ones(3))`, so the
  reference-face metric scaling is carried by the face flux.

Updated diagnostics to use `surface_mode="conservative_scaled"` by default:

- `test/test_sphere_full_rhs_constant.py`
- `test/test_sphere_full_rhs_smooth_snapshot.py`
- `test/test_sphere_flux_jump_diagnostic.py`
- `test/test_sphere_lsrk_short_sanity.py`
- `test/test_sphere_solid_body_rotation_extended.py`

The Phase 2.12b metric/energy diagnostic now also calls the core
`conservative_scaled` helper instead of a local diagnostic-only implementation.

Validation:

- `compileall core/rhs_sphere.py` and the updated sphere diagnostics passed.
- `test_sphere_full_rhs_constant.py` passed.
- `test_sphere_full_rhs_smooth_snapshot.py` passed.
- `test_sphere_flux_jump_diagnostic.py` passed.
- `test_sphere_lsrk_short_sanity.py` passed.
- `test_sphere_face_metric_and_energy_diagnostic.py` passed.
- `test_sphere_solid_body_rotation_extended.py` completed as a diagnostic.

Constant-state q=1 full RHS, `nsub=16`, `N=4`:

```text
flux       alpha_lf   max_abs_rhs    rms_rhs       max_abs_surface  rms_surface    max_abs_volume
central    1.0000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
upwind     1.0000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
lf         1.0000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
lf         1.5000    2.164720e-06   3.913355e-07  0.000000e+00     0.000000e+00   2.164720e-06
```

Smooth continuous q snapshot, `nsub=16`, `N=4`:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      max_abs_surface  rms_surface   mass_rate     energy_rate
central    1.0000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
upwind     1.0000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.0000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.5000    7.288674e-01  4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
```

Jump-flux snapshot, `nsub=16`, `N=4`, `eps=1e-2`:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      max_abs_surface  rms_surface   mass_rate     energy_rate    max_jump_q
central    1.0000    3.291933e+00  6.271320e-01 2.786195e+00     4.649838e-01  6.821800e-15  1.244864e-04  2.000000e-02
upwind     1.0000    6.221044e+00  7.851208e-01 5.715306e+00     6.626976e-01  5.987181e-15 -1.022069e-02  2.000000e-02
lf         1.0000    6.221044e+00  7.851208e-01 5.715306e+00     6.626976e-01  5.987181e-15 -1.022069e-02  2.000000e-02
lf         1.5000    7.685600e+00  9.494938e-01 7.179861e+00     8.510058e-01  8.012091e-15 -1.539328e-02  2.000000e-02
```

Jump-flux checks:

```text
max upwind/LF(alpha=1) RHS difference = 0.000000e+00
max upwind/LF(alpha=1.5) RHS difference = 1.464555e+00
max_face_match_error = 0.000000e+00
max_abs_penalty = 1.631816e-03
```

Short-time LSRK sanity, `T=1e-3`, `dt=2.5e-4`, `nsub=4`, `N=4`:

```text
flux       alpha_lf   q_min          q_max          mass_change    rel_mass      energy_change  max_abs_dq    nonfinite
central    1.0000   -1.144630e+00   1.144630e+00 -7.745811e-12  1.175041e+05 -4.051596e-09  7.290438e-04 False
upwind     1.0000   -1.144630e+00   1.144630e+00 -6.671684e-12  1.012096e+05 -4.026150e-09  7.290438e-04 False
lf         1.0000   -1.144630e+00   1.144630e+00 -6.671684e-12  1.012096e+05 -4.026150e-09  7.290438e-04 False
lf         1.5000   -1.144630e+00   1.144630e+00 -6.146070e-12  9.323600e+04 -4.013531e-09  7.290438e-04 False
```

Note: the relative mass value is not meaningful in the short linear-field test
because the initial mass is near zero.  The absolute mass change is the useful
quantity.

Extended-time Gaussian bell, `T=1.0`, `dt=5e-3`, `nsub=4`, `N=4`:

```text
flux       alpha_lf   q_min          q_max          mass_change    rel_mass      energy_change  max_abs_dq    nonfinite
central    1.0000   -4.254033e+25   2.138257e+26  1.319698e+24  8.391423e+24 1.031347e+50  2.138257e+26 False
upwind     1.0000   -2.084422e+01   4.169237e+02  4.477646e+01  2.847154e+02 3.979253e+03  4.167888e+02 False
lf         1.0000   -2.084422e+01   4.169237e+02  4.477646e+01  2.847154e+02 3.979253e+03  4.167888e+02 False
lf         1.5000   -1.207574e+01   3.046612e+01  2.429280e-01  1.544681e+00 1.120198e+01  3.046512e+01 False
```

Extended-time comparison:

```text
old helper upwind mass_change              ~= -1.091925e-02
raw conservative helper upwind mass_change ~=  4.862797e+01
conservative_scaled upwind mass_change     =  4.477646e+01
```

Interpretation:

- `conservative_scaled` successfully moves the shared-face conservative helper
  onto the same reference metric scale as the volume/reference operators.
- Static q=1, smooth, and jumped diagnostics remain finite and consistent.
- Upwind and LF with `alpha_lf=1` remain exactly identical in the diagnostics.
- LF with `alpha_lf=1.5` is more dissipative than upwind/LF(1), as expected.
- However, the extended-time `T=1.0` result is still not acceptable:
  upwind/LF(1) still grow to large overshoot/undershoot and large positive mass
  change, and central remains catastrophically unstable.
- The scaled helper improves the raw conservative upwind mass change only from
  about `+4.86e+01` to `+4.48e+01`; this is not a meaningful stabilization.

Status:

- Phase 2.12c is **needs investigation**.
- Do not proceed to Phase 2.13 one-period baseline yet.

Next investigation target:

- The remaining instability is likely not only the reference-face-length
  scaling.  Investigate the volume-surface sign convention, SAT sign, and
  whether the lifted conservative penalty is compatible with the
  skew-symmetric volume RHS and the LSRK callback sign convention.
- Keep `old`, `conservative`, and `conservative_scaled` modes available for
  side-by-side diagnostics.

### Phase 2.12d completed: small-mesh operator spectrum diagnostic

Added `test/test_sphere_operator_spectrum_diagnostic.py`.

This diagnostic does not run time integration and does not modify the RHS.  It
builds the linear semi-discrete sphere RHS operator

```text
q_t = L q
```

by applying the existing sphere RHS callback to every basis vector on a small
projected sphere mesh.

Setup:

- projected 3D octahedron topology
- `nsub=2`, `N=2`
- total degrees of freedom: `320`
- solid-body rotation velocity
- surface modes: `old`, `conservative`, `conservative_scaled`
- fluxes: `central`, `upwind`, `lf alpha=1`, `lf alpha=1.5`

Spectrum summary:

```text
surface_mode          flux       alpha   ndof   max_real      min_real       spectral_radius  num_Re>1e-10
old                   central    1.0     320    6.952305e-01 -6.952305e-01   6.236560e+00    133
old                   upwind     1.0     320    3.107124e-01 -4.551445e+00   5.734149e+00     95
old                   lf         1.0     320    3.107124e-01 -4.551445e+00   5.734149e+00     95
old                   lf         1.5     320    2.508078e-01 -7.358048e+00   7.358048e+00     93
conservative          central    1.0     320    1.311416e+01 -1.278436e+01   1.311416e+01    167
conservative          upwind     1.0     320    1.777764e+00 -2.362833e+01   2.362833e+01    134
conservative          lf         1.0     320    1.777764e+00 -2.362833e+01   2.362833e+01    134
conservative          lf         1.5     320    1.092738e+00 -2.910993e+01   2.910993e+01    124
conservative_scaled   central    1.0     320    7.627138e+00 -7.615905e+00   7.627138e+00    162
conservative_scaled   upwind     1.0     320    1.655778e+00 -1.301352e+01   1.301352e+01    132
conservative_scaled   lf         1.0     320    1.655778e+00 -1.301352e+01   1.301352e+01    132
conservative_scaled   lf         1.5     320    1.362608e+00 -1.572878e+01   1.572878e+01    139
```

Top rightmost eigenvalues:

```text
old upwind:
  3.107124e-01 +/- 2.700554e+00 i
  2.626088e-01 +/- 3.352593e+00 i

conservative upwind:
  1.777764e+00 + 0.000000e+00 i
  1.520862e+00 +/- 2.936715e+00 i

conservative_scaled upwind:
  1.655778e+00 + 0.000000e+00 i
  1.545549e+00 +/- 2.756956e+00 i
```

Upwind versus LF(1) operator equality:

```text
old:                   max |L_upwind - L_lf1| = 0.000000e+00
conservative:          max |L_upwind - L_lf1| = 0.000000e+00
conservative_scaled:   max |L_upwind - L_lf1| = 0.000000e+00
```

Validation:

- `test_sphere_operator_spectrum_diagnostic.py` passed.
- `compileall test/test_sphere_operator_spectrum_diagnostic.py` passed.
- All assembled operators and eigenvalues were finite.

Interpretation:

- The extended-time instability is consistent with positive-real-part
  semi-discrete operator modes, not primarily with timestep size.
- All tested modes have some positive-real-part eigenvalues on this small
  diagnostic mesh, including `old`.
- The conservative modes are much more unstable in the operator spectrum:
  raw conservative upwind has `max_real = 1.777764e+00`, and
  conservative_scaled upwind has `max_real = 1.655778e+00`, compared with
  old upwind `max_real = 3.107124e-01`.
- LF(1.5) shifts many eigenvalues left and reduces `max_real` for the
  conservative modes, but it still leaves positive-real-part modes:
  `1.092738e+00` for raw conservative and `1.362608e+00` for
  conservative_scaled.
- This supports the Phase 2.12c conclusion that the problem is in the
  semi-discrete volume-surface/SAT operator structure, not in one-period
  integration, h-convergence, or RK convergence.

Next investigation target:

- Compare the sign convention of the volume RHS and surface SAT by assembling
  volume-only and surface-only spectra.
- Check whether the time RHS should use `volume_rhs + surface_rhs` or
  `volume_rhs - surface_penalty` under the current lift convention.
- Still do not proceed to one-period rotation until the positive-real-part
  operator modes are understood and reduced.

### Final report focus update: sphere convergence plus planar verification

The sphere one-period / long-time conservative SAT path is now documented as a
limitation and future-work item.  Phase 2.12d showed that the sphere
extended-time instability is consistent with positive-real-part semi-discrete
operator modes.  Do not continue chasing one-period sphere rotation,
h-convergence, or RK convergence on the current sphere RHS until the
volume-surface SAT operator issue is resolved.

For the final report, use:

- sphere short-time Gaussian rotation convergence as the main sphere result;
- planar periodic advection RK and h-convergence as prerequisite verification
  of the planar split-form RHS and LSRK54 implementation;
- sphere geometry / topology / metric / flux diagnostics as supporting
  manifold validation and limitation analysis;
- the failed long-time / one-period conservative SAT path as explicit future
  work, not as the final reported solver result.

### Final planar convergence diagnostics completed

Added `test/test_final_planar_convergence.py`.

This file contains two final-report-oriented planar periodic advection
diagnostics.  Both use the existing planar split-form RHS, periodic trace
exchange, upwind flux, and LSRK54 time stepper.  No sphere RHS, one-period
sphere test, conservative SAT fix, or flux-switching refactor was performed.

#### Final RK temporal convergence

Setup:

- planar periodic advection
- exact profile: `sin(2*pi*(x + y - 2t))`
- fixed mesh: `divisions=4`
- polynomial degree: `N=3`
- quadrature parameter: `n_quad=4`
- final time: `T=5.0e-2`
- reference solution: same semi-discrete operator with
  `reference_dt = 1.25e-4`

Results:

```text
dt             steps   L2_error       order   Linf_error     order
3.846154e-03      13   3.289287e-08   -       1.904000e-07   -
2.000000e-03      25   2.311961e-09   4.0603  1.322944e-08   4.0780
1.000000e-03      50   1.415729e-10   4.0295  8.015456e-10   4.0448
5.000000e-04     100   8.729499e-12   4.0195  4.910317e-11   4.0289
```

Interpretation:

- The LSRK54 path shows the expected approximately fourth-order temporal
  convergence on the fixed planar periodic semi-discrete problem.

#### Final h-convergence

Setup:

- planar periodic advection
- exact profile: `sin(2*pi*(x + y - 2t))`
- polynomial degree: `N=3`
- quadrature parameter: `n_quad=4`
- refinement: `n_div = [4, 8, 16, 32]`
- final time: `T=1.0e-2`
- time step from CFL formula with `CFL=5.0e-2`

Results:

```text
n_div   h             dt            steps   actual_time   L2_error       order   Linf_error     order
4       2.500000e-01  9.090909e-04     11   1.000000e-02  6.766985e-04   -       4.182607e-03   -
8       1.250000e-01  4.761905e-04     21   1.000000e-02  5.377724e-05   3.6534  5.296815e-04   2.9812
16      6.250000e-02  2.439024e-04     41   1.000000e-02  3.811468e-06   3.8186  3.548681e-05   3.8998
32      3.125000e-02  1.219512e-04     82   1.000000e-02  2.478163e-07   3.9430  2.214076e-06   4.0025
```

Interpretation:

- The planar periodic h-refinement result shows high-order spatial
  convergence for `N=3`, approaching fourth order on the finest refinement
  pair.
- The finest-pair observed orders are `3.9430` in L2 and `4.0025` in Linf.

Validation:

- `test_final_planar_convergence.py` passed.
- `compileall test/test_final_planar_convergence.py` passed.

Final-report status:

- Completed prerequisite numerical results: planar RK temporal convergence and
  planar h-convergence.
- Sphere results should be presented as successful manifold geometry,
  topology, q=1 RHS, smooth/jump flux diagnostics, and short-time sanity
  checks, with the one-period conservative SAT path explicitly marked as
  future work due to positive-real-part semi-discrete operator modes.

### Final sphere convergence diagnostics completed

Added `test/test_final_sphere_convergence.py`.

This file provides the final-report sphere convergence diagnostics using the
projected 3D octahedron sphere topology.  It intentionally uses
`surface_mode="old"` as the main reported sphere path because that helper is
the currently more stable long/extended-time option and is closest to the
external `manifold_rhs.py` penalty structure.  The conservative and
conservative_scaled helpers remain documented as limitations/future work due
to positive-real-part semi-discrete operator modes.

No one-period `T=2*pi` sphere test was run.  No conservative SAT fix was made.
No unfolded topology was used.

Common setup:

- projected 3D octahedron sphere topology
- solid-body rotation velocity `V3D = omega x X`, `||omega|| = 1`
- Gaussian bell initial condition

```text
q0(X) = exp(-20 * ||X - normalize([1,1,1])||^2)
```

- exact short-time rotation solution

```text
q_exact(X,t) = exp(-20 * ||X - R(t) Xc||^2)
```

where `Xc = normalize([1,1,1])`.

#### Final sphere RK temporal convergence

Setup:

- fixed mesh: `nsub=4`
- polynomial degree: `N=3`
- quadrature parameter: `n_quad=4`
- final time: `T=1.0e-2`
- surface mode: `old`
- reference solution: same sphere semi-discrete operator with
  `reference_dt = 6.25e-5`

Results:

```text
dt             steps   L2_ref        order   Linf_ref      order   L2_exact      Linf_exact
2.000000e-03       5   8.860098e-12  -       2.311320e-10  -       1.127114e-03  3.013824e-02
1.000000e-03      10   5.499335e-13  4.0100  1.435607e-11  4.0090  1.127114e-03  3.013824e-02
5.000000e-04      20   3.425284e-14  4.0050  8.938961e-13  4.0054  1.127114e-03  3.013824e-02
2.500000e-04      40   2.139700e-15  4.0007  5.589973e-14  3.9992  1.127114e-03  3.013824e-02
```

Interpretation:

- The sphere LSRK54 path shows approximately fourth-order temporal convergence
  against the very-small-dt reference solution.
- The exact-solution error is saturated at about `L2_exact = 1.127114e-03`
  on this fixed `nsub=4`, `N=3` sphere mesh, so the reference solution is used
  to isolate temporal convergence.

#### Final sphere short-time h-convergence

Setup:

- polynomial degree: `N=3`
- quadrature parameter: `n_quad=4`
- refinement: `nsub = [2, 4, 8, 16]`
- final time: `T=5.0e-3`
- surface mode: `old`
- exact Gaussian rotation solution
- time step capped by `dt <= 2.5e-4` and further reduced on the finest mesh
  using `dt ~ h^2`

Results:

```text
nsub   h             dt            steps   actual_time   L2_error       order   Linf_error     order
2      7.653669e-01  2.500000e-04     20   5.000000e-03  1.366956e-03   -       1.986507e-02   -
4      3.203645e-01  2.500000e-04     20   5.000000e-03  5.689061e-04   1.0066  1.563801e-02   0.2747
8      1.417780e-01  2.500000e-04     20   5.000000e-03  1.096485e-04   2.0197  6.042480e-03   1.1665
16     6.655587e-02  8.771930e-05     57   5.000000e-03  1.296339e-05   2.8234  8.911974e-04   2.5310
```

Interpretation:

- This is a short-time sphere h-convergence diagnostic, not a long-time or
  one-period sphere benchmark.
- The error decreases monotonically with refinement.
- The finest refinement pair reaches observed order `2.8234` in L2 and
  `2.5310` in Linf.
- This is not yet ideal fourth-order spatial convergence for `N=3`; treat it
  as a successful short-time sphere convergence trend but still pre-asymptotic
  / limited by the current sphere surface/metric/RHS path.

Validation:

- `test_final_sphere_convergence.py` passed.
- `compileall test/test_final_sphere_convergence.py` passed.

Final-report status:

- Main sphere result: short-time Gaussian rotation convergence on projected
  sphere topology using the old surface helper.
- Supporting result: sphere RK temporal convergence shows the expected
  fourth-order LSRK54 behavior against a small-dt reference.
- Limitation: sphere h-convergence is monotone but not yet ideal high-order;
  one-period / long-time conservative SAT remains future work.

### Exchange-cache old-mode diagnostic completed

Added `test/test_sphere_exchange_cache_old_diagnostic.py`.

This is a diagnostic-only comparison inspired by the external
`manifold_rhs_exchange` pipeline.  It does not replace the main RHS and does
not modify the conservative or conservative_scaled helpers.

The new diagnostic builds a small old-style sphere exchange cache containing:

- face trace pairing from the projected 3D topology;
- paired high-order face-node ordering;
- face quadrature weights;
- side-local `J_face`, `u_tilde_face`, and `v_tilde_face`;
- reference face normals used by the current old mode.

It then keeps the old penalty formula unchanged:

```text
vn_sJ   = J_face * (nr*u_tilde + ns*v_tilde)
C       = 0, |vn_sJ|, or alpha_lf*|vn_sJ|
penalty = 0.5 * (vn_sJ - C) * (qM - qP)
```

Comparison results:

```text
q=1 RHS comparison, nsub=8, N=4:
flux       alpha   current_max_rhs  exchange_max_rhs  max_rhs_diff  max_surface_diff
central    1.0    4.666533e-05     4.666533e-05     0.000000e+00  0.000000e+00
upwind     1.0    4.666533e-05     4.666533e-05     0.000000e+00  0.000000e+00
lf         1.0    4.666533e-05     4.666533e-05     0.000000e+00  0.000000e+00
lf         1.5    4.666533e-05     4.666533e-05     0.000000e+00  0.000000e+00
```

```text
jump-flux comparison, nsub=8, N=4, eps=1e-2:
max_abs_jump_q = 2.000000e-02
rms_jump_q     = 1.632993e-02

flux       alpha   current_surf   exchange_surf   current_mass   exchange_mass   max_rhs_diff
central    1.0    1.025478e+00   1.025478e+00   3.346282e-15   3.346282e-15   0.000000e+00
upwind     1.0    1.992467e+00   1.992467e+00  -4.973044e-03  -4.973044e-03   0.000000e+00
lf         1.0    1.992467e+00   1.992467e+00  -4.973044e-03  -4.973044e-03   0.000000e+00
lf         1.5    2.498639e+00   2.498639e+00  -7.459566e-03  -7.459566e-03   0.000000e+00
```

```text
small-mesh spectrum comparison, nsub=2, N=2:
flux       alpha   ndof   current_max_real  exchange_max_real  max_operator_diff
central    1.0     320    6.952305e-01     6.952305e-01      0.000000e+00
upwind     1.0     320    3.107124e-01     3.107124e-01      0.000000e+00
lf         1.0     320    3.107124e-01     3.107124e-01      0.000000e+00
lf         1.5     320    2.508078e-01     2.508078e-01      0.000000e+00
```

```text
short-time Gaussian comparison, nsub=4, N=3, n_quad=4, T=1e-2, dt=2.5e-4:
flux       alpha   current_L2     exchange_L2    current_Linf   exchange_Linf  max_q_diff
upwind     1.0    1.127114e-03   1.127114e-03   3.013824e-02   3.013824e-02   0.000000e+00
lf         1.0    1.127114e-03   1.127114e-03   3.013824e-02   3.013824e-02   0.000000e+00
lf         1.5    1.113667e-03   1.113667e-03   2.972679e-02   2.972679e-02   0.000000e+00
```

Interpretation:

- The exchange-cache old-mode diagnostic exactly reproduces the current
  old-mode RHS, spectrum, and short-time Gaussian result to roundoff.
- This confirms that adding a cache/trace-policy layer alone does not change
  the operator; it is primarily an organization and performance/stability
  infrastructure step.
- The external pipeline may still differ through its global LF definition,
  fixed N=4 operators, and trace/lift implementation details, but the current
  old-mode formula is already equivalent to the minimal cached version tested
  here.

Validation:

- `test_sphere_exchange_cache_old_diagnostic.py` passed.
- `compileall test/test_sphere_exchange_cache_old_diagnostic.py` passed.
