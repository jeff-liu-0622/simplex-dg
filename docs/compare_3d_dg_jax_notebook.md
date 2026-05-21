# Comparison: `otherpeople/3D_DG_JAX.ipynb` vs current sphere RHS path

This note compares the newly added notebook:

- `otherpeople/3D_DG_JAX.ipynb`

against the current project sphere path:

- `core/rhs_sphere.py`
- `test/test_sphere_exchange_cache_old_diagnostic.py`
- `test/test_final_sphere_convergence.py`

Per request, this comparison ignores pure acceleration differences such as
JAX JIT compilation, GPU execution, `lax.scan`, reduced dispatch overhead, and
host/device transfer strategy.  The focus here is on mathematical, geometric,
operator, flux, and diagnostic differences.

## Short conclusion

The notebook is not just a faster version of our current old-mode sphere RHS.
It uses the same general old-style side-local SAT idea,

```text
penalty = 0.5 * (vn_sJ - C) * (qM - qP),
```

but several important non-speed choices differ:

1. It uses a pure divergence-form volume RHS, not our skew-symmetric/split
   manifold volume RHS.
2. It uses an analytic projected-sphere metric formula, while our main path
   currently differentiates mapped nodal coordinates.
3. It builds a custom SBP/collocation reference operator and lift convention,
   not the same `ReferenceElement` lift path used in `core/operators.py`.
4. Its face normal convention is non-unit reference conormal style, whereas
   our old mode uses the unit-normal convention in `REFERENCE_FACE_NORMALS`.
5. Its LF flux is a global-speed LF, while our current `lf` is pointwise and
   exactly equals upwind when `alpha_lf=1`.
6. It runs a physically scaled 12-day solid-body rotation problem; our final
   sphere results intentionally remain short-time diagnostics.

So the notebook is best read as a different old-style manifold DG pipeline,
not merely as an accelerated drop-in for the current solver.

## 1. Mesh and topology

Both implementations use a projected 3D octahedron sphere topology.

Notebook:

- builds the six octahedron vertices directly;
- subdivides each original octahedron face in barycentric coordinates;
- projects every generated vertex onto the sphere;
- merges shared vertices with a rounded coordinate key:

```text
key = (round(x, 8), round(y, 8), round(z, 8))
```

Current project:

- uses `create_projected_octahedron_sphere_mesh(...)`;
- keeps `patch_ids`;
- constructs connectivity from projected 3D shared vertices;
- has already verified shared-face high-order node mismatch equals zero.

Main non-speed difference:

- The notebook relies on rounded projected coordinate keys for vertex merging.
- The current core topology has a more formal projected mesh interface and
  keeps patch identifiers for diagnostics.

This difference is probably not the source of the main RHS behavior difference
as long as shared faces are continuous, but it is a topology implementation
difference.

## 2. Face trace pairing

Notebook:

- builds global trace maps `vmapM` and `vmapP`;
- stores trace indices as flattened global nodal indices;
- pairs the neighbor trace by reversing neighbor face nodes:

```text
vmapP[face] = neighbor_indices[::-1]
```

Current project:

- current old mode finds neighbor face ordering by comparing physical high-order
  face coordinates with direct/reversed order;
- the new exchange-cache diagnostic stores this pairing in advance, but still
  uses physical coordinate alignment.

Main non-speed difference:

- The notebook assumes a consistent topological face orientation and always
  reverses neighbor face nodes.
- The current project verifies the direct/reverse choice using the actual 3D
  physical face nodes.

This is a robustness difference.  Our latest exchange-cache diagnostic showed
that caching the current physical pairing gives exactly the same operator as
current old mode.

## 3. Geometry metrics

Notebook has two metric paths:

1. `compute_metrics_3d(...)` differentiates mapped coordinates.
2. `compute_exact_metrics_3d(...)` derives analytic metrics for the radial
   projection from a flat octahedron triangle to the sphere.

The simulation path uses:

```text
J, a1, a2 = compute_exact_metrics_3d(nodes, EToV, xi_ref, eta_ref, R_sphere=1.0)
```

Current project:

- uses `compute_manifold_geometry(...)`;
- computes `a1 = Dr X`, `a2 = Ds X`;
- computes `J`, surface normal, and contravariant bases from the nodal mapped
  coordinates.

Main non-speed difference:

- The notebook's main simulation uses analytic projected-sphere metrics.
- Our main path uses numerical differentiation of the projected nodal map.

This can affect free-stream preservation, q=1 metric divergence, and long-time
stability.  It is one of the largest mathematical differences.

## 4. Physical scale and velocity convention

Notebook:

- stores the physical Earth radius:

```text
R_sphere = 6.37122e6
```

- builds the mesh and metrics on a unit sphere;
- computes physical coordinates `X,Y,Z` scaled by `R_sphere`;
- computes velocity on the unit sphere with a dimensionless angular speed:

```text
u0 = 2*pi / (12*86400)
```

- uses this to represent one revolution in 12 days.

Current project:

- final sphere diagnostics use `R=1`;
- velocity convention is `V3D = omega x X` with `||omega||=1`;
- final reported sphere convergence is intentionally short-time:
  `T=1e-2` for temporal convergence and `T=5e-3` for h-convergence.

Main non-speed difference:

- The notebook is a physically scaled long-time solid-body rotation setup.
- The current final sphere result is a short-time unit-sphere diagnostic because
  long-time/one-period behavior is still a documented limitation.

## 5. Volume RHS form

Notebook RHS:

```text
c_r = J * u_tilde
c_s = J * v_tilde
volume_term = -[Dr(c_r*q) + Ds(c_s*q)] / J
```

This is a pure divergence-form volume term.

Current project manifold volume RHS:

```text
R_vol =
  -1/(2J) [Dr(J u_tilde q) + Ds(J v_tilde q)]
  -1/2 [u_tilde Dr(q) + v_tilde Ds(q)]
  -q/(2J) [Dr(J u_tilde) + Ds(J v_tilde)]
```

This is the skew-symmetric/split form.

Main non-speed difference:

- The notebook does not use our current skew-symmetric volume formula.
- It uses a divergence form with precomputed metric fluxes.

This is a fundamental operator difference.  It can change energy behavior,
aliasing behavior, and the spectrum even if the surface SAT formula looks
similar.

## 6. Reference operator construction

Notebook:

- builds raw Dubiner Vandermonde and gradient Vandermonde matrices;
- orthonormalizes the nodal basis through a Cholesky factor of the approximate
  modal mass;
- builds face extraction matrix `E`;
- builds boundary matrices:

```text
B1 = E.T @ diag(face_weights * nr) @ E
B2 = E.T @ diag(face_weights * ns) @ E
```

- constructs corrected derivative matrices:

```text
D_r = 0.5*(M_inv + VVT) @ B1 @ (I - VVT*M_diag) + Vr @ V.T @ M_diag
D_s = 0.5*(M_inv + VVT) @ B2 @ (I - VVT*M_diag) + Vs @ V.T @ M_diag
```

Current project:

- uses `build_local_operators(...)`;
- stores `Dr`, `Ds`, modal mass inverse, edge slices, and boundary lift inside
  `ReferenceElement`;
- the surface lift uses:

```text
engine.lift_boundary_penalty(p_boundary, edge_lengths=np.ones(3))
```

Main non-speed difference:

- The notebook builds a custom SBP/collocation-style derivative and lift
  convention.
- Our current path uses the existing `ReferenceElement` operator/lift
  convention.

This is another major non-speed difference.  Even with the same face penalty
formula, the volume derivative and boundary lift may not represent the same
semi-discrete operator.

## 7. Face normal / metric flux convention

Notebook face normals:

```text
nr_face = [ 0,  1, -1]
ns_face = [-1,  1,  0]
```

The hypotenuse face uses `(1,1)`, not a unit normal.

Current old mode:

```text
REFERENCE_FACE_NORMALS =
[
  (0, -1),
  (1/sqrt(2), 1/sqrt(2)),
  (-1, 0),
]
```

Current old surface metric flux:

```text
vn_sJ = J_face * (nr*u_tilde + ns*v_tilde)
```

Notebook surface metric flux:

```text
vn_sJ = J_M * (nr*u_tilde_M + ns*v_tilde_M)
```

Main non-speed difference:

- Both use reference contravariant metric flux, not the raw physical conormal.
- But the face normal scaling convention differs on the hypotenuse face.

The notebook's normal convention is tied to its custom face weights, `B1/B2`,
and lift convention.  Our current old mode uses unit reference normals and
passes unit edge lengths to the lift.  These may be internally consistent in
their own operator systems, but they are not identical conventions.

## 8. Surface penalty / numerical flux

Both use the old-style side-local penalty family:

```text
penalty = 0.5 * (vn_sJ - C) * (qM - qP)
```

Notebook:

```text
upwind: C = alpha_lf * abs(vn_sJ)
LF:     C = alpha_lf * global_V_max * J_M
central/default: C = 0
```

Current project:

```text
central: C = 0
upwind:  C = abs(vn_sJ)
lf:      C = alpha_lf * abs(vn_sJ)
```

Main non-speed differences:

- Notebook upwind honors `alpha_lf`; current upwind ignores `alpha_lf` and is
  always the standard pointwise upwind coefficient.
- Notebook LF uses a global speed times `J_M`.
- Current LF uses pointwise `alpha_lf*abs(vn_sJ)`.
- Therefore, in the current project `lf alpha=1` exactly equals upwind.
- In the notebook, LF is not the same as upwind unless the global-speed
  coefficient happens to match the local `abs(vn_sJ)` everywhere.

This is a central flux-comparison difference and should not be treated as a
speed-only implementation detail.

## 9. Surface lift and weights

Notebook:

```text
scaled_penalty = penalty * weights_1d
surface_integral = E.T @ scaled_penalty
surface_term = (1/J) * (M_inv_mat @ surface_integral)
```

Current project:

```text
p_boundary[edge_slice] = penalty
lifted = engine.lift_boundary_penalty(p_boundary, edge_lengths=np.ones(3))
surface_rhs = lifted / J
```

Main non-speed difference:

- The notebook explicitly multiplies the face penalty by 1D face weights and
  applies a diagonal inverse nodal mass.
- The current project delegates face weights and modal/nodal lift to
  `ReferenceElement.lift_boundary_penalty`.

The two may be analogous, but they are not literally the same implementation.
This matters because earlier conservative-SAT investigations showed that
surface scaling and lift convention strongly affect stability.

## 10. Initial condition and exact solution

Notebook Gaussian:

```text
q = exp(-10 * (geodesic_distance / R)^2)
```

with the center effectively along the positive y-axis via:

```text
dot_product = Y_rot / R
```

Current final sphere Gaussian:

```text
q = exp(-20 * ||X - normalize([1,1,1])||^2)
```

and the exact solution rotates the Gaussian center by Rodrigues' formula.

Main non-speed difference:

- Notebook uses geodesic angular distance on the sphere.
- Current project uses Euclidean chord distance in 3D.
- The center and width are also different.

This affects reported L2/Linf errors and makes direct numerical comparison
invalid unless both are made consistent.

## 11. Time step / CFL

Notebook:

```text
dt = CFL * h_min / (V_max * (k_degree + 1)^2)
```

then adjusts the step to align with final time and sampling intervals.

Current project:

- short-time final sphere temporal test uses prescribed dt values;
- h-convergence uses a cap and an `h^2` reduction on the finest mesh;
- long-time one-period is intentionally not part of the current final result.

Main non-speed difference:

- The notebook uses a CFL-driven long-time physical simulation.
- Current project uses short-time diagnostic dt choices to isolate RK and h
  behavior under the known sphere RHS limitations.

## 12. Error and mass norms

Notebook:

```text
L2 = sqrt(sum(error^2 * weights_ref * J))
mass_error = (mass - mass_initial) / abs(mass_initial)
```

Current final sphere convergence:

```text
L2 = sqrt(sum(error^2 * weights_ref * J) / sum(weights_ref * J))
Linf = max(abs(error))
```

Main non-speed difference:

- Notebook L2 is unnormalized.
- Current final sphere L2 is area-normalized.

The two L2 values differ by approximately a factor of `sqrt(surface area)` for
the same error field.

## 13. What matches our current old/exchange-cache path

The closest shared pieces are:

- projected 3D octahedron topology;
- side-local face traces;
- reference metric flux `J*(nr*u_tilde + ns*v_tilde)`;
- old-style penalty `0.5*(vn_sJ-C)*(qM-qP)`;
- closed-sphere solid-body rotation setup;
- LSRK54 coefficients.

This supports our previous conclusion that the notebook belongs to the same
general old-style SAT family, not the pairwise conservative SAT family.

## 14. What differs most from our current code

Most important non-speed differences, in likely importance order:

1. Pure divergence-form volume RHS vs our skew/split volume RHS.
2. Analytic projected-sphere metrics vs numerical nodal metric
   differentiation.
3. Custom SBP/collocation derivative and lift vs current `ReferenceElement`
   derivative/lift.
4. Non-unit reference face normal convention vs current old-mode unit normal
   convention.
5. Global LF coefficient vs current pointwise LF coefficient.
6. Geodesic Gaussian exact solution vs chord-distance Gaussian.
7. Long-time physical 12-day rotation vs short-time unit-sphere diagnostics.
8. Unnormalized L2 norm vs area-normalized L2 norm.

## 15. Practical implication for our next steps

If we want to compare this notebook against our project without mixing too many
variables, the safest order would be:

1. Keep topology fixed to our projected 3D sphere mesh.
2. Keep the current old-mode surface formula.
3. Add a diagnostic variant that swaps only the volume RHS from split/skew to
   divergence form.
4. Separately add analytic projected-sphere metrics as a diagnostic option.
5. Separately compare the custom SBP derivative/lift convention.
6. Only after those are isolated, compare global LF vs pointwise LF.

The notebook should not be treated as evidence that acceleration alone fixes
our long-time sphere issue.  Its mathematical/operator choices differ in
several stability-relevant places.
