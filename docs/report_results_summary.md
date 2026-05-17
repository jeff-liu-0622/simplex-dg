# Results Summary

This document summarizes the completed diagnostic phases recorded in
`README_codex.md`.  It is a results-only report: no new functionality is
introduced here.

## 1. Planar Flux Switching Status

Status: completed.

Implemented planar skew-symmetric DG surface-flux switching:

- `flux_type="central"`
- `flux_type="upwind"`
- `flux_type="lf"`
- `alpha_lf`
- legacy `tau=0/1` compatibility

Validation recorded in README:

- `test_flux_comparison_sinxy.py` passed.
- `test_lsrk_stationary_xy.py` passed.
- `test_lsrk_temporal_sinx.py` passed.
- `compileall core test/test_flux_comparison_sinxy.py` passed.
- `LF(alpha=1)` equals upwind for scalar linear advection.

Planar periodic `sin(2*pi*(x+y-(cx+cy)t))` flux comparison:

```text
flux      alpha    L2_final      Linf_final    mass_error     energy_change
central   1.000   5.941421e-05  4.042163e-04  1.370432e-16   1.247120e-10
upwind    1.000   2.801526e-05  2.666571e-04 -2.602085e-16  -3.837877e-08
lf        1.000   2.801526e-05  2.666571e-04 -2.602085e-16  -3.837877e-08
lf        1.500   2.900091e-05  2.926705e-04 -3.920475e-16  -4.097218e-08
```

Key check:

```text
max_abs_difference_between_upwind_and_lf_alpha1 = 0.000000e+00
```

Environment notes:

- `test_exchange_h_refinement.py` was not run because `matplotlib` is
  unavailable.
- `test_trace_exchange_h_convergence.py` exceeded the 180-second runtime
  limit.

## 2. Sphere Geometry Area Convergence

Status: completed.

File:

- `test/test_manifold_geometry_sphere.py`

Verified:

- sphere nodes satisfy `||X|| = R`
- covariant bases `a1 = Dr X`, `a2 = Ds X` are finite
- surface Jacobian `J = ||a1 x a2||` is positive and finite
- surface normals are finite and unit length
- contravariant bases satisfy `a^i dot a_j ~= delta^i_j`
- global surface area approximates `4*pi*R^2`

For `nsub=4`, `N=4`:

```text
computed area = 1.256703739491e+01
exact area    = 1.256637061436e+01
rel_error     = 5.306071e-05
```

Surface area convergence:

```text
nsub   K      h             area              rel_error     order
2      32     5.0000e-01    1.2568437495e+01  1.644772e-04  -
4      128    2.5000e-01    1.2567037395e+01  5.306071e-05  1.6322
8      512    1.2500e-01    1.2566544217e+01  1.381483e-05  1.9414
16     2048   6.2500e-02    1.2566414315e+01  3.477605e-06  1.9901
```

## 3. Velocity Diagnostic

Status: completed.

File:

- `test/test_manifold_velocity_sphere.py`

Setup:

```text
nsub  = 8
N     = 4
R     = 1
u0    = 1
alpha = pi/4
V3D   = omega x X
```

Results:

```text
max_tangency_error       = 1.387779e-16
max_reconstruction_error = 8.320434e-03
max_speed                = 9.999999e-01
```

Interpretation:

- The solid-body velocity is tangent to the sphere to near roundoff.
- Contravariant velocity components are finite.
- Reconstructing `V_rec = u_tilde a1 + v_tilde a2` matches `V3D` within the
  diagnostic tolerance.

## 4. Topology Bug And Fix

Status: completed.

Problem found:

- The old unfolded 2D octahedral layout connectivity is not suitable as the
  physical sphere face pairing.
- It can pair faces that are adjacent in the layout but far apart on the 3D
  sphere.
- Prior diagnostics showed shared-face `xyz` mismatch up to about `1.999`.

Fix:

- Added `core/geometry/sphere_manifold_topology.py`.
- New mesh uses projected 3D octahedron vertices and builds connectivity from
  shared 3D vertices.
- Existing sphere diagnostics can use the projected topology path.

Projected mesh face-continuity diagnostic:

```text
nsub   Nv     K      shared   max_face_match_error   rms_face_match_error
2      18     32     48       0.000000e+00           0.000000e+00
4      66     128    192      0.000000e+00           0.000000e+00
8      258    512    768      0.000000e+00           0.000000e+00
16     1026   2048   3072     0.000000e+00           0.000000e+00
```

Projected-topology q=1 metric divergence:

```text
max_abs_divJv = 2.164720e-06
rms_divJv     = 3.913355e-07
mean_divJv    = 0.000000e+00
max_abs_rhs   = 2.164720e-06
rms_rhs       = 3.913355e-07
```

Interpretation:

- The projected 3D topology removes the physical face-pairing mismatch.
- The q=1 volume-only metric divergence improves from the old layout-level
  pointwise error to about `2e-6`.

## 5. q=1 Full RHS Diagnostic

Status: completed.

File:

- `test/test_sphere_full_rhs_constant.py`

Setup:

```text
q     = 1
nsub  = 16
N     = 4
R     = 1
u0    = 1
alpha = pi/4
```

Surface penalty skeleton:

```text
P = 0.5 * (v_n_sJ - C) * (qM - qP)

central: C = 0
upwind:  C = |v_n_sJ|
lf:      C = alpha_lf * |v_n_sJ|
```

Results:

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

Interpretation:

- Since `qM - qP = 0`, all surface terms are zero.
- The full RHS equals the projected-topology volume-only metric-divergence
  level.
- Upwind and `LF(alpha=1)` are identical.

## 6. Smooth RHS Diagnostic

Status: completed with diagnostic note.

File:

- `test/test_sphere_full_rhs_smooth_snapshot.py`

Smooth field:

```text
q = X + 0.5Y - 0.25Z
```

Setup:

```text
nsub  = 16
N     = 4
R     = 1
u0    = 1
alpha = pi/4
```

Results:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      mean_rhs      max_abs_volume  rms_volume   max_abs_surface  rms_surface   mass_rate     energy_rate
central    1.0000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
upwind     1.0000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.0000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
lf         1.5000    7.288674e-01  4.208125e-01 1.786450e-15  7.288674e-01    4.208125e-01 0.000000e+00     0.000000e+00  8.165777e-15 -1.803462e-15
```

Additional checks:

```text
max upwind/LF(alpha=1) RHS difference       = 0.000000e+00
max upwind/LF(alpha=1.5) surface difference = 0.000000e+00
max_face_match_error                        = 0.000000e+00
max_abs_penalty                             = 0.000000e+00
```

Interpretation:

- The smooth field is evaluated from the same 3D physical coordinates on both
  sides of each shared face.
- Therefore `qM - qP = 0` and the jump-based surface penalty is zero for all
  fluxes.
- This validates the smooth full-RHS volume path and continuous-trace
  consistency, but it does not test dissipative flux separation.

## 7. Jump-Flux Diagnostic

Status: completed.

File:

- `test/test_sphere_flux_jump_diagnostic.py`

Jumped diagnostic field:

```text
q_base = X + 0.5Y - 0.25Z
q      = q_base + eps * sign_K
eps    = 1e-2
sign_K = +1 for even element index, -1 for odd element index
```

Setup:

```text
nsub  = 16
N     = 4
R     = 1
u0    = 1
alpha = pi/4
```

Results:

```text
flux       alpha_lf   max_abs_rhs   rms_rhs      max_abs_volume  rms_volume   max_abs_surface  rms_surface   mass_rate     energy_rate    max_jump_q    rms_jump_q
central    1.0000    2.989002e+00  6.125412e-01 7.288674e-01    4.208125e-01 2.482967e+00     4.453583e-01  6.641606e-15  1.109175e-10  2.000000e-02 1.632993e-02
upwind     1.0000    4.853700e+00  7.816204e-01 7.288674e-01    4.208125e-01 4.875371e+00     6.592748e-01 -4.989931e-03 -1.115925e-02  2.000000e-02 1.632993e-02
lf         1.0000    4.853700e+00  7.816204e-01 7.288674e-01    4.208125e-01 4.875371e+00     6.592748e-01 -4.989931e-03 -1.115925e-02  2.000000e-02 1.632993e-02
lf         1.5000    5.941709e+00  9.518491e-01 7.288674e-01    4.208125e-01 6.114180e+00     8.544105e-01 -7.484896e-03 -1.673887e-02  2.000000e-02 1.632993e-02
```

Additional checks:

```text
max upwind/LF(alpha=1) RHS difference       = 0.000000e+00
max upwind/LF(alpha=1.5) RHS difference     = 1.335862e+00
max upwind/LF(alpha=1.5) surface difference = 1.335862e+00
max_face_match_error                        = 0.000000e+00
max_abs_penalty                             = 1.911791e-03
```

Interpretation:

- The face jump is active: `max_abs_jump_q = 2.000000e-02`.
- Upwind and `LF(alpha=1)` match exactly.
- `LF(alpha=1.5)` separates from upwind.
- `LF(alpha=1.5)` produces larger surface RMS than upwind.

## 8. Short-Time LSRK Sanity Test

Status: completed.

File:

- `test/test_sphere_lsrk_short_sanity.py`

This is a very short-time sanity test.  It is not a production sphere solver,
not a one-period solid-body rotation benchmark, and not a convergence study.

Setup:

```text
nsub    = 4
N       = 4
q0      = X + 0.5Y - 0.25Z
T_final = 1e-3
dt      = 2.5e-4
```

Results:

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

Interpretation:

- All flux cases ran to completion.
- No final-state NaN/Inf was detected.
- Upwind and `LF(alpha=1)` produced identical final states.
- Long-time sphere simulation, one-period rotation, and convergence studies
  are still future work.

