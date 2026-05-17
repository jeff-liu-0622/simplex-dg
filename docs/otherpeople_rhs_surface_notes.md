# Otherpeople RHS / Surface Flux Notes

This note is a reference inspection only.  No main solver code was changed.

Inspected files:

- `otherpeople/manifold_rhs.py`
- `otherpeople/rhs_split_conservative_exchange.py`
- `otherpeople/rhs_split_conservative_exact_trace.py`
- `otherpeople/split_form.py`
- `otherpeople/divergence_split.py`
- `otherpeople/geometry/face_metrics.py`
- `otherpeople/geometry/edges.py`
- `otherpeople/operators/boundary.py`
- `otherpeople/operators/exchange.py`
- `otherpeople/operators/trace_policy.py`
- `otherpeople/operators/manifold_sbp_chan2019.py`

## 1. Split / Skew Volume RHS

There are three related volume formulations.

### Planar advective split form

`otherpeople/split_form.py` defines a split advective operator:

```text
L(v) =
  1/2 [ d_x(a v) + a d_x(v) - d_x(a) v ]
+ 1/2 [ d_y(b v) + b d_y(v) - d_y(b) v ]
```

This targets the advective form `a v_x + b v_y`.

### Planar conservative mapped split divergence

`otherpeople/divergence_split.py` defines the mapped conservative split
divergence.  With

```text
alpha = ys * a - xs * b
beta  = -yr * a + xr * b
```

the split divergence is

```text
div_h^split(F) =
1/J * [
  1/2 (D_r(alpha v) + alpha D_r(v) + v D_r(alpha))
+ 1/2 (D_s(beta  v) + beta  D_s(v) + v D_s(beta ))
]
```

`rhs_split_conservative_exchange.py` and
`rhs_split_conservative_exact_trace.py` use

```text
RHS_vol = - mapped_divergence_split_2d(...)
```

### Manifold split volume divergence

`otherpeople/manifold_rhs.py` defines:

```text
u_tilde = a^1 dot U
v_tilde = a^2 dot U

term1 = [D_r(J u_tilde q) + D_s(J v_tilde q)] / J
term2 = u_tilde D_r(q) + v_tilde D_s(q)
term3 = (q/J) [D_r(J u_tilde) + D_s(J v_tilde)]

volume_divergence = 0.5 * (term1 + term2 + term3)
RHS = -volume_divergence + surface
```

This matches our current manifold skew volume formula in sign convention:
the helper returns a positive divergence, and the RHS applies a leading minus.

## 2. Surface Penalty / Numerical Flux

The planar conservative exchange files define the numerical flux as:

```text
f      = ndotV * qM
fstar  = 0.5 * (ndotV*qM + ndotV*qP)
       + 0.5 * (1 - tau) * |ndotV| * (qM - qP)
p      = f - fstar
```

The simplified pure-upwind penalty is:

```text
p = min(ndotV, 0) * (qM - qP)
```

Equivalently, for the tau family:

```text
p = 0.5 * [ndotV - (1 - tau) |ndotV|] * (qM - qP)
```

`tau=0` is pure upwind.  `tau=1` is central.

`otherpeople/manifold_rhs.py` uses the same element-local penalty style:

```text
vn_sJ = J_face * (nr*u_tilde + ns*v_tilde)

central:         c_val = 0
upwind:          c_val = alpha_lf * |vn_sJ|
lax_friedrichs:  c_val = alpha_lf * max(|vn_sJ|)

penalty = 0.5 * (vn_sJ - c_val) * (qM - qP)
surface = lift(face_weights * penalty) / J
```

Important detail: in `manifold_rhs.py`, `lax_friedrichs` is not the same as
pointwise upwind when `alpha_lf=1`; it uses a global maximum face speed.  Our
current `lf alpha=1` is pointwise and therefore matches upwind.

## 3. Upwind / LF Sign

There are two equivalent sign conventions in the inspected files.

### Planar exchange sign

The planar exchange path computes:

```text
p = fM - fstar
surface_rhs = lift(p)
RHS = volume_rhs + surface_rhs
```

For upwind with `tau=0`:

```text
p = 0                      if ndotV >= 0
p = ndotV * (qM - qP)      if ndotV < 0
```

This is the same as:

```text
p = 0.5 * (ndotV - |ndotV|) * (qM - qP)
```

### Chan2019 SBP sign

`otherpeople/operators/manifold_sbp_chan2019.py` uses:

```text
f_star = vn_M * 0.5 * (qM + qP) + 0.5 * c_val * (qM - qP)
penalty = (f_star - vn_M * qM) * face_weights
return -(vol_term + surface_integral) / (weights_2d * J)
```

Here the SAT is `fstar - fM`, but it is inside an overall negative RHS.
So the net contribution has the same effective sign as `fM - fstar` in a
`RHS = volume + surface` convention.

This is a key difference from our current `core/rhs_sphere.py` conservative
helper: our helper constructs an equal-and-opposite boundary penalty in the
`fM - fstar` style, then adds it as `volume_rhs + surface_rhs`.  The mass
diagnostic is fixed, but the time instability suggests the energy/SAT sign or
metric scaling may not be equivalent to the SBP form.

## 4. Face Normal Velocity / Conormal / sJ

### Planar affine metrics

`otherpeople/geometry/face_metrics.py` computes:

```text
tangent = p1 - p0
length  = ||tangent||
normal  = (dy, -dx) / length
ndotV   = nx*u + ny*v
```

The physical line measure is not inside `ndotV`; it is stored separately as
`length`.

### Planar surface scaling

The planar lift path multiplies by physical edge length separately:

```text
surface contribution ~ (length / area) * face_weight_ratio * p
```

or, in projected inverse-mass form:

```text
surface_integral += length * face_weight * p
surface_rhs = surface_integral @ inverse_mass_T
surface_rhs /= area
```

So planar code keeps:

```text
normal velocity = unit normal dot velocity
surface metric  = edge length
```

as separate factors.

### Manifold metric flux

`otherpeople/manifold_rhs.py` does not compute a separate 3D conormal vector.
It uses reference contravariant metric flux:

```text
vn_sJ = J_face * (nr*u_tilde + ns*v_tilde)
```

Then it multiplies by `face_weights` and divides by `geom.J` after lifting.

`otherpeople/operators/manifold_sbp_chan2019.py` similarly uses:

```text
Ju = J * u_tilde
Jv = J * v_tilde
vn_M = nr * Ju_face + ns * Jv_face
```

For LF in the Chan2019 helper:

```text
c_val = alpha_lf * global_vmax * J_face
```

So the inspected manifold code treats `J * contravariant velocity` as the
surface metric flux in reference coordinates.  It does not use a physical
`tau x n_surf` conormal in these helpers.

## 5. Lift Back To Volume Nodes

There are two distinct lift conventions.

### Collocated planar lift

For Table1 embedded face nodes, the planar path can do a collocated lift:

```text
surface_rhs[:, ids] += length_over_area * (face_weight / volume_weight_at_face_node) * p
```

That is:

```text
edge length * edge quadrature weight / element area / volume nodal weight
```

### Projected inverse-mass lift

The exact-trace / exchange path can instead form:

```text
surface_integral[:, ids] += length * face_weight * p
surface_rhs = surface_integral @ surface_inverse_mass_T
surface_rhs /= area
```

### Manifold lift

`otherpeople/manifold_rhs.py` builds:

```text
M_modal = V.T @ diag(weights_2d) @ V
lift = V @ inv(M_modal) @ V.T

surface_integral = face_extraction.T @ (face_weights * penalty)
surface = (lift @ surface_integral).T / geom.J
```

Notice the weights convention: the comment says Table1 weights sum to one,
matching `M_modal = V.T @ diag(weights) @ V`.

Our current `core/operators.py` lift is:

```text
V @ invM_modal @ V.T @ full_boundary_weighted
```

where `invM_modal` was built from:

```text
M_modal = area * V.T @ W @ V
```

and `lift_boundary_penalty(edge_lengths=np.ones(3))` multiplies by edge
weights but not physical/reference edge length.  This differs from
`otherpeople/manifold_rhs.py` in the reference area / weight normalization
details.

## 6. Conservative Exchange

Yes, there is a conservative exchange style, but it is not a single shared
`Fstar` implementation in the same way as our Phase 2.11 helper.

`otherpeople/operators/exchange.py` provides:

- `evaluate_all_face_values`
- `pair_face_traces`
- `unique_interior_face_pairs`
- `interior_face_pair_mismatches`

`rhs_split_conservative_exchange.py` has a face-major pair kernel that loops
over unique face pairs:

```text
for each pair fa, fb:
    compute penalty on side A using ndotV_a, qA, qB
    compute penalty on side B using ndotV_b, qB, qA
```

This relies on the two sides having opposite normals and consistent
`length_over_area` / weight scaling.  It does not replace the two side metrics
by one averaged shared metric.  Pairing is used for efficient and consistent
exchange, not necessarily for forcing equal-and-opposite SAT values.

For manifold code, `manifold_rhs_exchange` uses `pair_face_traces`, but the
surface formula remains side-local:

```text
vn_sJ side M
qM, qP from exchange
penalty_M = 0.5 * (vn_sJ - c_val) * (qM - qP)
```

## 7. Energy-rate / Stability Diagnostics

The inspected files contain limited built-in diagnostics:

- `manifold_rhs_constant_field` returns `max_surface_abs` and `max_rhs_abs`.
- `constant_state_rhs_diagnostic` in `manifold_sbp_chan2019.py` returns
  `max_rhs_abs` and `weighted_rhs_mass`.
- `rhs_split_conservative_exchange.py` returns diagnostics such as `qM`,
  `qP`, `ndotV`, `p`, `tau_face`, and interior mismatches.

I did not find a full energy-rate diagnostic equivalent to our
`sum J*w*q*RHS` tables.  Stability is mostly encoded structurally through the
SBP / hybridized Chan2019 formulation and the upwind/LF SAT sign.

## 8. Differences From Current `core/rhs_sphere.py`

Main differences:

1. **Manifold face metric choice**

   - External manifold helpers use:

     ```text
     vn_sJ = J_face * (nr*u_tilde + ns*v_tilde)
     ```

   - Our conservative helper uses:

     ```text
     vn = V3D dot (tau x n_surf)
     ```

   These should be related geometrically, but they are not the same discrete
   object in the current implementation.  This may explain why the mass-rate
   diagnostic improved while the time RHS became unstable.

2. **SAT sign convention**

   - External planar/manifold old-style helpers use `p = fM - fstar` and add
     the lifted term.
   - Chan2019 uses `penalty = fstar - fM`, but places it inside an overall
     negative RHS.
   - Our conservative helper uses an equal-and-opposite pair penalty in the
     `fM - fstar` style.  The mass cancellation is enforced, but the resulting
     energy sign may not match the Chan2019 SAT.

3. **Single paired metric vs side-local metrics**

   - External exchange paths use side-local `ndotV` / `vn_sJ` and rely on
     connectivity, normals, and geometry to provide opposite signs.
   - Our Phase 2.11 helper averages the two side conormal fluxes:

     ```text
     vn = 0.5 * (vnM - vnP)
     ```

     then assigns equal and opposite penalties.

   This is conservative by construction, but may break the intended upwind
   dissipation direction if the sign/orientation is not exactly aligned with
   each element's local SAT.

4. **Lift normalization**

   - External planar collocated lift explicitly uses
     `length/area * face_weight/volume_weight`.
   - External manifold lift uses `face_weights`, a modal inverse mass built
     without the same explicit reference-area factor, then divides by `J`.
   - Our current core lift uses `M_modal = area * V.T W V` and
     `edge_lengths=np.ones(3)`, so the normalization may differ by a reference
     area / face-length factor from the external manifold reference code.

5. **LF definition**

   - External `manifold_rhs.py` `lax_friedrichs` uses a global max
     `max(abs(vn_sJ))`.
   - External Chan2019 LF uses `global_vmax * J_face`.
   - Our current `lf` uses pointwise `alpha_lf * abs(vn)`, so `LF(alpha=1)`
     intentionally matches upwind.  This is useful for diagnostics, but it is
     not the same LF definition as some external files.

6. **SBP hybridized structure**

   `manifold_sbp_chan2019.py` does not simply combine our current volume RHS
   and a lifted SAT.  It builds corrected SBP derivative matrices and hybrid
   `Q_tilde_r`, `Q_tilde_s` operators, then applies a SAT in that structure:

   ```text
   q_t = - M_J^{-1} [ volume_two_point_flux + surface_integral ]
   ```

   This is likely the most relevant reference for stability, because the
   surface sign and volume operator are designed together.

## Practical Takeaways For The Current Issue

1. The old helper is closer to `otherpeople/manifold_rhs.py` in face metric
   and penalty formula:

   ```text
   J_face * (nr*u_tilde + ns*v_tilde)
   0.5 * (vn_sJ - C) * jump
   ```

   That may explain why it behaved more stably in long integrations.

2. The conservative helper fixed global mass by forcing pair cancellation, but
   differs from the external stable-looking formulations in three ways:

   - physical conormal instead of `J * contravariant velocity`;
   - averaged shared metric instead of side-local metric;
   - equal/opposite SAT assignment not embedded in a Chan2019 SBP volume form.

3. Before another one-period test, the next diagnostic should compare, face by
   face:

   ```text
   V3D dot (tau x n_surf)
   vs
   J * (nr*u_tilde + ns*v_tilde)
   ```

   including sign, scale, and edge/reference parameter normalization.

4. A second useful diagnostic is an energy-rate table for the conservative
   helper:

   ```text
   sum J*w*q*surface_rhs
   ```

   split by central/upwind/LF and by old/conservative/sbp-style sign.  The
   current long-time blow-up is more consistent with a surface energy sign or
   scaling error than with the already-fixed global mass cancellation alone.

