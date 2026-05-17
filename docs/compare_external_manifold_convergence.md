# Comparison: external manifold convergence vs current project

This note compares the newly added external reference files:

- `otherpeople/manifold_div_h_convergence.py`
- `otherpeople/manifold_lsrk_convergence.py`

against the current project state recorded in `README_codex.md`, especially
Phase 2.5/2.6 and `test/test_final_sphere_convergence.py`.

This is comparison only. No solver changes are proposed here.

## 1. q=1 divergence h-convergence vs our q=1 RHS diagnostic

External `manifold_div_h_convergence.py` evaluates a constant-field manifold
RHS through:

```text
manifold_rhs_constant_field(...)
```

That routine computes:

```text
volume_div = manifold_volume_divergence(q=1, ...)
surface    = manifold_surface_term(q=1, ...)
rhs        = -volume_div + surface
```

and reports weighted norms of both `rhs` and `divergence` over
`mesh_levels = (2, 4, 8, 16, 32)`.

Our Phase 2.5 q=1 diagnostic is similar in goal but different in scope:

- It uses the projected 3D topology and checks that q=1 full RHS remains at the
  volume metric-divergence level.
- For q=1, `qM - qP = 0`, so all surface modes have zero surface contribution.
- The reported Phase 2.5 value was:

```text
max_abs_rhs = 2.164720e-06
rms_rhs     = 3.913355e-07
surface     = 0
```

Main difference:

- External q=1 convergence is a true h-study over multiple meshes and reports
  weighted `L2`/`Linf`.
- Our Phase 2.5/2.6 q=1 path was a diagnostic snapshot at fixed high
  resolution, mainly validating projected topology, metric continuity, and
  surface-zero behavior for constant traces.

External `manifold_div_h_convergence.py` therefore fills a gap: it turns the
q=1 metric/RHS check into a mesh-refinement study. Our current q=1 result is
strong evidence that the projected topology fixed the large unfolded-layout
error, but it is not the same convergence table.

## 2. external Gaussian LSRK h-convergence vs our final sphere convergence

External `manifold_lsrk_convergence.py` runs a full sphere Gaussian/constant
LSRK convergence pipeline:

```text
generate_spherical_octahedron_mesh
build_manifold_geometry_cache
build_manifold_exchange_cache
manifold_rhs_exchange
integrate_lsrk54
exact_gaussian_bell_xyz
```

Our `test/test_final_sphere_convergence.py` also runs Gaussian solid-body
rotation on the projected octahedron sphere, but with our existing core/test
helpers:

```text
create_projected_octahedron_sphere_mesh
compute_manifold_geometry
compute_sphere_full_rhs_for_state
core/rhs_sphere.py surface_mode="old"
lsrk54_step
gaussian_exact_on_state
```

Key conceptual difference:

- External pipeline is built around a cached manifold exchange RHS:
  `manifold_rhs_exchange`.
- Our final sphere convergence is built around the existing test callback and
  the current `core/rhs_sphere.py` old helper.

Both use projected sphere topology and a Gaussian exact rotation target, but
the RHS and exchange infrastructure are not identical.

## 3. Detailed comparison table

| Category | External files | Current project |
|---|---|---|
| Mesh generator | `generate_spherical_octahedron_mesh(n_div, R)` from `sphere_manifold_mesh.py`; returns `nodes_xyz, EToV`; 3D projected octahedron connectivity by merged projected vertices. | `create_projected_octahedron_sphere_mesh(nsub, R)` from `core/geometry/sphere_manifold_topology.py`; returns `VX,VY,VZ,EToV,patch_ids,nodes_xyz`; same essential projected-octahedron idea plus patch ids. |
| h measure | `spherical_mesh_hmin(nodes_xyz, EToV)`, minimum chord length. | `projected_sphere_mesh_hmin(nodes_xyz, EToV)`, also minimum chord length. |
| Reference operators | Fixed `build_manifold_table1_k4_reference_operators()`, hard-coded Table1 k=4 / N=4. | `build_local_operators(N=order, n=n_quad, rule="table1")`; final sphere test uses `N=3`, `n_quad=4`. |
| Geometry cache | `build_manifold_geometry_cache` returns packed arrays `X,Y,Z,J,nx,ny,nz,a1x,...,a2z`. | Per-element list of dicts from `compute_manifold_geometry`, including `a1`, `a2`, `J`, `n`, `a_contra_1`, `a_contra_2`. |
| Velocity | `solid_body_velocity_xyz(geom.X, geom.Y, geom.Z, u0, alpha0)`, default `alpha0=-pi/4`. | `solid_body_rotation_velocity(xyz, u0, alpha=pi/4)` with omega convention `(-u0 sin(alpha),0,u0 cos(alpha))`. Sign/angle convention should be checked when comparing exact solutions. |
| Volume RHS | External `manifold_volume_divergence` returns the split-form divergence without leading RHS sign; `manifold_rhs_exchange` returns `-div + surface`. | `compute_manifold_skew_volume_rhs` returns the skew volume RHS directly; time callback adds `volume_rhs + surface_rhs`. This sign convention is one place to compare carefully. |
| Surface RHS | External `manifold_surface_term_from_exchange` uses `vn_sJ = J_face*(nr*u_tilde + ns*v_tilde)`, penalty `0.5*(vn_sJ-C)*(qM-qP)`, face weights, lift, then `/J`. | Old mode uses the same basic local formula `J_face*(nr*u_tilde+ns*v_tilde)` and `0.5*(vn-C)*(qM-qP)`, then `engine.lift_boundary_penalty(... edge_lengths=ones)/J`. Conservative modes use paired equal/opposite SAT and are not the main final mode. |
| Exchange cache | `build_manifold_exchange_cache` stores topology, trace policy, face weights/normals, and optionally precomputed `J_face`, `u_tilde_face`, `v_tilde_face`. | Old mode recomputes/reads face data from `state` each call and aligns neighbor face nodes by matching 3D coordinates. No persistent exchange cache for old mode. |
| Face pairing | External uses topological `build_face_connectivity` plus `face_flip`, then `pair_face_traces`. | Current old mode uses `EToE/EToF` plus coordinate-based direct/reverse ordering check through `aligned_neighbor_face_indices`. |
| Flux names | `upwind`, `central`, `lax_friedrichs`. | `upwind`, `central`, `lf`. |
| Upwind definition | `C = alpha_lf * abs(vn_sJ)` for `upwind`. | `C = abs(vn)` for upwind; `alpha_lf` is ignored for upwind in `sphere_flux_coefficient`. In normal use alpha is 1, so this matches. |
| LF definition | `lax_friedrichs`: `C = alpha_lf * max(abs(vn_sJ))`, a global speed over the face metric field. In the time-convergence config, CFL speed is also scaled by `alpha_lf` for LF. | `lf`: `C = alpha_lf * abs(vn)` pointwise. Thus `lf alpha=1` exactly matches upwind. This is different from external LF. |
| dt/CFL | `dt = cfl_dt_from_h(cfl, h, N+1, vmax*flux_speed_scale)`, with comment matching `dt = CFL*h/(vmax*(k+1)^2)`. Default config has `cfl=0.25`, `N=4`, `tf=1.0`. | Final sphere test uses manual short-time dt targets: temporal reference `dt_ref=6.25e-5`; h-study `dt <= 2.5e-4` and finest mesh `dt ~ h^2`. No one-period/long-time CFL run. |
| Exact Gaussian | External calls `exact_gaussian_bell_xyz(..., width=1/sqrt(10), center_xyz=...)`; default center `(1,0,0)`, with presets available. | Current final test uses `exp(-20*||X-Xc||^2)`, `Xc=normalize([1,1,1])`, and rotates center with Rodrigues formula. This corresponds to a narrower/different Gaussian unless external width maps to the same beta by convention. |
| Weighted L2 | External `manifold_weighted_norms` returns unnormalized `sqrt(sum w*J*err^2)`. | Current `sphere_l2_error` normalizes by total sphere area: `sqrt(sum w*J*err^2 / sum w*J)`. Values are therefore scaled by about `1/sqrt(area)` relative to unnormalized norms. |
| Linf | Both use max absolute nodal error. | Same. |
| Mass | External reports `manifold_weighted_mass` and relative mass error in LSRK rows. | Current final sphere convergence does not report mass in the final table, although earlier short/extended diagnostics do. |

## 4. Is `manifold_rhs_exchange` closer to a stable version than our old mode?

It is closer to our **old** mode than to our conservative modes.

Shared structure with our old helper:

```text
vn_sJ = J_face * (nr*u_tilde + ns*v_tilde)
penalty = 0.5 * (vn_sJ - C) * (qM - qP)
surface = lift(face_weights * penalty) / J
rhs = -div + surface
```

This is exactly the family of formulas that behaved better than our paired
conservative helper in extended-time diagnostics.

Important differences from our old mode:

1. External has a real exchange cache.

   `build_manifold_exchange_cache` precomputes:

   ```text
   conn, trace, face_weights, nr, ns, J_face, u_tilde_face, v_tilde_face
   ```

   Our old helper computes face ordering and uses the state arrays every RHS
   call.  It is simpler and less structured.

2. External pairing is topological and trace-policy driven.

   `pair_face_traces` uses `EToE`, `EToF`, and `face_flip` to align traces.
   Our old helper aligns by comparing physical face-node coordinates and
   reversing if needed.

3. External lift normalization is not obviously identical.

   External forms:

   ```text
   scaled = penalty * exchange_cache.face_weights
   surface_integral = face_extraction.T @ scaled_penalty
   return (ref_ops.lift @ surface_integral).T / geom.J
   ```

   Our old helper passes packed penalties into:

   ```text
   engine.lift_boundary_penalty(p_boundary, edge_lengths=np.ones(3)) / J
   ```

   Both are face-weighted lift paths, but their modal mass conventions and
   reference-area normalization should be compared before claiming exact
   equivalence.

4. External LF is different.

   External `lax_friedrichs` uses a global `max(abs(vn_sJ))`; our `lf` is
   pointwise and intentionally equals upwind when `alpha_lf=1`.

5. External config is fixed to N=4.

   `_validate_config` rejects `N != 4`, because the reference operators are
   hard-wired to Table1 k=4.  Our final test uses `N=3`, `n_quad=4`.

Conclusion:

- `manifold_rhs_exchange` is likely the best external reference for a stable
  sphere RHS path because it combines the old-style side-local metric flux
  with a cleaner exchange cache.
- It does **not** implement the pairwise equal/opposite conservative SAT that
  caused positive-real-part modes in our Phase 2.12 diagnostics.
- It therefore supports the current decision to report `surface_mode="old"` as
  the main short-time sphere result while marking conservative SAT as future
  work.

## 5. Direct answers to the requested questions

### 1. Difference between external q=1 divergence h-convergence and Phase 2.5/2.6

External:

- h-study over `(2,4,8,16,32)`;
- uses `manifold_rhs_constant_field`;
- reports weighted `L2`/`Linf` of both full RHS and volume divergence;
- includes `max_surface_abs`, which should be zero/negligible for constant q.

Ours:

- Phase 2.5 validates q=1 full RHS on the projected topology at fixed setup;
- Phase 2.6 validates smooth continuous-trace RHS, where surface term is zero
  because `qM-qP=0`;
- the q=1 value is a diagnostic magnitude, not a full h-convergence study.

### 2. Difference between external Gaussian LSRK h-convergence and our final sphere convergence

External:

- N fixed to 4;
- uses cached geometry and cached exchange;
- uses `manifold_rhs_exchange`;
- default final time is `tf=1.0`;
- uses `dt = CFL*h/(vmax*(N+1)^2)` through helper;
- exact Gaussian is supplied by `exact_gaussian_bell_xyz`;
- reports mass and optional time histories/snapshots.

Ours:

- N=3, `n_quad=4`;
- uses existing core/test geometry and RHS callback;
- uses `surface_mode="old"`;
- intentionally short final times: `T=1e-2` for temporal, `T=5e-3` for h;
- uses manual dt/reference choices to avoid the known long-time instability;
- exact Gaussian is implemented locally by rotating the center with Rodrigues
  formula;
- reports area-normalized L2 and nodal Linf.

### 3. Summary of requested categories

See the detailed comparison table above.  The highest-impact differences are:

- external N=4 fixed Table1 operators vs our flexible `build_local_operators`;
- external exchange cache vs our coordinate-aligned per-call old helper;
- external global LF vs our pointwise LF;
- external unnormalized weighted L2 vs our area-normalized L2;
- external `-div + surface` convention vs our callback `volume_rhs + surface`
  where `volume_rhs` already includes the sign.

### 4. Is external `manifold_rhs_exchange` more stable than our old mode?

From code structure, it is plausibly a cleaner and more stable version of the
same old-mode idea:

- it uses the same reference metric flux family,
  `J_face*(nr*u_tilde+ns*v_tilde)`;
- it uses side-local SAT rather than pairwise equal/opposite conservative SAT;
- it has a proper exchange cache and trace policy;
- it avoids the raw/physical conormal scaling issue that motivated
  `conservative_scaled`.

However, this note does not run the external code.  The conclusion is a code
comparison, not a measured stability result.

## 6. Recommended future comparison, not implemented here

If this becomes a next phase, the cleanest comparison would be:

1. Port only the external `build_manifold_exchange_cache` / `pair_face_traces`
   idea into a diagnostic branch.
2. Keep the old-mode formula unchanged.
3. Compare old current vs exchange-cache old-style RHS on:
   q=1, Gaussian short-time h-convergence, and small-mesh spectra.
4. Do not revisit one-period testing until the small-mesh spectra show no
   significant positive-real-part modes.

