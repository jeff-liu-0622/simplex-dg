import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.manifold_metrics import compute_manifold_geometry
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
    projected_sphere_mesh_hmin,
)
from core.operators import build_local_operators
from core.operators_sphere import (
    compute_manifold_skew_volume_rhs,
    compute_sphere_rhs,
)
from core.rhs_sphere import compute_sphere_surface_penalty
from core.time_integration import lsrk54_step


REFERENCE_FACE_LENGTHS = np.array([2.0, 2.0 * np.sqrt(2.0), 2.0], dtype=float)

# Face orientation used by core.rhs_sphere.physical_conormal_flux_on_face:
#
# face 0: r varies, s constant,       tau =  2 a1
# face 1: along r + s constant,       tau =  2(a2 - a1)
# face 2: s varies, r constant,       tau = -2 a2
#
# Therefore tau = drdt * a1 + dsdt * a2 with:
FACE_DRDT = np.array([2.0, -2.0, 0.0], dtype=float)
FACE_DSDT = np.array([0.0, 2.0, -2.0], dtype=float)


def reference_style_omega(u0=1.0, alpha0=-np.pi / 4.0):
    return u0 * np.array(
        [-np.sin(alpha0), 0.0, np.cos(alpha0)],
        dtype=float,
    )


def reference_style_velocity(xyz, u0=1.0, alpha0=-np.pi / 4.0):
    omega = reference_style_omega(u0=u0, alpha0=alpha0)
    return np.cross(omega[None, :], xyz)


def rotate_xyz_rodrigues(xyz, omega, t):
    xyz = np.asarray(xyz, dtype=float)
    omega = np.asarray(omega, dtype=float).reshape(3)
    speed = float(np.linalg.norm(omega))

    if speed == 0.0 or t == 0.0:
        return xyz.copy()

    axis = omega / speed
    angle = speed * float(t)
    c = np.cos(angle)
    s = np.sin(angle)
    dot = xyz @ axis
    cross = np.cross(axis[None, :], xyz)

    return xyz * c + cross * s + axis[None, :] * dot[:, None] * (1.0 - c)


def gaussian_bell_reference_style(
    xyz,
    R=1.0,
    center_xyz=(1.0, 0.0, 0.0),
    width=1.0 / np.sqrt(10.0),
):
    center = np.asarray(center_xyz, dtype=float)
    center = R * center / np.linalg.norm(center)
    dot = np.clip((xyz @ center) / (R * R), -1.0, 1.0)
    distance = R * np.arccos(dot)
    return np.exp(-((distance / float(width)) ** 2))


def exact_gaussian_bell_reference_style(
    xyz,
    t,
    R=1.0,
    u0=1.0,
    alpha0=-np.pi / 4.0,
    center_xyz=(1.0, 0.0, 0.0),
    width=1.0 / np.sqrt(10.0),
):
    omega = reference_style_omega(u0=u0, alpha0=alpha0)
    xyz0 = rotate_xyz_rodrigues(xyz, omega=omega, t=-float(t))

    return gaussian_bell_reference_style(
        xyz0,
        R=R,
        center_xyz=center_xyz,
        width=width,
    )


def face_node_indices(engine, face_id):
    face_slice = engine.edge_slices[face_id]
    return np.arange(face_slice.start, face_slice.stop)


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def build_projected_flux_state(nsub, order=4, R=1.0, u0=1.0, alpha0=-np.pi / 4.0):
    engine = build_local_operators(N=order, n=order, rule="table1")
    _, _, _, EToV, patch_ids, nodes_xyz = create_projected_octahedron_sphere_mesh(
        nsub=nsub,
        R=R,
    )
    EToE, EToF = build_connectivity(EToV)
    element_xyz = map_reference_nodes_to_projected_sphere(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        r=engine.r,
        s=engine.s,
        R=R,
    )

    geometries = []
    velocities = []
    u_tilde = []
    v_tilde = []

    for xyz in element_xyz:
        geometry = compute_manifold_geometry(engine, xyz)
        V3D = reference_style_velocity(xyz, u0=u0, alpha0=alpha0)

        geometries.append(geometry)
        velocities.append(V3D)
        u_tilde.append(np.sum(geometry["a_contra_1"] * V3D, axis=1))
        v_tilde.append(np.sum(geometry["a_contra_2"] * V3D, axis=1))

    return {
        "engine": engine,
        "EToE": EToE,
        "EToF": EToF,
        "patch_ids": patch_ids,
        "xyz": element_xyz,
        "geometry": geometries,
        "V3D": np.asarray(velocities),
        "u_tilde": np.asarray(u_tilde),
        "v_tilde": np.asarray(v_tilde),
        "R": R,
        "h": projected_sphere_mesh_hmin(nodes_xyz, EToV),
    }


def q_values_for_case(state, q_case):
    q = np.zeros((state["xyz"].shape[0], state["engine"].num_nodes), dtype=float)

    if q_case == "constant":
        q[:, :] = 1.0
        return q

    if q_case == "gaussian":
        for k, xyz in enumerate(state["xyz"]):
            q[k, :] = gaussian_bell_reference_style(xyz, R=state["R"])
        return q

    if q_case == "jump":
        signs = np.where(np.arange(q.shape[0]) % 2 == 0, 1.0, -1.0)
        q[:, :] = 1.0 + 0.1 * signs[:, None]
        return q

    raise ValueError(f"unknown q_case: {q_case}")


def physical_conormal_flux_on_face(state, elem_id, face_id, nodes):
    geometry = state["geometry"][elem_id]
    a1 = geometry["a1"][nodes, :]
    a2 = geometry["a2"][nodes, :]
    n_surf = geometry["n"][nodes, :]
    V3D = state["V3D"][elem_id, nodes, :]

    tau = FACE_DRDT[face_id] * a1 + FACE_DSDT[face_id] * a2
    conormal_sJ = np.cross(tau, n_surf)

    return np.sum(V3D * conormal_sJ, axis=1)


def aligned_neighbor_face_indices(xyz_M, xyz_P):
    direct = np.max(np.linalg.norm(xyz_M - xyz_P, axis=1))
    reverse = np.max(np.linalg.norm(xyz_M - xyz_P[::-1], axis=1))

    if reverse < direct:
        return np.arange(xyz_P.shape[0] - 1, -1, -1), reverse

    return np.arange(xyz_P.shape[0]), direct


def flux_coefficient(vn, flux_type):
    if flux_type == "central":
        return 0.0
    if flux_type == "upwind":
        return np.abs(vn)
    raise ValueError(f"unsupported diagnostic flux_type: {flux_type}")


def prepare_surface_state(state, q):
    q = np.asarray(q, dtype=float)
    prepared = dict(state)
    prepared["q"] = q
    prepared["volume_rhs"] = np.zeros_like(q)
    return prepared


def diagnostic_surface_rhs(
    state,
    q,
    *,
    speed_scaling,
    lift_edge_lengths,
    flux_type="upwind",
    face_match_tol=1.0e-12,
):
    engine = state["engine"]
    q = np.asarray(q, dtype=float)
    p_boundary = np.zeros((q.shape[0], engine.num_boundary_nodes), dtype=float)
    max_face_match_error = 0.0

    for kM in range(q.shape[0]):
        for fM in range(3):
            kP = int(state["EToE"][kM, fM])
            fP = int(state["EToF"][kM, fM])

            if kP == kM:
                raise AssertionError(f"unexpected boundary face ({kM}, {fM})")

            if (kM, fM) > (kP, fP):
                continue

            nodes_M = face_node_indices(engine, fM)
            nodes_P = face_node_indices(engine, fP)
            ordering, face_match_error = aligned_neighbor_face_indices(
                state["xyz"][kM, nodes_M, :],
                state["xyz"][kP, nodes_P, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            qM = q[kM, nodes_M]
            qP = q[kP, nodes_P[ordering]]

            aM = physical_conormal_flux_on_face(state, kM, fM, nodes_M)
            aP = physical_conormal_flux_on_face(state, kP, fP, nodes_P)[ordering]

            if speed_scaling == "divLref":
                aM = aM / REFERENCE_FACE_LENGTHS[fM]
                aP = aP / REFERENCE_FACE_LENGTHS[fP]
            elif speed_scaling == "noDiv":
                pass
            else:
                raise ValueError(f"unknown speed_scaling: {speed_scaling}")

            vn = 0.5 * (aM - aP)
            C = flux_coefficient(vn, flux_type=flux_type)
            penalty_M = 0.5 * (vn - C) * (qM - qP)
            penalty_P_aligned = -penalty_M

            penalty_P_native = np.empty_like(penalty_P_aligned)
            penalty_P_native[ordering] = penalty_P_aligned

            p_boundary[kM, engine.edge_slices[fM]] = penalty_M
            p_boundary[kP, engine.edge_slices[fP]] = penalty_P_native

    if max_face_match_error > face_match_tol:
        raise AssertionError(
            "projected sphere face pairing is not physically continuous: "
            f"max face match error = {max_face_match_error:.3e}"
        )

    surface_rhs = np.zeros_like(q)
    for k in range(q.shape[0]):
        lifted = engine.lift_boundary_penalty(
            p_boundary[k],
            edge_lengths=lift_edge_lengths,
        )
        surface_rhs[k, :] = lifted / state["geometry"][k]["J"]

    return surface_rhs


def weighted_integral(state, values):
    total = 0.0
    for k, geometry in enumerate(state["geometry"]):
        total += state["engine"].area * np.sum(
            state["engine"].w_s * geometry["J"] * values[k]
        )
    return float(total)


def rhs_relative_l2(state, diff, reference):
    num = weighted_integral(state, diff * diff)
    den = weighted_integral(state, reference * reference)
    if den <= 0.0:
        if num <= 0.0:
            return 0.0
        return np.inf
    return float(np.sqrt(num / den))


def surface_rhs_equivalence_diagnostic(nsub, q_case, flux_type="upwind", order=4):
    state = build_projected_flux_state(nsub=nsub, order=order)
    q = q_values_for_case(state, q_case)
    core_state = prepare_surface_state(state, q)
    surface_A = compute_sphere_surface_penalty(
        core_state,
        flux_type=flux_type,
        alpha_lf=1.0,
        surface_mode="conservative_scaled",
    )["surface_rhs"]

    cases = {
        "case1_divLref_lift1": diagnostic_surface_rhs(
            state,
            q,
            speed_scaling="divLref",
            lift_edge_lengths=np.ones(3),
            flux_type=flux_type,
        ),
        "case2_noDiv_lift1": diagnostic_surface_rhs(
            state,
            q,
            speed_scaling="noDiv",
            lift_edge_lengths=np.ones(3),
            flux_type=flux_type,
        ),
        "case3_noDiv_liftRef": diagnostic_surface_rhs(
            state,
            q,
            speed_scaling="noDiv",
            lift_edge_lengths=REFERENCE_FACE_LENGTHS,
            flux_type=flux_type,
        ),
    }

    rows = {}
    for name, surface_B in cases.items():
        diff = surface_A - surface_B
        rows[name] = {
            "max_abs": float(np.max(np.abs(diff))),
            "rms": rms(diff),
            "relative_L2": rhs_relative_l2(state, diff, surface_A),
            "mass_A": weighted_integral(state, surface_A),
            "mass_case": weighted_integral(state, surface_B),
            "energy_A": weighted_integral(state, q * surface_A),
            "energy_case": weighted_integral(state, q * surface_B),
        }

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "q_case": q_case,
        "flux_type": flux_type,
        "cases": rows,
    }


def compute_projected_line_rhs(q, t, *, state, flux_type="upwind"):
    del t
    engine = state["engine"]
    q = np.asarray(q, dtype=float)
    volume_rhs = np.zeros_like(q)

    for k, geometry in enumerate(state["geometry"]):
        rhs_vol, _, u_local, v_local = compute_manifold_skew_volume_rhs(
            engine=engine,
            geometry=geometry,
            V3D=state["V3D"][k],
            q=q[k],
        )
        volume_rhs[k, :] = rhs_vol
        state["u_tilde"][k, :] = u_local
        state["v_tilde"][k, :] = v_local

    state["q"] = q
    state["volume_rhs"] = volume_rhs

    surface_rhs = diagnostic_surface_rhs(
        state,
        q,
        speed_scaling="noDiv",
        lift_edge_lengths=np.ones(3),
        flux_type=flux_type,
    )

    return volume_rhs + surface_rhs


def initialize_reference_gaussian_state(nsub, order=4):
    state = build_projected_flux_state(nsub=nsub, order=order)
    q0 = q_values_for_case(state, "gaussian")
    state["q"] = q0.copy()
    state["volume_rhs"] = np.zeros_like(q0)
    return state


def exact_gaussian_on_state(state, t):
    q_exact = np.zeros_like(state["q"])
    for k, xyz in enumerate(state["xyz"]):
        q_exact[k, :] = exact_gaussian_bell_reference_style(
            xyz,
            t=t,
            R=state["R"],
        )
    return q_exact


def weighted_l2_error(state, error):
    area = weighted_integral(state, np.ones_like(error))
    return float(np.sqrt(weighted_integral(state, error * error) / area))


def observed_rate(previous_error, current_error, previous_h, current_h):
    if previous_error is None:
        return None
    return float(np.log(previous_error / current_error) / np.log(previous_h / current_h))


def sphere_mesh_h_from_state(state):
    return float(state["h"])


def run_gaussian_t01_case(nsub, rhs_mode, dt=1.0e-3, final_time=1.0e-1):
    state = initialize_reference_gaussian_state(nsub=nsub, order=4)
    q = state["q"].copy()
    q_initial = q.copy()
    res = np.zeros_like(q)
    t = 0.0

    if rhs_mode == "core_conservative_scaled":
        rhs_func = compute_sphere_rhs
        rhs_kwargs = {
            "state": state,
            "flux_type": "upwind",
            "surface_mode": "conservative_scaled",
        }
    elif rhs_mode == "projected_line_noDiv":
        rhs_func = compute_projected_line_rhs
        rhs_kwargs = {
            "state": state,
            "flux_type": "upwind",
        }
    else:
        raise ValueError(f"unknown rhs_mode: {rhs_mode}")

    while t < final_time - 1.0e-15:
        dt_step = min(dt, final_time - t)
        q, res = lsrk54_step(
            q,
            res,
            t,
            dt_step,
            rhs_func,
            **rhs_kwargs,
        )
        t += dt_step

    state["q"] = q
    q_exact = exact_gaussian_on_state(state, final_time)
    error = q - q_exact
    mass_initial = weighted_integral(state, q_initial)
    mass_final = weighted_integral(state, q)
    energy_initial = weighted_integral(state, q_initial * q_initial)
    energy_final = weighted_integral(state, q * q)

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "h": sphere_mesh_h_from_state(state),
        "L2_error": weighted_l2_error(state, error),
        "max_error": float(np.max(np.abs(error))),
        "mass_error": float(mass_final - mass_initial),
        "energy_change": float(energy_final - energy_initial),
        "has_nonfinite": bool(not np.all(np.isfinite(q))),
    }


def flux_compatibility_diagnostic(nsub, q_case, order=4):
    state = build_projected_flux_state(nsub=nsub, order=order)
    engine = state["engine"]
    q = q_values_for_case(state, q_case)

    differences_div_lref = []
    differences_no_div = []
    projected_values = []
    per_face_diff_div_lref = {face_id: [] for face_id in range(3)}
    per_face_diff_no_div = {face_id: [] for face_id in range(3)}

    for k in range(q.shape[0]):
        geometry = state["geometry"][k]
        A = geometry["J"] * state["u_tilde"][k] * q[k]
        B = geometry["J"] * state["v_tilde"][k] * q[k]

        for face_id in range(3):
            nodes = face_node_indices(engine, face_id)
            q_face = q[k, nodes]

            conormal_flux = physical_conormal_flux_on_face(state, k, face_id, nodes)
            F_current_divLref = (
                conormal_flux / REFERENCE_FACE_LENGTHS[face_id]
            ) * q_face
            F_current_noDiv = conormal_flux * q_face

            drdt = FACE_DRDT[face_id]
            dsdt = FACE_DSDT[face_id]
            F_projected = dsdt * A[nodes] - drdt * B[nodes]

            diff_div_lref = F_current_divLref - F_projected
            diff_no_div = F_current_noDiv - F_projected
            differences_div_lref.append(diff_div_lref)
            differences_no_div.append(diff_no_div)
            projected_values.append(F_projected)
            per_face_diff_div_lref[face_id].append(diff_div_lref)
            per_face_diff_no_div[face_id].append(diff_no_div)

    differences_div_lref = np.concatenate(differences_div_lref)
    differences_no_div = np.concatenate(differences_no_div)
    projected_values = np.concatenate(projected_values)
    denom = float(np.sqrt(np.sum(projected_values**2)))
    relative_l2_div_lref = (
        float(np.sqrt(np.sum(differences_div_lref**2)) / denom)
        if denom > 0.0
        else np.inf
    )
    relative_l2_no_div = (
        float(np.sqrt(np.sum(differences_no_div**2)) / denom)
        if denom > 0.0
        else np.inf
    )

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "q_case": q_case,
        "max_abs_difference_divLref": float(np.max(np.abs(differences_div_lref))),
        "rms_difference_divLref": rms(differences_div_lref),
        "relative_L2_difference_divLref": relative_l2_div_lref,
        "max_abs_difference_noDiv": float(np.max(np.abs(differences_no_div))),
        "rms_difference_noDiv": rms(differences_no_div),
        "relative_L2_difference_noDiv": relative_l2_no_div,
        "per_face_max_difference_divLref": {
            face_id: float(
                np.max(np.abs(np.concatenate(per_face_diff_div_lref[face_id])))
            )
            for face_id in range(3)
        },
        "per_face_max_difference_noDiv": {
            face_id: float(
                np.max(np.abs(np.concatenate(per_face_diff_no_div[face_id])))
            )
            for face_id in range(3)
        },
    }


def test_sphere_projected_flux_compatibility():
    print("\n" + "=" * 132)
    print("Sphere projected line flux compatibility diagnostic")
    print("=" * 132)
    print("F_current_divLref = (V3D dot (tau x n_surf) / Lref) * q_face")
    print("F_current_noDiv   = (V3D dot (tau x n_surf)) * q_face")
    print("F_projected       = dsdt * E(J*u_tilde*q) - drdt * E(J*v_tilde*q)")
    print("face 0: drdt= 2, dsdt= 0 | r varies, s constant")
    print("face 1: drdt=-2, dsdt= 2 | along r+s constant")
    print("face 2: drdt= 0, dsdt=-2 | s varies, r constant")
    print(
        "reference lengths: "
        f"face 0 Lref={REFERENCE_FACE_LENGTHS[0]:.12e}, "
        f"face 1 Lref={REFERENCE_FACE_LENGTHS[1]:.12e}, "
        f"face 2 Lref={REFERENCE_FACE_LENGTHS[2]:.12e}"
    )
    print("-" * 132)
    print(
        f"{'q_case':>10s} {'nsub':>6s} {'K':>8s} "
        f"{'max_divLref':>16s} {'rel_divLref':>16s} "
        f"{'max_noDiv':>16s} {'rel_noDiv':>16s}"
    )
    print("-" * 132)

    rows = []
    for q_case in ("constant", "gaussian"):
        for nsub in (4, 8, 16):
            row = flux_compatibility_diagnostic(nsub=nsub, q_case=q_case)
            rows.append(row)
            print(
                f"{q_case:>10s} {row['nsub']:6d} {row['K']:8d} "
                f"{row['max_abs_difference_divLref']:16.6e} "
                f"{row['relative_L2_difference_divLref']:16.6e} "
                f"{row['max_abs_difference_noDiv']:16.6e} "
                f"{row['relative_L2_difference_noDiv']:16.6e}"
            )

    print("-" * 132)
    print("Per-face max abs difference")
    print("-" * 132)
    print(
        f"{'q_case':>10s} {'nsub':>6s} "
        f"{'divLref_f0':>16s} {'divLref_f1':>16s} {'divLref_f2':>16s} "
        f"{'noDiv_f0':>16s} {'noDiv_f1':>16s} {'noDiv_f2':>16s}"
    )
    print("-" * 132)
    for row in rows:
        div_lref = row["per_face_max_difference_divLref"]
        no_div = row["per_face_max_difference_noDiv"]
        print(
            f"{row['q_case']:>10s} {row['nsub']:6d} "
            f"{div_lref[0]:16.6e} "
            f"{div_lref[1]:16.6e} "
            f"{div_lref[2]:16.6e} "
            f"{no_div[0]:16.6e} "
            f"{no_div[1]:16.6e} "
            f"{no_div[2]:16.6e}"
        )

    print("=" * 132)

    for row in rows:
        assert np.isfinite(row["max_abs_difference_divLref"])
        assert np.isfinite(row["rms_difference_divLref"])
        assert np.isfinite(row["relative_L2_difference_divLref"])
        assert np.isfinite(row["max_abs_difference_noDiv"])
        assert np.isfinite(row["rms_difference_noDiv"])
        assert np.isfinite(row["relative_L2_difference_noDiv"])
        for value in row["per_face_max_difference_divLref"].values():
            assert np.isfinite(value)
        for value in row["per_face_max_difference_noDiv"].values():
            assert np.isfinite(value)


def test_sphere_projected_surface_rhs_equivalence():
    print("\n" + "=" * 148)
    print("Sphere projected-line surface RHS equivalence diagnostic")
    print("=" * 148)
    print("A: core conservative_scaled surface RHS")
    print("case 1: divLref face speed, lift edge_lengths=ones(3)")
    print("case 2: noDiv face speed,   lift edge_lengths=ones(3)")
    print("case 3: noDiv face speed,   lift edge_lengths=reference face lengths")
    print(
        "reference lengths: "
        f"face 0 Lref={REFERENCE_FACE_LENGTHS[0]:.12e}, "
        f"face 1 Lref={REFERENCE_FACE_LENGTHS[1]:.12e}, "
        f"face 2 Lref={REFERENCE_FACE_LENGTHS[2]:.12e}"
    )

    rows = []
    for q_case in ("constant", "gaussian"):
        for nsub in (4, 8, 16):
            rows.append(
                surface_rhs_equivalence_diagnostic(
                    nsub=nsub,
                    q_case=q_case,
                    flux_type="upwind",
                )
            )

    for case_name in (
        "case1_divLref_lift1",
        "case2_noDiv_lift1",
        "case3_noDiv_liftRef",
    ):
        print("-" * 148)
        print(case_name)
        print("-" * 148)
        print(
            f"{'q_case':>10s} {'nsub':>6s} {'K':>8s} "
            f"{'max_abs(A-case)':>18s} {'rms(A-case)':>16s} "
            f"{'rel_L2':>14s} {'mass_A':>14s} {'mass_case':>14s} "
            f"{'energy_A':>14s} {'energy_case':>14s}"
        )
        print("-" * 148)
        for row in rows:
            stats = row["cases"][case_name]
            print(
                f"{row['q_case']:>10s} {row['nsub']:6d} {row['K']:8d} "
                f"{stats['max_abs']:18.6e} "
                f"{stats['rms']:16.6e} "
                f"{stats['relative_L2']:14.6e} "
                f"{stats['mass_A']:14.6e} "
                f"{stats['mass_case']:14.6e} "
                f"{stats['energy_A']:14.6e} "
                f"{stats['energy_case']:14.6e}"
            )

    print("=" * 148)

    for row in rows:
        for stats in row["cases"].values():
            assert np.isfinite(stats["max_abs"])
            assert np.isfinite(stats["rms"])
            assert np.isfinite(stats["relative_L2"])
            assert np.isfinite(stats["mass_A"])
            assert np.isfinite(stats["mass_case"])
            assert np.isfinite(stats["energy_A"])
            assert np.isfinite(stats["energy_case"])


def test_sphere_projected_jump_surface_rhs_equivalence():
    print("\n" + "=" * 148)
    print("Sphere projected-line jump-state surface RHS equivalence diagnostic")
    print("=" * 148)
    print("q[k, :] = 1.0 + 0.1 * (-1)**k")
    print("A: core conservative_scaled surface RHS")
    print("case 1: divLref face speed, lift edge_lengths=ones(3)")
    print("case 2: noDiv face speed,   lift edge_lengths=ones(3)")
    print("case 3: noDiv face speed,   lift edge_lengths=reference face lengths")
    print(
        "reference lengths: "
        f"face 0 Lref={REFERENCE_FACE_LENGTHS[0]:.12e}, "
        f"face 1 Lref={REFERENCE_FACE_LENGTHS[1]:.12e}, "
        f"face 2 Lref={REFERENCE_FACE_LENGTHS[2]:.12e}"
    )

    rows = []
    for flux_type in ("upwind", "central"):
        for nsub in (4, 8, 16):
            rows.append(
                surface_rhs_equivalence_diagnostic(
                    nsub=nsub,
                    q_case="jump",
                    flux_type=flux_type,
                )
            )

    for case_name in (
        "case1_divLref_lift1",
        "case2_noDiv_lift1",
        "case3_noDiv_liftRef",
    ):
        print("-" * 148)
        print(case_name)
        print("-" * 148)
        print(
            f"{'flux':>8s} {'nsub':>6s} {'K':>8s} "
            f"{'max_abs(A-case)':>18s} {'rms(A-case)':>16s} "
            f"{'rel_L2':>14s} {'mass_A':>14s} {'mass_case':>14s} "
            f"{'energy_A':>14s} {'energy_case':>14s}"
        )
        print("-" * 148)
        for row in rows:
            stats = row["cases"][case_name]
            print(
                f"{row['flux_type']:>8s} {row['nsub']:6d} {row['K']:8d} "
                f"{stats['max_abs']:18.6e} "
                f"{stats['rms']:16.6e} "
                f"{stats['relative_L2']:14.6e} "
                f"{stats['mass_A']:14.6e} "
                f"{stats['mass_case']:14.6e} "
                f"{stats['energy_A']:14.6e} "
                f"{stats['energy_case']:14.6e}"
            )

    print("=" * 148)

    for row in rows:
        for stats in row["cases"].values():
            assert np.isfinite(stats["max_abs"])
            assert np.isfinite(stats["rms"])
            assert np.isfinite(stats["relative_L2"])
            assert np.isfinite(stats["mass_A"])
            assert np.isfinite(stats["mass_case"])
            assert np.isfinite(stats["energy_A"])
            assert np.isfinite(stats["energy_case"])


def test_reference_gaussian_t01_projected_line_comparison():
    print("\n" + "=" * 132)
    print("Reference-style Gaussian T=0.1 RHS comparison diagnostic")
    print("=" * 132)
    print("q0=exp(-(dist/width)^2), width=1/sqrt(10), center=(1,0,0)")
    print("Omega=(-sin(-pi/4), 0, cos(-pi/4)), dt=1e-3, flux_type=upwind")
    print("A: core compute_sphere_rhs(surface_mode='conservative_scaled')")
    print("B: diagnostic projected_line surface RHS with noDiv face convention")

    for rhs_mode in ("core_conservative_scaled", "projected_line_noDiv"):
        print("-" * 132)
        print(rhs_mode)
        print("-" * 132)
        print(
            f"{'nsub':>6s} {'K':>8s} {'h':>13s} "
            f"{'L2_error':>16s} {'L2_rate':>10s} "
            f"{'max_error':>16s} {'max_rate':>10s} "
            f"{'mass_error':>16s} {'energy_change':>16s}"
        )
        print("-" * 132)

        previous = None
        rows = []
        for nsub in (4, 8, 16):
            row = run_gaussian_t01_case(nsub=nsub, rhs_mode=rhs_mode)
            L2_rate = observed_rate(
                None if previous is None else previous["L2_error"],
                row["L2_error"],
                None if previous is None else previous["h"],
                row["h"],
            )
            max_rate = observed_rate(
                None if previous is None else previous["max_error"],
                row["max_error"],
                None if previous is None else previous["h"],
                row["h"],
            )
            rows.append(row)
            previous = row

            print(
                f"{row['nsub']:6d} {row['K']:8d} {row['h']:13.6e} "
                f"{row['L2_error']:16.6e} "
                f"{'---' if L2_rate is None else f'{L2_rate:.4f}':>10s} "
                f"{row['max_error']:16.6e} "
                f"{'---' if max_rate is None else f'{max_rate:.4f}':>10s} "
                f"{row['mass_error']:16.6e} "
                f"{row['energy_change']:16.6e}"
            )

        for row in rows:
            assert not row["has_nonfinite"]
            assert np.isfinite(row["L2_error"])
            assert np.isfinite(row["max_error"])
            assert np.isfinite(row["mass_error"])
            assert np.isfinite(row["energy_change"])

    print("=" * 132)


if __name__ == "__main__":
    test_sphere_projected_flux_compatibility()
    test_sphere_projected_surface_rhs_equivalence()
    test_sphere_projected_jump_surface_rhs_equivalence()
    test_reference_gaussian_t01_projected_line_comparison()
