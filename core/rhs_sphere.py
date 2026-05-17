import numpy as np


REFERENCE_FACE_NORMALS = np.array(
    [
        [0.0, -1.0],
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        [-1.0, 0.0],
    ],
    dtype=float,
)

REFERENCE_FACE_LENGTHS = np.array([2.0, 2.0 * np.sqrt(2.0), 2.0], dtype=float)


def aligned_neighbor_face_indices(xyz_M, xyz_P):
    direct = np.max(np.linalg.norm(xyz_M - xyz_P, axis=1))
    reverse = np.max(np.linalg.norm(xyz_M - xyz_P[::-1], axis=1))

    if reverse < direct:
        return np.arange(xyz_P.shape[0] - 1, -1, -1), reverse

    return np.arange(xyz_P.shape[0]), direct


def sphere_flux_coefficient(vn, flux_type, alpha_lf):
    if flux_type == "central":
        return 0.0
    if flux_type == "upwind":
        return np.abs(vn)
    if flux_type == "lf":
        return alpha_lf * np.abs(vn)
    raise ValueError(f"unknown flux_type: {flux_type}")


def face_node_indices(engine, face_id):
    face_slice = engine.edge_slices[face_id]
    return np.arange(face_slice.start, face_slice.stop)


def physical_conormal_flux_on_face(state, elem_id, face_id, nodes):
    """
    Compute physical line-metric normal velocity for one sphere face.

    The tangent is dX/dlambda for lambda in [0, 1] following the local face
    node ordering.  The conormal tau x n_surf is outward for the reference
    triangle face ordering used by the boundary quadrature nodes.
    """
    geometry = state["geometry"][elem_id]
    a1 = geometry["a1"][nodes, :]
    a2 = geometry["a2"][nodes, :]
    n_surf = geometry["n"][nodes, :]
    V3D = state["V3D"][elem_id, nodes, :]

    if face_id == 0:
        tau = 2.0 * a1
    elif face_id == 1:
        tau = 2.0 * (a2 - a1)
    elif face_id == 2:
        tau = -2.0 * a2
    else:
        raise ValueError(f"unknown face_id: {face_id}")

    conormal_sJ = np.cross(tau, n_surf)
    return np.sum(V3D * conormal_sJ, axis=1)


def compute_sphere_surface_penalty_old(
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    face_match_tol=1.0e-12,
):
    """
    Original projected-sphere surface penalty skeleton.

    This mode is kept for diagnostics and comparison.  It uses
    J_face * (nr*u_tilde + ns*v_tilde) as the face metric flux and computes
    each element face independently.
    """
    engine = state["engine"]
    EToE = state["EToE"]
    EToF = state["EToF"]
    xyz = state["xyz"]
    geometries = state["geometry"]
    q = state["q"]
    u_tilde = state["u_tilde"]
    v_tilde = state["v_tilde"]

    surface_rhs = np.zeros_like(q)
    max_face_match_error = 0.0
    max_abs_penalty = 0.0

    for kM in range(q.shape[0]):
        p_boundary = np.zeros(engine.num_boundary_nodes)

        for fM in range(3):
            kP = int(EToE[kM, fM])
            fP = int(EToF[kM, fM])

            if kP == kM:
                raise AssertionError(f"unexpected boundary face ({kM}, {fM})")

            nodes_M = face_node_indices(engine, fM)
            nodes_P = face_node_indices(engine, fP)
            ordering, face_match_error = aligned_neighbor_face_indices(
                xyz[kM, nodes_M, :],
                xyz[kP, nodes_P, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            qM = q[kM, nodes_M]
            qP = q[kP, nodes_P[ordering]]

            nr, ns = REFERENCE_FACE_NORMALS[fM]
            J_face = geometries[kM]["J"][nodes_M]
            v_n_sJ = J_face * (nr * u_tilde[kM, nodes_M] + ns * v_tilde[kM, nodes_M])
            C = sphere_flux_coefficient(v_n_sJ, flux_type=flux_type, alpha_lf=alpha_lf)

            penalty = 0.5 * (v_n_sJ - C) * (qM - qP)
            max_abs_penalty = max(max_abs_penalty, float(np.max(np.abs(penalty))))
            p_boundary[engine.edge_slices[fM]] = penalty

        lifted = engine.lift_boundary_penalty(
            p_boundary,
            edge_lengths=np.ones(3),
        )
        surface_rhs[kM, :] = lifted / geometries[kM]["J"]

    if max_face_match_error > face_match_tol:
        raise AssertionError(
            "projected sphere face pairing is not physically continuous: "
            f"max face match error = {max_face_match_error:.3e}"
        )

    return {
        "surface_rhs": surface_rhs,
        "max_face_match_error": max_face_match_error,
        "max_abs_penalty": max_abs_penalty,
        "max_abs_vn_orientation_error": np.nan,
        "surface_mode": "old",
    }


def compute_sphere_surface_penalty_conservative(
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    face_match_tol=1.0e-12,
):
    """
    Conservative shared-face sphere surface penalty.

    The physical line metric is included through V3D dot (tau x n_surf).
    The lift is called with unit edge lengths so it applies edge quadrature
    weights only.  Each shared face contributes equal and opposite penalties
    in paired face-node ordering, giving global mass cancellation.
    """
    engine = state["engine"]
    EToE = state["EToE"]
    EToF = state["EToF"]
    xyz = state["xyz"]
    q = state["q"]

    p_boundary = np.zeros((q.shape[0], engine.num_boundary_nodes))
    max_face_match_error = 0.0
    max_abs_penalty = 0.0
    max_abs_vn_orientation_error = 0.0

    for kM in range(q.shape[0]):
        for fM in range(3):
            kP = int(EToE[kM, fM])
            fP = int(EToF[kM, fM])

            if kP == kM:
                raise AssertionError(f"unexpected boundary face ({kM}, {fM})")

            if (kM, fM) > (kP, fP):
                continue

            nodes_M = face_node_indices(engine, fM)
            nodes_P = face_node_indices(engine, fP)
            ordering, face_match_error = aligned_neighbor_face_indices(
                xyz[kM, nodes_M, :],
                xyz[kP, nodes_P, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            qM = q[kM, nodes_M]
            qP = q[kP, nodes_P[ordering]]
            vnM = physical_conormal_flux_on_face(state, kM, fM, nodes_M)
            vnP = physical_conormal_flux_on_face(state, kP, fP, nodes_P)[ordering]
            max_abs_vn_orientation_error = max(
                max_abs_vn_orientation_error,
                float(np.max(np.abs(vnM + vnP))),
            )

            vn = 0.5 * (vnM - vnP)
            C = sphere_flux_coefficient(vn, flux_type=flux_type, alpha_lf=alpha_lf)
            penalty_M = 0.5 * (vn - C) * (qM - qP)
            penalty_P_aligned = -penalty_M

            penalty_P_native = np.empty_like(penalty_P_aligned)
            penalty_P_native[ordering] = penalty_P_aligned

            p_boundary[kM, engine.edge_slices[fM]] = penalty_M
            p_boundary[kP, engine.edge_slices[fP]] = penalty_P_native
            max_abs_penalty = max(
                max_abs_penalty,
                float(np.max(np.abs(penalty_M))),
            )

    if max_face_match_error > face_match_tol:
        raise AssertionError(
            "projected sphere face pairing is not physically continuous: "
            f"max face match error = {max_face_match_error:.3e}"
        )

    surface_rhs = np.zeros_like(q)
    for k in range(q.shape[0]):
        lifted = engine.lift_boundary_penalty(
            p_boundary[k],
            edge_lengths=np.ones(3),
        )
        surface_rhs[k, :] = lifted / state["geometry"][k]["J"]

    return {
        "surface_rhs": surface_rhs,
        "max_face_match_error": max_face_match_error,
        "max_abs_penalty": max_abs_penalty,
        "max_abs_vn_orientation_error": max_abs_vn_orientation_error,
        "surface_mode": "conservative",
    }


def compute_sphere_surface_penalty_conservative_scaled(
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    face_match_tol=1.0e-12,
):
    """
    Conservative shared-face sphere surface penalty in reference-face scaling.

    This mode uses the physical conormal flux divided by the local reference
    face length,

        [V3D dot (tau x n_surf)] / Lref,

    which matches the reference contravariant metric flux
    J * (nr*u_tilde + ns*v_tilde).  Each physical shared face is visited once,
    then equal and opposite paired penalties are assigned before lifting.
    """
    engine = state["engine"]
    EToE = state["EToE"]
    EToF = state["EToF"]
    xyz = state["xyz"]
    q = state["q"]

    p_boundary = np.zeros((q.shape[0], engine.num_boundary_nodes))
    max_face_match_error = 0.0
    max_abs_penalty = 0.0
    max_abs_vn_orientation_error = 0.0

    for kM in range(q.shape[0]):
        for fM in range(3):
            kP = int(EToE[kM, fM])
            fP = int(EToF[kM, fM])

            if kP == kM:
                raise AssertionError(f"unexpected boundary face ({kM}, {fM})")

            if (kM, fM) > (kP, fP):
                continue

            nodes_M = face_node_indices(engine, fM)
            nodes_P = face_node_indices(engine, fP)
            ordering, face_match_error = aligned_neighbor_face_indices(
                xyz[kM, nodes_M, :],
                xyz[kP, nodes_P, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            qM = q[kM, nodes_M]
            qP = q[kP, nodes_P[ordering]]
            vnM = (
                physical_conormal_flux_on_face(state, kM, fM, nodes_M)
                / REFERENCE_FACE_LENGTHS[fM]
            )
            vnP = (
                physical_conormal_flux_on_face(state, kP, fP, nodes_P)[ordering]
                / REFERENCE_FACE_LENGTHS[fP]
            )
            max_abs_vn_orientation_error = max(
                max_abs_vn_orientation_error,
                float(np.max(np.abs(vnM + vnP))),
            )

            vn = 0.5 * (vnM - vnP)
            C = sphere_flux_coefficient(vn, flux_type=flux_type, alpha_lf=alpha_lf)
            penalty_M = 0.5 * (vn - C) * (qM - qP)
            penalty_P_aligned = -penalty_M

            penalty_P_native = np.empty_like(penalty_P_aligned)
            penalty_P_native[ordering] = penalty_P_aligned

            p_boundary[kM, engine.edge_slices[fM]] = penalty_M
            p_boundary[kP, engine.edge_slices[fP]] = penalty_P_native
            max_abs_penalty = max(
                max_abs_penalty,
                float(np.max(np.abs(penalty_M))),
            )

    if max_face_match_error > face_match_tol:
        raise AssertionError(
            "projected sphere face pairing is not physically continuous: "
            f"max face match error = {max_face_match_error:.3e}"
        )

    surface_rhs = np.zeros_like(q)
    for k in range(q.shape[0]):
        lifted = engine.lift_boundary_penalty(
            p_boundary[k],
            edge_lengths=np.ones(3),
        )
        surface_rhs[k, :] = lifted / state["geometry"][k]["J"]

    return {
        "surface_rhs": surface_rhs,
        "max_face_match_error": max_face_match_error,
        "max_abs_penalty": max_abs_penalty,
        "max_abs_vn_orientation_error": max_abs_vn_orientation_error,
        "surface_mode": "conservative_scaled",
    }


def compute_sphere_surface_penalty(
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    surface_mode="conservative",
    face_match_tol=1.0e-12,
):
    if surface_mode == "old":
        return compute_sphere_surface_penalty_old(
            state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            face_match_tol=face_match_tol,
        )
    if surface_mode == "conservative":
        return compute_sphere_surface_penalty_conservative(
            state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            face_match_tol=face_match_tol,
        )
    if surface_mode == "conservative_scaled":
        return compute_sphere_surface_penalty_conservative_scaled(
            state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            face_match_tol=face_match_tol,
        )
    raise ValueError(f"unknown surface_mode: {surface_mode}")
