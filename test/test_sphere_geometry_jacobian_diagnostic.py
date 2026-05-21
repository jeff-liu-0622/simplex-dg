import numpy as np

from core.geometry.manifold_metrics import compute_manifold_geometry
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
    projected_sphere_mesh_hmin,
)
from core.operators import build_local_operators


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


def reference_shape_functions(r, s):
    phi1 = -0.5 * (r + s)
    phi2 = 0.5 * (1.0 + r)
    phi3 = 0.5 * (1.0 + s)
    return phi1, phi2, phi3


def affine_preprojection(rs, vertices):
    r = rs[:, 0]
    s = rs[:, 1]
    phi1, phi2, phi3 = reference_shape_functions(r, s)

    v1 = vertices[0]
    v2 = vertices[1]
    v3 = vertices[2]

    Y = (
        phi1[:, None] * v1[None, :]
        + phi2[:, None] * v2[None, :]
        + phi3[:, None] * v3[None, :]
    )
    Yr = 0.5 * (v2 - v1)
    Ys = 0.5 * (v3 - v1)

    return Y, Yr, Ys


def radial_projection_derivative(Y, dY, R):
    rho = np.linalg.norm(Y, axis=1, keepdims=True)
    Y_dot_dY = np.sum(Y * dY, axis=1, keepdims=True)
    return R * (dY / rho - Y * Y_dot_dY / (rho**3))


def analytic_volume_geometry(rs, vertices, R=1.0):
    Y, Yr, Ys = affine_preprojection(rs, vertices)
    rho = np.linalg.norm(Y, axis=1)
    triple = Y @ np.cross(Yr, Ys)
    J = (R**2) * np.abs(triple) / (rho**3)

    Xr = radial_projection_derivative(
        Y,
        np.broadcast_to(Yr, Y.shape),
        R,
    )
    Xs = radial_projection_derivative(
        Y,
        np.broadcast_to(Ys, Y.shape),
        R,
    )
    cross = np.cross(Xr, Xs)
    n = cross / np.linalg.norm(cross, axis=1)[:, None]

    return J, Xr, Xs, n


def analytic_face_conormal_flux(rs_face, vertices, face_id, R=1.0):
    Y, Yr, Ys = affine_preprojection(rs_face, vertices)
    Xr = radial_projection_derivative(
        Y,
        np.broadcast_to(Yr, Y.shape),
        R,
    )
    Xs = radial_projection_derivative(
        Y,
        np.broadcast_to(Ys, Y.shape),
        R,
    )
    cross = np.cross(Xr, Xs)
    n = cross / np.linalg.norm(cross, axis=1)[:, None]
    tau = FACE_DRDT[face_id] * Xr + FACE_DSDT[face_id] * Xs
    conormal = np.cross(tau, n)
    X = R * Y / np.linalg.norm(Y, axis=1)[:, None]
    V = reference_style_velocity(X)
    return np.sum(V * conormal, axis=1), np.linalg.norm(tau, axis=1)


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def relative_l2(diff, reference):
    num = float(np.sqrt(np.sum(diff * diff)))
    den = float(np.sqrt(np.sum(reference * reference)))
    if den <= 0.0:
        return 0.0 if num <= 0.0 else np.inf
    return num / den


def observed_rate(prev_error, error, prev_h, h):
    if prev_error is None:
        return None
    return float(np.log(prev_error / error) / np.log(prev_h / h))


def face_node_indices(engine, face_id):
    face_slice = engine.edge_slices[face_id]
    return np.arange(face_slice.start, face_slice.stop)


def build_geometry_diagnostic_row(nsub, order=4, R=1.0):
    engine = build_local_operators(N=order, n=order, rule="table1")
    _, _, _, EToV, _, nodes_xyz = create_projected_octahedron_sphere_mesh(
        nsub=nsub,
        R=R,
    )
    element_xyz = map_reference_nodes_to_projected_sphere(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        r=engine.r,
        s=engine.s,
        R=R,
    )

    rs = np.column_stack([engine.r, engine.s])
    h = projected_sphere_mesh_hmin(nodes_xyz, EToV)

    j_diff = []
    j_ref = []
    conormal_flux_diff = []
    conormal_flux_ref = []
    face_jacobian_diff = []
    face_jacobian_ref = []
    per_face_flux_diff = {face_id: [] for face_id in range(3)}

    for k, elem in enumerate(EToV):
        vertices = nodes_xyz[elem]
        geometry = compute_manifold_geometry(engine, element_xyz[k])

        J_analytic, _, _, _ = analytic_volume_geometry(rs, vertices, R=R)
        j_diff.append(geometry["J"] - J_analytic)
        j_ref.append(J_analytic)

        for face_id in range(3):
            nodes = face_node_indices(engine, face_id)
            a1 = geometry["a1"][nodes, :]
            a2 = geometry["a2"][nodes, :]
            n_numeric = geometry["n"][nodes, :]
            tau_numeric = FACE_DRDT[face_id] * a1 + FACE_DSDT[face_id] * a2
            conormal_numeric = np.cross(tau_numeric, n_numeric)
            V_numeric = reference_style_velocity(element_xyz[k, nodes, :])
            flux_numeric = np.sum(V_numeric * conormal_numeric, axis=1)
            face_j_numeric = np.linalg.norm(tau_numeric, axis=1)

            rs_face = rs[nodes, :]
            flux_analytic, face_j_analytic = analytic_face_conormal_flux(
                rs_face,
                vertices,
                face_id,
                R=R,
            )

            flux_diff = flux_numeric - flux_analytic
            conormal_flux_diff.append(flux_diff)
            conormal_flux_ref.append(flux_analytic)
            face_jacobian_diff.append(face_j_numeric - face_j_analytic)
            face_jacobian_ref.append(face_j_analytic)
            per_face_flux_diff[face_id].append(flux_diff)

    j_diff = np.concatenate(j_diff)
    j_ref = np.concatenate(j_ref)
    conormal_flux_diff = np.concatenate(conormal_flux_diff)
    conormal_flux_ref = np.concatenate(conormal_flux_ref)
    face_jacobian_diff = np.concatenate(face_jacobian_diff)
    face_jacobian_ref = np.concatenate(face_jacobian_ref)

    return {
        "nsub": nsub,
        "K": int(EToV.shape[0]),
        "h": float(h),
        "J_max_abs": float(np.max(np.abs(j_diff))),
        "J_rms": rms(j_diff),
        "J_rel_L2": relative_l2(j_diff, j_ref),
        "face_flux_max_abs": float(np.max(np.abs(conormal_flux_diff))),
        "face_flux_rms": rms(conormal_flux_diff),
        "face_flux_rel_L2": relative_l2(conormal_flux_diff, conormal_flux_ref),
        "face_jac_max_abs": float(np.max(np.abs(face_jacobian_diff))),
        "face_jac_rms": rms(face_jacobian_diff),
        "face_jac_rel_L2": relative_l2(face_jacobian_diff, face_jacobian_ref),
        "per_face_flux_max_abs": {
            face_id: float(np.max(np.abs(np.concatenate(per_face_flux_diff[face_id]))))
            for face_id in range(3)
        },
    }


def test_sphere_geometry_jacobian_diagnostic():
    rows = []
    prev = None

    print("\n" + "=" * 132)
    print("Sphere geometry Jacobian diagnostic")
    print("=" * 132)
    print("J_numeric  = ||Dr X x Ds X|| from compute_manifold_geometry")
    print("J_analytic = R^2 |Y dot (Y_r x Y_s)| / ||Y||^3 for radial projection")
    print("-" * 132)
    print(
        f"{'nsub':>6s} {'K':>8s} {'h':>13s} "
        f"{'J_max_abs':>14s} {'J_rms':>14s} {'J_rel_L2':>14s} {'rate':>9s} "
        f"{'faceFlux_rel':>14s} {'faceJ_rel':>14s}"
    )
    print("-" * 132)

    for nsub in (2, 4, 8, 16):
        row = build_geometry_diagnostic_row(nsub=nsub)
        rate = observed_rate(
            None if prev is None else prev["J_rel_L2"],
            row["J_rel_L2"],
            None if prev is None else prev["h"],
            row["h"],
        )
        rows.append(row)
        prev = row

        print(
            f"{row['nsub']:6d} {row['K']:8d} {row['h']:13.6e} "
            f"{row['J_max_abs']:14.6e} "
            f"{row['J_rms']:14.6e} "
            f"{row['J_rel_L2']:14.6e} "
            f"{'---' if rate is None else f'{rate:.4f}':>9s} "
            f"{row['face_flux_rel_L2']:14.6e} "
            f"{row['face_jac_rel_L2']:14.6e}"
        )

    print("-" * 132)
    print("Per-face max abs current V dot (tau x n) minus analytic V dot (tau x n)")
    print("-" * 132)
    print(
        f"{'nsub':>6s} {'face0':>14s} {'face1':>14s} {'face2':>14s}"
    )
    print("-" * 132)

    for row in rows:
        per_face = row["per_face_flux_max_abs"]
        print(
            f"{row['nsub']:6d} "
            f"{per_face[0]:14.6e} "
            f"{per_face[1]:14.6e} "
            f"{per_face[2]:14.6e}"
        )

    print("=" * 132)

    for row in rows:
        for key in (
            "J_max_abs",
            "J_rms",
            "J_rel_L2",
            "face_flux_max_abs",
            "face_flux_rms",
            "face_flux_rel_L2",
            "face_jac_max_abs",
            "face_jac_rms",
            "face_jac_rel_L2",
        ):
            assert np.isfinite(row[key])


if __name__ == "__main__":
    test_sphere_geometry_jacobian_diagnostic()
