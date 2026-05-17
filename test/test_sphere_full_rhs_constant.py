import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
)
from core.operators import build_local_operators
from core.rhs_sphere import compute_sphere_surface_penalty as _core_surface_penalty
from test.test_manifold_constant_rhs_sphere import compute_manifold_skew_volume_rhs
from test.test_manifold_geometry_sphere import compute_manifold_geometry
from test.test_manifold_velocity_sphere import solid_body_rotation_velocity


REFERENCE_FACE_NORMALS = np.array(
    [
        [0.0, -1.0],
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        [-1.0, 0.0],
    ],
    dtype=float,
)


def build_projected_sphere_rhs_state(nsub=16, order=4, R=1.0, u0=1.0, alpha=np.pi / 4.0):
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
    volume_rhs = []
    q = np.ones((EToV.shape[0], engine.num_nodes))

    for k in range(EToV.shape[0]):
        geometry = compute_manifold_geometry(engine, element_xyz[k])
        V3D = solid_body_rotation_velocity(element_xyz[k], u0=u0, alpha=alpha)
        rhs_vol, _, u_local, v_local = compute_manifold_skew_volume_rhs(
            engine=engine,
            geometry=geometry,
            V3D=V3D,
            q=q[k],
        )

        geometries.append(geometry)
        velocities.append(V3D)
        u_tilde.append(u_local)
        v_tilde.append(v_local)
        volume_rhs.append(rhs_vol)

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
        "q": q,
        "volume_rhs": np.asarray(volume_rhs),
    }


def _aligned_neighbor_face_indices(xyz_M, xyz_P):
    direct = np.max(np.linalg.norm(xyz_M - xyz_P, axis=1))
    reverse = np.max(np.linalg.norm(xyz_M - xyz_P[::-1], axis=1))

    if reverse < direct:
        return np.arange(xyz_P.shape[0] - 1, -1, -1), reverse

    return np.arange(xyz_P.shape[0]), direct


def _flux_coefficient(v_n_sJ, flux_type, alpha_lf):
    if flux_type == "central":
        return 0.0
    if flux_type == "upwind":
        return np.abs(v_n_sJ)
    if flux_type == "lf":
        return alpha_lf * np.abs(v_n_sJ)
    raise ValueError(f"unknown flux_type: {flux_type}")


def compute_sphere_surface_penalty(
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    surface_mode="old",
    face_match_tol=1.0e-12,
):
    """
    Compute the projected-sphere face penalty skeleton.

    The face penalty uses v_n_sJ as requested by the manifold diagnostic note.
    We pass unit edge lengths to the existing lift routine because the face
    metric flux already carries the geometric scaling for this skeleton.
    """
    return _core_surface_penalty(
        state,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        surface_mode=surface_mode,
        face_match_tol=face_match_tol,
    )


def compute_sphere_full_rhs_constant_diagnostic(
    nsub=16,
    order=4,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
    flux_type="upwind",
    alpha_lf=1.0,
    surface_mode="conservative_scaled",
):
    state = build_projected_sphere_rhs_state(
        nsub=nsub,
        order=order,
        R=R,
        u0=u0,
        alpha=alpha,
    )
    surface = compute_sphere_surface_penalty(
        state,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        surface_mode=surface_mode,
    )

    volume_rhs = state["volume_rhs"]
    surface_rhs = surface["surface_rhs"]
    full_rhs = volume_rhs + surface_rhs

    return {
        "nsub": nsub,
        "order": order,
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "surface_mode": surface_mode,
        "rhs": full_rhs,
        "surface_rhs": surface_rhs,
        "volume_rhs": volume_rhs,
        "max_abs_rhs": float(np.max(np.abs(full_rhs))),
        "rms_rhs": float(np.sqrt(np.mean(full_rhs**2))),
        "mean_rhs": float(np.mean(full_rhs)),
        "max_abs_surface": float(np.max(np.abs(surface_rhs))),
        "rms_surface": float(np.sqrt(np.mean(surface_rhs**2))),
        "max_abs_volume": float(np.max(np.abs(volume_rhs))),
        "max_face_match_error": surface["max_face_match_error"],
        "max_abs_penalty": surface["max_abs_penalty"],
        "max_abs_vn_orientation_error": surface["max_abs_vn_orientation_error"],
    }


def test_projected_sphere_full_rhs_constant_state_flux_skeleton():
    cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    results = []

    print("\n" + "=" * 118)
    print("Projected sphere full RHS constant-state diagnostic")
    print("=" * 118)
    print(
        f"{'flux':>10s} "
        f"{'alpha_lf':>10s} "
        f"{'max_abs_rhs':>16s} "
        f"{'rms_rhs':>16s} "
        f"{'mean_rhs':>16s} "
        f"{'max_abs_surface':>18s} "
        f"{'rms_surface':>16s} "
        f"{'max_abs_volume':>16s}"
    )
    print("-" * 118)

    for flux_type, alpha_lf in cases:
        result = compute_sphere_full_rhs_constant_diagnostic(
            nsub=16,
            order=4,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            surface_mode="conservative_scaled",
        )
        results.append(result)

        print(
            f"{flux_type:>10s} "
            f"{alpha_lf:10.4f} "
            f"{result['max_abs_rhs']:16.6e} "
            f"{result['rms_rhs']:16.6e} "
            f"{result['mean_rhs']:16.6e} "
            f"{result['max_abs_surface']:18.6e} "
            f"{result['rms_surface']:16.6e} "
            f"{result['max_abs_volume']:16.6e}"
        )

    print("-" * 118)
    print(
        "max_face_match_error = "
        f"{max(row['max_face_match_error'] for row in results):.6e}"
    )
    print(
        "max_abs_penalty = "
        f"{max(row['max_abs_penalty'] for row in results):.6e}"
    )
    print("=" * 118)

    upwind = next(row for row in results if row["flux_type"] == "upwind")
    lf_alpha_1 = next(
        row
        for row in results
        if row["flux_type"] == "lf" and abs(row["alpha_lf"] - 1.0) < 1.0e-14
    )
    max_upwind_lf_difference = np.max(np.abs(upwind["rhs"] - lf_alpha_1["rhs"]))

    for result in results:
        assert np.all(np.isfinite(result["rhs"]))
        assert np.all(np.isfinite(result["surface_rhs"]))
        assert result["max_face_match_error"] < 1.0e-12
        assert result["max_abs_surface"] < 1.0e-13
        assert result["rms_surface"] < 1.0e-13
        assert result["max_abs_rhs"] < 1.0e-4
        assert result["max_abs_volume"] < 1.0e-4

    assert max_upwind_lf_difference < 1.0e-14, (
        "upwind and LF(alpha=1) should match for q=1: "
        f"max difference = {max_upwind_lf_difference:.3e}"
    )


def run_all_tests():
    test_projected_sphere_full_rhs_constant_state_flux_skeleton()
    print("test_sphere_full_rhs_constant.py passed")


if __name__ == "__main__":
    run_all_tests()
