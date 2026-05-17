import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
)
from core.operators import build_local_operators
from test.test_manifold_constant_rhs_sphere import compute_manifold_skew_volume_rhs
from test.test_manifold_geometry_sphere import compute_manifold_geometry
from test.test_manifold_velocity_sphere import solid_body_rotation_velocity
from test.test_sphere_full_rhs_constant import compute_sphere_surface_penalty


def smooth_sphere_snapshot_q(xyz):
    """
    Smooth nonconstant scalar field on the sphere.

    This linear field is intentionally simple and numerically mild. Because it
    is evaluated from physical 3D coordinates, matching shared-face nodes should
    produce matching traces on the projected sphere topology.
    """
    return xyz[:, 0] + 0.5 * xyz[:, 1] - 0.25 * xyz[:, 2]


def build_projected_sphere_smooth_state(
    nsub=16,
    order=4,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
):
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
    q = np.zeros((EToV.shape[0], engine.num_nodes))

    for k in range(EToV.shape[0]):
        xyz = element_xyz[k]
        q[k, :] = smooth_sphere_snapshot_q(xyz)
        geometry = compute_manifold_geometry(engine, xyz)
        V3D = solid_body_rotation_velocity(xyz, u0=u0, alpha=alpha)
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


def _weighted_integral(state, values):
    engine = state["engine"]
    total = 0.0

    for k, geometry in enumerate(state["geometry"]):
        total += engine.area * np.sum(engine.w_s * geometry["J"] * values[k])

    return float(total)


def _rhs_summary(state, rhs, volume_rhs, surface_rhs, flux_type, alpha_lf):
    q = state["q"]

    return {
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "rhs": rhs,
        "volume_rhs": volume_rhs,
        "surface_rhs": surface_rhs,
        "max_abs_rhs": float(np.max(np.abs(rhs))),
        "rms_rhs": float(np.sqrt(np.mean(rhs**2))),
        "mean_rhs": float(np.mean(rhs)),
        "max_abs_volume": float(np.max(np.abs(volume_rhs))),
        "rms_volume": float(np.sqrt(np.mean(volume_rhs**2))),
        "max_abs_surface": float(np.max(np.abs(surface_rhs))),
        "rms_surface": float(np.sqrt(np.mean(surface_rhs**2))),
        "mass_rate": _weighted_integral(state, rhs),
        "energy_rate": _weighted_integral(state, q * rhs),
    }


def compute_sphere_full_rhs_smooth_snapshot(
    nsub=16,
    order=4,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
    flux_type="upwind",
    alpha_lf=1.0,
    surface_mode="conservative_scaled",
):
    state = build_projected_sphere_smooth_state(
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
    rhs = volume_rhs + surface_rhs
    result = _rhs_summary(
        state=state,
        rhs=rhs,
        volume_rhs=volume_rhs,
        surface_rhs=surface_rhs,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
    )
    result["state"] = state
    result["surface_mode"] = surface_mode
    result["max_face_match_error"] = surface["max_face_match_error"]
    result["max_abs_penalty"] = surface["max_abs_penalty"]
    result["max_abs_vn_orientation_error"] = surface["max_abs_vn_orientation_error"]

    return result


def test_projected_sphere_full_rhs_smooth_snapshot_flux_diagnostic():
    cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    results = []

    print("\n" + "=" * 154)
    print("Projected sphere full RHS smooth nonconstant snapshot diagnostic")
    print("=" * 154)
    print("q = X + 0.5Y - 0.25Z, surface_mode=conservative_scaled")
    print(
        f"{'flux':>10s} "
        f"{'alpha_lf':>10s} "
        f"{'max_abs_rhs':>15s} "
        f"{'rms_rhs':>15s} "
        f"{'mean_rhs':>15s} "
        f"{'max_abs_volume':>16s} "
        f"{'rms_volume':>15s} "
        f"{'max_abs_surface':>18s} "
        f"{'rms_surface':>15s} "
        f"{'mass_rate':>15s} "
        f"{'energy_rate':>15s}"
    )
    print("-" * 154)

    for flux_type, alpha_lf in cases:
        result = compute_sphere_full_rhs_smooth_snapshot(
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
            f"{result['max_abs_rhs']:15.6e} "
            f"{result['rms_rhs']:15.6e} "
            f"{result['mean_rhs']:15.6e} "
            f"{result['max_abs_volume']:16.6e} "
            f"{result['rms_volume']:15.6e} "
            f"{result['max_abs_surface']:18.6e} "
            f"{result['rms_surface']:15.6e} "
            f"{result['mass_rate']:15.6e} "
            f"{result['energy_rate']:15.6e}"
        )

    print("-" * 154)

    upwind = next(row for row in results if row["flux_type"] == "upwind")
    lf_alpha_1 = next(
        row
        for row in results
        if row["flux_type"] == "lf" and abs(row["alpha_lf"] - 1.0) < 1.0e-14
    )
    lf_alpha_15 = next(
        row
        for row in results
        if row["flux_type"] == "lf" and abs(row["alpha_lf"] - 1.5) < 1.0e-14
    )

    max_upwind_lf1_rhs_diff = float(np.max(np.abs(upwind["rhs"] - lf_alpha_1["rhs"])))
    max_upwind_lf15_surface_diff = float(
        np.max(np.abs(upwind["surface_rhs"] - lf_alpha_15["surface_rhs"]))
    )

    print(f"max upwind/LF(alpha=1) RHS difference = {max_upwind_lf1_rhs_diff:.6e}")
    print(
        "max upwind/LF(alpha=1.5) surface difference = "
        f"{max_upwind_lf15_surface_diff:.6e}"
    )
    print(
        "max_face_match_error = "
        f"{max(row['max_face_match_error'] for row in results):.6e}"
    )
    print(
        "max_abs_penalty = "
        f"{max(row['max_abs_penalty'] for row in results):.6e}"
    )
    print("=" * 154)

    for result in results:
        scalar_values = [
            result["max_abs_rhs"],
            result["rms_rhs"],
            result["mean_rhs"],
            result["max_abs_volume"],
            result["rms_volume"],
            result["max_abs_surface"],
            result["rms_surface"],
            result["mass_rate"],
            result["energy_rate"],
            result["max_face_match_error"],
            result["max_abs_penalty"],
            result["max_abs_vn_orientation_error"],
        ]

        assert np.all(np.isfinite(result["rhs"]))
        assert np.all(np.isfinite(result["volume_rhs"]))
        assert np.all(np.isfinite(result["surface_rhs"]))
        assert np.all(np.isfinite(scalar_values))
        assert result["max_face_match_error"] < 1.0e-12

    assert max_upwind_lf1_rhs_diff < 1.0e-14, (
        "upwind and LF(alpha=1) should match: "
        f"max difference = {max_upwind_lf1_rhs_diff:.3e}"
    )


def run_all_tests():
    test_projected_sphere_full_rhs_smooth_snapshot_flux_diagnostic()
    print("test_sphere_full_rhs_smooth_snapshot.py passed")


if __name__ == "__main__":
    run_all_tests()
