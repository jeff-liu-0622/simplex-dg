import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.manifold_metrics import compute_manifold_geometry
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
)
from core.operators import build_local_operators
from core.operators_sphere import compute_sphere_rhs
from test.test_manifold_velocity_sphere import solid_body_rotation_velocity
from test.test_sphere_full_rhs_smooth_snapshot import _weighted_integral


def build_projected_constant_state(nsub=16, order=4, R=1.0, u0=1.0, alpha=np.pi / 4.0):
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
    q = np.ones((EToV.shape[0], engine.num_nodes))

    for k in range(EToV.shape[0]):
        geometries.append(compute_manifold_geometry(engine, element_xyz[k]))
        velocities.append(solid_body_rotation_velocity(element_xyz[k], u0=u0, alpha=alpha))

    return {
        "engine": engine,
        "EToE": EToE,
        "EToF": EToF,
        "patch_ids": patch_ids,
        "xyz": element_xyz,
        "geometry": geometries,
        "V3D": np.asarray(velocities),
        "u_tilde": np.zeros_like(q),
        "v_tilde": np.zeros_like(q),
        "q": q,
        "volume_rhs": np.zeros_like(q),
    }


def rms(values):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values**2))


def run_core_api_case(flux_type):
    state = build_projected_constant_state()
    q = state["q"].copy()

    rhs = compute_sphere_rhs(
        q,
        0.0,
        state=state,
        flux_type=flux_type,
        surface_mode="conservative_scaled",
    )

    return {
        "flux_type": flux_type,
        "max_abs_rhs": float(np.max(np.abs(rhs))),
        "rms_rhs": float(rms(rhs)),
        "mass_total": float(_weighted_integral(state, rhs)),
    }


def test_sphere_core_api_constant_state_consistency():
    results = [run_core_api_case("central"), run_core_api_case("upwind")]

    for result in results:
        assert np.isfinite(result["max_abs_rhs"])
        assert np.isfinite(result["rms_rhs"])
        assert np.isfinite(result["mass_total"])
        assert result["max_abs_rhs"] < 1.0e-4
        assert result["rms_rhs"] < 1.0e-4
        assert abs(result["mass_total"]) < 1.0e-12


def run_all_tests():
    print("\n" + "=" * 90)
    print("sphere core API consistency | projected 3D manifold route | q=1")
    print("=" * 90)
    print(f"{'flux':>10s} {'max_abs_rhs':>16s} {'rms_rhs':>16s} {'mass_total':>16s}")
    print("-" * 90)

    results = [run_core_api_case("central"), run_core_api_case("upwind")]

    for result in results:
        print(
            f"{result['flux_type']:>10s} "
            f"{result['max_abs_rhs']:16.6e} "
            f"{result['rms_rhs']:16.6e} "
            f"{result['mass_total']:16.6e}"
        )

    print("-" * 90)

    for result in results:
        if result["max_abs_rhs"] >= 1.0e-4:
            raise AssertionError(
                f"{result['flux_type']} max_abs_rhs too large: "
                f"{result['max_abs_rhs']:.6e}"
            )
        if result["rms_rhs"] >= 1.0e-4:
            raise AssertionError(
                f"{result['flux_type']} rms_rhs too large: "
                f"{result['rms_rhs']:.6e}"
            )
        if abs(result["mass_total"]) >= 1.0e-12:
            raise AssertionError(
                f"{result['flux_type']} mass_total too large: "
                f"{result['mass_total']:.6e}"
            )

    print("sphere core API consistency passed")


if __name__ == "__main__":
    run_all_tests()
