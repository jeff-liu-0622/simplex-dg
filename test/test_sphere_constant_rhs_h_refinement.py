import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.manifold_metrics import compute_manifold_geometry
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
    projected_sphere_mesh_hmin,
)
from core.operators import build_local_operators
from core.operators_sphere import compute_sphere_rhs
from test.test_manifold_velocity_sphere import solid_body_rotation_velocity
from test.test_sphere_full_rhs_smooth_snapshot import _weighted_integral


def build_projected_constant_state(nsub, order=4, R=1.0, u0=1.0, alpha=np.pi / 4.0):
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
        "h": projected_sphere_mesh_hmin(nodes_xyz, EToV),
    }


def rms(values):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values**2))


def weighted_l2_norm(state, values):
    area = _weighted_integral(state, np.ones_like(values))
    return np.sqrt(_weighted_integral(state, values * values) / area)


def rate(previous_error, current_error, previous_h, current_h):
    if previous_error is None:
        return None
    return np.log(previous_error / current_error) / np.log(previous_h / current_h)


def run_refinement_case(nsub):
    state = build_projected_constant_state(nsub=nsub)
    q = state["q"].copy()

    rhs = compute_sphere_rhs(
        q,
        0.0,
        state=state,
        flux_type="central",
        surface_mode="conservative_scaled",
    )

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "h": float(state["h"]),
        "max_abs_rhs": float(np.max(np.abs(rhs))),
        "rms_rhs": float(rms(rhs)),
        "weighted_l2": float(weighted_l2_norm(state, rhs)),
        "mass_total": float(_weighted_integral(state, rhs)),
    }


def test_sphere_constant_rhs_h_refinement():
    levels = [2, 4, 8, 16]
    results = [run_refinement_case(nsub) for nsub in levels]

    assert all(np.isfinite(row["max_abs_rhs"]) for row in results)
    assert all(np.isfinite(row["rms_rhs"]) for row in results)
    assert all(np.isfinite(row["weighted_l2"]) for row in results)
    assert all(np.isfinite(row["mass_total"]) for row in results)
    assert results[-1]["max_abs_rhs"] < results[0]["max_abs_rhs"]
    assert results[-1]["rms_rhs"] < results[0]["rms_rhs"]
    assert results[-1]["weighted_l2"] < results[0]["weighted_l2"]
    assert abs(results[-1]["mass_total"]) < 1.0e-12


def run_all_tests():
    levels = [2, 4, 8, 16]
    previous = None
    results = []

    print("\n" + "=" * 118)
    print("sphere projected manifold constant RHS h-refinement | q=1 | central flux")
    print("=" * 118)
    print(
        f"{'nsub':>8s} {'K':>8s} {'h':>13s} "
        f"{'max_abs_rhs':>16s} {'rate':>10s} "
        f"{'rms_rhs':>16s} {'rate':>10s} "
        f"{'weighted_L2':>16s} {'rate':>10s} "
        f"{'mass_total':>16s}"
    )
    print("-" * 118)

    for nsub in levels:
        row = run_refinement_case(nsub)

        max_rate = rate(
            None if previous is None else previous["max_abs_rhs"],
            row["max_abs_rhs"],
            None if previous is None else previous["h"],
            row["h"],
        )
        rms_rate = rate(
            None if previous is None else previous["rms_rhs"],
            row["rms_rhs"],
            None if previous is None else previous["h"],
            row["h"],
        )
        l2_rate = rate(
            None if previous is None else previous["weighted_l2"],
            row["weighted_l2"],
            None if previous is None else previous["h"],
            row["h"],
        )

        print(
            f"{row['nsub']:8d} {row['K']:8d} {row['h']:13.6e} "
            f"{row['max_abs_rhs']:16.6e} "
            f"{'---' if max_rate is None else f'{max_rate:.4f}':>10s} "
            f"{row['rms_rhs']:16.6e} "
            f"{'---' if rms_rate is None else f'{rms_rate:.4f}':>10s} "
            f"{row['weighted_l2']:16.6e} "
            f"{'---' if l2_rate is None else f'{l2_rate:.4f}':>10s} "
            f"{row['mass_total']:16.6e}"
        )

        results.append(row)
        previous = row

    print("-" * 118)

    if not (
        results[-1]["max_abs_rhs"] < results[0]["max_abs_rhs"]
        and results[-1]["rms_rhs"] < results[0]["rms_rhs"]
        and results[-1]["weighted_l2"] < results[0]["weighted_l2"]
        and abs(results[-1]["mass_total"]) < 1.0e-12
    ):
        raise AssertionError("sphere constant RHS h-refinement diagnostic failed")

    print("sphere constant RHS h-refinement diagnostic passed")


if __name__ == "__main__":
    run_all_tests()
