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


def solid_body_z_velocity(xyz):
    omega = np.array([0.0, 0.0, 1.0], dtype=float)
    return np.cross(omega[None, :], xyz)


def q_linear_x(xyz):
    return xyz[:, 0]


def rhs_linear_x(xyz):
    return xyz[:, 1]


def q_sin_x_minus_y(xyz):
    return np.sin(xyz[:, 0] - xyz[:, 1])


def rhs_sin_x_minus_y(xyz):
    return (xyz[:, 0] + xyz[:, 1]) * np.cos(xyz[:, 0] - xyz[:, 1])


def weighted_integral(state, values):
    engine = state["engine"]
    total = 0.0

    for k, geometry in enumerate(state["geometry"]):
        total += engine.area * np.sum(engine.w_s * geometry["J"] * values[k])

    return float(total)


def weighted_l2_error(state, error):
    area = weighted_integral(state, np.ones_like(error))
    return np.sqrt(weighted_integral(state, error * error) / area)


def build_projected_smooth_state(nsub, q_func, rhs_func, order=4, R=1.0):
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
    q = np.zeros((EToV.shape[0], engine.num_nodes))
    target = np.zeros_like(q)

    for k in range(EToV.shape[0]):
        xyz = element_xyz[k]
        geometries.append(compute_manifold_geometry(engine, xyz))
        velocities.append(solid_body_z_velocity(xyz))
        q[k, :] = q_func(xyz)
        target[k, :] = rhs_func(xyz)

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
        "target_rhs": target,
        "h": projected_sphere_mesh_hmin(nodes_xyz, EToV),
    }


def rate(previous_error, current_error, previous_h, current_h):
    if previous_error is None:
        return None

    return np.log(previous_error / current_error) / np.log(previous_h / current_h)


def run_refinement_case(nsub, q_func, rhs_func):
    state = build_projected_smooth_state(
        nsub=nsub,
        q_func=q_func,
        rhs_func=rhs_func,
    )
    q = state["q"].copy()

    rhs = compute_sphere_rhs(
        q,
        0.0,
        state=state,
        flux_type="central",
        surface_mode="conservative_scaled",
    )

    error = rhs - state["target_rhs"]

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "h": float(state["h"]),
        "L2_error": float(weighted_l2_error(state, error)),
        "max_error": float(np.max(np.abs(error))),
        "mass_total": float(weighted_integral(state, rhs)),
    }


def run_case_table(case_name, q_func, rhs_func):
    levels = [2, 4, 8, 16]
    previous = None
    results = []

    print("\n" + "=" * 104)
    print(
        "sphere smooth RHS h-refinement | projected 3D manifold | "
        f"{case_name} | omega=(0,0,1)"
    )
    print("=" * 104)
    print(
        f"{'nsub':>8s} {'K':>8s} {'h':>13s} "
        f"{'L2_error':>16s} {'L2_rate':>10s} "
        f"{'max_error':>16s} {'max_rate':>10s} "
        f"{'mass_total':>16s}"
    )
    print("-" * 104)

    for nsub in levels:
        row = run_refinement_case(nsub, q_func=q_func, rhs_func=rhs_func)

        L2_rate = rate(
            None if previous is None else previous["L2_error"],
            row["L2_error"],
            None if previous is None else previous["h"],
            row["h"],
        )
        max_rate = rate(
            None if previous is None else previous["max_error"],
            row["max_error"],
            None if previous is None else previous["h"],
            row["h"],
        )

        print(
            f"{row['nsub']:8d} {row['K']:8d} {row['h']:13.6e} "
            f"{row['L2_error']:16.6e} "
            f"{'---' if L2_rate is None else f'{L2_rate:.4f}':>10s} "
            f"{row['max_error']:16.6e} "
            f"{'---' if max_rate is None else f'{max_rate:.4f}':>10s} "
            f"{row['mass_total']:16.6e}"
        )

        results.append(row)
        previous = row

    print("-" * 104)

    assert all(np.isfinite(row["L2_error"]) for row in results)
    assert all(np.isfinite(row["max_error"]) for row in results)
    assert all(np.isfinite(row["mass_total"]) for row in results)
    assert results[-1]["L2_error"] < results[0]["L2_error"]
    assert results[-1]["max_error"] < results[0]["max_error"]
    assert abs(results[-1]["mass_total"]) < 1.0e-10

    return results


def test_sphere_smooth_rhs_h_refinement():
    run_case_table(
        "q=X",
        q_func=q_linear_x,
        rhs_func=rhs_linear_x,
    )
    run_case_table(
        "q=sin(X-Y)",
        q_func=q_sin_x_minus_y,
        rhs_func=rhs_sin_x_minus_y,
    )


def run_all_tests():
    test_sphere_smooth_rhs_h_refinement()
    print("sphere smooth RHS h-refinement diagnostic passed")


if __name__ == "__main__":
    run_all_tests()
