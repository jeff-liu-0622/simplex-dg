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
from core.time_integration import lsrk54_step


def solid_body_z_velocity(xyz):
    omega = np.array([0.0, 0.0, 1.0], dtype=float)
    return np.cross(omega[None, :], xyz)


def gaussian_bell(xyz, beta=20.0):
    center = np.array([1.0, 0.0, 0.0], dtype=float)
    dot = np.clip(xyz @ center, -1.0, 1.0)
    distance = np.arccos(dot)

    return np.exp(-beta * distance * distance)


def gaussian_bell_exact_after_z_rotation(xyz, final_time, beta=20.0):
    c = np.cos(final_time)
    s = np.sin(final_time)

    xyz0 = np.empty_like(xyz)
    xyz0[:, 0] = c * xyz[:, 0] + s * xyz[:, 1]
    xyz0[:, 1] = -s * xyz[:, 0] + c * xyz[:, 1]
    xyz0[:, 2] = xyz[:, 2]

    return gaussian_bell(xyz0, beta=beta)


def weighted_integral(state, values):
    engine = state["engine"]
    total = 0.0

    for k, geometry in enumerate(state["geometry"]):
        total += engine.area * np.sum(engine.w_s * geometry["J"] * values[k])

    return float(total)


def weighted_l2_error(state, error):
    area = weighted_integral(state, np.ones_like(error))

    return np.sqrt(weighted_integral(state, error * error) / area)


def build_projected_gaussian_state(nsub, beta=20.0, order=4, R=1.0):
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

    for k in range(EToV.shape[0]):
        xyz = element_xyz[k]
        geometries.append(compute_manifold_geometry(engine, xyz))
        velocities.append(solid_body_z_velocity(xyz))
        q[k, :] = gaussian_bell(xyz, beta=beta)

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


def rate(previous_error, current_error, previous_h, current_h):
    if previous_error is None:
        return None

    return np.log(previous_error / current_error) / np.log(previous_h / current_h)


def run_gaussian_short_time_case(
    nsub,
    final_time=1.0e-2,
    beta=20.0,
    order=4,
):
    dt_by_nsub = {
        4: 2.5e-4,
        8: 2.5e-4,
        16: 2.5e-4,
    }
    state = build_projected_gaussian_state(
        nsub=nsub,
        beta=beta,
        order=order,
    )
    q_initial = state["q"].copy()
    q = q_initial.copy()
    res = np.zeros_like(q)
    t = 0.0
    dt = dt_by_nsub[nsub]
    num_steps = 0

    mass_initial = weighted_integral(state, q_initial)
    energy_initial = weighted_integral(state, q_initial * q_initial)

    while t < final_time - 1.0e-15:
        dt_step = min(dt, final_time - t)
        q, res = lsrk54_step(
            q,
            res,
            t,
            dt_step,
            compute_sphere_rhs,
            state=state,
            flux_type="central",
            surface_mode="conservative_scaled",
        )
        t += dt_step
        num_steps += 1

    q_exact = np.zeros_like(q)
    for k, xyz in enumerate(state["xyz"]):
        q_exact[k, :] = gaussian_bell_exact_after_z_rotation(
            xyz,
            final_time=final_time,
            beta=beta,
        )

    state["q"] = q
    mass_final = weighted_integral(state, q)
    energy_final = weighted_integral(state, q * q)
    error = q - q_exact

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "h": float(state["h"]),
        "dt": dt,
        "num_steps": num_steps,
        "L2_error": float(weighted_l2_error(state, error)),
        "max_error": float(np.max(np.abs(error))),
        "mass_change": float(mass_final - mass_initial),
        "energy_change": float(energy_final - energy_initial),
        "has_nonfinite": bool(not np.all(np.isfinite(q))),
    }


def test_sphere_gaussian_bell_short_time():
    levels = [4, 8, 16]
    final_time = 1.0
    beta = 20.0
    previous = None
    results = []

    print("\n" + "=" * 126)
    print(
        "sphere Gaussian bell short-time rotation | projected 3D manifold | "
        "LSRK54 | central flux"
    )
    print("=" * 126)
    print(
        "q0 = exp(-20 * arccos(x dot [1,0,0])^2), "
        "omega=(0,0,1), T_final=1e-2"
    )
    print(
        f"{'nsub':>8s} {'K':>8s} {'h':>13s} {'dt':>12s} "
        f"{'steps':>8s} {'L2_error':>16s} {'L2_rate':>10s} "
        f"{'max_error':>16s} {'max_rate':>10s} "
        f"{'mass_change':>16s} {'energy_change':>16s}"
    )
    print("-" * 126)

    for nsub in levels:
        row = run_gaussian_short_time_case(
            nsub=nsub,
            final_time=final_time,
            beta=beta,
        )
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
            f"{row['dt']:12.6e} {row['num_steps']:8d} "
            f"{row['L2_error']:16.6e} "
            f"{'---' if L2_rate is None else f'{L2_rate:.4f}':>10s} "
            f"{row['max_error']:16.6e} "
            f"{'---' if max_rate is None else f'{max_rate:.4f}':>10s} "
            f"{row['mass_change']:16.6e} "
            f"{row['energy_change']:16.6e}"
        )

        results.append(row)
        previous = row

    print("-" * 126)

    for row in results:
        assert not row["has_nonfinite"]
        assert np.isfinite(row["L2_error"])
        assert np.isfinite(row["max_error"])
        assert np.isfinite(row["mass_change"])
        assert np.isfinite(row["energy_change"])

    assert results[-1]["L2_error"] < results[0]["L2_error"]
    assert results[-1]["max_error"] < results[0]["max_error"]


def run_all_tests():
    test_sphere_gaussian_bell_short_time()
    print("sphere Gaussian bell short-time rotation diagnostic passed")


if __name__ == "__main__":
    run_all_tests()
