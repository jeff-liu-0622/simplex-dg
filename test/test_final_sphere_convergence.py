import time

import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
    projected_sphere_mesh_hmin,
)
from core.operators import build_local_operators
from core.time_integration import lsrk54_step
from test.test_manifold_geometry_sphere import compute_manifold_geometry
from test.test_manifold_velocity_sphere import solid_body_rotation_velocity
from test.test_sphere_full_rhs_smooth_snapshot import _weighted_integral
from test.test_sphere_lsrk_short_sanity import compute_sphere_full_rhs_for_state


def omega_vector(u0=1.0, alpha=np.pi / 4.0):
    return np.array([-u0 * np.sin(alpha), 0.0, u0 * np.cos(alpha)], dtype=float)


def rotate_vectors(vectors, angle, omega=None):
    vectors = np.asarray(vectors, dtype=float)
    if omega is None:
        omega = omega_vector()
    axis = np.asarray(omega, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cross = np.cross(axis[None, :], vectors)
    dot = np.sum(vectors * axis[None, :], axis=1)
    return (
        cos_a * vectors
        + sin_a * cross
        + (1.0 - cos_a) * dot[:, None] * axis[None, :]
    )


def gaussian_center():
    center = np.array([1.0, 1.0, 1.0], dtype=float)
    return center / np.linalg.norm(center)


def gaussian_bell(xyz, center=None, beta=20.0):
    if center is None:
        center = gaussian_center()
    distance_squared = np.sum((xyz - center[None, :]) ** 2, axis=1)
    return np.exp(-beta * distance_squared)


def gaussian_exact_on_state(state, t, beta=20.0):
    center_t = rotate_vectors(gaussian_center()[None, :], t)[0]
    q = np.zeros_like(state["q"])
    for k, xyz in enumerate(state["xyz"]):
        q[k, :] = gaussian_bell(xyz, center=center_t, beta=beta)
    return q


def build_projected_sphere_state(
    nsub=4,
    order=3,
    n_quad=None,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
):
    if n_quad is None:
        n_quad = order + 1
    engine = build_local_operators(N=order, n=n_quad, rule="table1")
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
    for xyz in element_xyz:
        geometries.append(compute_manifold_geometry(engine, xyz))
        velocities.append(solid_body_rotation_velocity(xyz, u0=u0, alpha=alpha))

    q = np.zeros((EToV.shape[0], engine.num_nodes), dtype=float)
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
        "n_quad": n_quad,
    }


def sphere_l2_error(state, error):
    area = _weighted_integral(state, np.ones_like(error))
    return float(np.sqrt(_weighted_integral(state, error * error) / area))


def observed_order(previous_error, current_error, previous_size, current_size):
    if previous_error is None:
        return np.nan
    return float(np.log(previous_error / current_error) / np.log(previous_size / current_size))


def integrate_sphere_gaussian(
    nsub=4,
    order=3,
    n_quad=None,
    final_time=1.0e-2,
    dt_target=1.0e-3,
    beta=20.0,
    surface_mode="old",
    flux_type="upwind",
    alpha_lf=1.0,
):
    state = build_projected_sphere_state(nsub=nsub, order=order, n_quad=n_quad)
    q = gaussian_exact_on_state(state, 0.0, beta=beta)
    state["q"] = q.copy()
    res = np.zeros_like(q)
    steps = int(np.ceil(final_time / dt_target))
    dt = final_time / steps
    t = 0.0

    for _ in range(steps):
        q, res = lsrk54_step(
            q,
            res,
            t,
            dt,
            compute_sphere_full_rhs_for_state,
            state=state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            surface_mode=surface_mode,
        )
        t += dt

    state["q"] = q
    q_exact = gaussian_exact_on_state(state, t, beta=beta)
    error = q - q_exact
    return {
        "state": state,
        "q": q,
        "q_exact": q_exact,
        "dt": dt,
        "steps": steps,
        "actual_time": t,
        "L2_exact": sphere_l2_error(state, error),
        "Linf_exact": float(np.max(np.abs(error))),
        "has_nonfinite": bool(not np.all(np.isfinite(q))),
    }


def final_sphere_rk_temporal_convergence():
    nsub = 4
    order = 3
    n_quad = 4
    final_time = 1.0e-2
    dt_targets = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4]
    dt_reference = 6.25e-5

    reference = integrate_sphere_gaussian(
        nsub=nsub,
        order=order,
        n_quad=n_quad,
        final_time=final_time,
        dt_target=dt_reference,
        surface_mode="old",
    )

    rows = []
    previous_l2 = None
    previous_linf = None
    previous_dt = None
    for dt_target in dt_targets:
        result = integrate_sphere_gaussian(
            nsub=nsub,
            order=order,
            n_quad=n_quad,
            final_time=final_time,
            dt_target=dt_target,
            surface_mode="old",
        )
        ref_error = result["q"] - reference["q"]
        l2_ref = sphere_l2_error(result["state"], ref_error)
        linf_ref = float(np.max(np.abs(ref_error)))
        rows.append(
            {
                "dt": result["dt"],
                "steps": result["steps"],
                "L2_error": l2_ref,
                "Linf_error": linf_ref,
                "L2_exact": result["L2_exact"],
                "Linf_exact": result["Linf_exact"],
                "order_L2": observed_order(previous_l2, l2_ref, previous_dt, result["dt"]),
                "order_Linf": observed_order(
                    previous_linf,
                    linf_ref,
                    previous_dt,
                    result["dt"],
                ),
                "has_nonfinite": result["has_nonfinite"],
            }
        )
        previous_l2 = l2_ref
        previous_linf = linf_ref
        previous_dt = result["dt"]

    return {
        "nsub": nsub,
        "order": order,
        "n_quad": n_quad,
        "final_time": final_time,
        "surface_mode": "old",
        "reference_dt": reference["dt"],
        "reference_steps": reference["steps"],
        "rows": rows,
    }


def final_sphere_h_convergence():
    order = 3
    n_quad = 4
    final_time = 5.0e-3
    nsubs = [2, 4, 8, 16]
    dt_cap = 2.5e-4

    rows = []
    previous_l2 = None
    previous_linf = None
    previous_h = None
    for nsub in nsubs:
        state_for_h = build_projected_sphere_state(
            nsub=nsub,
            order=order,
            n_quad=n_quad,
        )
        h = state_for_h["h"]
        dt_target = min(dt_cap, 2.0e-2 * h * h)
        start = time.time()
        result = integrate_sphere_gaussian(
            nsub=nsub,
            order=order,
            n_quad=n_quad,
            final_time=final_time,
            dt_target=dt_target,
            surface_mode="old",
        )
        rows.append(
            {
                "nsub": nsub,
                "h": h,
                "dt": result["dt"],
                "steps": result["steps"],
                "actual_time": result["actual_time"],
                "L2_error": result["L2_exact"],
                "Linf_error": result["Linf_exact"],
                "order_L2": observed_order(previous_l2, result["L2_exact"], previous_h, h),
                "order_Linf": observed_order(
                    previous_linf,
                    result["Linf_exact"],
                    previous_h,
                    h,
                ),
                "elapsed": time.time() - start,
                "has_nonfinite": result["has_nonfinite"],
            }
        )
        previous_l2 = result["L2_exact"]
        previous_linf = result["Linf_exact"]
        previous_h = h

    return {
        "order": order,
        "n_quad": n_quad,
        "final_time": final_time,
        "surface_mode": "old",
        "rows": rows,
    }


def _order_string(value):
    if not np.isfinite(value):
        return "-"
    return f"{value:.4f}"


def test_final_sphere_rk_temporal_convergence():
    result = final_sphere_rk_temporal_convergence()
    print("\n" + "=" * 132)
    print("Final sphere RK temporal convergence")
    print("=" * 132)
    print(
        "projected sphere, Gaussian exact rotation, "
        f"surface_mode={result['surface_mode']}, "
        f"nsub={result['nsub']}, N={result['order']}, "
        f"n_quad={result['n_quad']}, "
        f"T={result['final_time']:.3e}"
    )
    print(
        f"reference_dt={result['reference_dt']:.6e}, "
        f"reference_steps={result['reference_steps']}"
    )
    print(
        f"{'dt':>13s} {'steps':>8s} {'L2_ref':>15s} {'order':>9s} "
        f"{'Linf_ref':>15s} {'order':>9s} {'L2_exact':>15s} {'Linf_exact':>15s}"
    )
    print("-" * 132)
    for row in result["rows"]:
        print(
            f"{row['dt']:13.6e} {row['steps']:8d} "
            f"{row['L2_error']:15.6e} {_order_string(row['order_L2']):>9s} "
            f"{row['Linf_error']:15.6e} {_order_string(row['order_Linf']):>9s} "
            f"{row['L2_exact']:15.6e} {row['Linf_exact']:15.6e}"
        )
    print("=" * 132)

    for row in result["rows"]:
        assert not row["has_nonfinite"]
        assert np.isfinite(row["L2_error"])
        assert np.isfinite(row["Linf_error"])
        assert np.isfinite(row["L2_exact"])
        assert np.isfinite(row["Linf_exact"])
    assert result["rows"][-1]["order_L2"] > 3.0


def test_final_sphere_h_convergence():
    result = final_sphere_h_convergence()
    print("\n" + "=" * 132)
    print("Final sphere short-time h-convergence")
    print("=" * 132)
    print(
        "projected sphere, Gaussian exact rotation, "
        f"surface_mode={result['surface_mode']}, "
        f"N={result['order']}, n_quad={result['n_quad']}, "
        f"T={result['final_time']:.3e}"
    )
    print(
        f"{'nsub':>6s} {'h':>13s} {'dt':>13s} {'steps':>8s} "
        f"{'actual_time':>13s} {'L2_error':>15s} {'order':>9s} "
        f"{'Linf_error':>15s} {'order':>9s} {'time(s)':>9s}"
    )
    print("-" * 132)
    for row in result["rows"]:
        print(
            f"{row['nsub']:6d} {row['h']:13.6e} {row['dt']:13.6e} "
            f"{row['steps']:8d} {row['actual_time']:13.6e} "
            f"{row['L2_error']:15.6e} {_order_string(row['order_L2']):>9s} "
            f"{row['Linf_error']:15.6e} {_order_string(row['order_Linf']):>9s} "
            f"{row['elapsed']:9.2f}"
        )
    print("=" * 132)

    for row in result["rows"]:
        assert not row["has_nonfinite"]
        assert np.isfinite(row["L2_error"])
        assert np.isfinite(row["Linf_error"])
    assert result["rows"][-1]["L2_error"] < result["rows"][0]["L2_error"]


def run_all_tests():
    test_final_sphere_rk_temporal_convergence()
    test_final_sphere_h_convergence()
    print("test_final_sphere_convergence.py passed")


if __name__ == "__main__":
    run_all_tests()
