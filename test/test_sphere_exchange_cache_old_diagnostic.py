import numpy as np

from core.rhs_sphere import (
    REFERENCE_FACE_NORMALS,
    aligned_neighbor_face_indices,
    compute_sphere_surface_penalty,
    face_node_indices,
    sphere_flux_coefficient,
)
from core.time_integration import lsrk54_step
from test.test_final_sphere_convergence import (
    build_projected_sphere_state,
    gaussian_exact_on_state,
    sphere_l2_error,
)
from test.test_manifold_constant_rhs_sphere import compute_manifold_skew_volume_rhs
from test.test_sphere_flux_jump_diagnostic import (
    apply_elementwise_jump_field,
    collect_face_jump_stats,
)
from test.test_sphere_full_rhs_constant import build_projected_sphere_rhs_state
from test.test_sphere_full_rhs_smooth_snapshot import (
    _rhs_summary,
    build_projected_sphere_smooth_state,
)
from test.test_sphere_lsrk_short_sanity import compute_sphere_full_rhs_for_state
from test.test_sphere_operator_spectrum_diagnostic import spectrum_summary


def build_sphere_old_exchange_cache(state, face_match_tol=1.0e-12):
    """
    Diagnostic-only exchange cache for the old sphere SAT path.

    The cache mirrors the external old-style pipeline: it stores face trace
    pairing, reference face weights, and side-local metric flux ingredients.
    It intentionally keeps the current old penalty formula unchanged.
    """
    engine = state["engine"]
    num_elements = state["q"].shape[0]
    num_faces = 3
    num_face_nodes = engine.num_edge_nodes

    local_nodes = np.zeros((num_elements, num_faces, num_face_nodes), dtype=int)
    neighbor_elements = np.zeros((num_elements, num_faces), dtype=int)
    neighbor_faces = np.zeros((num_elements, num_faces), dtype=int)
    neighbor_nodes = np.zeros((num_elements, num_faces, num_face_nodes), dtype=int)
    nr = np.zeros((num_elements, num_faces, num_face_nodes), dtype=float)
    ns = np.zeros_like(nr)
    face_weights = np.zeros_like(nr)
    J_face = np.zeros_like(nr)
    u_tilde_face = np.zeros_like(nr)
    v_tilde_face = np.zeros_like(nr)
    max_face_match_error = 0.0

    for kM in range(num_elements):
        for fM in range(num_faces):
            kP = int(state["EToE"][kM, fM])
            fP = int(state["EToF"][kM, fM])
            if kP == kM:
                raise AssertionError(f"unexpected boundary face ({kM}, {fM})")

            nodes_M = face_node_indices(engine, fM)
            nodes_P_native = face_node_indices(engine, fP)
            ordering, face_match_error = aligned_neighbor_face_indices(
                state["xyz"][kM, nodes_M, :],
                state["xyz"][kP, nodes_P_native, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            local_nodes[kM, fM, :] = nodes_M
            neighbor_elements[kM, fM] = kP
            neighbor_faces[kM, fM] = fP
            neighbor_nodes[kM, fM, :] = nodes_P_native[ordering]
            nr[kM, fM, :] = REFERENCE_FACE_NORMALS[fM, 0]
            ns[kM, fM, :] = REFERENCE_FACE_NORMALS[fM, 1]
            face_weights[kM, fM, :] = engine.w_e
            J_face[kM, fM, :] = state["geometry"][kM]["J"][nodes_M]
            u_tilde_face[kM, fM, :] = state["u_tilde"][kM, nodes_M]
            v_tilde_face[kM, fM, :] = state["v_tilde"][kM, nodes_M]

    if max_face_match_error > face_match_tol:
        raise AssertionError(
            "projected sphere face pairing is not physically continuous: "
            f"max face match error = {max_face_match_error:.3e}"
        )

    return {
        "local_nodes": local_nodes,
        "neighbor_elements": neighbor_elements,
        "neighbor_faces": neighbor_faces,
        "neighbor_nodes": neighbor_nodes,
        "face_weights": face_weights,
        "nr": nr,
        "ns": ns,
        "J_face": J_face,
        "u_tilde_face": u_tilde_face,
        "v_tilde_face": v_tilde_face,
        "max_face_match_error": max_face_match_error,
    }


def refresh_sphere_old_exchange_cache_metrics(state, cache):
    for k in range(state["q"].shape[0]):
        for f in range(3):
            nodes = cache["local_nodes"][k, f]
            cache["J_face"][k, f, :] = state["geometry"][k]["J"][nodes]
            cache["u_tilde_face"][k, f, :] = state["u_tilde"][k, nodes]
            cache["v_tilde_face"][k, f, :] = state["v_tilde"][k, nodes]


def compute_sphere_surface_penalty_exchange_old(
    state,
    cache,
    flux_type="upwind",
    alpha_lf=1.0,
):
    engine = state["engine"]
    q = state["q"]
    surface_rhs = np.zeros_like(q)
    max_abs_penalty = 0.0

    for kM in range(q.shape[0]):
        p_boundary = np.zeros(engine.num_boundary_nodes, dtype=float)

        for fM in range(3):
            nodes_M = cache["local_nodes"][kM, fM]
            kP = cache["neighbor_elements"][kM, fM]
            nodes_P = cache["neighbor_nodes"][kM, fM]

            qM = q[kM, nodes_M]
            qP = q[kP, nodes_P]
            vn_sJ = cache["J_face"][kM, fM] * (
                cache["nr"][kM, fM] * cache["u_tilde_face"][kM, fM]
                + cache["ns"][kM, fM] * cache["v_tilde_face"][kM, fM]
            )
            C = sphere_flux_coefficient(vn_sJ, flux_type=flux_type, alpha_lf=alpha_lf)
            penalty = 0.5 * (vn_sJ - C) * (qM - qP)

            p_boundary[engine.edge_slices[fM]] = penalty
            max_abs_penalty = max(max_abs_penalty, float(np.max(np.abs(penalty))))

        lifted = engine.lift_boundary_penalty(
            p_boundary,
            edge_lengths=np.ones(3),
        )
        surface_rhs[kM, :] = lifted / state["geometry"][kM]["J"]

    return {
        "surface_rhs": surface_rhs,
        "max_face_match_error": cache["max_face_match_error"],
        "max_abs_penalty": max_abs_penalty,
        "max_abs_vn_orientation_error": np.nan,
        "surface_mode": "exchange_old",
    }


def recompute_volume_rhs_for_q(state, q):
    engine = state["engine"]
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
    return volume_rhs


def compute_sphere_full_rhs_for_state_exchange_old(
    q,
    _t,
    state,
    exchange_cache,
    flux_type="upwind",
    alpha_lf=1.0,
):
    q = np.asarray(q, dtype=float)
    volume_rhs = recompute_volume_rhs_for_q(state, q)
    refresh_sphere_old_exchange_cache_metrics(state, exchange_cache)
    surface = compute_sphere_surface_penalty_exchange_old(
        state,
        exchange_cache,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
    )
    return volume_rhs + surface["surface_rhs"]


def compare_old_and_exchange_for_state(
    state,
    exchange_cache,
    flux_type="upwind",
    alpha_lf=1.0,
):
    current_surface = compute_sphere_surface_penalty(
        state,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        surface_mode="old",
    )
    exchange_surface = compute_sphere_surface_penalty_exchange_old(
        state,
        exchange_cache,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
    )
    current_rhs = state["volume_rhs"] + current_surface["surface_rhs"]
    exchange_rhs = state["volume_rhs"] + exchange_surface["surface_rhs"]
    return {
        "current_surface": current_surface,
        "exchange_surface": exchange_surface,
        "current_rhs": current_rhs,
        "exchange_rhs": exchange_rhs,
        "max_abs_rhs_diff": float(np.max(np.abs(current_rhs - exchange_rhs))),
        "max_abs_surface_diff": float(
            np.max(np.abs(current_surface["surface_rhs"] - exchange_surface["surface_rhs"]))
        ),
        "current_summary": _rhs_summary(
            state,
            current_rhs,
            state["volume_rhs"],
            current_surface["surface_rhs"],
            flux_type,
            alpha_lf,
        ),
        "exchange_summary": _rhs_summary(
            state,
            exchange_rhs,
            state["volume_rhs"],
            exchange_surface["surface_rhs"],
            flux_type,
            alpha_lf,
        ),
    }


def test_exchange_cache_old_q1_rhs_comparison():
    cases = [("central", 1.0), ("upwind", 1.0), ("lf", 1.0), ("lf", 1.5)]
    state = build_projected_sphere_rhs_state(nsub=8, order=4)
    cache = build_sphere_old_exchange_cache(state)

    print("\n" + "=" * 132)
    print("Exchange-cache old-mode q=1 RHS comparison")
    print("=" * 132)
    print(
        f"{'flux':>10s} {'alpha':>8s} {'current_max_rhs':>16s} "
        f"{'exchange_max_rhs':>16s} {'max_rhs_diff':>15s} "
        f"{'max_surface_diff':>17s} {'max_face_match':>15s}"
    )
    print("-" * 132)

    for flux_type, alpha_lf in cases:
        result = compare_old_and_exchange_for_state(
            state,
            cache,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        print(
            f"{flux_type:>10s} {alpha_lf:8.3f} "
            f"{result['current_summary']['max_abs_rhs']:16.6e} "
            f"{result['exchange_summary']['max_abs_rhs']:16.6e} "
            f"{result['max_abs_rhs_diff']:15.6e} "
            f"{result['max_abs_surface_diff']:17.6e} "
            f"{cache['max_face_match_error']:15.6e}"
        )
        assert np.all(np.isfinite(result["exchange_rhs"]))
        assert result["max_abs_rhs_diff"] < 1.0e-13
        assert result["max_abs_surface_diff"] < 1.0e-13

    print("=" * 132)


def test_exchange_cache_old_jump_flux_comparison():
    cases = [("central", 1.0), ("upwind", 1.0), ("lf", 1.0), ("lf", 1.5)]
    state = build_projected_sphere_smooth_state(nsub=8, order=4)
    apply_elementwise_jump_field(state, eps=1.0e-2)
    cache = build_sphere_old_exchange_cache(state)
    jump_stats = collect_face_jump_stats(state)

    print("\n" + "=" * 150)
    print("Exchange-cache old-mode jump-flux comparison")
    print("=" * 150)
    print(
        f"max_abs_jump_q={jump_stats['max_abs_jump_q']:.6e}, "
        f"rms_jump_q={jump_stats['rms_jump_q']:.6e}"
    )
    print(
        f"{'flux':>10s} {'alpha':>8s} {'current_surf':>15s} "
        f"{'exchange_surf':>15s} {'current_mass':>15s} "
        f"{'exchange_mass':>15s} {'max_rhs_diff':>15s} "
        f"{'max_surface_diff':>17s}"
    )
    print("-" * 150)

    results = []
    for flux_type, alpha_lf in cases:
        result = compare_old_and_exchange_for_state(
            state,
            cache,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        results.append((flux_type, alpha_lf, result))
        print(
            f"{flux_type:>10s} {alpha_lf:8.3f} "
            f"{result['current_summary']['max_abs_surface']:15.6e} "
            f"{result['exchange_summary']['max_abs_surface']:15.6e} "
            f"{result['current_summary']['mass_rate']:15.6e} "
            f"{result['exchange_summary']['mass_rate']:15.6e} "
            f"{result['max_abs_rhs_diff']:15.6e} "
            f"{result['max_abs_surface_diff']:17.6e}"
        )
        assert np.all(np.isfinite(result["exchange_rhs"]))
        assert result["max_abs_rhs_diff"] < 1.0e-13
        assert result["max_abs_surface_diff"] < 1.0e-13

    upwind = next(row for row in results if row[0] == "upwind")[2]
    lf1 = next(row for row in results if row[0] == "lf" and abs(row[1] - 1.0) < 1.0e-14)[2]
    lf15 = next(row for row in results if row[0] == "lf" and abs(row[1] - 1.5) < 1.0e-14)[2]
    upwind_lf1_diff = float(np.max(np.abs(upwind["exchange_rhs"] - lf1["exchange_rhs"])))
    upwind_lf15_diff = float(np.max(np.abs(upwind["exchange_rhs"] - lf15["exchange_rhs"])))
    print("-" * 150)
    print(f"exchange max upwind/LF(alpha=1) RHS difference = {upwind_lf1_diff:.6e}")
    print(f"exchange max upwind/LF(alpha=1.5) RHS difference = {upwind_lf15_diff:.6e}")
    print("=" * 150)
    assert upwind_lf1_diff < 1.0e-14
    assert upwind_lf15_diff > 1.0e-12


def build_exchange_old_rhs_operator(flux_type, alpha_lf=1.0, nsub=2, order=2):
    state = build_projected_sphere_smooth_state(nsub=nsub, order=order)
    cache = build_sphere_old_exchange_cache(state)
    shape = state["q"].shape
    ndof = int(np.prod(shape))
    operator = np.zeros((ndof, ndof), dtype=float)

    for col in range(ndof):
        q = np.zeros(shape, dtype=float)
        q.flat[col] = 1.0
        rhs = compute_sphere_full_rhs_for_state_exchange_old(
            q,
            0.0,
            state,
            cache,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        operator[:, col] = rhs.ravel()

    return operator, state, cache


def test_exchange_cache_old_small_mesh_spectrum_comparison():
    cases = [("central", 1.0), ("upwind", 1.0), ("lf", 1.0), ("lf", 1.5)]

    print("\n" + "=" * 150)
    print("Exchange-cache old-mode small-mesh spectrum comparison")
    print("=" * 150)
    print(
        f"{'flux':>10s} {'alpha':>8s} {'ndof':>7s} "
        f"{'current_max_real':>17s} {'exchange_max_real':>18s} "
        f"{'current_radius':>16s} {'exchange_radius':>16s} "
        f"{'max_operator_diff':>18s}"
    )
    print("-" * 150)

    for flux_type, alpha_lf in cases:
        exchange_operator, state, cache = build_exchange_old_rhs_operator(
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        shape = state["q"].shape
        current_operator = np.zeros_like(exchange_operator)
        for col in range(exchange_operator.shape[1]):
            q = np.zeros(shape, dtype=float)
            q.flat[col] = 1.0
            rhs = compute_sphere_full_rhs_for_state(
                q,
                0.0,
                state,
                flux_type=flux_type,
                alpha_lf=alpha_lf,
                surface_mode="old",
            )
            current_operator[:, col] = rhs.ravel()

        current_summary = spectrum_summary(current_operator, "old", flux_type, alpha_lf)
        exchange_summary = spectrum_summary(
            exchange_operator,
            "exchange_old",
            flux_type,
            alpha_lf,
        )
        max_operator_diff = float(np.max(np.abs(current_operator - exchange_operator)))
        print(
            f"{flux_type:>10s} {alpha_lf:8.3f} {exchange_operator.shape[0]:7d} "
            f"{current_summary['max_real_eigenvalue']:17.6e} "
            f"{exchange_summary['max_real_eigenvalue']:18.6e} "
            f"{current_summary['spectral_radius']:16.6e} "
            f"{exchange_summary['spectral_radius']:16.6e} "
            f"{max_operator_diff:18.6e}"
        )
        assert cache["max_face_match_error"] < 1.0e-12
        assert current_summary["all_finite"]
        assert exchange_summary["all_finite"]
        assert max_operator_diff < 1.0e-13

    print("=" * 150)


def integrate_gaussian_exchange_old(
    nsub=4,
    order=3,
    n_quad=4,
    final_time=1.0e-2,
    dt_target=2.5e-4,
    flux_type="upwind",
    alpha_lf=1.0,
):
    state = build_projected_sphere_state(nsub=nsub, order=order, n_quad=n_quad)
    q = gaussian_exact_on_state(state, 0.0)
    state["q"] = q.copy()
    recompute_volume_rhs_for_q(state, q)
    cache = build_sphere_old_exchange_cache(state)
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
            compute_sphere_full_rhs_for_state_exchange_old,
            state=state,
            exchange_cache=cache,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        t += dt

    state["q"] = q
    q_exact = gaussian_exact_on_state(state, t)
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


def integrate_gaussian_current_old(
    nsub=4,
    order=3,
    n_quad=4,
    final_time=1.0e-2,
    dt_target=2.5e-4,
    flux_type="upwind",
    alpha_lf=1.0,
):
    state = build_projected_sphere_state(nsub=nsub, order=order, n_quad=n_quad)
    q = gaussian_exact_on_state(state, 0.0)
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
            surface_mode="old",
        )
        t += dt

    state["q"] = q
    q_exact = gaussian_exact_on_state(state, t)
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


def test_exchange_cache_old_short_time_gaussian_comparison():
    cases = [("upwind", 1.0), ("lf", 1.0), ("lf", 1.5)]

    print("\n" + "=" * 132)
    print("Exchange-cache old-mode short-time Gaussian comparison")
    print("=" * 132)
    print("projected sphere, nsub=4, N=3, n_quad=4, T=1e-2, dt=2.5e-4")
    print(
        f"{'flux':>10s} {'alpha':>8s} {'current_L2':>15s} "
        f"{'exchange_L2':>15s} {'current_Linf':>15s} "
        f"{'exchange_Linf':>15s} {'max_q_diff':>15s}"
    )
    print("-" * 132)

    results = []
    for flux_type, alpha_lf in cases:
        current = integrate_gaussian_current_old(
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        exchange = integrate_gaussian_exchange_old(
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        max_q_diff = float(np.max(np.abs(current["q"] - exchange["q"])))
        results.append((flux_type, alpha_lf, current, exchange, max_q_diff))
        print(
            f"{flux_type:>10s} {alpha_lf:8.3f} "
            f"{current['L2_exact']:15.6e} {exchange['L2_exact']:15.6e} "
            f"{current['Linf_exact']:15.6e} {exchange['Linf_exact']:15.6e} "
            f"{max_q_diff:15.6e}"
        )
        assert not current["has_nonfinite"]
        assert not exchange["has_nonfinite"]
        assert np.isfinite(current["L2_exact"])
        assert np.isfinite(exchange["L2_exact"])
        assert max_q_diff < 1.0e-13

    upwind = next(row for row in results if row[0] == "upwind")
    lf1 = next(row for row in results if row[0] == "lf" and abs(row[1] - 1.0) < 1.0e-14)
    upwind_lf1_diff = float(np.max(np.abs(upwind[3]["q"] - lf1[3]["q"])))
    print("-" * 132)
    print(f"exchange max upwind/LF(alpha=1) final q difference = {upwind_lf1_diff:.6e}")
    print("=" * 132)
    assert upwind_lf1_diff < 1.0e-13


def run_all_tests():
    test_exchange_cache_old_q1_rhs_comparison()
    test_exchange_cache_old_jump_flux_comparison()
    test_exchange_cache_old_small_mesh_spectrum_comparison()
    test_exchange_cache_old_short_time_gaussian_comparison()
    print("test_sphere_exchange_cache_old_diagnostic.py passed")


if __name__ == "__main__":
    run_all_tests()
