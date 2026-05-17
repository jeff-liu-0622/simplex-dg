import numpy as np

from test.test_sphere_flux_jump_diagnostic import (
    apply_elementwise_jump_field,
    collect_face_jump_stats,
    recompute_volume_rhs_for_state,
)
from test.test_sphere_full_rhs_constant import (
    _aligned_neighbor_face_indices,
    compute_sphere_surface_penalty,
)
from test.test_sphere_full_rhs_smooth_snapshot import (
    _weighted_integral,
    build_projected_sphere_smooth_state,
)


SURFACE_SCALING_NOTE = (
    "Current surface skeleton uses v_n_sJ = J_face * (nr*u_tilde + ns*v_tilde). "
    "It does not compute a separate physical face line Jacobian sJ. "
    "engine.lift_boundary_penalty is called with edge_lengths=np.ones(3), "
    "so the lift still applies boundary quadrature weights but not the "
    "reference/physical edge lengths."
)

CONSERVATIVE_SURFACE_SCALING_NOTE = (
    "Conservative diagnostic helper uses the physical face conormal metric "
    "V3D dot (tau x n_surf), where tau is dX/dlambda for the unit edge "
    "quadrature coordinate. The lifted penalty is passed with "
    "edge_lengths=np.ones(3), so engine.lift_boundary_penalty applies only "
    "the edge quadrature weights; the physical line metric is already inside "
    "the conormal flux. Interior face penalties are assigned as equal and "
    "opposite values in the paired face-node ordering."
)


def build_state_for_q_case(q_case, nsub=4, order=4, eps=1.0e-2):
    state = build_projected_sphere_smooth_state(nsub=nsub, order=order)

    if q_case == "constant":
        state["q"] = np.ones_like(state["q"])
        state["volume_rhs"] = recompute_volume_rhs_for_state(state)
    elif q_case == "smooth":
        state["volume_rhs"] = recompute_volume_rhs_for_state(state)
    elif q_case == "jump":
        apply_elementwise_jump_field(state, eps=eps)
    else:
        raise ValueError(f"unknown q_case: {q_case}")

    return state


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def _flux_coefficient(vn, flux_type, alpha_lf):
    if flux_type == "central":
        return 0.0
    if flux_type == "upwind":
        return np.abs(vn)
    if flux_type == "lf":
        return alpha_lf * np.abs(vn)
    raise ValueError(f"unknown flux_type: {flux_type}")


def _face_node_indices(engine, face_id):
    face_slice = engine.edge_slices[face_id]
    return np.arange(face_slice.start, face_slice.stop)


def _physical_conormal_flux_on_face(state, elem_id, face_id, nodes):
    """
    Compute the line-metric normal velocity for one reference face.

    The tangent is dX/dlambda with lambda in [0, 1] following the local face
    node ordering.  The conormal tau x n_surf points outward for the current
    reference-triangle face ordering used by the SDG boundary nodes.
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


def compute_sphere_surface_penalty_conservative(
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    face_match_tol=1.0e-12,
):
    """
    Conservative shared-face surface penalty diagnostic.

    This helper keeps the old RHS volume form untouched, but replaces the
    face-local penalty skeleton with one shared-face computation.  Each
    interior face contributes equal and opposite boundary penalties in the
    paired node ordering, so the global lifted surface mass cancels up to
    roundoff under the same edge quadrature weights.
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

            nodes_M = _face_node_indices(engine, fM)
            nodes_P = _face_node_indices(engine, fP)
            ordering, face_match_error = _aligned_neighbor_face_indices(
                xyz[kM, nodes_M, :],
                xyz[kP, nodes_P, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            qM = q[kM, nodes_M]
            qP = q[kP, nodes_P[ordering]]
            vnM = _physical_conormal_flux_on_face(state, kM, fM, nodes_M)
            vnP = _physical_conormal_flux_on_face(state, kP, fP, nodes_P)[ordering]
            max_abs_vn_orientation_error = max(
                max_abs_vn_orientation_error,
                float(np.max(np.abs(vnM + vnP))),
            )

            # Use a single paired metric flux in the M orientation. Averaging
            # protects the conservative SAT from tiny opposite-side metric
            # roundoff while still reporting the raw orientation mismatch.
            vn = 0.5 * (vnM - vnP)
            C = _flux_coefficient(vn, flux_type=flux_type, alpha_lf=alpha_lf)
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
    }


def compute_global_conservation_diagnostic(
    q_case,
    flux_type,
    alpha_lf=1.0,
    nsub=4,
    order=4,
    eps=1.0e-2,
    surface_mode="old",
):
    state = build_state_for_q_case(
        q_case=q_case,
        nsub=nsub,
        order=order,
        eps=eps,
    )
    if surface_mode == "old":
        surface = compute_sphere_surface_penalty(
            state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        max_abs_vn_orientation_error = np.nan
    elif surface_mode == "conservative":
        surface = compute_sphere_surface_penalty_conservative(
            state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        max_abs_vn_orientation_error = surface["max_abs_vn_orientation_error"]
    else:
        raise ValueError(f"unknown surface_mode: {surface_mode}")

    volume_rhs = state["volume_rhs"]
    surface_rhs = surface["surface_rhs"]
    rhs = volume_rhs + surface_rhs
    jump_stats = collect_face_jump_stats(state)

    return {
        "q_case": q_case,
        "surface_mode": surface_mode,
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "mass_rate_total": _weighted_integral(state, rhs),
        "mass_rate_volume": _weighted_integral(state, volume_rhs),
        "mass_rate_surface": _weighted_integral(state, surface_rhs),
        "max_abs_volume": float(np.max(np.abs(volume_rhs))),
        "max_abs_surface": float(np.max(np.abs(surface_rhs))),
        "rms_volume": rms(volume_rhs),
        "rms_surface": rms(surface_rhs),
        "max_abs_jump_q": jump_stats["max_abs_jump_q"],
        "rms_jump_q": jump_stats["rms_jump_q"],
        "max_face_match_error": max(
            surface["max_face_match_error"],
            jump_stats["max_face_match_error"],
        ),
        "max_abs_penalty": surface["max_abs_penalty"],
        "max_abs_vn_orientation_error": max_abs_vn_orientation_error,
        "all_finite": bool(
            np.all(np.isfinite(rhs))
            and np.all(np.isfinite(volume_rhs))
            and np.all(np.isfinite(surface_rhs))
        ),
    }


def test_sphere_global_conservation_and_surface_scaling_diagnostic():
    q_cases = ["constant", "smooth", "jump"]
    flux_cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    surface_modes = ["old", "conservative"]
    results = []

    print("\n" + "=" * 168)
    print("Projected sphere global conservation and surface-scaling diagnostic")
    print("=" * 168)
    print("nsub=4, N=4")
    print("Old surface scaling note:")
    print(SURFACE_SCALING_NOTE)
    print("Conservative surface scaling note:")
    print(CONSERVATIVE_SURFACE_SCALING_NOTE)

    for surface_mode in surface_modes:
        print("\n" + "#" * 168)
        print(f"surface_mode = {surface_mode}")
        print("#" * 168)

        for q_case in q_cases:
            print("\n" + "-" * 168)
            print(f"q_case = {q_case}")
            print("-" * 168)
            print(
                f"{'flux':>10s} "
                f"{'alpha_lf':>10s} "
                f"{'mass_total':>15s} "
                f"{'mass_volume':>15s} "
                f"{'mass_surface':>15s} "
                f"{'max_surf':>13s} "
                f"{'rms_surf':>13s} "
                f"{'max_jump_q':>13s} "
                f"{'vn_orient':>13s} "
                f"{'finite':>8s}"
            )

            for flux_type, alpha_lf in flux_cases:
                result = compute_global_conservation_diagnostic(
                    q_case=q_case,
                    flux_type=flux_type,
                    alpha_lf=alpha_lf,
                    surface_mode=surface_mode,
                )
                results.append(result)

                print(
                    f"{flux_type:>10s} "
                    f"{alpha_lf:10.4f} "
                    f"{result['mass_rate_total']:15.6e} "
                    f"{result['mass_rate_volume']:15.6e} "
                    f"{result['mass_rate_surface']:15.6e} "
                    f"{result['max_abs_surface']:13.6e} "
                    f"{result['rms_surface']:13.6e} "
                    f"{result['max_abs_jump_q']:13.6e} "
                    f"{result['max_abs_vn_orientation_error']:13.6e} "
                    f"{str(result['all_finite']):>8s}"
                )

    print("-" * 168)
    print(
        "max_face_match_error = "
        f"{max(row['max_face_match_error'] for row in results):.6e}"
    )
    print(
        "max_abs_penalty = "
        f"{max(row['max_abs_penalty'] for row in results):.6e}"
    )
    conservative_vn_errors = [
        row["max_abs_vn_orientation_error"]
        for row in results
        if row["surface_mode"] == "conservative"
    ]
    print(
        "max_abs_vn_orientation_error(conservative) = "
        f"{max(conservative_vn_errors):.6e}"
    )
    print("=" * 168)

    for result in results:
        scalar_values = [
            result["mass_rate_total"],
            result["mass_rate_volume"],
            result["mass_rate_surface"],
            result["max_abs_volume"],
            result["max_abs_surface"],
            result["rms_volume"],
            result["rms_surface"],
            result["max_abs_jump_q"],
            result["rms_jump_q"],
            result["max_face_match_error"],
            result["max_abs_penalty"],
        ]
        if result["surface_mode"] == "conservative":
            scalar_values.append(result["max_abs_vn_orientation_error"])

        assert result["all_finite"]
        assert np.all(np.isfinite(scalar_values))
        assert result["max_face_match_error"] < 1.0e-12

    conservative_results = [
        row for row in results if row["surface_mode"] == "conservative"
    ]
    old_jump_upwind = next(
        row
        for row in results
        if row["surface_mode"] == "old"
        and row["q_case"] == "jump"
        and row["flux_type"] == "upwind"
    )
    cons_jump_upwind = next(
        row
        for row in conservative_results
        if row["q_case"] == "jump" and row["flux_type"] == "upwind"
    )
    cons_jump_lf1 = next(
        row
        for row in conservative_results
        if row["q_case"] == "jump"
        and row["flux_type"] == "lf"
        and abs(row["alpha_lf"] - 1.0) < 1.0e-14
    )
    cons_jump_lf15 = next(
        row
        for row in conservative_results
        if row["q_case"] == "jump"
        and row["flux_type"] == "lf"
        and abs(row["alpha_lf"] - 1.5) < 1.0e-14
    )

    assert abs(old_jump_upwind["mass_rate_total"]) > 1.0e-5

    for result in conservative_results:
        assert abs(result["mass_rate_total"]) < 1.0e-12

    assert abs(
        cons_jump_upwind["max_abs_surface"] - cons_jump_lf1["max_abs_surface"]
    ) < 1.0e-14
    assert abs(
        cons_jump_upwind["rms_surface"] - cons_jump_lf1["rms_surface"]
    ) < 1.0e-14
    assert abs(
        cons_jump_lf15["rms_surface"] - cons_jump_upwind["rms_surface"]
    ) > 1.0e-4


def run_all_tests():
    test_sphere_global_conservation_and_surface_scaling_diagnostic()
    print("test_sphere_global_conservation_diagnostic.py passed")


if __name__ == "__main__":
    run_all_tests()
