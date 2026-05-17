import numpy as np

from test.test_manifold_constant_rhs_sphere import compute_manifold_skew_volume_rhs
from test.test_sphere_full_rhs_constant import (
    _aligned_neighbor_face_indices,
    compute_sphere_surface_penalty,
)
from test.test_sphere_full_rhs_smooth_snapshot import (
    _rhs_summary,
    build_projected_sphere_smooth_state,
    smooth_sphere_snapshot_q,
)


def apply_elementwise_jump_field(state, eps=1.0e-2):
    """
    Replace the smooth q field by q_base + eps * sign_K.

    The jump is diagnostic-only: it deliberately creates nonzero face jumps so
    the surface flux skeleton can be distinguished without time integration.
    """
    xyz = state["xyz"]
    q = np.zeros_like(state["q"])

    for k in range(q.shape[0]):
        sign_K = 1.0 if k % 2 == 0 else -1.0
        q[k, :] = smooth_sphere_snapshot_q(xyz[k]) + eps * sign_K

    state["q"] = q
    state["volume_rhs"] = recompute_volume_rhs_for_state(state)

    return state


def recompute_volume_rhs_for_state(state):
    engine = state["engine"]
    volume_rhs = np.zeros_like(state["q"])
    u_tilde = np.zeros_like(state["q"])
    v_tilde = np.zeros_like(state["q"])

    for k, geometry in enumerate(state["geometry"]):
        rhs_vol, _, u_local, v_local = compute_manifold_skew_volume_rhs(
            engine=engine,
            geometry=geometry,
            V3D=state["V3D"][k],
            q=state["q"][k],
        )
        volume_rhs[k, :] = rhs_vol
        u_tilde[k, :] = u_local
        v_tilde[k, :] = v_local

    state["u_tilde"] = u_tilde
    state["v_tilde"] = v_tilde

    return volume_rhs


def collect_face_jump_stats(state):
    engine = state["engine"]
    EToE = state["EToE"]
    EToF = state["EToF"]
    xyz = state["xyz"]
    q = state["q"]
    jumps = []
    max_face_match_error = 0.0

    for kM in range(q.shape[0]):
        for fM in range(3):
            kP = int(EToE[kM, fM])
            fP = int(EToF[kM, fM])

            if (kM, fM) > (kP, fP):
                continue

            nodes_M = np.arange(
                engine.edge_slices[fM].start,
                engine.edge_slices[fM].stop,
            )
            nodes_P = np.arange(
                engine.edge_slices[fP].start,
                engine.edge_slices[fP].stop,
            )
            ordering, face_match_error = _aligned_neighbor_face_indices(
                xyz[kM, nodes_M, :],
                xyz[kP, nodes_P, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            jump_q = q[kM, nodes_M] - q[kP, nodes_P[ordering]]
            jumps.extend(jump_q.tolist())

    jumps = np.asarray(jumps, dtype=float)

    return {
        "max_abs_jump_q": float(np.max(np.abs(jumps))),
        "rms_jump_q": float(np.sqrt(np.mean(jumps**2))),
        "max_face_match_error": max_face_match_error,
    }


def compute_sphere_flux_jump_diagnostic(
    nsub=16,
    order=4,
    eps=1.0e-2,
    flux_type="upwind",
    alpha_lf=1.0,
    surface_mode="conservative_scaled",
):
    state = build_projected_sphere_smooth_state(
        nsub=nsub,
        order=order,
    )
    apply_elementwise_jump_field(state, eps=eps)

    surface = compute_sphere_surface_penalty(
        state,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        surface_mode=surface_mode,
    )
    jump_stats = collect_face_jump_stats(state)

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
    result["eps"] = eps
    result["surface_mode"] = surface_mode
    result["max_abs_jump_q"] = jump_stats["max_abs_jump_q"]
    result["rms_jump_q"] = jump_stats["rms_jump_q"]
    result["max_face_match_error"] = max(
        surface["max_face_match_error"],
        jump_stats["max_face_match_error"],
    )
    result["max_abs_penalty"] = surface["max_abs_penalty"]
    result["max_abs_vn_orientation_error"] = surface["max_abs_vn_orientation_error"]

    return result


def test_projected_sphere_flux_jump_diagnostic():
    cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    results = []

    print("\n" + "=" * 156)
    print("Projected sphere jump-flux diagnostic")
    print("=" * 156)
    print(
        "q = X + 0.5Y - 0.25Z + eps * sign_K, eps = 1e-2, "
        "surface_mode=conservative_scaled"
    )
    print(
        f"{'flux':>10s} "
        f"{'alpha_lf':>10s} "
        f"{'max_abs_rhs':>15s} "
        f"{'rms_rhs':>15s} "
        f"{'max_abs_volume':>16s} "
        f"{'rms_volume':>15s} "
        f"{'max_abs_surface':>18s} "
        f"{'rms_surface':>15s} "
        f"{'mass_rate':>15s} "
        f"{'energy_rate':>15s} "
        f"{'max_jump_q':>14s} "
        f"{'rms_jump_q':>14s}"
    )
    print("-" * 156)

    for flux_type, alpha_lf in cases:
        result = compute_sphere_flux_jump_diagnostic(
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
            f"{result['max_abs_volume']:16.6e} "
            f"{result['rms_volume']:15.6e} "
            f"{result['max_abs_surface']:18.6e} "
            f"{result['rms_surface']:15.6e} "
            f"{result['mass_rate']:15.6e} "
            f"{result['energy_rate']:15.6e} "
            f"{result['max_abs_jump_q']:14.6e} "
            f"{result['rms_jump_q']:14.6e}"
        )

    print("-" * 156)

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
    max_upwind_lf15_rhs_diff = float(np.max(np.abs(upwind["rhs"] - lf_alpha_15["rhs"])))
    max_upwind_lf15_surface_diff = float(
        np.max(np.abs(upwind["surface_rhs"] - lf_alpha_15["surface_rhs"]))
    )

    print(f"max upwind/LF(alpha=1) RHS difference = {max_upwind_lf1_rhs_diff:.6e}")
    print(f"max upwind/LF(alpha=1.5) RHS difference = {max_upwind_lf15_rhs_diff:.6e}")
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
    print("=" * 156)

    for result in results:
        scalar_values = [
            result["max_abs_rhs"],
            result["rms_rhs"],
            result["max_abs_volume"],
            result["rms_volume"],
            result["max_abs_surface"],
            result["rms_surface"],
            result["mass_rate"],
            result["energy_rate"],
            result["max_abs_jump_q"],
            result["rms_jump_q"],
            result["max_face_match_error"],
            result["max_abs_penalty"],
            result["max_abs_vn_orientation_error"],
        ]

        assert np.all(np.isfinite(result["rhs"]))
        assert np.all(np.isfinite(result["volume_rhs"]))
        assert np.all(np.isfinite(result["surface_rhs"]))
        assert np.all(np.isfinite(scalar_values))
        assert result["max_face_match_error"] < 1.0e-12
        assert result["max_abs_jump_q"] > 0.0
        assert result["rms_jump_q"] > 0.0

    assert upwind["max_abs_surface"] > 0.0
    assert lf_alpha_1["max_abs_surface"] > 0.0
    assert lf_alpha_15["max_abs_surface"] > 0.0
    assert max_upwind_lf1_rhs_diff < 1.0e-14, (
        "upwind and LF(alpha=1) should match: "
        f"max difference = {max_upwind_lf1_rhs_diff:.3e}"
    )
    assert max_upwind_lf15_rhs_diff > 1.0e-12, (
        "LF(alpha=1.5) should differ from upwind for a jumped field"
    )
    assert max_upwind_lf15_surface_diff > 1.0e-12, (
        "LF(alpha=1.5) surface contribution should differ from upwind"
    )
    assert lf_alpha_15["rms_surface"] > upwind["rms_surface"], (
        "LF(alpha=1.5) should produce larger surface RMS than upwind"
    )


def run_all_tests():
    test_projected_sphere_flux_jump_diagnostic()
    print("test_sphere_flux_jump_diagnostic.py passed")


if __name__ == "__main__":
    run_all_tests()
