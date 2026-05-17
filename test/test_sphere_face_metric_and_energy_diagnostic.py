import numpy as np

from core.rhs_sphere import (
    REFERENCE_FACE_LENGTHS,
    REFERENCE_FACE_NORMALS,
    aligned_neighbor_face_indices,
    compute_sphere_surface_penalty,
    face_node_indices,
    physical_conormal_flux_on_face,
)
from test.test_sphere_flux_jump_diagnostic import (
    apply_elementwise_jump_field,
    recompute_volume_rhs_for_state,
)
from test.test_sphere_full_rhs_smooth_snapshot import (
    _weighted_integral,
    build_projected_sphere_smooth_state,
)


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def gaussian_bell_q(xyz, beta=20.0):
    center = np.array([1.0, 1.0, 1.0], dtype=float)
    center /= np.linalg.norm(center)
    distance_squared = np.sum((xyz - center[None, :]) ** 2, axis=1)
    return np.exp(-beta * distance_squared)


def build_state_for_q_case(q_case, nsub=4, order=4, eps=1.0e-2):
    state = build_projected_sphere_smooth_state(nsub=nsub, order=order)

    if q_case == "constant":
        state["q"] = np.ones_like(state["q"])
    elif q_case == "smooth":
        pass
    elif q_case == "jump":
        apply_elementwise_jump_field(state, eps=eps)
        return state
    elif q_case == "gaussian":
        q = np.zeros_like(state["q"])
        for k, xyz in enumerate(state["xyz"]):
            q[k, :] = gaussian_bell_q(xyz)
        state["q"] = q
    else:
        raise ValueError(f"unknown q_case: {q_case}")

    state["volume_rhs"] = recompute_volume_rhs_for_state(state)
    return state


def face_metric_comparison_diagnostic(nsub=4, order=4):
    state = build_projected_sphere_smooth_state(nsub=nsub, order=order)
    engine = state["engine"]
    records = []
    differences = []
    scaled_differences = []
    ratios = []
    scaled_ratios = []
    sign_mismatch_count = 0
    scaled_sign_mismatch_count = 0
    max_face_match_error = 0.0
    per_face = {
        face_id: {
            "diff": [],
            "scaled_diff": [],
            "ratio": [],
            "scaled_ratio": [],
            "sign_mismatch": 0,
            "scaled_sign_mismatch": 0,
        }
        for face_id in range(3)
    }

    for kM in range(state["q"].shape[0]):
        for fM in range(3):
            kP = int(state["EToE"][kM, fM])
            fP = int(state["EToF"][kM, fM])
            if (kM, fM) > (kP, fP):
                continue

            nodes_M = face_node_indices(engine, fM)
            nodes_P = face_node_indices(engine, fP)
            ordering, face_match_error = aligned_neighbor_face_indices(
                state["xyz"][kM, nodes_M, :],
                state["xyz"][kP, nodes_P, :],
            )
            max_face_match_error = max(max_face_match_error, face_match_error)

            A = physical_conormal_flux_on_face(state, kM, fM, nodes_M)
            A_scaled = A / REFERENCE_FACE_LENGTHS[fM]
            nr, ns = REFERENCE_FACE_NORMALS[fM]
            B = state["geometry"][kM]["J"][nodes_M] * (
                nr * state["u_tilde"][kM, nodes_M]
                + ns * state["v_tilde"][kM, nodes_M]
            )

            for local_i, node in enumerate(nodes_M):
                a_val = float(A[local_i])
                b_val = float(B[local_i])
                a_scaled_val = float(A_scaled[local_i])
                diff = a_val - b_val
                scaled_diff = a_scaled_val - b_val
                if abs(b_val) > 1.0e-14:
                    ratio = a_val / b_val
                    scaled_ratio = a_scaled_val / b_val
                    ratios.append(ratio)
                    scaled_ratios.append(scaled_ratio)
                    per_face[fM]["ratio"].append(ratio)
                    per_face[fM]["scaled_ratio"].append(scaled_ratio)
                else:
                    ratio = np.nan
                    scaled_ratio = np.nan

                if abs(a_val) > 1.0e-14 and abs(b_val) > 1.0e-14:
                    if np.sign(a_val) != np.sign(b_val):
                        sign_mismatch_count += 1
                        per_face[fM]["sign_mismatch"] += 1
                if abs(a_scaled_val) > 1.0e-14 and abs(b_val) > 1.0e-14:
                    if np.sign(a_scaled_val) != np.sign(b_val):
                        scaled_sign_mismatch_count += 1
                        per_face[fM]["scaled_sign_mismatch"] += 1

                records.append(
                    {
                        "element": kM,
                        "face": fM,
                        "node": int(node),
                        "patch_id": int(state["patch_ids"][kM]),
                        "X": state["xyz"][kM, node, :],
                        "A": a_val,
                        "A_scaled": a_scaled_val,
                        "B": b_val,
                        "diff": diff,
                        "scaled_diff": scaled_diff,
                        "ratio": ratio,
                        "scaled_ratio": scaled_ratio,
                        "sign": f"{np.sign(a_val):.0f}/{np.sign(b_val):.0f}",
                        "scaled_sign": f"{np.sign(a_scaled_val):.0f}/{np.sign(b_val):.0f}",
                        "neighbor_element": kP,
                        "neighbor_face": fP,
                        "neighbor_node": int(nodes_P[ordering][local_i]),
                    }
                )
                differences.append(diff)
                scaled_differences.append(scaled_diff)
                per_face[fM]["diff"].append(diff)
                per_face[fM]["scaled_diff"].append(scaled_diff)

    differences = np.asarray(differences, dtype=float)
    scaled_differences = np.asarray(scaled_differences, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    scaled_ratios = np.asarray(scaled_ratios, dtype=float)
    finite_ratios = ratios[np.isfinite(ratios)]
    finite_scaled_ratios = scaled_ratios[np.isfinite(scaled_ratios)]
    records.sort(key=lambda row: abs(row["diff"]), reverse=True)
    scaled_records = sorted(records, key=lambda row: abs(row["scaled_diff"]), reverse=True)

    per_face_stats = {}
    for face_id, values in per_face.items():
        diff = np.asarray(values["diff"], dtype=float)
        scaled_diff = np.asarray(values["scaled_diff"], dtype=float)
        ratio = np.asarray(values["ratio"], dtype=float)
        scaled_ratio = np.asarray(values["scaled_ratio"], dtype=float)
        ratio = ratio[np.isfinite(ratio)]
        scaled_ratio = scaled_ratio[np.isfinite(scaled_ratio)]
        per_face_stats[face_id] = {
            "count": int(diff.size),
            "Lref": float(REFERENCE_FACE_LENGTHS[face_id]),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "rms_diff": rms(diff),
            "max_abs_scaled_diff": float(np.max(np.abs(scaled_diff))),
            "rms_scaled_diff": rms(scaled_diff),
            "ratio_min": float(np.min(ratio)),
            "ratio_max": float(np.max(ratio)),
            "scaled_ratio_min": float(np.min(scaled_ratio)),
            "scaled_ratio_max": float(np.max(scaled_ratio)),
            "scaled_ratio_mean": float(np.mean(scaled_ratio)),
            "sign_mismatch": values["sign_mismatch"],
            "scaled_sign_mismatch": values["scaled_sign_mismatch"],
        }

    return {
        "max_abs_diff": float(np.max(np.abs(differences))),
        "rms_diff": rms(differences),
        "max_abs_scaled_diff": float(np.max(np.abs(scaled_differences))),
        "rms_scaled_diff": rms(scaled_differences),
        "ratio_min": float(np.min(finite_ratios)),
        "ratio_max": float(np.max(finite_ratios)),
        "ratio_mean": float(np.mean(finite_ratios)),
        "max_abs_ratio": float(np.max(np.abs(finite_ratios))),
        "scaled_ratio_min": float(np.min(finite_scaled_ratios)),
        "scaled_ratio_max": float(np.max(finite_scaled_ratios)),
        "scaled_ratio_mean": float(np.mean(finite_scaled_ratios)),
        "max_abs_scaled_ratio": float(np.max(np.abs(finite_scaled_ratios))),
        "sign_mismatch_count": sign_mismatch_count,
        "scaled_sign_mismatch_count": scaled_sign_mismatch_count,
        "num_face_nodes": int(differences.size),
        "max_face_match_error": max_face_match_error,
        "per_face_stats": per_face_stats,
        "top_records": records[:10],
        "top_scaled_records": scaled_records[:10],
    }


def energy_rate_diagnostic(
    q_case,
    flux_type,
    alpha_lf=1.0,
    surface_mode="old",
    nsub=4,
    order=4,
):
    state = build_state_for_q_case(q_case=q_case, nsub=nsub, order=order)
    surface = compute_sphere_surface_penalty(
        state,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        surface_mode=surface_mode,
    )

    volume_rhs = state["volume_rhs"]
    surface_rhs = surface["surface_rhs"]
    rhs = volume_rhs + surface_rhs
    q = state["q"]

    return {
        "q_case": q_case,
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "surface_mode": surface_mode,
        "mass_rate_total": _weighted_integral(state, rhs),
        "energy_rate_total": _weighted_integral(state, q * rhs),
        "energy_rate_volume": _weighted_integral(state, q * volume_rhs),
        "energy_rate_surface": _weighted_integral(state, q * surface_rhs),
        "max_abs_surface": float(np.max(np.abs(surface_rhs))),
        "rms_surface": rms(surface_rhs),
        "max_face_match_error": surface["max_face_match_error"],
        "max_abs_penalty": surface["max_abs_penalty"],
        "all_finite": bool(
            np.all(np.isfinite(rhs))
            and np.all(np.isfinite(volume_rhs))
            and np.all(np.isfinite(surface_rhs))
        ),
    }


def test_sphere_face_metric_and_energy_diagnostic():
    metric = face_metric_comparison_diagnostic()

    print("\n" + "=" * 148)
    print("Projected sphere face metric comparison diagnostic")
    print("=" * 148)
    print("A = V3D dot (tau x n_surf)")
    print("B = J * (nr*u_tilde + ns*v_tilde)")
    print(f"num_face_nodes       = {metric['num_face_nodes']}")
    print(f"max_abs(A-B)        = {metric['max_abs_diff']:.6e}")
    print(f"rms(A-B)            = {metric['rms_diff']:.6e}")
    print(f"max_abs(A/Lref-B)   = {metric['max_abs_scaled_diff']:.6e}")
    print(f"rms(A/Lref-B)       = {metric['rms_scaled_diff']:.6e}")
    print(f"ratio_min           = {metric['ratio_min']:.6e}")
    print(f"ratio_max           = {metric['ratio_max']:.6e}")
    print(f"ratio_mean          = {metric['ratio_mean']:.6e}")
    print(f"max_abs_ratio       = {metric['max_abs_ratio']:.6e}")
    print(f"scaled_ratio_min    = {metric['scaled_ratio_min']:.6e}")
    print(f"scaled_ratio_max    = {metric['scaled_ratio_max']:.6e}")
    print(f"scaled_ratio_mean   = {metric['scaled_ratio_mean']:.6e}")
    print(f"max_abs_scaled_ratio= {metric['max_abs_scaled_ratio']:.6e}")
    print(f"sign_mismatch_count = {metric['sign_mismatch_count']}")
    print(f"scaled_sign_mismatch= {metric['scaled_sign_mismatch_count']}")
    print(f"max_face_match_error= {metric['max_face_match_error']:.6e}")
    print("-" * 148)
    print("Per-face scaling statistics")
    print(
        f"{'face':>4s} {'Lref':>12s} {'count':>7s} "
        f"{'max|A-B|':>14s} {'max|A/L-B|':>14s} "
        f"{'rms(A/L-B)':>14s} {'ratio':>17s} {'scaled_ratio':>17s}"
    )
    for face_id, stats in metric["per_face_stats"].items():
        ratio_range = f"{stats['ratio_min']:.6g}..{stats['ratio_max']:.6g}"
        scaled_ratio_range = (
            f"{stats['scaled_ratio_min']:.6g}..{stats['scaled_ratio_max']:.6g}"
        )
        print(
            f"{face_id:4d} {stats['Lref']:12.6e} {stats['count']:7d} "
            f"{stats['max_abs_diff']:14.6e} "
            f"{stats['max_abs_scaled_diff']:14.6e} "
            f"{stats['rms_scaled_diff']:14.6e} "
            f"{ratio_range:>17s} {scaled_ratio_range:>17s}"
        )
    print("-" * 148)
    print(
        f"{'rank':>4s} {'elem':>6s} {'face':>4s} {'node':>4s} {'patch':>5s} "
        f"{'X':>38s} {'A':>14s} {'B':>14s} {'ratio':>14s} {'sign':>8s}"
    )
    for rank, row in enumerate(metric["top_records"], start=1):
        x = row["X"]
        print(
            f"{rank:4d} {row['element']:6d} {row['face']:4d} {row['node']:4d} "
            f"{row['patch_id']:5d} "
            f"({x[0]: .4e},{x[1]: .4e},{x[2]: .4e}) "
            f"{row['A']:14.6e} {row['B']:14.6e} "
            f"{row['ratio']:14.6e} {row['sign']:>8s}"
        )

    q_cases = ["constant", "smooth", "jump", "gaussian"]
    flux_cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    surface_modes = ["old", "conservative"]
    results = []

    print("\n" + "=" * 170)
    print("Projected sphere semi-discrete energy-rate diagnostic")
    print("=" * 170)
    print("nsub=4, N=4; no time integration")

    for q_case in q_cases:
        print("\n" + "-" * 170)
        print(f"q_case = {q_case}")
        print("-" * 170)
        print(
            f"{'mode':>13s} {'flux':>10s} {'alpha':>8s} "
            f"{'mass_total':>15s} {'energy_total':>15s} "
            f"{'energy_vol':>15s} {'energy_surf':>15s} "
            f"{'max_surf':>13s} {'rms_surf':>13s} {'finite':>8s} {'flag':>18s}"
        )

        for surface_mode in surface_modes:
            for flux_type, alpha_lf in flux_cases:
                row = energy_rate_diagnostic(
                    q_case=q_case,
                    flux_type=flux_type,
                    alpha_lf=alpha_lf,
                    surface_mode=surface_mode,
                )
                flag = ""
                if (
                    surface_mode == "conservative"
                    and flux_type in ("upwind", "lf")
                    and row["energy_rate_surface"] > 1.0e-12
                ):
                    flag = "anti-dissipative"
                row["flag"] = flag
                results.append(row)
                print(
                    f"{surface_mode:>13s} {flux_type:>10s} {alpha_lf:8.4f} "
                    f"{row['mass_rate_total']:15.6e} "
                    f"{row['energy_rate_total']:15.6e} "
                    f"{row['energy_rate_volume']:15.6e} "
                    f"{row['energy_rate_surface']:15.6e} "
                    f"{row['max_abs_surface']:13.6e} "
                    f"{row['rms_surface']:13.6e} "
                    f"{str(row['all_finite']):>8s} "
                    f"{flag:>18s}"
                )

    anti_dissipative = [
        row
        for row in results
        if row["surface_mode"] == "conservative"
        and row["flux_type"] in ("upwind", "lf")
        and row["energy_rate_surface"] > 1.0e-12
    ]
    print("-" * 170)
    print(f"conservative upwind/LF anti-dissipative rows = {len(anti_dissipative)}")
    print("=" * 170)

    scaled_results = []
    print("\n" + "=" * 154)
    print("Diagnostic-only conservative_scaled surface mode")
    print("=" * 154)
    print("Uses A_scaled = [V3D dot (tau x n_surf)] / Lref as shared metric flux")
    print(
        f"{'q_case':>10s} {'flux':>10s} {'alpha':>8s} "
        f"{'mass_total':>15s} {'energy_surf':>15s} "
        f"{'max_surf':>13s} {'rms_surf':>13s} {'finite':>8s} {'flag':>18s}"
    )
    for q_case in ("jump", "gaussian"):
        for flux_type, alpha_lf in flux_cases:
            row = energy_rate_diagnostic(
                q_case=q_case,
                flux_type=flux_type,
                alpha_lf=alpha_lf,
                surface_mode="conservative_scaled",
            )
            flag = ""
            if flux_type in ("upwind", "lf") and row["energy_rate_surface"] > 1.0e-12:
                flag = "anti-dissipative"
            row["flag"] = flag
            scaled_results.append(row)
            print(
                f"{q_case:>10s} {flux_type:>10s} {alpha_lf:8.4f} "
                f"{row['mass_rate_total']:15.6e} "
                f"{row['energy_rate_surface']:15.6e} "
                f"{row['max_abs_surface']:13.6e} "
                f"{row['rms_surface']:13.6e} "
                f"{str(row['all_finite']):>8s} "
                f"{flag:>18s}"
            )
    scaled_bad_energy = [
        row
        for row in scaled_results
        if row["flux_type"] in ("upwind", "lf")
        and row["energy_rate_surface"] > 1.0e-12
    ]
    print("-" * 154)
    print(f"conservative_scaled upwind/LF anti-dissipative rows = {len(scaled_bad_energy)}")
    print("=" * 154)

    assert metric["max_face_match_error"] < 1.0e-12
    assert np.isfinite(metric["max_abs_diff"])
    assert np.isfinite(metric["rms_diff"])
    assert metric["max_abs_scaled_diff"] < 1.0e-12
    assert metric["scaled_sign_mismatch_count"] == 0
    for row in results:
        scalar_values = [
            row["mass_rate_total"],
            row["energy_rate_total"],
            row["energy_rate_volume"],
            row["energy_rate_surface"],
            row["max_abs_surface"],
            row["rms_surface"],
            row["max_face_match_error"],
            row["max_abs_penalty"],
        ]
        assert row["all_finite"]
        assert np.all(np.isfinite(scalar_values))
        assert row["max_face_match_error"] < 1.0e-12
    for row in scaled_results:
        scalar_values = [
            row["mass_rate_total"],
            row["energy_rate_total"],
            row["energy_rate_volume"],
            row["energy_rate_surface"],
            row["max_abs_surface"],
            row["rms_surface"],
            row["max_face_match_error"],
            row["max_abs_penalty"],
        ]
        assert row["all_finite"]
        assert np.all(np.isfinite(scalar_values))
        assert row["max_face_match_error"] < 1.0e-12
        assert abs(row["mass_rate_total"]) < 1.0e-12
    for row in scaled_results:
        if row["flux_type"] in ("upwind", "lf"):
            assert row["energy_rate_surface"] <= 1.0e-12


def run_all_tests():
    test_sphere_face_metric_and_energy_diagnostic()
    print("test_sphere_face_metric_and_energy_diagnostic.py passed")


if __name__ == "__main__":
    run_all_tests()
