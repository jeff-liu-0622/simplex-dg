import numpy as np

from core.time_integration import lsrk54_step
from core.operators_sphere import compute_sphere_rhs
from test.test_sphere_full_rhs_smooth_snapshot import (
    _weighted_integral,
    build_projected_sphere_smooth_state,
)


compute_sphere_full_rhs_for_state = compute_sphere_rhs


def discrete_mass(state, q):
    return _weighted_integral(state, q)


def discrete_energy(state, q):
    return 0.5 * _weighted_integral(state, q * q)


def run_short_lsrk_sphere_case(
    flux_type,
    alpha_lf=1.0,
    nsub=4,
    order=4,
    T_final=1.0e-3,
    dt=2.5e-4,
    surface_mode="conservative_scaled",
):
    state = build_projected_sphere_smooth_state(
        nsub=nsub,
        order=order,
    )
    q_initial = state["q"].copy()
    q = q_initial.copy()
    res = np.zeros_like(q)
    t = 0.0

    mass_initial = discrete_mass(state, q_initial)
    energy_initial = discrete_energy(state, q_initial)

    while t < T_final - 1.0e-15:
        dt_step = min(dt, T_final - t)
        q, res = lsrk54_step(
            q,
            res,
            t,
            dt_step,
            compute_sphere_full_rhs_for_state,
            state=state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            surface_mode=surface_mode,
        )
        t += dt_step

    state["q"] = q
    mass_final = discrete_mass(state, q)
    energy_final = discrete_energy(state, q)

    return {
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "nsub": nsub,
        "order": order,
        "T_final": T_final,
        "dt": dt,
        "surface_mode": surface_mode,
        "num_steps": int(round(T_final / dt)),
        "q_final": q,
        "q_initial": q_initial,
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "mass_final": float(mass_final),
        "mass_change": float(mass_final - mass_initial),
        "relative_mass_change": float((mass_final - mass_initial) / mass_initial)
        if abs(mass_initial) > 1.0e-30
        else np.nan,
        "energy_final": float(energy_final),
        "energy_change": float(energy_final - energy_initial),
        "max_abs_delta_q": float(np.max(np.abs(q - q_initial))),
        "has_nonfinite": bool(not np.all(np.isfinite(q))),
    }


def test_short_time_sphere_lsrk_sanity():
    cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    results = []

    print("\n" + "=" * 132)
    print("Projected sphere short-time LSRK sanity diagnostic")
    print("=" * 132)
    print(
        "q0 = X + 0.5Y - 0.25Z, nsub=4, N=4, "
        "T_final=1e-3, dt=2.5e-4, surface_mode=conservative_scaled"
    )
    print(
        f"{'flux':>10s} "
        f"{'alpha_lf':>10s} "
        f"{'q_min':>15s} "
        f"{'q_max':>15s} "
        f"{'mass_final':>16s} "
        f"{'mass_change':>16s} "
        f"{'rel_mass':>12s} "
        f"{'energy_final':>16s} "
        f"{'energy_change':>16s} "
        f"{'max_abs_dq':>15s} "
        f"{'nonfinite':>10s}"
    )
    print("-" * 132)

    for flux_type, alpha_lf in cases:
        result = run_short_lsrk_sphere_case(
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )
        results.append(result)

        print(
            f"{flux_type:>10s} "
            f"{alpha_lf:10.4f} "
            f"{result['q_min']:15.6e} "
            f"{result['q_max']:15.6e} "
            f"{result['mass_final']:16.6e} "
            f"{result['mass_change']:16.6e} "
            f"{result['relative_mass_change']:12.6e} "
            f"{result['energy_final']:16.6e} "
            f"{result['energy_change']:16.6e} "
            f"{result['max_abs_delta_q']:15.6e} "
            f"{str(result['has_nonfinite']):>10s}"
        )

    print("-" * 132)

    upwind = next(row for row in results if row["flux_type"] == "upwind")
    lf_alpha_1 = next(
        row
        for row in results
        if row["flux_type"] == "lf" and abs(row["alpha_lf"] - 1.0) < 1.0e-14
    )
    max_upwind_lf1_final_diff = float(
        np.max(np.abs(upwind["q_final"] - lf_alpha_1["q_final"]))
    )

    print(
        "max upwind/LF(alpha=1) final q difference = "
        f"{max_upwind_lf1_final_diff:.6e}"
    )
    print("=" * 132)

    for result in results:
        scalar_values = [
            result["q_min"],
            result["q_max"],
            result["mass_final"],
            result["mass_change"],
            result["relative_mass_change"],
            result["energy_final"],
            result["energy_change"],
            result["max_abs_delta_q"],
        ]

        assert not result["has_nonfinite"]
        assert np.all(np.isfinite(result["q_final"]))
        assert np.all(np.isfinite(scalar_values))
        assert result["max_abs_delta_q"] > 0.0

    assert max_upwind_lf1_final_diff < 1.0e-13, (
        "upwind and LF(alpha=1) should produce the same short-time state: "
        f"max difference = {max_upwind_lf1_final_diff:.3e}"
    )


def run_all_tests():
    test_short_time_sphere_lsrk_sanity()
    print("test_sphere_lsrk_short_sanity.py passed")


if __name__ == "__main__":
    run_all_tests()
