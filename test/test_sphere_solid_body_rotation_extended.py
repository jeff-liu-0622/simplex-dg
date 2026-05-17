import numpy as np

from core.time_integration import lsrk54_step
from test.test_sphere_full_rhs_smooth_snapshot import (
    _weighted_integral,
    build_projected_sphere_smooth_state,
)
from test.test_sphere_lsrk_short_sanity import compute_sphere_full_rhs_for_state


def gaussian_bell_on_sphere(xyz, beta=20.0):
    center = np.array([1.0, 1.0, 1.0], dtype=float)
    center /= np.linalg.norm(center)
    distance_squared = np.sum((xyz - center[None, :]) ** 2, axis=1)

    return np.exp(-beta * distance_squared)


def set_gaussian_initial_condition(state, beta=20.0):
    q = np.zeros_like(state["q"])

    for k, xyz in enumerate(state["xyz"]):
        q[k, :] = gaussian_bell_on_sphere(xyz, beta=beta)

    state["q"] = q
    return q


def discrete_mass(state, q):
    return _weighted_integral(state, q)


def discrete_energy(state, q):
    return 0.5 * _weighted_integral(state, q * q)


def weighted_l2_norm(state, values):
    return np.sqrt(_weighted_integral(state, values * values))


def run_extended_solid_body_case(
    final_time,
    flux_type,
    alpha_lf=1.0,
    nsub=4,
    order=4,
    beta=20.0,
    dt=5.0e-3,
    surface_mode="conservative_scaled",
):
    state = build_projected_sphere_smooth_state(
        nsub=nsub,
        order=order,
        u0=1.0,
    )
    q_initial = set_gaussian_initial_condition(state, beta=beta).copy()
    q = q_initial.copy()
    res = np.zeros_like(q)
    t = 0.0

    mass_initial = discrete_mass(state, q_initial)
    energy_initial = discrete_energy(state, q_initial)

    num_steps = 0
    while t < final_time - 1.0e-15:
        dt_step = min(dt, final_time - t)
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
        num_steps += 1

    state["q"] = q
    mass_final = discrete_mass(state, q)
    energy_final = discrete_energy(state, q)
    mass_change = mass_final - mass_initial
    energy_change = energy_final - energy_initial
    return_difference = q - q_initial

    return {
        "final_time": final_time,
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "nsub": nsub,
        "order": order,
        "beta": beta,
        "dt": dt,
        "surface_mode": surface_mode,
        "num_steps": num_steps,
        "q_initial": q_initial,
        "q_final": q,
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "mass_initial": float(mass_initial),
        "mass_final": float(mass_final),
        "mass_change": float(mass_change),
        "relative_mass_change": float(mass_change / mass_initial),
        "energy_initial": float(energy_initial),
        "energy_final": float(energy_final),
        "energy_change": float(energy_change),
        "relative_energy_change": float(energy_change / energy_initial),
        "l2_return_error": float(weighted_l2_norm(state, return_difference)),
        "linf_return_error": float(np.max(np.abs(return_difference))),
        "max_abs_delta_q": float(np.max(np.abs(return_difference))),
        "has_nonfinite": bool(not np.all(np.isfinite(q))),
    }


def test_sphere_solid_body_rotation_extended_time_diagnostic():
    final_times = [1.0e-2, 1.0e-1, 1.0]
    cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    all_results = []

    print("\n" + "=" * 164)
    print("Projected sphere solid-body rotation extended-time diagnostic")
    print("=" * 164)
    print(
        "q0 = exp(-20 * ||X - normalize([1,1,1])||^2), "
        "nsub=4, N=4, dt=5e-3, surface_mode=conservative_scaled"
    )

    for final_time in final_times:
        print("\n" + "-" * 164)
        print(f"T_final = {final_time:.6e}")
        print("-" * 164)
        print(
            f"{'flux':>10s} "
            f"{'alpha_lf':>10s} "
            f"{'steps':>7s} "
            f"{'q_min':>14s} "
            f"{'q_max':>14s} "
            f"{'mass_initial':>16s} "
            f"{'mass_final':>16s} "
            f"{'mass_change':>16s} "
            f"{'rel_mass':>12s} "
            f"{'energy_initial':>16s} "
            f"{'energy_final':>16s} "
            f"{'energy_change':>16s} "
            f"{'max_abs_dq':>14s} "
            f"{'nonfinite':>10s}"
        )

        time_results = []
        for flux_type, alpha_lf in cases:
            result = run_extended_solid_body_case(
                final_time=final_time,
                flux_type=flux_type,
                alpha_lf=alpha_lf,
            )
            time_results.append(result)
            all_results.append(result)

            print(
                f"{flux_type:>10s} "
                f"{alpha_lf:10.4f} "
                f"{result['num_steps']:7d} "
                f"{result['q_min']:14.6e} "
                f"{result['q_max']:14.6e} "
                f"{result['mass_initial']:16.6e} "
                f"{result['mass_final']:16.6e} "
                f"{result['mass_change']:16.6e} "
                f"{result['relative_mass_change']:12.6e} "
                f"{result['energy_initial']:16.6e} "
                f"{result['energy_final']:16.6e} "
                f"{result['energy_change']:16.6e} "
                f"{result['max_abs_delta_q']:14.6e} "
                f"{str(result['has_nonfinite']):>10s}"
            )

        upwind = next(row for row in time_results if row["flux_type"] == "upwind")
        lf_alpha_1 = next(
            row
            for row in time_results
            if row["flux_type"] == "lf" and abs(row["alpha_lf"] - 1.0) < 1.0e-14
        )
        max_upwind_lf1_final_diff = float(
            np.max(np.abs(upwind["q_final"] - lf_alpha_1["q_final"]))
        )
        print(
            "max upwind/LF(alpha=1) final q difference = "
            f"{max_upwind_lf1_final_diff:.6e}"
        )
        assert max_upwind_lf1_final_diff < 1.0e-13, (
            "upwind and LF(alpha=1) should produce identical final states: "
            f"max difference = {max_upwind_lf1_final_diff:.3e}"
        )

    print("=" * 164)

    for result in all_results:
        scalar_values = [
            result["q_min"],
            result["q_max"],
            result["mass_initial"],
            result["mass_final"],
            result["mass_change"],
            result["relative_mass_change"],
            result["energy_initial"],
            result["energy_final"],
            result["energy_change"],
            result["max_abs_delta_q"],
        ]

        assert not result["has_nonfinite"]
        assert np.all(np.isfinite(result["q_final"]))
        assert np.all(np.isfinite(scalar_values))


def test_sphere_solid_body_rotation_one_period_return_diagnostic():
    final_time = 2.0 * np.pi
    dt = 5.0e-3
    cases = [
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    results = []

    print("\n" + "=" * 188)
    print("Projected sphere solid-body rotation one-period return diagnostic")
    print("=" * 188)
    print("central flux skipped: Phase 2.9a showed large oscillation/energy growth by T=1.0")
    print("q0 = exp(-20 * ||X - normalize([1,1,1])||^2), nsub=4, N=4")
    print(f"T_period = {final_time:.12e}, dt = {dt:.4e}")
    print("-" * 188)
    print(
        f"{'flux':>10s} "
        f"{'alpha_lf':>10s} "
        f"{'T_final':>14s} "
        f"{'steps':>7s} "
        f"{'q_min':>13s} "
        f"{'q_max':>13s} "
        f"{'mass_initial':>15s} "
        f"{'mass_final':>15s} "
        f"{'mass_change':>15s} "
        f"{'rel_mass':>12s} "
        f"{'energy_initial':>15s} "
        f"{'energy_final':>15s} "
        f"{'energy_change':>15s} "
        f"{'rel_energy':>12s} "
        f"{'L2_return':>13s} "
        f"{'Linf_return':>13s} "
        f"{'nonfinite':>10s}"
    )

    for flux_type, alpha_lf in cases:
        result = run_extended_solid_body_case(
            final_time=final_time,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            dt=dt,
        )
        results.append(result)

        print(
            f"{flux_type:>10s} "
            f"{alpha_lf:10.4f} "
            f"{result['final_time']:14.6e} "
            f"{result['num_steps']:7d} "
            f"{result['q_min']:13.6e} "
            f"{result['q_max']:13.6e} "
            f"{result['mass_initial']:15.6e} "
            f"{result['mass_final']:15.6e} "
            f"{result['mass_change']:15.6e} "
            f"{result['relative_mass_change']:12.6e} "
            f"{result['energy_initial']:15.6e} "
            f"{result['energy_final']:15.6e} "
            f"{result['energy_change']:15.6e} "
            f"{result['relative_energy_change']:12.6e} "
            f"{result['l2_return_error']:13.6e} "
            f"{result['linf_return_error']:13.6e} "
            f"{str(result['has_nonfinite']):>10s}"
        )

    print("-" * 188)

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
    print("=" * 188)

    for result in results:
        scalar_values = [
            result["q_min"],
            result["q_max"],
            result["mass_initial"],
            result["mass_final"],
            result["mass_change"],
            result["relative_mass_change"],
            result["energy_initial"],
            result["energy_final"],
            result["energy_change"],
            result["relative_energy_change"],
            result["l2_return_error"],
            result["linf_return_error"],
        ]

        assert not result["has_nonfinite"]
        assert np.all(np.isfinite(result["q_final"]))
        assert np.all(np.isfinite(scalar_values))

    assert max_upwind_lf1_final_diff < 1.0e-13, (
        "upwind and LF(alpha=1) should produce identical final states: "
        f"max difference = {max_upwind_lf1_final_diff:.3e}"
    )


def run_all_tests():
    test_sphere_solid_body_rotation_extended_time_diagnostic()
    print("test_sphere_solid_body_rotation_extended.py passed")


if __name__ == "__main__":
    run_all_tests()
