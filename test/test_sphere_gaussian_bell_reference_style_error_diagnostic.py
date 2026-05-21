import numpy as np

from core.operators_sphere import compute_sphere_rhs
from core.time_integration import lsrk54_step
from test.test_sphere_gaussian_bell_reference_style_short_time import (
    build_reference_style_state,
    exact_gaussian_bell_reference_style,
    rate,
    weighted_integral,
    weighted_l2_error,
)


def run_reference_style_case(nsub, final_time, dt, flux_type):
    state = build_reference_style_state(nsub=nsub)
    q_initial = state["q"].copy()
    q = q_initial.copy()
    res = np.zeros_like(q)
    t = 0.0
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
            flux_type=flux_type,
            surface_mode="conservative_scaled",
        )
        t += dt_step
        num_steps += 1

    q_exact = np.zeros_like(q)
    for k, xyz in enumerate(state["xyz"]):
        q_exact[k, :] = exact_gaussian_bell_reference_style(
            xyz,
            t=final_time,
            u0=state["u0"],
            alpha0=state["alpha0"],
            R=state["R"],
            center_xyz=state["center_xyz"],
            width=state["width"],
        )

    state["q"] = q
    error = q - q_exact
    mass_final = weighted_integral(state, q)
    energy_final = weighted_integral(state, q * q)

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "h": float(state["h"]),
        "dt": float(dt),
        "steps": num_steps,
        "flux_type": flux_type,
        "T_final": float(final_time),
        "L2_error": float(weighted_l2_error(state, error)),
        "max_error": float(np.max(np.abs(error))),
        "mass_error": float(mass_final - mass_initial),
        "energy_change": float(energy_final - energy_initial),
        "has_nonfinite": bool(not np.all(np.isfinite(q))),
    }


def print_flux_comparison():
    print("\n" + "=" * 116, flush=True)
    print("A. central vs upwind | reference-style Gaussian | T=0.1 | dt=5e-4", flush=True)
    print("=" * 116, flush=True)
    print(
        f"{'flux':>10s} {'nsub':>8s} {'K':>8s} {'h':>13s} "
        f"{'L2_error':>16s} {'max_error':>16s} "
        f"{'mass_error':>16s} {'energy_change':>16s}",
        flush=True,
    )
    print("-" * 116, flush=True)

    rows = []
    for flux_type in ["upwind", "central"]:
        for nsub in [2, 4, 8]:
            row = run_reference_style_case(
                nsub=nsub,
                final_time=1.0e-1,
                dt=5.0e-4,
                flux_type=flux_type,
            )
            rows.append(row)
            print(
                f"{flux_type:>10s} {row['nsub']:8d} {row['K']:8d} "
                f"{row['h']:13.6e} {row['L2_error']:16.6e} "
                f"{row['max_error']:16.6e} {row['mass_error']:16.6e} "
                f"{row['energy_change']:16.6e}",
                flush=True,
            )

    return rows


def print_dt_refinement():
    print("\n" + "=" * 104, flush=True)
    print("B. dt refinement | reference-style Gaussian | upwind | nsub=8 | T=0.1", flush=True)
    print("=" * 104, flush=True)
    print(
        f"{'dt':>12s} {'steps':>8s} {'L2_error':>16s} "
        f"{'max_error':>16s} {'mass_error':>16s} {'energy_change':>16s}",
        flush=True,
    )
    print("-" * 104, flush=True)

    rows = []
    for dt in [1.0e-3, 5.0e-4, 2.5e-4]:
        row = run_reference_style_case(
            nsub=8,
            final_time=1.0e-1,
            dt=dt,
            flux_type="upwind",
        )
        rows.append(row)
        print(
            f"{row['dt']:12.6e} {row['steps']:8d} "
            f"{row['L2_error']:16.6e} {row['max_error']:16.6e} "
            f"{row['mass_error']:16.6e} {row['energy_change']:16.6e}",
            flush=True,
        )

    return rows


def print_time_length_comparison():
    print("\n" + "=" * 132, flush=True)
    print("C. shorter time comparison | reference-style Gaussian | upwind | dt=5e-4", flush=True)
    print("=" * 132, flush=True)
    print(
        f"{'T_final':>10s} {'nsub':>8s} {'K':>8s} {'h':>13s} "
        f"{'L2_error':>16s} {'L2_rate':>10s} "
        f"{'max_error':>16s} {'max_rate':>10s} "
        f"{'mass_error':>16s} {'energy_change':>16s}",
        flush=True,
    )
    print("-" * 132, flush=True)

    all_rows = []
    for final_time in [3.0e-2, 5.0e-2, 1.0e-1]:
        previous = None
        for nsub in [2, 4, 8]:
            row = run_reference_style_case(
                nsub=nsub,
                final_time=final_time,
                dt=5.0e-4,
                flux_type="upwind",
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
                f"{final_time:10.3e} {row['nsub']:8d} {row['K']:8d} "
                f"{row['h']:13.6e} {row['L2_error']:16.6e} "
                f"{'---' if L2_rate is None else f'{L2_rate:.4f}':>10s} "
                f"{row['max_error']:16.6e} "
                f"{'---' if max_rate is None else f'{max_rate:.4f}':>10s} "
                f"{row['mass_error']:16.6e} {row['energy_change']:16.6e}",
                flush=True,
            )
            all_rows.append(row)
            previous = row

    return all_rows


def test_sphere_gaussian_bell_reference_style_error_diagnostic():
    rows = []
    rows.extend(print_flux_comparison())
    rows.extend(print_dt_refinement())
    rows.extend(print_time_length_comparison())

    for row in rows:
        assert not row["has_nonfinite"]
        assert np.isfinite(row["L2_error"])
        assert np.isfinite(row["max_error"])
        assert np.isfinite(row["mass_error"])
        assert np.isfinite(row["energy_change"])


def run_all_tests():
    test_sphere_gaussian_bell_reference_style_error_diagnostic()
    print("sphere Gaussian bell reference-style error diagnostic passed")


if __name__ == "__main__":
    run_all_tests()
