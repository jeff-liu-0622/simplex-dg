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


NSUB8_H = 1.417780e-01
NSUB8_REFERENCE = {
    3.0e-2: {
        "L2_error": 1.104249e-05,
        "max_error": 4.348506e-04,
    },
    5.0e-2: {
        "L2_error": 2.226118e-05,
        "max_error": 1.048006e-03,
    },
    1.0e-1: {
        "L2_error": 9.125198e-05,
        "max_error": 4.614021e-03,
    },
}


def run_nsub16_reference_style_case(final_time):
    nsub = 16
    dt = 1.0e-3
    flux_type = "upwind"

    state = build_reference_style_state(nsub=nsub)
    q_initial = state["q"].copy()
    q = q_initial.copy()
    res = np.zeros_like(q)
    t = 0.0
    num_steps = 0

    mass_initial = weighted_integral(state, q_initial)
    energy_initial = weighted_integral(state, q_initial * q_initial)

    print("\n" + "=" * 112, flush=True)
    print(
        "reference-style Gaussian nsub=16 diagnostic | "
        "projected 3D manifold | LSRK54 | upwind",
        flush=True,
    )
    print("=" * 112, flush=True)
    print(
        "q0=exp(-(dist/width)^2), width=1/sqrt(10), center=(1,0,0), "
        f"Omega=(-sin(-pi/4),0,cos(-pi/4)), T_final={final_time:g}, dt=1e-3",
        flush=True,
    )
    print(
        f"start: nsub={nsub}, K={q.shape[0]}, h={state['h']:.6e}, "
        f"expected steps={int(round(final_time / dt))}",
        flush=True,
    )

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

        if num_steps % 10 == 0 or t >= final_time - 1.0e-15:
            print(
                f"progress: step={num_steps:4d}, t={t:.6e}/{final_time:.6e}",
                flush=True,
            )

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
    L2_error = float(weighted_l2_error(state, error))
    max_error = float(np.max(np.abs(error)))
    nsub8 = NSUB8_REFERENCE[float(final_time)]
    L2_rate = rate(nsub8["L2_error"], L2_error, NSUB8_H, state["h"])
    max_rate = rate(nsub8["max_error"], max_error, NSUB8_H, state["h"])
    mass_error = float(mass_final - mass_initial)
    energy_change = float(energy_final - energy_initial)

    print("-" * 112, flush=True)
    print(f"T_final = {final_time:.6e}", flush=True)
    print(f"nsub = {nsub}", flush=True)
    print(f"K = {q.shape[0]}", flush=True)
    print(f"h = {state['h']:.6e}", flush=True)
    print(f"dt = {dt:.6e}", flush=True)
    print(f"steps = {num_steps}", flush=True)
    print(f"L2_error = {L2_error:.6e}", flush=True)
    print(f"max_error = {max_error:.6e}", flush=True)
    print(f"mass_error = {mass_error:.6e}", flush=True)
    print(f"energy_change = {energy_change:.6e}", flush=True)
    print(f"L2_rate_from_8_to_16 = {L2_rate:.6f}", flush=True)
    print(f"max_rate_from_8_to_16 = {max_rate:.6f}", flush=True)
    print("=" * 112, flush=True)

    assert np.all(np.isfinite(q))
    assert np.isfinite(L2_error)
    assert np.isfinite(max_error)
    assert np.isfinite(mass_error)
    assert np.isfinite(energy_change)

    return {
        "T_final": float(final_time),
        "steps": num_steps,
        "L2_error": L2_error,
        "max_error": max_error,
        "mass_error": mass_error,
        "energy_change": energy_change,
        "L2_rate_from_8_to_16": float(L2_rate),
        "max_rate_from_8_to_16": float(max_rate),
    }


def run_all_tests():
    rows = []
    for final_time in [3.0e-2, 5.0e-2, 1.0e-1]:
        rows.append(run_nsub16_reference_style_case(final_time))

    print("\n" + "=" * 128, flush=True)
    print("nsub=16 time-length comparison summary", flush=True)
    print("=" * 128, flush=True)
    print(
        f"{'T_final':>12s} {'steps':>8s} "
        f"{'L2_error':>16s} {'max_error':>16s} "
        f"{'mass_error':>16s} {'energy_change':>16s} "
        f"{'L2_rate_8_16':>16s} {'max_rate_8_16':>16s}",
        flush=True,
    )
    print("-" * 128, flush=True)
    for row in rows:
        print(
            f"{row['T_final']:12.6e} {row['steps']:8d} "
            f"{row['L2_error']:16.6e} {row['max_error']:16.6e} "
            f"{row['mass_error']:16.6e} {row['energy_change']:16.6e} "
            f"{row['L2_rate_from_8_to_16']:16.6f} "
            f"{row['max_rate_from_8_to_16']:16.6f}",
            flush=True,
        )
    print("=" * 128, flush=True)


if __name__ == "__main__":
    run_all_tests()
