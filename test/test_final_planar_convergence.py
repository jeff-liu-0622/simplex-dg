import time

import numpy as np

from core.cfl import compute_advection_dt
from core.operators_split import compute_split_rhs
from core.time_integration import lsrk54_step
from test.test_flux_comparison_sinxy import (
    build_periodic_planar_case,
    q_exact_sinxy,
    weighted_l2_error,
)


def observed_order(previous_error, current_error, previous_size, current_size):
    if previous_error is None:
        return np.nan
    return np.log(previous_error / current_error) / np.log(previous_size / current_size)


def integrate_planar_periodic(
    N_poly,
    n_quad,
    divisions,
    final_time,
    dt_target,
    flux_type="upwind",
    alpha_lf=1.0,
):
    cx, cy = 1.0, 1.0
    engine, x_nodes, y_nodes, J, kwargs = build_periodic_planar_case(
        N_poly=N_poly,
        n_quad=n_quad,
        divisions=divisions,
        cx=cx,
        cy=cy,
    )
    kwargs = {
        **kwargs,
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
    }

    steps = int(np.ceil(final_time / dt_target))
    dt = final_time / steps
    q = q_exact_sinxy(x_nodes, y_nodes, 0.0, cx=cx, cy=cy)
    res = np.zeros_like(q)
    t = 0.0

    for _ in range(steps):
        q, res = lsrk54_step(
            q,
            res,
            t,
            dt,
            compute_split_rhs,
            **kwargs,
        )
        t += dt

    return {
        "q": q,
        "engine": engine,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "J": J,
        "dt": dt,
        "steps": steps,
        "actual_time": t,
        "cx": cx,
        "cy": cy,
    }


def final_rk_temporal_convergence():
    N_poly = 3
    n_quad = 4
    divisions = 4
    final_time = 5.0e-2
    dt_targets = [4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4]
    dt_reference = 1.25e-4

    reference = integrate_planar_periodic(
        N_poly=N_poly,
        n_quad=n_quad,
        divisions=divisions,
        final_time=final_time,
        dt_target=dt_reference,
    )

    rows = []
    previous_l2 = None
    previous_linf = None
    previous_dt = None

    for dt_target in dt_targets:
        result = integrate_planar_periodic(
            N_poly=N_poly,
            n_quad=n_quad,
            divisions=divisions,
            final_time=final_time,
            dt_target=dt_target,
        )
        error = result["q"] - reference["q"]
        l2_error = weighted_l2_error(error, result["J"], result["engine"].w_s)
        linf_error = float(np.max(np.abs(error)))
        rows.append(
            {
                "dt": result["dt"],
                "steps": result["steps"],
                "L2_error": float(l2_error),
                "Linf_error": linf_error,
                "order_L2": observed_order(previous_l2, l2_error, previous_dt, result["dt"]),
                "order_Linf": observed_order(
                    previous_linf,
                    linf_error,
                    previous_dt,
                    result["dt"],
                ),
            }
        )
        previous_l2 = l2_error
        previous_linf = linf_error
        previous_dt = result["dt"]

    return {
        "N_poly": N_poly,
        "divisions": divisions,
        "final_time": final_time,
        "reference_dt": reference["dt"],
        "reference_steps": reference["steps"],
        "rows": rows,
    }


def final_h_convergence():
    N_poly = 3
    n_quad = 4
    final_time = 1.0e-2
    cfl = 0.05
    n_divisions = [4, 8, 16, 32]
    speed = np.sqrt(2.0)

    rows = []
    previous_l2 = None
    previous_linf = None
    previous_h = None

    for divisions in n_divisions:
        h = 1.0 / divisions
        dt_target = compute_advection_dt(h=h, N=N_poly, speed=speed, cfl=cfl)
        start = time.time()
        result = integrate_planar_periodic(
            N_poly=N_poly,
            n_quad=n_quad,
            divisions=divisions,
            final_time=final_time,
            dt_target=dt_target,
        )
        q_ref = q_exact_sinxy(
            result["x_nodes"],
            result["y_nodes"],
            result["actual_time"],
            cx=result["cx"],
            cy=result["cy"],
        )
        error = result["q"] - q_ref
        l2_error = weighted_l2_error(error, result["J"], result["engine"].w_s)
        linf_error = float(np.max(np.abs(error)))
        rows.append(
            {
                "n_div": divisions,
                "h": h,
                "dt": result["dt"],
                "steps": result["steps"],
                "actual_time": result["actual_time"],
                "L2_error": float(l2_error),
                "Linf_error": linf_error,
                "order_L2": observed_order(previous_l2, l2_error, previous_h, h),
                "order_Linf": observed_order(previous_linf, linf_error, previous_h, h),
                "elapsed": time.time() - start,
            }
        )
        previous_l2 = l2_error
        previous_linf = linf_error
        previous_h = h

    return {
        "N_poly": N_poly,
        "final_time": final_time,
        "cfl": cfl,
        "rows": rows,
    }


def _order_string(value):
    if not np.isfinite(value):
        return "-"
    return f"{value:.4f}"


def test_final_planar_rk_temporal_convergence():
    result = final_rk_temporal_convergence()

    print("\n" + "=" * 120)
    print("Final planar RK temporal convergence")
    print("=" * 120)
    print(
        "periodic sin(2pi(x+y-2t)), fixed mesh, upwind flux, "
        f"N={result['N_poly']}, divisions={result['divisions']}, "
        f"T={result['final_time']:.3e}"
    )
    print(
        f"reference_dt={result['reference_dt']:.6e}, "
        f"reference_steps={result['reference_steps']}"
    )
    print(
        f"{'dt':>13s} {'steps':>8s} {'L2_error':>15s} "
        f"{'order':>9s} {'Linf_error':>15s} {'order':>9s}"
    )
    print("-" * 120)

    for row in result["rows"]:
        print(
            f"{row['dt']:13.6e} {row['steps']:8d} "
            f"{row['L2_error']:15.6e} {_order_string(row['order_L2']):>9s} "
            f"{row['Linf_error']:15.6e} {_order_string(row['order_Linf']):>9s}"
        )

    print("=" * 120)

    for row in result["rows"]:
        assert np.isfinite(row["L2_error"])
        assert np.isfinite(row["Linf_error"])
    assert result["rows"][-1]["order_L2"] > 3.0
    assert result["rows"][-1]["order_Linf"] > 3.0


def test_final_planar_h_convergence():
    result = final_h_convergence()

    print("\n" + "=" * 132)
    print("Final planar h-convergence")
    print("=" * 132)
    print(
        "periodic sin(2pi(x+y-2t)), upwind flux, "
        f"N={result['N_poly']}, T={result['final_time']:.3e}, "
        f"CFL={result['cfl']:.3e}"
    )
    print(
        f"{'n_div':>7s} {'h':>13s} {'dt':>13s} {'steps':>8s} "
        f"{'actual_time':>13s} {'L2_error':>15s} {'order':>9s} "
        f"{'Linf_error':>15s} {'order':>9s} {'time(s)':>9s}"
    )
    print("-" * 132)

    for row in result["rows"]:
        print(
            f"{row['n_div']:7d} {row['h']:13.6e} {row['dt']:13.6e} "
            f"{row['steps']:8d} {row['actual_time']:13.6e} "
            f"{row['L2_error']:15.6e} {_order_string(row['order_L2']):>9s} "
            f"{row['Linf_error']:15.6e} {_order_string(row['order_Linf']):>9s} "
            f"{row['elapsed']:9.2f}"
        )

    print("=" * 132)

    for row in result["rows"]:
        assert np.isfinite(row["L2_error"])
        assert np.isfinite(row["Linf_error"])
    assert result["rows"][-1]["L2_error"] < result["rows"][0]["L2_error"]
    assert result["rows"][-1]["Linf_error"] < result["rows"][0]["Linf_error"]


def run_all_tests():
    test_final_planar_rk_temporal_convergence()
    test_final_planar_h_convergence()
    print("test_final_planar_convergence.py passed")


if __name__ == "__main__":
    run_all_tests()
