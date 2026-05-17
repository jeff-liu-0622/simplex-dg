import time
import numpy as np

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.geometry.connectivity import (
    build_connectivity,
    apply_periodic_conditions,
    build_maps,
)
from core.operators_split import compute_split_rhs
from core.time_integration import lsrk54_step
from core.cfl import compute_advection_dt


def weighted_mass(q, J, w_s):
    return np.sum(J[:, None] * w_s[None, :] * q)


def weighted_energy(q, J, w_s):
    return np.sum(J[:, None] * w_s[None, :] * q**2)


def weighted_l2_error(error, J, w_s):
    numerator = np.sum(J[:, None] * w_s[None, :] * error**2)
    denominator = np.sum(J[:, None] * w_s[None, :])
    return np.sqrt(numerator / denominator)


def q_exact_sinxy(x, y, t, cx=1.0, cy=1.0):
    return np.sin(2.0 * np.pi * (x + y - (cx + cy) * t))


def build_periodic_planar_case(N_poly=4, n_quad=4, divisions=6, cx=1.0, cy=1.0):
    engine = build_local_operators(N_poly, n_quad, rule="table1")

    VX, VY, EToV = create_square_mesh(divisions, divisions)

    EToV_x = VX[EToV]
    EToV_y = VY[EToV]

    rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)
    nx, ny, edge_lengths, _ = compute_face_metrics(EToV_x, EToV_y)

    r, s = engine.r, engine.s

    x_nodes = 0.5 * (
        -(r + s) * EToV_x[:, 0:1]
        + (1.0 + r) * EToV_x[:, 1:2]
        + (1.0 + s) * EToV_x[:, 2:3]
    )

    y_nodes = 0.5 * (
        -(r + s) * EToV_y[:, 0:1]
        + (1.0 + r) * EToV_y[:, 1:2]
        + (1.0 + s) * EToV_y[:, 2:3]
    )

    EToE, EToF = build_connectivity(EToV)
    EToE, EToF = apply_periodic_conditions(
        EToE,
        EToF,
        x_nodes,
        y_nodes,
        engine,
    )

    vmapM, vmapP = build_maps(
        engine,
        EToV,
        EToE,
        EToF,
        x_nodes,
        y_nodes,
    )

    base_kwargs = {
        "engine": engine,
        "rx": rx,
        "sx": sx,
        "ry": ry,
        "sy": sy,
        "J": J,
        "nx": nx,
        "ny": ny,
        "edge_lengths": edge_lengths,
        "vmapM": vmapM,
        "vmapP": vmapP,
        "cx": cx,
        "cy": cy,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "lift_mode": "physical",
    }

    return engine, x_nodes, y_nodes, J, base_kwargs


def run_flux_case(flux_type, alpha_lf=1.0, T_final=1.0):
    N_poly = 4
    n_quad = 4
    divisions = 6
    cx, cy = 1.0, 1.0

    engine, x_nodes, y_nodes, J, kwargs = build_periodic_planar_case(
        N_poly=N_poly,
        n_quad=n_quad,
        divisions=divisions,
        cx=cx,
        cy=cy,
    )

    h = 1.0 / divisions
    speed = np.sqrt(cx**2 + cy**2)
    dt_nominal = compute_advection_dt(h, N_poly, speed, cfl=0.25)
    steps = int(np.ceil(T_final / dt_nominal))
    dt = T_final / steps

    kwargs = {
        **kwargs,
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
    }

    q0 = q_exact_sinxy(x_nodes, y_nodes, 0.0, cx=cx, cy=cy)
    q = q0.copy()
    res = np.zeros_like(q)

    mass_initial = weighted_mass(q, J, engine.w_s)
    energy_initial = weighted_energy(q, J, engine.w_s)

    t = 0.0
    start = time.time()

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

    elapsed = time.time() - start

    q_ref = q_exact_sinxy(x_nodes, y_nodes, T_final, cx=cx, cy=cy)
    error = q - q_ref

    return {
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "q": q,
        "dt": dt,
        "steps": steps,
        "L2_final": weighted_l2_error(error, J, engine.w_s),
        "Linf_final": np.max(np.abs(error)),
        "mass_initial": mass_initial,
        "mass_final": weighted_mass(q, J, engine.w_s),
        "energy_initial": energy_initial,
        "energy_final": weighted_energy(q, J, engine.w_s),
        "elapsed": elapsed,
    }


def print_result(result):
    mass_error = result["mass_final"] - result["mass_initial"]
    energy_change = result["energy_final"] - result["energy_initial"]

    print(
        f"{result['flux_type']:<10s} "
        f"{result['alpha_lf']:8.3f} "
        f"{result['L2_final']:14.6e} "
        f"{result['Linf_final']:14.6e} "
        f"{result['mass_initial']:14.6e} "
        f"{result['mass_final']:14.6e} "
        f"{mass_error:14.6e} "
        f"{result['energy_initial']:14.6e} "
        f"{result['energy_final']:14.6e} "
        f"{energy_change:14.6e} "
        f"{result['steps']:8d} "
        f"{result['dt']:12.4e} "
        f"{result['elapsed']:9.2f}"
    )


def run_flux_comparison_sinxy():
    print("\n" + "=" * 176)
    print("Planar skew-symmetric DG flux comparison | sin2pi(x+y-(cx+cy)t) | periodic")
    print("=" * 176)
    print(
        f"{'flux':<10s} "
        f"{'alpha':>8s} "
        f"{'L2_final':>14s} "
        f"{'Linf_final':>14s} "
        f"{'mass_initial':>14s} "
        f"{'mass_final':>14s} "
        f"{'mass_error':>14s} "
        f"{'energy_initial':>14s} "
        f"{'energy_final':>14s} "
        f"{'energy_change':>14s} "
        f"{'steps':>8s} "
        f"{'dt':>12s} "
        f"{'time(s)':>9s}"
    )
    print("-" * 176)

    cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]

    results = []
    for flux_type, alpha_lf in cases:
        result = run_flux_case(flux_type=flux_type, alpha_lf=alpha_lf)
        results.append(result)
        print_result(result)

    print("-" * 176)

    by_key = {
        (result["flux_type"], result["alpha_lf"]): result
        for result in results
    }
    upwind = by_key[("upwind", 1.0)]
    lf_alpha1 = by_key[("lf", 1.0)]

    max_abs_diff = np.max(np.abs(upwind["q"] - lf_alpha1["q"]))
    print(
        "max_abs_difference_between_upwind_and_lf_alpha1 = "
        f"{max_abs_diff:.6e}"
    )

    assert max_abs_diff < 1.0e-13, (
        "LF with alpha_lf=1 should match upwind for scalar linear advection; "
        f"got max difference {max_abs_diff:.6e}."
    )

    print("=" * 176)


if __name__ == "__main__":
    run_flux_comparison_sinxy()
