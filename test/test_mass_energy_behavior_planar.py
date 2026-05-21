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


def initial_condition(x, y):
    return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def integral(field, J, weights):
    return np.sum(J[:, None] * weights[None, :] * field)


def mass(q, J, weights):
    return integral(q, J, weights)


def energy(q, J, weights):
    return integral(q**2, J, weights)


def build_periodic_planar_case(N_poly=4, n_quad=4, NX=8, NY=8, cx=1.0, cy=1.0):
    engine = build_local_operators(N_poly, n_quad, rule="table1")

    VX, VY, EToV = create_square_mesh(NX, NY)
    EToV_x = VX[EToV]
    EToV_y = VY[EToV]

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)
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
    EToE, EToF = apply_periodic_conditions(EToE, EToF, x_nodes, y_nodes, engine)
    vmapM, vmapP = build_maps(engine, EToV, EToE, EToF, x_nodes, y_nodes)

    kwargs = {
        "engine": engine,
        "xr": xr,
        "xs": xs,
        "yr": yr,
        "ys": ys,
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

    return engine, kwargs, x_nodes, y_nodes


def run_simulation(tau, T_final=0.5, dt=0.0025):
    engine, kwargs, x_nodes, y_nodes = build_periodic_planar_case()
    kwargs["tau"] = tau

    q = initial_condition(x_nodes, y_nodes)
    res = np.zeros_like(q)

    steps = int(round(T_final / dt))
    dt = T_final / steps

    J = kwargs["J"]
    weights = engine.w_s

    history = []

    def record(step, t):
        current_mass = mass(q, J, weights)
        current_energy = energy(q, J, weights)
        history.append((step, t, current_mass, current_energy))

    record(0, 0.0)

    t = 0.0
    report_every = max(1, steps // 5)

    for step in range(1, steps + 1):
        q, res = lsrk54_step(q, res, t, dt, compute_split_rhs, **kwargs)
        t += dt

        if step % report_every == 0 or step == steps:
            record(step, t)

    return history


def print_history(name, tau, history):
    initial_mass = history[0][2]
    initial_energy = history[0][3]

    print(f"\n{name} flux | tau={tau:.1f}")
    print("-" * 96)
    print(
        f"{'step':>8s} {'t':>12s} {'mass':>18s} {'mass_delta':>18s} "
        f"{'energy':>18s} {'energy_delta':>18s}"
    )
    print("-" * 96)

    for step, t, current_mass, current_energy in history:
        print(
            f"{step:8d} {t:12.6e} "
            f"{current_mass:18.10e} {current_mass - initial_mass:18.10e} "
            f"{current_energy:18.10e} {current_energy - initial_energy:18.10e}"
        )

    final_mass = history[-1][2]
    final_energy = history[-1][3]
    return initial_mass, final_mass, initial_energy, final_energy


def run_all_tests():
    print("\n" + "=" * 96)
    print("mass/energy diagnostic | planar affine periodic mesh | LSRK54 | compute_split_rhs")
    print("=" * 96)

    central = run_simulation(tau=1.0)
    central_m0, central_mf, central_e0, central_ef = print_history(
        "central", 1.0, central
    )

    upwind = run_simulation(tau=0.0)
    upwind_m0, upwind_mf, upwind_e0, upwind_ef = print_history(
        "upwind", 0.0, upwind
    )

    central_mass_delta = abs(central_mf - central_m0)
    upwind_mass_delta = abs(upwind_mf - upwind_m0)
    central_energy_rel = abs(central_ef - central_e0) / central_e0
    upwind_energy_delta = upwind_ef - upwind_e0

    print("\nsummary")
    print("-" * 96)
    print(f"central mass delta      = {central_mass_delta:.10e}")
    print(f"central energy rel diff = {central_energy_rel:.10e}")
    print(f"upwind mass delta       = {upwind_mass_delta:.10e}")
    print(f"upwind energy delta     = {upwind_energy_delta:.10e}")
    print("-" * 96)

    assert central_mass_delta < 1.0e-11
    assert upwind_mass_delta < 1.0e-11
    assert central_energy_rel < 1.0e-3
    assert upwind_energy_delta <= 1.0e-12

    print("mass/energy diagnostic passed")


if __name__ == "__main__":
    run_all_tests()
