import time
import numpy as np
import matplotlib.pyplot as plt

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.geometry.connectivity import build_connectivity, apply_periodic_conditions, build_maps
from core.operators_split import compute_split_rhs
from core.time_integration import lsrk54_step


def weighted_l2_error(error_field, J, w_s):
    """
    Physical L2 error averaged over the domain.

    error_field shape = (K, Np)
    J shape = (K,)
    w_s shape = (Np,)
    """
    numerator = np.sum(J[:, None] * w_s[None, :] * error_field**2)
    denominator = np.sum(J[:, None] * w_s[None, :])
    return np.sqrt(numerator / denominator)


def compute_order(errors, hs):
    """
    Compute pairwise convergence orders.
    """
    orders = []

    for i in range(1, len(errors)):
        order = np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i])
        orders.append(order)

    return orders


def run_single_mesh(n, engine, N_poly, cx, cy, T_final, cfl_like, sample_count):
    """
    Run one mesh level and record error evolution.
    """
    NX = NY = n
    h_size = 1.0 / n

    # Match the previous benchmark style:
    # dt ~= 0.5 * h / N^2.
    dt_nominal = cfl_like * h_size / (N_poly**2)

    steps = int(np.ceil(T_final / dt_nominal))
    dt = T_final / steps

    VX, VY, EToV = create_square_mesh(NX, NY)

    EToV_x = VX[EToV]
    EToV_y = VY[EToV]

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)
    nx, ny, edge_lengths, sJ = compute_face_metrics(EToV_x, EToV_y)

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

    def q_exact(x, y, t):
        return np.sin(2.0 * np.pi * (x + y - (cx + cy) * t))

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
        "tau": 0.0,
    }

    q = q_exact(x_nodes, y_nodes, 0.0)
    res = np.zeros_like(q)

    t = 0.0

    sample_every = max(1, steps // sample_count)

    times = []
    l2_errors = []
    linf_errors = []

    def record_error(current_t):
        q_ref = q_exact(x_nodes, y_nodes, current_t)
        error_field = q - q_ref

        l2 = weighted_l2_error(error_field, J, engine.w_s)
        linf = np.max(np.abs(error_field))

        times.append(current_t)
        l2_errors.append(l2)
        linf_errors.append(linf)

    record_error(t)

    start = time.time()

    for step in range(1, steps + 1):
        q, res = lsrk54_step(
            q,
            res,
            t,
            dt,
            compute_split_rhs,
            **kwargs,
        )

        t += dt

        if step % sample_every == 0 or step == steps:
            record_error(t)

    elapsed = time.time() - start

    return {
        "n": n,
        "K": EToV.shape[0],
        "h": h_size,
        "dt": dt,
        "steps": steps,
        "times": np.array(times),
        "l2_errors": np.array(l2_errors),
        "linf_errors": np.array(linf_errors),
        "elapsed": elapsed,
    }


def run_error_evolution_sinxy():
    print("\n" + "=" * 96)
    print("LSRK L2 error vs time | sin2pi_xy | trace-exchange | tau=0 | periodic")
    print("=" * 96)

    N_poly = 4
    n_quad = 4

    cx, cy = 1.0, 1.0
    T_final = 5.0

    cfl_like = 0.5
    n_levels = [4, 8, 16]

    # Number of recorded samples per mesh.
    # This does not affect dt; it only affects how many points are plotted.
    sample_count = 250

    engine = build_local_operators(N_poly, n_quad, rule="table1")

    results = []

    print(
        f"{'n':>6s} "
        f"{'K':>8s} "
        f"{'h':>12s} "
        f"{'dt':>12s} "
        f"{'steps':>8s} "
        f"{'L2_final':>14s} "
        f"{'Linf_final':>14s} "
        f"{'time(s)':>9s}"
    )
    print("-" * 96)

    for n in n_levels:
        result = run_single_mesh(
            n=n,
            engine=engine,
            N_poly=N_poly,
            cx=cx,
            cy=cy,
            T_final=T_final,
            cfl_like=cfl_like,
            sample_count=sample_count,
        )

        results.append(result)

        print(
            f"{result['n']:6d} "
            f"{result['K']:8d} "
            f"{result['h']:12.4e} "
            f"{result['dt']:12.4e} "
            f"{result['steps']:8d} "
            f"{result['l2_errors'][-1]:14.6e} "
            f"{result['linf_errors'][-1]:14.6e} "
            f"{result['elapsed']:9.2f}"
        )

    print("-" * 96)

    hs = np.array([r["h"] for r in results])

    final_l2 = np.array([r["l2_errors"][-1] for r in results])
    final_linf = np.array([r["linf_errors"][-1] for r in results])

    avg_l2 = np.array([np.mean(r["l2_errors"]) for r in results])
    avg_linf = np.array([np.mean(r["linf_errors"]) for r in results])

    final_l2_orders = compute_order(final_l2, hs)
    final_linf_orders = compute_order(final_linf, hs)

    avg_l2_orders = compute_order(avg_l2, hs)
    avg_linf_orders = compute_order(avg_linf, hs)

    print("Final-time spatial orders:")
    for i in range(1, len(results)):
        print(
            f"  n={results[i-1]['n']} -> n={results[i]['n']}: "
            f"L2 order = {final_l2_orders[i-1]:.4f}, "
            f"Linf order = {final_linf_orders[i-1]:.4f}"
        )

    print("\nAverage-in-time spatial orders:")
    for i in range(1, len(results)):
        print(
            f"  n={results[i-1]['n']} -> n={results[i]['n']}: "
            f"L2 order = {avg_l2_orders[i-1]:.4f}, "
            f"Linf order = {avg_linf_orders[i-1]:.4f}"
        )

    if final_l2_orders:
        print(
            "\nSummary:"
            f"\n  p_L2(last)  = {final_l2_orders[-1]:.4f}"
            f"\n  p_L2(avg)   = {avg_l2_orders[-1]:.4f}"
            f"\n  p_Linf(last)= {final_linf_orders[-1]:.4f}"
            f"\n  p_Linf(avg) = {avg_linf_orders[-1]:.4f}"
        )

    # ------------------------------------------------------------
    # Plot L2 error evolution
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 7))

    for result in results:
        plt.semilogy(
            result["times"],
            result["l2_errors"],
            linewidth=2,
            label=f"L2 error (n={result['n']})",
        )

    plt.xlabel("time")
    plt.ylabel("L2 error (log scale)")
    plt.title(
        "LSRK L2 error vs time | "
        f"tf={T_final}, n={','.join(str(n) for n in n_levels)}, "
        "tau=0, trace=exchange"
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.8)
    plt.legend()

    text = (
        "final-time spatial order\n"
        f"p_L2(last)={final_l2_orders[-1]:.3f}, "
        f"p_L2(avg)={avg_l2_orders[-1]:.3f}\n"
        f"p_Linf(last)={final_linf_orders[-1]:.3f}, "
        f"p_Linf(avg)={avg_linf_orders[-1]:.3f}"
    )

    plt.text(
        0.03,
        0.05,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.show()

    print("\n✅ error evolution sinxy benchmark finished")
    print("=" * 96)


if __name__ == "__main__":
    run_error_evolution_sinxy()