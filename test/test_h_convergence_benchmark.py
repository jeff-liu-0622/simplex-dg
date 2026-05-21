import time
import numpy as np

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.operators_split import compute_split_rhs
from core.geometry.connectivity import build_connectivity, apply_periodic_conditions, build_maps
from core.time_integration import lsrk54_step
from core.cfl import compute_advection_dt

def get_fmask(engine):
    """
    Get face-node masks from ReferenceElement.
    """
    if hasattr(engine, "fmask"):
        return engine.fmask

    return [
        np.arange(s.start, s.stop, dtype=int)
        for s in engine.edge_slices
    ]


def weighted_l2_error(error_field, J, w_s):
    """
    Compute physical L2 error.

    error_field shape:
        (K, Np)

    J shape:
        (K,)

    w_s:
        reference volume quadrature weights, normalized to sum 1.

    Physical integral on element k:

        ∫_D e^2 dx dy
        ≈ J_k * |T_ref| * sum_i w_i e_i^2

    Since |T_ref| = 2, but it appears in both numerator and denominator
    if we compute relative domain average, it cancels.
    """
    numerator = np.sum(J[:, None] * w_s[None, :] * error_field**2)
    denominator = np.sum(J[:, None] * w_s[None, :])

    return np.sqrt(numerator / denominator)


def run_h_convergence_benchmark():
    print("\n" + "=" * 110)
    print("啟動 SDG solution h-convergence benchmark: periodic sin(x)")
    print("PDE: q_t + q_x = 0, periodic BC, final time T=1")
    print("=" * 110)

    print(
        f"{'n':>6s} "
        f"{'K':>9s} "
        f"{'h':>13s} "
        f"{'dt':>13s} "
        f"{'steps':>8s} "
        f"{'L2_error':>14s} "
        f"{'rate':>8s} "
        f"{'Linf_error':>14s} "
        f"{'rate':>8s} "
        f"{'time(s)':>8s}"
    )
    print("-" * 110)

    # ------------------------------------------------------------
    # Polynomial degree and advection velocity
    # ------------------------------------------------------------
    N_poly = 4
    n_quad = 4

    cx, cy = 1.0, 0.0
    T_final = 1.0

    engine = build_local_operators(N_poly, n_quad, rule="table1")
    fmask = get_fmask(engine)

    # Mesh levels
    n_levels = [1, 2, 4, 8, 16]

    prev_L2 = None
    prev_Linf = None

    for n in n_levels:
        start_time = time.time()

        NX = NY = n
        K_total = 2 * NX * NY
        h_size = 1.0 / n

        # Conservative time step.
        # You can tune this later using CFL.
        h_size = 1.0 / n
        dt = compute_advection_dt(h_size, N_poly, cx, cy, cfl=0.05)

        steps = int(np.ceil(T_final / dt))
        dt = T_final / steps

        # --------------------------------------------------------
        # Mesh and metrics
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Connectivity and periodic maps
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Exact solution
        # --------------------------------------------------------
        domain_width = np.max(x_nodes) - np.min(x_nodes)

        def q_exact_sinx(x, t):
            return np.sin(2.0 * np.pi * (x - cx * t) / domain_width)

        # --------------------------------------------------------
        # RHS kwargs
        # --------------------------------------------------------
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

            # Periodic problem: boundary pairing is already encoded in vmapP.
            "lift_mode": "physical",

            # Upwind flux.
            "tau": 0.0,
        }

        # --------------------------------------------------------
        # Initial condition and time stepping
        # --------------------------------------------------------
        q = q_exact_sinx(x_nodes, 0.0)
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

        # --------------------------------------------------------
        # Error
        # --------------------------------------------------------
        q_ref = q_exact_sinx(x_nodes, T_final)

        error_field = q - q_ref

        L2_error = weighted_l2_error(error_field, J, engine.w_s)
        Linf_error = np.max(np.abs(error_field))

        elapsed_time = time.time() - start_time

        rate_L2_str = "-"
        rate_Linf_str = "-"

        if prev_L2 is not None:
            rate_L2 = np.log2(prev_L2 / L2_error)
            rate_Linf = np.log2(prev_Linf / Linf_error)

            rate_L2_str = f"{rate_L2:7.3f}"
            rate_Linf_str = f"{rate_Linf:7.3f}"

        print(
            f"{n:6d} "
            f"{K_total:9d} "
            f"{h_size:13.4e} "
            f"{dt:13.4e} "
            f"{steps:8d} "
            f"{L2_error:14.6e} "
            f"{rate_L2_str:>8s} "
            f"{Linf_error:14.6e} "
            f"{rate_Linf_str:>8s} "
            f"{elapsed_time:8.2f}"
        )

        prev_L2 = L2_error
        prev_Linf = Linf_error

    print("-" * 110)
    print("✅ SDG solution h-convergence benchmark finished")
    print("=" * 110)


if __name__ == "__main__":
    run_h_convergence_benchmark()