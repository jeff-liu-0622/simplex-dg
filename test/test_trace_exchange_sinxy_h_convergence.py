import time
import numpy as np

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.geometry.connectivity import build_connectivity, apply_periodic_conditions, build_maps
from core.operators_split import compute_split_rhs
from core.time_integration import lsrk54_step


def weighted_l2_error(error_field, J, w_s):
    numerator = np.sum(J[:, None] * w_s[None, :] * error_field**2)
    denominator = np.sum(J[:, None] * w_s[None, :])
    return np.sqrt(numerator / denominator)


def run_trace_exchange_sinxy_h_convergence():
    print("\n" + "=" * 110)
    print("[run] tau_role=surface numerical flux parameter; tau=0 is pure upwind")
    print("[run] test=sin2pi_xy | trace-exchange | periodic BC")
    print("=" * 110)

    print(
        f"{'n':>6s} "
        f"{'K':>9s} "
        f"{'h':>13s} "
        f"{'status':>8s} "
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
    # Method setup
    # ------------------------------------------------------------
    N_poly = 4
    n_quad = 4

    cx, cy = 1.0, 1.0
    T_final = 3.0 * np.pi

    # 為了接近你截圖的 dt：
    # n=1 時 h=1, N=4:
    # dt = 0.5 * h / N^2 = 0.03125
    cfl_like = 0.5

    tau = 0.0

    engine = build_local_operators(N_poly, n_quad, rule="table1")

    n_levels = [1, 2, 4, 8, 16]

    prev_L2 = None
    prev_Linf = None

    for n in n_levels:
        start_time = time.time()

        NX = NY = n
        K_total = 2 * NX * NY
        h_size = 1.0 / n

        dt_nominal = cfl_like * h_size / (N_poly**2)

        # 用 ceil，保證實際 dt 不大於 nominal dt
        steps = int(np.ceil(T_final / dt_nominal))
        dt = T_final / steps

        # --------------------------------------------------------
        # Mesh and geometry
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
        # Connectivity and periodic trace exchange
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
        #
        # q_t + q_x + q_y = 0
        #
        # q(x,y,t) = sin(2π(x+y-2t))
        # --------------------------------------------------------
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

            # Periodic trace exchange:
            # qP is supplied by vmapP.
            "lift_mode": "physical",

            # tau=0 means full upwind.
            "tau": tau,
        }

        # --------------------------------------------------------
        # Time integration
        # --------------------------------------------------------
        q = q_exact(x_nodes, y_nodes, 0.0)
        res = np.zeros_like(q)

        t = 0.0
        status = "ok"

        try:
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

                if not np.all(np.isfinite(q)):
                    status = "nan"
                    break

        except Exception:
            status = "fail"
            raise

        # --------------------------------------------------------
        # Error
        # --------------------------------------------------------
        q_ref = q_exact(x_nodes, y_nodes, T_final)
        error_field = q - q_ref

        L2_error = weighted_l2_error(error_field, J, engine.w_s)
        Linf_error = np.max(np.abs(error_field))

        elapsed = time.time() - start_time

        if prev_L2 is None:
            rate_L2_str = "-"
            rate_Linf_str = "-"
        else:
            rate_L2 = np.log2(prev_L2 / L2_error)
            rate_Linf = np.log2(prev_Linf / Linf_error)

            rate_L2_str = f"{rate_L2:7.3f}"
            rate_Linf_str = f"{rate_Linf:7.3f}"

        print(
            f"{n:6d} "
            f"{K_total:9d} "
            f"{h_size:13.4e} "
            f"{status:>8s} "
            f"{dt:13.4e} "
            f"{steps:8d} "
            f"{L2_error:14.6e} "
            f"{rate_L2_str:>8s} "
            f"{Linf_error:14.6e} "
            f"{rate_Linf_str:>8s} "
            f"{elapsed:8.2f}"
        )

        prev_L2 = L2_error
        prev_Linf = Linf_error

    print("-" * 110)
    print("✅ sin2pi_xy trace-exchange h-convergence test finished")
    print("=" * 110)


if __name__ == "__main__":
    run_trace_exchange_sinxy_h_convergence()