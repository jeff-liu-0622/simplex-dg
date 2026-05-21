import numpy as np

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.operators_split import compute_split_rhs
from core.time_integration import lsrk54_step


def get_fmask(engine):
    """
    Get face-node masks from the ReferenceElement.

    Prefer engine.fmask if operators.py provides it.
    Otherwise build from edge_slices.
    """
    if hasattr(engine, "fmask"):
        return engine.fmask

    return [
        np.arange(s.start, s.stop, dtype=int)
        for s in engine.edge_slices
    ]


def build_pure_geometric_maps(x_nodes, y_nodes, fmask):
    """
    Build vmapM and vmapP by geometric matching.

    Boundary faces are mapped to themselves initially.
    Periodic pairing is applied later by apply_periodic_boundary().
    """
    K, Np = x_nodes.shape
    Nfaces = 3
    Nfp = len(fmask[0])

    vmapM = np.zeros((K, Nfaces, Nfp), dtype=int)
    vmapP = np.zeros((K, Nfaces, Nfp), dtype=int)

    x_flat = x_nodes.reshape(-1)
    y_flat = y_nodes.reshape(-1)

    for k in range(K):
        for f in range(Nfaces):
            vmapM[k, f, :] = k * Np + fmask[f]

    face_centers = {}

    for k in range(K):
        for f in range(Nfaces):
            m_idx = vmapM[k, f, :]
            cx = np.mean(x_flat[m_idx])
            cy = np.mean(y_flat[m_idx])
            face_centers[(k, f)] = (round(cx, 10), round(cy, 10))

    center_to_face = {}

    for key, coords in face_centers.items():
        center_to_face.setdefault(coords, []).append(key)

    for k1 in range(K):
        for f1 in range(Nfaces):
            coords = face_centers[(k1, f1)]
            neighbors = [nf for nf in center_to_face[coords] if nf != (k1, f1)]

            if not neighbors:
                vmapP[k1, f1, :] = vmapM[k1, f1, :]
                continue

            k2, f2 = neighbors[0]

            nodes1 = vmapM[k1, f1, :]
            nodes2 = vmapM[k2, f2, :]

            for i, n1 in enumerate(nodes1):
                dist = (
                    (x_flat[nodes2] - x_flat[n1]) ** 2
                    + (y_flat[nodes2] - y_flat[n1]) ** 2
                )
                vmapP[k1, f1, i] = nodes2[np.argmin(dist)]

    return vmapM, vmapP


def apply_periodic_boundary(vmapM, vmapP, x_nodes, y_nodes):
    """
    Pair physical boundary faces periodically.

    This assumes a rectangular domain.
    """
    x_flat = x_nodes.reshape(-1)
    y_flat = y_nodes.reshape(-1)

    K, Nfaces, Nfp = vmapM.shape

    TOL = 1e-8

    min_x, max_x = np.min(x_flat), np.max(x_flat)
    min_y, max_y = np.min(y_flat), np.max(y_flat)

    bnd_faces = []

    for k in range(K):
        for f in range(Nfaces):
            if np.array_equal(vmapM[k, f, :], vmapP[k, f, :]):
                bnd_faces.append((k, f))

    for k1, f1 in bnd_faces:
        m1 = vmapM[k1, f1, :]

        cx1 = np.mean(x_flat[m1])
        cy1 = np.mean(y_flat[m1])

        target_cx = cx1
        target_cy = cy1

        shift_x = 0.0
        shift_y = 0.0

        if cx1 < min_x + TOL:
            target_cx = max_x
            shift_x = max_x - min_x
        elif cx1 > max_x - TOL:
            target_cx = min_x
            shift_x = min_x - max_x

        if cy1 < min_y + TOL:
            target_cy = max_y
            shift_y = max_y - min_y
        elif cy1 > max_y - TOL:
            target_cy = min_y
            shift_y = min_y - max_y

        for k2, f2 in bnd_faces:
            if k1 == k2 and f1 == f2:
                continue

            m2 = vmapM[k2, f2, :]

            cx2 = np.mean(x_flat[m2])
            cy2 = np.mean(y_flat[m2])

            if abs(cx2 - target_cx) < TOL and abs(cy2 - target_cy) < TOL:
                for i, n1 in enumerate(m1):
                    dist = (
                        (x_flat[m2] - (x_flat[n1] + shift_x)) ** 2
                        + (y_flat[m2] - (y_flat[n1] + shift_y)) ** 2
                    )
                    vmapP[k1, f1, i] = m2[np.argmin(dist)]

                break

    return vmapP, min_x, max_x, min_y, max_y


def run_temporal_order_test():
    print("\n" + "=" * 72)
    print("啟動 SDG + LSRK54 temporal convergence test: periodic sin(x)")
    print("=" * 72)

    # ------------------------------------------------------------
    # Spatial setup.
    #
    # We compare each dt solution to a very small-dt reference solution
    # on the same spatial grid. This isolates time error.
    # ------------------------------------------------------------
    N, n = 4, 4
    NX, NY = 4, 4

    cx, cy = 1.0, 0.0
    T_final = 0.5

    engine = build_local_operators(N, n, rule="table1")
    fmask = get_fmask(engine)

    VX, VY, EToV = create_square_mesh(NX, NY)

    EToV_x = VX[EToV]
    EToV_y = VY[EToV]

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)

    # New interface:
    #   nx, ny, edge_lengths, sJ
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

    vmapM, vmapP = build_pure_geometric_maps(x_nodes, y_nodes, fmask)
    vmapP, min_x, max_x, min_y, max_y = apply_periodic_boundary(
        vmapM,
        vmapP,
        x_nodes,
        y_nodes,
    )

    domain_width = max_x - min_x

    def q_exact_sinx(x, y, t):
        return np.sin(2.0 * np.pi * (x - cx * t) / domain_width)

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

        # Periodic problem: qP is supplied by vmapP.
        # No q_exact boundary overwrite is needed.
        "lift_mode": "physical",

        # upwind flux
        "tau": 0.0,
    }

    def run_simulation(dt):
        q = q_exact_sinx(x_nodes, y_nodes, 0.0)
        res = np.zeros_like(q)

        t = 0.0
        steps = int(round(T_final / dt))
        dt_actual = T_final / steps

        for _ in range(steps):
            q, res = lsrk54_step(
                q,
                res,
                t,
                dt_actual,
                compute_split_rhs,
                **kwargs,
            )
            t += dt_actual

        return q

    # ------------------------------------------------------------
    # Reference solution.
    #
    # Choose small dt but not ridiculously tiny.
    # If dt_ref is too small, runtime gets high.
    # ------------------------------------------------------------
    dt_ref = 2.5e-4
    print(f"正在計算小時間步參考解 dt_ref = {dt_ref:.3e} ...")
    q_ref = run_simulation(dt_ref)
    print("參考解完成\n")

    dt_tests = [
        0.04,
        0.02,
        0.01,
        0.005,
    ]

    errors = []

    print("-" * 78)
    print(
        f"{'dt':>12s} | "
        f"{'steps':>8s} | "
        f"{'max error vs ref':>20s} | "
        f"{'rate':>10s}"
    )
    print("-" * 78)

    for i, dt in enumerate(dt_tests):
        q_num = run_simulation(dt)

        err = np.max(np.abs(q_num - q_ref))
        errors.append(err)

        steps = int(round(T_final / dt))

        if i == 0:
            rate_str = "---"
        else:
            rate = np.log(errors[i - 1] / errors[i]) / np.log(dt_tests[i - 1] / dt_tests[i])
            rate_str = f"{rate:.4f}"

        print(
            f"{dt:12.6e} | "
            f"{steps:8d} | "
            f"{err:20.6e} | "
            f"{rate_str:>10s}"
        )

    print("-" * 78)

    last_rate = np.log(errors[-2] / errors[-1]) / np.log(dt_tests[-2] / dt_tests[-1])

    assert last_rate > 3.5, (
        f"Temporal convergence rate too low: last_rate={last_rate:.4f}"
    )

    print("✅ SDG + LSRK54 temporal convergence test passed")
    print("=" * 72)


if __name__ == "__main__":
    run_temporal_order_test()