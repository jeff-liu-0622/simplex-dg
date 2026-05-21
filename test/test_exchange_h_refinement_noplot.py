import numpy as np

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.operators_split import compute_split_rhs


def get_fmask(engine):
    if hasattr(engine, "fmask"):
        return engine.fmask

    return [
        np.arange(s.start, s.stop, dtype=int)
        for s in engine.edge_slices
    ]


def build_pure_geometric_maps(x_nodes, y_nodes, fmask):
    """
    Build vmapM/vmapP by geometric matching.
    Boundary faces are mapped to themselves.
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


def q_exact(x, y, t, cx=1.0, cy=1.0):
    return np.sin(2.0 * np.pi * (y - cy * t))


def qt_exact(x, y, t, cx=1.0, cy=1.0):
    return -2.0 * np.pi * cy * np.cos(2.0 * np.pi * (y - cy * t))


def weighted_l2_error(error_field, J, weights):
    numerator = np.sum(J[:, None] * weights[None, :] * error_field**2)
    denominator = np.sum(J[:, None] * weights[None, :])
    return np.sqrt(numerator / denominator)


def run_h_refinement_case(nx, ny):
    N, n = 4, 4
    cx, cy = 1.0, 1.0
    t0 = 0.0

    engine = build_local_operators(N, n, rule="table1")
    fmask = get_fmask(engine)

    VX, VY, EToV = create_square_mesh(nx, ny)
    EToV_x = VX[EToV]
    EToV_y = VY[EToV]

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)
    nx_f, ny_f, edge_lengths, _ = compute_face_metrics(EToV_x, EToV_y)

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

    q_current = q_exact(x_nodes, y_nodes, t0, cx=cx, cy=cy)
    qt_target = qt_exact(x_nodes, y_nodes, t0, cx=cx, cy=cy)

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
        "nx": nx_f,
        "ny": ny_f,
        "edge_lengths": edge_lengths,
        "vmapM": vmapM,
        "vmapP": vmapP,
        "cx": cx,
        "cy": cy,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "q_exact": q_exact,
        "lift_mode": "physical",
        "tau": 0.0,
    }

    rhs = compute_split_rhs(q_current, t0, **kwargs)
    error = rhs - qt_target

    l2_error = weighted_l2_error(error, J, engine.w_s)
    max_error = np.max(np.abs(error))

    return l2_error, max_error


def convergence_rate(previous_error, current_error):
    if previous_error is None:
        return None

    return np.log(previous_error / current_error) / np.log(2.0)


def run_all_tests():
    levels = [2, 4, 8, 16]

    print("\n" + "=" * 92)
    print("no-plot h-refinement diagnostic | smooth MMS | compute_split_rhs | tau=0")
    print("=" * 92)
    print(
        f"{'NX':>6s} {'K':>8s} {'h':>12s} "
        f"{'L2_error':>16s} {'L2_rate':>10s} "
        f"{'max_error':>16s} {'max_rate':>10s}"
    )
    print("-" * 92)

    results = []
    previous_l2 = None
    previous_max = None

    for nx in levels:
        l2_error, max_error = run_h_refinement_case(nx, nx)
        l2_rate = convergence_rate(previous_l2, l2_error)
        max_rate = convergence_rate(previous_max, max_error)

        print(
            f"{nx:6d} {2 * nx * nx:8d} {1.0 / nx:12.6e} "
            f"{l2_error:16.8e} "
            f"{'---' if l2_rate is None else f'{l2_rate:.4f}':>10s} "
            f"{max_error:16.8e} "
            f"{'---' if max_rate is None else f'{max_rate:.4f}':>10s}"
        )

        results.append((nx, l2_error, max_error, l2_rate, max_rate))
        previous_l2 = l2_error
        previous_max = max_error

    print("-" * 92)

    if len(results) >= 3:
        last_l2_rate = results[-1][3]
        last_max_rate = results[-1][4]
        assert last_l2_rate is not None and last_l2_rate > 3.0
        assert last_max_rate is not None and last_max_rate > 3.0

    print("no-plot h-refinement diagnostic passed")


if __name__ == "__main__":
    run_all_tests()
