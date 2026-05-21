import numpy as np
import matplotlib.pyplot as plt

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
    """
    Smooth manufactured solution:

        q(x,y,t) = sin(2π(y - cy t))

    This depends only on y.
    """
    return np.sin(2.0 * np.pi * (y - cy * t))


def qt_exact(x, y, t, cx=1.0, cy=1.0):
    """
    For q = sin(2π(y - cy t)),

        q_t = -2π cy cos(2π(y - cy t))
    """
    return -2.0 * np.pi * cy * np.cos(2.0 * np.pi * (y - cy * t))


def run_h_refinement_test(nx, ny):
    print(f"正在測試網格密度: {nx}x{ny} ({nx * ny * 2} 個三角形)")

    N, n = 4, 4
    cx, cy = 1.0, 1.0
    t0 = 0.0

    engine = build_local_operators(N, n, rule="table1")
    fmask = get_fmask(engine)

    # ------------------------------------------------------------
    # 1. Mesh and metrics
    # ------------------------------------------------------------
    VX, VY, EToV = create_square_mesh(nx, ny)

    EToV_x = VX[EToV]
    EToV_y = VY[EToV]

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)
    nx_f, ny_f, edge_lengths, sJ = compute_face_metrics(EToV_x, EToV_y)

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

    # ------------------------------------------------------------
    # 2. Face exchange maps
    # ------------------------------------------------------------
    vmapM, vmapP = build_pure_geometric_maps(x_nodes, y_nodes, fmask)

    # ------------------------------------------------------------
    # 3. Exact solution and RHS
    # ------------------------------------------------------------
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

        # physical mode:
        # internal faces use qP from neighbor.
        # physical boundary inflow uses q_exact.
        "lift_mode": "physical",

        # upwind flux
        "tau": 0.0,
    }

    total_rhs = compute_split_rhs(q_current, t0, **kwargs)

    max_err = np.max(np.abs(total_rhs - qt_target))

    # ------------------------------------------------------------
    # 4. Exchange mismatch diagnostic
    # ------------------------------------------------------------
    q_flat = q_current.reshape(-1)
    x_flat = x_nodes.reshape(-1)
    y_flat = y_nodes.reshape(-1)

    max_interior_mismatch = 0.0

    for f in range(3):
        qP = q_flat[vmapP[:, f, :]]
        is_boundary = np.all(vmapM[:, f, :] == vmapP[:, f, :], axis=1)
        interior = ~is_boundary

        if np.any(interior):
            xM = x_flat[vmapM[interior, f, :]]
            yM = y_flat[vmapM[interior, f, :]]
            qP_exact = q_exact(xM, yM, t0, cx=cx, cy=cy)

            mismatch = np.max(np.abs(qP[interior, :] - qP_exact))
            max_interior_mismatch = max(max_interior_mismatch, mismatch)

    print(
        f"  Result -> Exchange mismatch: {max_interior_mismatch:.2e} "
        f"| RHS Error: {max_err:.4e}"
    )

    return max_err


def plot_convergence(densities, errors):
    h_values = np.array([1.0 / d for d in densities])
    errors = np.array(errors)

    plt.figure(figsize=(8, 6))

    plt.loglog(
        h_values,
        errors,
        "o-",
        linewidth=2,
        markersize=7,
        label="SDG RHS error",
    )

    # For N=4, use O(h^4) as conservative reference for RHS derivative error.
    ref4 = errors[0] * (h_values / h_values[0]) ** 4

    plt.loglog(
        h_values,
        ref4,
        "k--",
        linewidth=2,
        label=r"reference $O(h^4)$",
    )

    plt.xlabel("Mesh size h")
    plt.ylabel(r"Max error in $q_t$")
    plt.title("h-Refinement RHS Convergence Test")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 SDG h-refinement RHS convergence test")
    print("=" * 72)

    densities = [4, 8, 16, 32]
    errors = []

    print("-" * 72)

    for d in densities:
        err = run_h_refinement_test(d, d)
        errors.append(err)

    print("-" * 72)
    print("誤差收斂總覽:")

    for i, d in enumerate(densities):
        if i == 0:
            rate = "---"
        else:
            rate_val = np.log(errors[i - 1] / errors[i]) / np.log(2.0)
            rate = f"{rate_val:.4f}"

        print(
            f"  mesh {d:3d}x{d:<3d} -> error = {errors[i]:.6e}, rate = {rate}"
        )

    if len(errors) >= 3:
        last_rate = np.log(errors[-2] / errors[-1]) / np.log(2.0)
        assert last_rate > 3.0, (
            f"h-refinement rate too low: last_rate={last_rate:.4f}"
        )

    print("✅ SDG h-refinement RHS convergence test passed")

    plot_convergence(densities, errors)

    print("=" * 72)


if __name__ == "__main__":
    run_all_tests()