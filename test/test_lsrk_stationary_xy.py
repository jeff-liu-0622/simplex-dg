import numpy as np

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.time_integration import lsrk54_step
from core.operators_split import compute_split_rhs


def build_pure_geometric_maps(x_nodes, y_nodes, fmask):
    """
    Geometry-only map builder.

    vmapM[k,f,:] gives current element face nodes.
    vmapP[k,f,:] gives matched neighbor face nodes.
    Boundary faces map to themselves.
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


def q_exact_xy(x, y, t=0.0):
    """
    Stationary exact solution for velocity (cx,cy)=(1,1):

        q(x,y,t) = x - y

    because:

        q_t + q_x + q_y = 0 + 1 - 1 = 0.
    """
    return x - y


def run_stationary_xy_test():
    print("\n" + "=" * 60)
    print("啟動 SDG stationary linear solution test: q=x-y, velocity=(1,1)")
    print("=" * 60)

    N, n = 4, 4
    NX, NY = 4, 4

    cx, cy = 1.0, 1.0

    T_final = 0.5
    dt = 0.002

    # ------------------------------------------------------------
    # 1. Build reference element
    # ------------------------------------------------------------
    engine = build_local_operators(N, n, rule="table1")

    # Use compatibility fmask property if you added it to operators.py.
    # If not, build it from edge_slices here.
    if hasattr(engine, "fmask"):
        fmask = engine.fmask
    else:
        fmask = [
            np.arange(s.start, s.stop, dtype=int)
            for s in engine.edge_slices
        ]

    # ------------------------------------------------------------
    # 2. Build mesh and metrics
    # ------------------------------------------------------------
    VX, VY, EToV = create_square_mesh(NX, NY)

    EToV_coords_x = VX[EToV]
    EToV_coords_y = VY[EToV]

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_coords_x, EToV_coords_y)

    # New face_metrics interface:
    #   nx, ny, edge_lengths, sJ
    nx, ny, edge_lengths, sJ = compute_face_metrics(EToV_coords_x, EToV_coords_y)

    r, s = engine.r, engine.s

    x_nodes = 0.5 * (
        -(r + s) * EToV_coords_x[:, 0:1]
        + (1.0 + r) * EToV_coords_x[:, 1:2]
        + (1.0 + s) * EToV_coords_x[:, 2:3]
    )

    y_nodes = 0.5 * (
        -(r + s) * EToV_coords_y[:, 0:1]
        + (1.0 + r) * EToV_coords_y[:, 1:2]
        + (1.0 + s) * EToV_coords_y[:, 2:3]
    )

    vmapM, vmapP = build_pure_geometric_maps(x_nodes, y_nodes, fmask)

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
        "q_exact": q_exact_xy,

        # exact_trace: overwrite qP on every face by exact trace.
        # This is a diagnostic mode. It is good for checking that the
        # volume + lift machinery preserves the stationary linear state.
        "lift_mode": "physical",

        # tau=0: upwind
        "tau": 0.0,
    }

    # ------------------------------------------------------------
    # 3. Time integration
    # ------------------------------------------------------------
    t = 0.0
    q = q_exact_xy(x_nodes, y_nodes, t)
    res = np.zeros_like(q)

    steps = int(round(T_final / dt))
    dt = T_final / steps

    print("-" * 60)

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

        if step % 50 == 0 or step == steps:
            q_ex = q_exact_xy(x_nodes, y_nodes, t)
            err = np.max(np.abs(q - q_ex))
            print(f"[Step {step:4d}] t = {t:.4f} | max error = {err:.6e}")

    final_exact = q_exact_xy(x_nodes, y_nodes, T_final)
    final_err = np.max(np.abs(q - final_exact))

    print("-" * 60)
    print(f"Final max error = {final_err:.6e}")

    assert final_err < 1e-9, (
        f"Stationary q=x-y test failed: final_err={final_err:.6e}"
    )

    print("✅ SDG stationary linear solution test passed")
    print("=" * 60)


if __name__ == "__main__":
    run_stationary_xy_test()