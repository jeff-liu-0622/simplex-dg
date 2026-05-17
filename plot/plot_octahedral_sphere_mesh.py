import numpy as np
import matplotlib.pyplot as plt

from core.mesh_octahedron import create_octahedral_layout_mesh
from core.geometry.sphere_mapping import map_unit_triangle_to_sphere


def plot_octahedral_sphere_mesh(nsub=5, R=1.0):
    VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")

    # 先畫一個淡淡的球面 wireframe
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 40)
    uu, vv = np.meshgrid(u, v)

    Xs = R * np.cos(vv) * np.cos(uu)
    Ys = R * np.cos(vv) * np.sin(uu)
    Zs = R * np.sin(vv)

    ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.25, alpha=0.10)

    # 畫每個小三角形
    for k in range(EToV.shape[0]):
        pid = patch_ids[k]          # 1..8
        patch0 = pid - 1            # sphere_mapping 用 0..7

        loc = element_local_coords[k]   # shape (3,2)
        xi = loc[:, 0]
        eta = loc[:, 1]

        X, Y, Z = map_unit_triangle_to_sphere(xi, eta, patch0, R=R)

        color = cmap((pid - 1) % 10)

        Xc = np.append(X, X[0])
        Yc = np.append(Y, Y[0])
        Zc = np.append(Z, Z[0])

        ax.plot(Xc, Yc, Zc, color=color, linewidth=0.8)

    # patch 中心文字
    centers = []
    for pid in range(1, 9):
        patch0 = pid - 1
        xi_c = np.array([1.0 / 3.0])
        eta_c = np.array([1.0 / 3.0])
        Xc, Yc, Zc = map_unit_triangle_to_sphere(xi_c, eta_c, patch0, R=R)
        centers.append((pid, Xc[0], Yc[0], Zc[0]))

    for pid, x, y, z in centers:
        ax.text(x, y, z, f"T{pid}", fontsize=12, weight="bold")

    ax.set_title(f"Octahedral sphere mesh (nsub={nsub})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_octahedral_sphere_mesh(nsub=5, R=1.0)