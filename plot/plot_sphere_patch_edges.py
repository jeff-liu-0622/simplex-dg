import numpy as np
import matplotlib.pyplot as plt

from core.geometry.sphere_mapping import map_unit_triangle_to_sphere


def edge_samples(n=100):
    """
    Return three edges of the unit triangle:

        edge 0: eta = 0, xi from 0 to 1
        edge 1: xi + eta = 1
        edge 2: xi = 0, eta from 1 to 0

    This follows the simplex face convention:

        face 0: v0 -> v1
        face 1: v1 -> v2
        face 2: v2 -> v0
    """
    t = np.linspace(0.0, 1.0, n)

    # face 0: (0,0) -> (1,0)
    xi0 = t
    eta0 = np.zeros_like(t)

    # face 1: (1,0) -> (0,1)
    xi1 = 1.0 - t
    eta1 = t

    # face 2: (0,1) -> (0,0)
    xi2 = np.zeros_like(t)
    eta2 = 1.0 - t

    return [
        (xi0, eta0),
        (xi1, eta1),
        (xi2, eta2),
    ]


def plot_sphere_patch_edges():
    R = 1.0
    edges = edge_samples(n=150)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw transparent reference sphere
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 40)
    uu, vv = np.meshgrid(u, v)

    Xs = R * np.cos(vv) * np.cos(uu)
    Ys = R * np.cos(vv) * np.sin(uu)
    Zs = R * np.sin(vv)

    ax.plot_wireframe(
        Xs,
        Ys,
        Zs,
        linewidth=0.25,
        alpha=0.12,
    )

    # Draw patch edges
    for patch_id in range(8):
        for edge_id, (xi, eta) in enumerate(edges):
            X, Y, Z = map_unit_triangle_to_sphere(
                xi,
                eta,
                patch_id,
                R=R,
            )

            label = f"patch {patch_id}" if edge_id == 0 else None

            ax.plot(
                X,
                Y,
                Z,
                linewidth=2.0,
                label=label,
            )

        # Mark patch center
        xi_c = np.array([1.0 / 3.0])
        eta_c = np.array([1.0 / 3.0])

        Xc, Yc, Zc = map_unit_triangle_to_sphere(
            xi_c,
            eta_c,
            patch_id,
            R=R,
        )

        ax.text(
            Xc[0],
            Yc[0],
            Zc[0],
            str(patch_id),
            fontsize=12,
            weight="bold",
        )

    ax.set_title("Octahedral sphere patch edges")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sphere_patch_edges()