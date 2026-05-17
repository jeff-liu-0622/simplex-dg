import numpy as np
import matplotlib.pyplot as plt

from core.geometry.sphere_mapping import map_unit_triangle_to_sphere


def sample_unit_triangle(n=25):
    """
    Sample points inside unit triangle:
        xi >= 0, eta >= 0, xi + eta <= 1
    """
    xi_list = []
    eta_list = []

    for i in range(n + 1):
        for j in range(n + 1 - i):
            xi_list.append(i / n)
            eta_list.append(j / n)

    return np.array(xi_list), np.array(eta_list)


def plot_sphere_mapping():
    R = 1.0
    xi, eta = sample_unit_triangle(n=40)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for patch_id in range(8):
        X, Y, Z = map_unit_triangle_to_sphere(xi, eta, patch_id, R=R)

        ax.scatter(
            X,
            Y,
            Z,
            s=8,
            label=f"Patch {patch_id}",
            alpha=0.75,
        )

    # Draw wireframe sphere for reference
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 40)

    uu, vv = np.meshgrid(u, v)

    Xs = R * np.cos(vv) * np.cos(uu)
    Ys = R * np.cos(vv) * np.sin(uu)
    Zs = R * np.sin(vv)

    ax.plot_wireframe(
        Xs,
        Ys,
        Zs,
        linewidth=0.3,
        alpha=0.15,
    )

    ax.set_title("Octahedral equal-area sphere mapping")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sphere_mapping()