import numpy as np
import matplotlib.pyplot as plt

from core.mesh_octahedron import create_octahedral_layout_mesh


def plot_octahedral_layout_mesh(nsub=5):
    VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

    plt.figure(figsize=(10, 10))

    cmap = plt.get_cmap("tab10")

    for k, elem in enumerate(EToV):
        pid = patch_ids[k]
        color = cmap((pid - 1) % 10)

        x = VX[elem]
        y = VY[elem]

        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])

        plt.plot(x_closed, y_closed, color=color, linewidth=0.8)

    # draw outer square and axes
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "b-", linewidth=2)
    plt.axhline(0.0, color="tab:blue", linewidth=1.0)
    plt.axvline(0.0, color="tab:blue", linewidth=1.0)

    # patch labels
    labels = {
        1: (0.35, 0.35),
        2: (0.65, 0.65),
        3: (-0.35, 0.35),
        4: (-0.65, 0.65),
        5: (-0.35, -0.35),
        6: (-0.65, -0.65),
        7: (0.35, -0.35),
        8: (0.65, -0.65),
    }

    for pid, (x, y) in labels.items():
        plt.text(
            x,
            y,
            f"T{pid}",
            fontsize=16,
            ha="center",
            va="center",
            weight="bold",
        )

    plt.title(f"2D Ti layout with uniform subdivision (nsub={nsub})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_octahedral_layout_mesh(nsub=5)