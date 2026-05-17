import numpy as np
import matplotlib.pyplot as plt

from core.mesh import create_unit_triangle_mesh


def plot_unit_triangle_mesh(n_div=4):
    VX, VY, EToV = create_unit_triangle_mesh(n_div)

    plt.figure(figsize=(7, 7))

    for k, elem in enumerate(EToV):
        x = VX[elem]
        y = VY[elem]

        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])

        plt.plot(x_closed, y_closed, "k-", linewidth=1)

        cx = np.mean(x)
        cy = np.mean(y)

        plt.text(cx, cy, str(k), fontsize=8, ha="center", va="center")

    plt.fill([0, 1, 0], [0, 0, 1], alpha=0.12)

    plt.title(f"Unit triangle refinement mesh, n_div={n_div}")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\eta$")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_unit_triangle_mesh(n_div=4)