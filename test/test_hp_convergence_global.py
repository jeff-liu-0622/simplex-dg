import numpy as np

from core import build_local_operators
from core.rhs import compute_volume_divergence


def build_uniform_mesh(divisions):
    """
    Build a uniform triangular mesh on [0,1] x [0,1].

    Each square is split into two counter-clockwise triangles.
    """
    x_1d = np.linspace(0.0, 1.0, divisions + 1)
    y_1d = np.linspace(0.0, 1.0, divisions + 1)

    elements = []

    for i in range(divisions):
        for j in range(divisions):
            x0, x1 = x_1d[i], x_1d[i + 1]
            y0, y1 = y_1d[j], y_1d[j + 1]

            elements.append(((x0, y0), (x1, y0), (x0, y1)))
            elements.append(((x1, y1), (x0, y1), (x1, y0)))

    return elements


def compute_global_error(N, divisions):
    """
    Compute max error of the conservative volume divergence operator.

    Test field:

        F = (sin(pi x) cos(pi y), cos(pi x) sin(pi y))

    Exact divergence:

        div F = 2 pi cos(pi x) cos(pi y)
    """
    engine = build_local_operators(N=N, n=N, rule="table1")

    elements = build_uniform_mesh(divisions)

    r = engine.r
    s = engine.s
    Np = engine.num_nodes

    global_max_err = 0.0

    for v1, v2, v3 in elements:
        x1, y1 = v1
        x2, y2 = v2
        x3, y3 = v3

        x = 0.5 * (
            -(r + s) * x1
            + (1.0 + r) * x2
            + (1.0 + s) * x3
        )

        y = 0.5 * (
            -(r + s) * y1
            + (1.0 + r) * y2
            + (1.0 + s) * y3
        )

        xr = 0.5 * (x2 - x1)
        xs = 0.5 * (x3 - x1)
        yr = 0.5 * (y2 - y1)
        ys = 0.5 * (y3 - y1)

        J_val = xr * ys - xs * yr

        if J_val <= 0.0:
            raise ValueError("Non-positive Jacobian detected.")

        rx = ys / J_val
        sx = -yr / J_val
        ry = -xs / J_val
        sy = xr / J_val

        rx_arr = np.full(Np, rx)
        sx_arr = np.full(Np, sx)
        ry_arr = np.full(Np, ry)
        sy_arr = np.full(Np, sy)
        J_arr = np.full(Np, J_val)

        Fx = np.sin(np.pi * x) * np.cos(np.pi * y)
        Fy = np.cos(np.pi * x) * np.sin(np.pi * y)

        exact_div = 2.0 * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)

        num_div = compute_volume_divergence(
            engine,
            Fx,
            Fy,
            rx_arr,
            sx_arr,
            ry_arr,
            sy_arr,
            J_arr,
        )

        element_max_err = np.max(np.abs(num_div - exact_div))
        global_max_err = max(global_max_err, element_max_err)

    return global_max_err, len(elements)


def run_tests():
    print("=" * 72)
    print("Nodal DG volume divergence h-p convergence test")
    print("=" * 72)

    # ------------------------------------------------------------
    # h-convergence
    # ------------------------------------------------------------
    print("\nTable 1: h-convergence, fixed polynomial degree N=3")
    print(f"{'Divisions':<12} {'Elements':<10} {'Max Error':<15} {'Rate':<10}")
    print("-" * 72)

    div_list = [2, 4, 8, 16, 32, 64]

    prev_err = None

    for divs in div_list:
        err, num_elem = compute_global_error(N=3, divisions=divs)

        if prev_err is None:
            rate_str = "-"
        else:
            rate = np.log2(prev_err / err)
            rate_str = f"{rate:.3f}"

        print(f"{divs:<12} {num_elem:<10} {err:<15.6e} {rate_str:<10}")

        prev_err = err

    # ------------------------------------------------------------
    # p-convergence
    # ------------------------------------------------------------
    print("\nTable 2: p-convergence, fixed mesh 4x4")
    print(f"{'Degree N':<12} {'Elements':<10} {'Max Error':<15} {'Ratio':<10}")
    print("-" * 72)

    N_list = [1, 2, 3, 4]

    prev_err = None

    for N in N_list:
        err, num_elem = compute_global_error(N=N, divisions=4)

        if prev_err is None:
            ratio_str = "-"
        else:
            ratio = prev_err / err
            ratio_str = f"{ratio:.3e}"

        print(f"{N:<12} {num_elem:<10} {err:<15.6e} {ratio_str:<10}")

        prev_err = err

    print("\nConclusion:")
    print("1. h-convergence checks the conservative volume divergence operator.")
    print("2. p-convergence should show rapid error reduction for this smooth field.")
    print("3. This test does not include surface fluxes or time integration.")

    print("=" * 72)


if __name__ == "__main__":
    run_tests()