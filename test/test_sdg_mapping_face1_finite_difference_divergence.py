import numpy as np

from core.geometry.sdg_sphere_mapping import sdg_mapping_from_xy_patch


def solid_body_velocity_lonlat(lam, theta, u0=1.0, alpha=np.pi / 4.0):
    """
    Same velocity convention as the SDG sphere notes:

        u = u0 (cos(alpha) cos(theta)
                + sin(alpha) cos(lambda) sin(theta))

        v = -u0 sin(alpha) sin(lambda)

    Here u, v are the two physical velocity components used by A^{-1}.
    """
    u = u0 * (
        np.cos(alpha) * np.cos(theta)
        + np.sin(alpha) * np.cos(lam) * np.sin(theta)
    )

    v = -u0 * np.sin(alpha) * np.sin(lam)

    return u, v


def contravariant_velocity_T1(x, y, alpha=np.pi / 4.0, R=1.0):
    """
    Compute (u1,u2) on T1 using SDG stable A^{-1}.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    result = sdg_mapping_from_xy_patch(
        x_arr,
        y_arr,
        patch_id=np.ones_like(x_arr, dtype=int),
        R=R,
    )

    lam = result.lambda_
    theta = result.theta

    u, v = solid_body_velocity_lonlat(
        lam,
        theta,
        u0=1.0,
        alpha=alpha,
    )

    uv = np.stack([u, v], axis=-1)

    contravariant = np.einsum(
        "...ij,...j->...i",
        result.Ainv,
        uv,
    )

    u1 = contravariant[..., 0]
    u2 = contravariant[..., 1]

    return u1, u2


def finite_difference_divergence(x, y, eps, alpha=np.pi / 4.0):
    """
    Central difference approximation:

        div = d_x u1 + d_y u2
    """
    u1_px, _ = contravariant_velocity_T1(x + eps, y, alpha=alpha)
    u1_mx, _ = contravariant_velocity_T1(x - eps, y, alpha=alpha)

    _, u2_py = contravariant_velocity_T1(x, y + eps, alpha=alpha)
    _, u2_my = contravariant_velocity_T1(x, y - eps, alpha=alpha)

    du1_dx = (u1_px - u1_mx) / (2.0 * eps)
    du2_dy = (u2_py - u2_my) / (2.0 * eps)

    return du1_dx + du2_dy


def sample_T1_interior_points(n=20, margin=0.08):
    """
    T1 in unfolded SDG coordinates is:

        x >= 0, y >= 0, x + y <= 1

    Avoid:
        pole x+y=0
        edges
    """
    xs = []
    ys = []

    grid = np.linspace(margin, 1.0 - margin, n)

    for x in grid:
        for y in grid:
            if x > margin and y > margin and x + y < 1.0 - margin:
                xs.append(x)
                ys.append(y)

    return np.array(xs), np.array(ys)


def run_finite_difference_test():
    print("\n" + "=" * 88)
    print("T1 finite-difference divergence check using SDG stable A^{-1}")
    print("=" * 88)

    x, y = sample_T1_interior_points(n=30, margin=0.05)

    for alpha in [0.0, np.pi / 4.0, np.pi / 2.0]:
        print(f"\nalpha = {alpha:.12f}")
        print("-" * 88)
        print(f"{'eps':>12s} {'max |div|':>16s} {'mean |div|':>16s} {'rms div':>16s}")
        print("-" * 88)

        for eps in [1.0e-3, 5.0e-4, 1.0e-4, 5.0e-5, 1.0e-5]:
            div = finite_difference_divergence(
                x,
                y,
                eps=eps,
                alpha=alpha,
            )

            abs_div = np.abs(div)

            print(
                f"{eps:12.1e} "
                f"{np.max(abs_div):16.6e} "
                f"{np.mean(abs_div):16.6e} "
                f"{np.sqrt(np.mean(div**2)):16.6e}"
            )

    print("\n✅ finite-difference divergence diagnostic finished")
    print("=" * 88)


if __name__ == "__main__":
    run_finite_difference_test()