import numpy as np

from core.operators import build_local_operators
from core.geometry.sphere_mapping import (
    map_unit_triangle_to_sphere,
    map_reference_to_sphere,
    constant_area_jacobian,
    numerical_surface_jacobian,
)


def test_points_lie_on_sphere():
    """
    Every mapped point should satisfy:

        X^2 + Y^2 + Z^2 = R^2
    """
    R = 2.0
    engine = build_local_operators(N=4, n=4, rule="table1")

    r = engine.r
    s = engine.s

    for patch_id in range(8):
        X, Y, Z = map_reference_to_sphere(r, s, patch_id, R=R)

        radius_error = np.max(np.abs(X**2 + Y**2 + Z**2 - R**2))

        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y))
        assert np.all(np.isfinite(Z))

        assert radius_error < 1.0e-12, (
            f"Patch {patch_id}: radius error = {radius_error:.3e}"
        )


def test_constant_area_jacobian_numeric():
    """
    Check numerically that

        |X_xi cross X_eta| = pi R^2

    away from patch vertices.
    """
    R = 1.5
    exact = constant_area_jacobian(R=R)

    # Interior points away from rho=0 and rho=1.
    points = [
        (0.20, 0.20),
        (0.50, 0.10),
        (0.10, 0.55),
        (0.30, 0.40),
    ]

    for patch_id in range(8):
        for xi, eta in points:
            numeric = numerical_surface_jacobian(
                xi,
                eta,
                patch_id,
                R=R,
                eps=1.0e-6,
            )

            err = abs(numeric - exact)

            assert err < 1.0e-5, (
                f"Patch {patch_id}, xi={xi}, eta={eta}: "
                f"numeric={numeric:.8e}, exact={exact:.8e}, err={err:.3e}"
            )


def test_adjacent_upper_patch_edges_match():
    """
    Upper hemisphere patches should match along shared meridian edges.

    Patch p edge xi=0 corresponds to longitude lambda_{p+1}.
    Patch p+1 edge eta=0 corresponds to the same longitude.
    """
    R = 1.0
    samples = np.linspace(0.0, 1.0, 11)

    for p in range(4):
        q = (p + 1) % 4

        # patch p edge xi=0, eta=rho
        xi_p = np.zeros_like(samples)
        eta_p = samples

        # patch q edge eta=0, xi=rho
        xi_q = samples
        eta_q = np.zeros_like(samples)

        Xp, Yp, Zp = map_unit_triangle_to_sphere(xi_p, eta_p, p, R=R)
        Xq, Yq, Zq = map_unit_triangle_to_sphere(xi_q, eta_q, q, R=R)

        err = np.max(
            np.sqrt((Xp - Xq) ** 2 + (Yp - Yq) ** 2 + (Zp - Zq) ** 2)
        )

        assert err < 1.0e-12, (
            f"Upper patch edge mismatch: patch {p} vs {q}, err={err:.3e}"
        )


def test_upper_lower_equator_edges_match():
    """
    Upper and lower patches should match on the equator edge xi+eta=1.
    """
    R = 1.0
    samples = np.linspace(0.0, 1.0, 11)

    xi = samples
    eta = 1.0 - samples

    for p in range(4):
        upper = p
        lower = p + 4

        Xu, Yu, Zu = map_unit_triangle_to_sphere(xi, eta, upper, R=R)
        Xl, Yl, Zl = map_unit_triangle_to_sphere(xi, eta, lower, R=R)

        err = np.max(
            np.sqrt((Xu - Xl) ** 2 + (Yu - Yl) ** 2 + (Zu - Zl) ** 2)
        )

        assert err < 1.0e-12, (
            f"Equator mismatch: upper patch {upper}, lower patch {lower}, "
            f"err={err:.3e}"
        )


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 sphere mapping 幾何測試")
    print("=" * 72)

    test_points_lie_on_sphere()
    print("✅ mapped points lie on sphere")

    test_constant_area_jacobian_numeric()
    print("✅ constant area Jacobian verified numerically")

    test_adjacent_upper_patch_edges_match()
    print("✅ adjacent upper patch edges match")

    test_upper_lower_equator_edges_match()
    print("✅ upper/lower equator edges match")

    print("=" * 72)
    print("🎉 test_sphere_mapping.py 全部通過")


if __name__ == "__main__":
    run_all_tests()