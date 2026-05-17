import numpy as np

from core.geometry.sphere_mapping import unit_triangle_to_lonlat
from core.geometry.sphere_velocity import (
    solid_body_velocity_lonlat,
    lonlat_velocity_to_angular_velocity,
    solid_body_contravariant_velocity,
)


def sample_interior_points():
    """
    Points away from rho=0 pole and rho=1 equator endpoints.
    """
    xi = np.array([0.20, 0.40, 0.15, 0.30, 0.55])
    eta = np.array([0.20, 0.10, 0.50, 0.35, 0.20])

    # Keep only xi+eta < 1
    mask = xi + eta < 1.0
    return xi[mask], eta[mask]


def test_solid_body_velocity_no_nan():
    xi, eta = sample_interior_points()

    for patch_id in range(8):
        lam, theta = unit_triangle_to_lonlat(xi, eta, patch_id)

        u, v = solid_body_velocity_lonlat(
            lam,
            theta,
            R=1.0,
            u0=1.0,
            alpha=0.0,
        )

        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))


def test_zonal_rotation_angular_velocity():
    """
    For alpha=0,

        u = u0 cos(theta)
        v = 0

    so

        lambda_dot = u0 / R
        theta_dot = 0.
    """
    R = 2.0
    u0 = 3.0

    xi, eta = sample_interior_points()

    for patch_id in range(8):
        lam, theta = unit_triangle_to_lonlat(xi, eta, patch_id)

        u, v = solid_body_velocity_lonlat(
            lam,
            theta,
            R=R,
            u0=u0,
            alpha=0.0,
        )

        lam_dot, theta_dot = lonlat_velocity_to_angular_velocity(
            u,
            v,
            theta,
            R=R,
        )

        assert np.max(np.abs(lam_dot - u0 / R)) < 1.0e-12
        assert np.max(np.abs(theta_dot)) < 1.0e-12


def test_contravariant_velocity_no_nan_alpha0():
    """
    Check patch-coordinate velocity is finite away from vertices.
    """
    xi, eta = sample_interior_points()

    for patch_id in range(8):
        u_xi, u_eta = solid_body_contravariant_velocity(
            xi,
            eta,
            patch_id,
            R=1.0,
            u0=1.0,
            alpha=0.0,
        )

        assert np.all(np.isfinite(u_xi))
        assert np.all(np.isfinite(u_eta))


def test_contravariant_velocity_no_nan_tilted():
    """
    Check tilted solid-body velocity is finite away from pole vertices.
    """
    xi, eta = sample_interior_points()

    alpha = np.pi / 4.0

    for patch_id in range(8):
        u_xi, u_eta = solid_body_contravariant_velocity(
            xi,
            eta,
            patch_id,
            R=1.0,
            u0=1.0,
            alpha=alpha,
        )

        assert np.all(np.isfinite(u_xi))
        assert np.all(np.isfinite(u_eta))


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 sphere velocity 測試")
    print("=" * 72)

    test_solid_body_velocity_no_nan()
    print("✅ solid-body velocity has no NaN")

    test_zonal_rotation_angular_velocity()
    print("✅ zonal angular velocity verified")

    test_contravariant_velocity_no_nan_alpha0()
    print("✅ contravariant velocity finite for alpha=0")

    test_contravariant_velocity_no_nan_tilted()
    print("✅ contravariant velocity finite for tilted flow")

    print("=" * 72)
    print("🎉 test_sphere_velocity.py 全部通過")


if __name__ == "__main__":
    run_all_tests()