import numpy as np

from core.geometry.sphere_mapping import unit_triangle_to_lonlat
from core.geometry.sphere_velocity import (
    solid_body_velocity_lonlat,
    lonlat_velocity_to_angular_velocity,
    lonlat_jacobian_wrt_unit_triangle,
    solid_body_contravariant_velocity,
)


def sample_safe_points():
    """
    Sample points inside the unit triangle, away from:
        rho = 0 pole
        xi + eta = 1 edge endpoints

    This avoids longitude singularity at the pole.
    """
    xi = []
    eta = []

    raw = [
        (0.20, 0.20),
        (0.35, 0.15),
        (0.15, 0.45),
        (0.45, 0.25),
        (0.25, 0.55),
        (0.60, 0.15),
        (0.15, 0.65),
    ]

    for a, b in raw:
        if a > 0.0 and b > 0.0 and a + b < 0.95:
            xi.append(a)
            eta.append(b)

    return np.array(xi), np.array(eta)


def check_reconstruction(alpha):
    R = 1.0
    u0 = 1.0

    xi, eta = sample_safe_points()

    for patch_id in range(8):
        lam, theta = unit_triangle_to_lonlat(xi, eta, patch_id)

        u, v = solid_body_velocity_lonlat(
            lam,
            theta,
            R=R,
            u0=u0,
            alpha=alpha,
        )

        lam_dot, theta_dot = lonlat_velocity_to_angular_velocity(
            u,
            v,
            theta,
            R=R,
        )

        u_xi, u_eta = solid_body_contravariant_velocity(
            xi,
            eta,
            patch_id,
            R=R,
            u0=u0,
            alpha=alpha,
        )

        lam_xi, lam_eta, theta_xi, theta_eta = lonlat_jacobian_wrt_unit_triangle(
            xi,
            eta,
            patch_id,
        )

        lam_dot_reconstructed = lam_xi * u_xi + lam_eta * u_eta
        theta_dot_reconstructed = theta_xi * u_xi + theta_eta * u_eta

        lam_err = np.max(np.abs(lam_dot_reconstructed - lam_dot))
        theta_err = np.max(np.abs(theta_dot_reconstructed - theta_dot))

        assert lam_err < 1.0e-11, (
            f"patch={patch_id}, alpha={alpha}: "
            f"lambda_dot reconstruction error = {lam_err:.3e}"
        )

        assert theta_err < 1.0e-11, (
            f"patch={patch_id}, alpha={alpha}: "
            f"theta_dot reconstruction error = {theta_err:.3e}"
        )


def test_reconstruction_zonal():
    """
    alpha = 0:
        simple zonal rotation.
    """
    check_reconstruction(alpha=0.0)


def test_reconstruction_tilted_pi_over_4():
    """
    alpha = pi/4:
        tilted solid-body rotation.
    """
    check_reconstruction(alpha=np.pi / 4.0)


def test_reconstruction_tilted_pi_over_2():
    """
    alpha = pi/2:
        another extreme tilted case.
    """
    check_reconstruction(alpha=np.pi / 2.0)


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 sphere contravariant velocity reconstruction 測試")
    print("=" * 72)

    test_reconstruction_zonal()
    print("✅ alpha=0 reconstruction passed")

    test_reconstruction_tilted_pi_over_4()
    print("✅ alpha=pi/4 reconstruction passed")

    test_reconstruction_tilted_pi_over_2()
    print("✅ alpha=pi/2 reconstruction passed")

    print("=" * 72)
    print("🎉 test_sphere_contravariant_reconstruction.py 全部通過")


if __name__ == "__main__":
    run_all_tests()