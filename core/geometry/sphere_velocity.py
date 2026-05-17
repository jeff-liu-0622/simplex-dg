import numpy as np

from core.geometry.sphere_mapping import (
    patch_info,
    unit_triangle_to_lonlat,
)


def solid_body_velocity_lonlat(lam, theta, R=1.0, u0=1.0, alpha=0.0):
    """
    Solid-body rotation velocity on the sphere.

    Common benchmark form:

        u = u0 * (cos(alpha) cos(theta)
                  + sin(alpha) cos(lambda) sin(theta))

        v = -u0 * sin(alpha) sin(lambda)

    where

        u = eastward physical velocity component
        v = northward physical velocity component

    Parameters
    ----------
    lam:
        Longitude.

    theta:
        Latitude.

    R:
        Sphere radius.

    u0:
        Velocity scale.

    alpha:
        Rotation-axis tilt angle.

        alpha = 0:
            simple zonal rotation.

    Returns
    -------
    u, v:
        Physical velocity components in longitude-latitude frame.
    """
    lam = np.asarray(lam, dtype=float)
    theta = np.asarray(theta, dtype=float)

    u = u0 * (
        np.cos(alpha) * np.cos(theta)
        + np.sin(alpha) * np.cos(lam) * np.sin(theta)
    )

    v = -u0 * np.sin(alpha) * np.sin(lam)

    return u, v


def lonlat_velocity_to_angular_velocity(u, v, theta, R=1.0):
    """
    Convert physical velocity components to angular coordinate velocities.

    On a sphere:

        eastward physical speed  = R cos(theta) * lambda_dot
        northward physical speed = R * theta_dot

    Therefore:

        lambda_dot = u / (R cos(theta))
        theta_dot  = v / R

    Notes
    -----
    Near the poles, longitude is singular. For alpha=0, the formula is still
    well behaved because u ~ cos(theta). For general tilted flow, avoid
    evaluating exactly at the pole in diagnostic tests.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    theta = np.asarray(theta, dtype=float)

    cos_theta = np.cos(theta)

    eps = 1.0e-14
    safe_cos = np.where(np.abs(cos_theta) < eps, np.nan, cos_theta)

    lam_dot = u / (R * safe_cos)
    theta_dot = v / R

    return lam_dot, theta_dot


def lonlat_jacobian_wrt_unit_triangle(xi, eta, patch_id):
    """
    Compute derivatives of (lambda, theta) with respect to unit triangle
    coordinates (xi, eta).

    Mapping:

        rho = xi + eta
        lambda = lambda0 + dlam * eta / rho
        sin(theta) = hemisphere * (1 - rho^2)

    Returns
    -------
    lam_xi, lam_eta, theta_xi, theta_eta
    """
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)

    hemisphere, lam0, lam1 = patch_info(patch_id)
    dlam = lam1 - lam0

    rho = xi + eta

    lam_xi = np.zeros_like(rho)
    lam_eta = np.zeros_like(rho)

    mask = rho > 1.0e-14

    lam_xi[mask] = -dlam * eta[mask] / rho[mask] ** 2
    lam_eta[mask] = dlam * xi[mask] / rho[mask] ** 2

    sin_theta = hemisphere * (1.0 - rho**2)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    cos_theta = np.sqrt(np.maximum(1.0 - sin_theta**2, 0.0))

    theta_rho = np.zeros_like(rho)

    safe = cos_theta > 1.0e-14
    theta_rho[safe] = hemisphere * (-2.0 * rho[safe]) / cos_theta[safe]

    # rho_xi = 1, rho_eta = 1
    theta_xi = theta_rho
    theta_eta = theta_rho

    return lam_xi, lam_eta, theta_xi, theta_eta


def angular_velocity_to_contravariant(
    lam_dot,
    theta_dot,
    xi,
    eta,
    patch_id,
):
    """
    Convert angular velocities (lambda_dot, theta_dot) to patch-coordinate
    velocities (xi_dot, eta_dot).

    We solve:

        [lambda_xi   lambda_eta ] [xi_dot ] = [lambda_dot]
        [theta_xi    theta_eta  ] [eta_dot]   [theta_dot ]

    Returns
    -------
    u_xi, u_eta:
        Contravariant velocity components in unit triangle coordinates.
    """
    lam_dot = np.asarray(lam_dot, dtype=float)
    theta_dot = np.asarray(theta_dot, dtype=float)

    lam_xi, lam_eta, theta_xi, theta_eta = lonlat_jacobian_wrt_unit_triangle(
        xi,
        eta,
        patch_id,
    )

    det = lam_xi * theta_eta - lam_eta * theta_xi

    eps = 1.0e-14
    safe_det = np.where(np.abs(det) < eps, np.nan, det)

    u_xi = (theta_eta * lam_dot - lam_eta * theta_dot) / safe_det
    u_eta = (-theta_xi * lam_dot + lam_xi * theta_dot) / safe_det

    return u_xi, u_eta


def solid_body_contravariant_velocity(
    xi,
    eta,
    patch_id,
    R=1.0,
    u0=1.0,
    alpha=0.0,
):
    """
    Full pipeline:

        (xi, eta)
        -> (lambda, theta)
        -> physical velocity (u, v)
        -> angular velocity (lambda_dot, theta_dot)
        -> patch-coordinate velocity (xi_dot, eta_dot)

    Returns
    -------
    u_xi, u_eta:
        Contravariant velocity components on the unit triangle.
    """
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

    u_xi, u_eta = angular_velocity_to_contravariant(
        lam_dot,
        theta_dot,
        xi,
        eta,
        patch_id,
    )

    return u_xi, u_eta

def solid_body_contravariant_velocity_face1_regularized(
    xi,
    eta,
    R=1.0,
    u0=1.0,
    alpha=0.0,
):
    """
    Regularized contravariant velocity for the first octahedron face.

    This avoids the apparent singularity in A^{-1} as theta -> pi/2.

    Valid for patch_id = 0, where lambda in [0, pi/2].
    """
    from core.geometry.sphere_mapping import unit_triangle_to_lonlat
    from core.geometry.sphere_velocity import solid_body_velocity_lonlat

    lam, theta = unit_triangle_to_lonlat(xi, eta, patch_id=0)

    u, v = solid_body_velocity_lonlat(
        lam,
        theta,
        R=R,
        u0=u0,
        alpha=alpha,
    )

    sin_theta = np.sin(theta)
    sqrt_factor = np.sqrt(1.0 + sin_theta)

    u_xi = (
        -2.0 * u / (np.pi * R * sqrt_factor)
        - ((np.pi - 2.0 * lam) * sqrt_factor * v) / (2.0 * np.pi * R)
    )

    u_eta = (
        2.0 * u / (np.pi * R * sqrt_factor)
        - (lam * sqrt_factor * v) / (np.pi * R)
    )

    return u_xi, u_eta