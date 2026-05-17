import numpy as np


def reference_to_unit_triangle(r, s):
    """
    Map reference triangle

        (-1,-1), (1,-1), (-1,1)

    to unit triangle

        (0,0), (1,0), (0,1)

    using

        xi  = (r+1)/2
        eta = (s+1)/2
    """
    r = np.asarray(r)
    s = np.asarray(s)

    xi = 0.5 * (r + 1.0)
    eta = 0.5 * (s + 1.0)

    return xi, eta


def patch_info(patch_id):
    """
    Return hemisphere sign and longitude interval for one octahedral patch.

    Patch convention:

        0,1,2,3 : northern hemisphere
        4,5,6,7 : southern hemisphere

    Each patch covers one quadrant in longitude.

    patch 0: lambda in [0, pi/2]
    patch 1: lambda in [pi/2, pi]
    patch 2: lambda in [pi, 3pi/2]
    patch 3: lambda in [3pi/2, 2pi]

    patches 4-7 use the same longitude intervals but southern latitude.
    """
    if patch_id < 0 or patch_id > 7:
        raise ValueError("patch_id must be between 0 and 7.")

    if patch_id < 4:
        hemisphere = +1.0
        sector = patch_id
    else:
        hemisphere = -1.0
        sector = patch_id - 4

    lam0 = sector * 0.5 * np.pi
    lam1 = (sector + 1) * 0.5 * np.pi

    return hemisphere, lam0, lam1


def unit_triangle_to_lonlat(xi, eta, patch_id):
    """
    Equal-area octahedral sphere mapping.

    Unit triangle coordinates:

        xi >= 0, eta >= 0, xi + eta <= 1

    Define:

        rho = xi + eta

    The pole is rho = 0.
    The equator edge is rho = 1.

    Longitude:

        lambda = lambda0 + (lambda1-lambda0) * eta / rho

    Latitude:

        sin(theta) = hemisphere * (1 - rho^2)

    This gives constant surface Jacobian:

        sqrt(G) = pi R^2

    before multiplying by R^2 in the sphere coordinate map.
    """
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)

    hemisphere, lam0, lam1 = patch_info(patch_id)

    rho = xi + eta
    dlam = lam1 - lam0

    # At rho = 0, longitude is mathematically irrelevant because it is the pole.
    ratio = np.zeros_like(rho, dtype=float)
    mask = rho > 1.0e-14
    ratio[mask] = eta[mask] / rho[mask]

    lam = lam0 + dlam * ratio

    sin_theta = hemisphere * (1.0 - rho**2)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    theta = np.arcsin(sin_theta)

    return lam, theta


def lonlat_to_xyz(lam, theta, R=1.0):
    """
    Convert longitude-latitude coordinates to 3D sphere coordinates.

        X = R cos(theta) cos(lambda)
        Y = R cos(theta) sin(lambda)
        Z = R sin(theta)
    """
    lam = np.asarray(lam, dtype=float)
    theta = np.asarray(theta, dtype=float)

    cos_theta = np.cos(theta)

    X = R * cos_theta * np.cos(lam)
    Y = R * cos_theta * np.sin(lam)
    Z = R * np.sin(theta)

    return X, Y, Z


def map_unit_triangle_to_sphere(xi, eta, patch_id, R=1.0):
    """
    Map unit triangle coordinates directly to sphere.
    """
    lam, theta = unit_triangle_to_lonlat(xi, eta, patch_id)
    X, Y, Z = lonlat_to_xyz(lam, theta, R=R)

    return X, Y, Z


def map_reference_to_sphere(r, s, patch_id, R=1.0):
    """
    Map reference triangle coordinates (r,s) to sphere.
    """
    xi, eta = reference_to_unit_triangle(r, s)
    return map_unit_triangle_to_sphere(xi, eta, patch_id, R=R)


def constant_area_jacobian(R=1.0):
    """
    The equal-area octahedral mapping has

        sqrt(G) = pi R^2

    with respect to unit triangle coordinates (xi, eta).
    """
    return np.pi * R**2


def numerical_surface_jacobian(xi, eta, patch_id, R=1.0, eps=1.0e-6):
    """
    Numerically compute

        |dX/dxi cross dX/deta|

    using central differences.

    This is only reliable away from patch singularities / vertices.
    """
    xi = float(xi)
    eta = float(eta)

    def F(a, b):
        X, Y, Z = map_unit_triangle_to_sphere(a, b, patch_id, R=R)
        return np.array([X, Y, Z], dtype=float)

    X_xi_plus = F(xi + eps, eta)
    X_xi_minus = F(xi - eps, eta)

    X_eta_plus = F(xi, eta + eps)
    X_eta_minus = F(xi, eta - eps)

    dX_dxi = (X_xi_plus - X_xi_minus) / (2.0 * eps)
    dX_deta = (X_eta_plus - X_eta_minus) / (2.0 * eps)

    return np.linalg.norm(np.cross(dX_dxi, dX_deta))