import numpy as np


def compute_volume_metrics(x, y):
    """
    Compute affine volume metrics for triangular elements.

    Parameters
    ----------
    x, y:
        Arrays of shape (K, 3), storing the physical coordinates
        of the three vertices of each triangle.

    Reference triangle:
        (-1,-1), (1,-1), (-1,1)

    Mapping:
        (r,s) -> (x,y)

    Returns
    -------
    rx, sx, ry, sy, J

    Notes
    -----
    J is the Jacobian determinant dxdy/drds.

    Since the reference triangle area is 2,

        physical element area = 2J.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x1, y1 = x[:, 0], y[:, 0]
    x2, y2 = x[:, 1], y[:, 1]
    x3, y3 = x[:, 2], y[:, 2]

    xr = 0.5 * (x2 - x1)
    xs = 0.5 * (x3 - x1)

    yr = 0.5 * (y2 - y1)
    ys = 0.5 * (y3 - y1)

    J = xr * ys - xs * yr

    if np.any(J <= 0):
        raise ValueError(
            "Geometry error: found non-positive Jacobian J <= 0. "
            "Check whether EToV vertices are counter-clockwise."
        )

    rx = ys / J
    sx = -yr / J
    ry = -xs / J
    sy = xr / J

    return rx, sx, ry, sy, J


def compute_face_metrics(x, y):
    """
    Compute outward unit normals and edge lengths.

    Face ordering matches the reference triangle and connectivity.py:

        face 0: v0 -> v1, reference edge s = -1
        face 1: v1 -> v2, reference edge r + s = 0
        face 2: v2 -> v0, reference edge r = -1

    For counter-clockwise elements, the outward unit normal of an edge
    with tangent (dx, dy) is:

        n = (dy, -dx) / L

    Returns
    -------
    nx, ny:
        Shape (K, 3), outward unit normal components.

    edge_lengths:
        Shape (K, 3), full physical edge lengths L.

    sJ:
        Shape (K, 3), surface Jacobian L/2.
        This is kept for compatibility with H&W-style formulas.

    Notes
    -----
    For the SDG lift

        V M^{-1} V^T E^T W_b p

    use edge_lengths, not sJ, because quadrature.py uses edge weights
    normalized so that sum(w_e) = 1.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x1, y1 = x[:, 0], y[:, 0]
    x2, y2 = x[:, 1], y[:, 1]
    x3, y3 = x[:, 2], y[:, 2]

    # Face 0: v0 -> v1
    dx1 = x2 - x1
    dy1 = y2 - y1

    # Face 1: v1 -> v2
    dx2 = x3 - x2
    dy2 = y3 - y2

    # Face 2: v2 -> v0
    dx3 = x1 - x3
    dy3 = y1 - y3

    L1 = np.sqrt(dx1**2 + dy1**2)
    L2 = np.sqrt(dx2**2 + dy2**2)
    L3 = np.sqrt(dx3**2 + dy3**2)

    edge_lengths = np.vstack([L1, L2, L3]).T
    sJ = 0.5 * edge_lengths

    nx = np.vstack([
        dy1 / L1,
        dy2 / L2,
        dy3 / L3,
    ]).T

    ny = np.vstack([
        -dx1 / L1,
        -dx2 / L2,
        -dx3 / L3,
    ]).T

    return nx, ny, edge_lengths, sJ