import numpy as np


def compute_manifold_geometry(engine, xyz_nodes):
    """
    Compute local manifold geometry directly from 3D nodal coordinates.

    This helper keeps the projected 3D manifold diagnostic geometry formula
    used by the sphere tests.
    """
    a1 = engine.Dr @ xyz_nodes
    a2 = engine.Ds @ xyz_nodes

    normal_area = np.cross(a1, a2)
    J = np.linalg.norm(normal_area, axis=1)
    n = normal_area / J[:, None]

    a_contra_1 = np.cross(a2, n) / J[:, None]
    a_contra_2 = np.cross(n, a1) / J[:, None]

    return {
        "a1": a1,
        "a2": a2,
        "J": J,
        "n": n,
        "a_contra_1": a_contra_1,
        "a_contra_2": a_contra_2,
    }
