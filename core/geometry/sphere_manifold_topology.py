from __future__ import annotations

import numpy as np


def _normalize_to_sphere(point: np.ndarray, R: float) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    norm = float(np.linalg.norm(point))

    if norm <= 0.0:
        raise ValueError("Cannot project the zero vector to the sphere.")

    return R * point / norm


def _add_projected_vertex(
    point: np.ndarray,
    R: float,
    vertex_map: dict[tuple[float, float, float], int],
    vertices: list[np.ndarray],
    ndigits: int,
) -> int:
    xyz = _normalize_to_sphere(point, R=R)
    key = tuple(round(float(v), ndigits) for v in xyz)

    if key in vertex_map:
        return vertex_map[key]

    idx = len(vertices)
    vertex_map[key] = idx
    vertices.append(xyz)

    return idx


def _subdivide_octahedron_face(
    base_vertices: np.ndarray,
    nsub: int,
    R: float,
    vertex_map: dict[tuple[float, float, float], int],
    vertices: list[np.ndarray],
    ndigits: int,
) -> list[tuple[int, int, int]]:
    local_to_global: dict[tuple[int, int], int] = {}
    v0, v1, v2 = np.asarray(base_vertices, dtype=float)

    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            a = 1.0 - (i + j) / nsub
            b = i / nsub
            c = j / nsub
            point = a * v0 + b * v1 + c * v2

            local_to_global[(i, j)] = _add_projected_vertex(
                point=point,
                R=R,
                vertex_map=vertex_map,
                vertices=vertices,
                ndigits=ndigits,
            )

    elems: list[tuple[int, int, int]] = []

    for i in range(nsub):
        for j in range(nsub - i):
            v00 = local_to_global[(i, j)]
            v10 = local_to_global[(i + 1, j)]
            v01 = local_to_global[(i, j + 1)]
            elems.append((v00, v10, v01))

            if i + j < nsub - 1:
                v11 = local_to_global[(i + 1, j + 1)]
                elems.append((v10, v11, v01))

    return elems


def create_projected_octahedron_sphere_mesh(
    nsub: int,
    R: float = 1.0,
    ndigits: int = 12,
):
    """
    Create a closed 3D projected octahedron sphere mesh.

    The connectivity is built from shared projected 3D vertices, not from an
    unfolded 2D layout. This makes shared faces physically continuous on the
    sphere.

    Returns
    -------
    VX, VY, VZ:
        Vertex coordinates on the sphere, each shape (Nv,).

    EToV:
        Element-to-vertex connectivity, shape (K,3).

    patch_ids:
        Base octahedron face id for each element, shape (K,), values 1..8.

    nodes_xyz:
        Full vertex coordinate array, shape (Nv,3).
    """
    if nsub < 1:
        raise ValueError("nsub must be >= 1.")
    if R <= 0.0:
        raise ValueError("R must be positive.")

    base_nodes = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    # Outward orientation for each octahedron face.
    base_faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=int,
    )

    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[np.ndarray] = []
    elems: list[tuple[int, int, int]] = []
    patch_ids: list[int] = []

    for patch_id, face in enumerate(base_faces, start=1):
        face_elems = _subdivide_octahedron_face(
            base_vertices=base_nodes[face],
            nsub=nsub,
            R=R,
            vertex_map=vertex_map,
            vertices=vertices,
            ndigits=ndigits,
        )
        elems.extend(face_elems)
        patch_ids.extend([patch_id] * len(face_elems))

    nodes_xyz = np.vstack(vertices).astype(float)
    EToV = np.asarray(elems, dtype=int)
    patch_ids_array = np.asarray(patch_ids, dtype=int)

    return (
        nodes_xyz[:, 0],
        nodes_xyz[:, 1],
        nodes_xyz[:, 2],
        EToV,
        patch_ids_array,
        nodes_xyz,
    )


def map_reference_nodes_to_projected_sphere(
    nodes_xyz: np.ndarray,
    EToV: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    R: float = 1.0,
) -> np.ndarray:
    """
    Map reference nodes to the sphere by linearly interpolating each 3D
    triangular element and radially projecting to radius R.
    """
    nodes_xyz = np.asarray(nodes_xyz, dtype=float)
    EToV = np.asarray(EToV, dtype=int)
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] != 3:
        raise ValueError("nodes_xyz must have shape (Nv,3).")
    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K,3).")
    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    L1 = -0.5 * (r + s)
    L2 = 0.5 * (1.0 + r)
    L3 = 0.5 * (1.0 + s)

    tri = nodes_xyz[EToV]
    xyz = (
        L1[None, :, None] * tri[:, None, 0, :]
        + L2[None, :, None] * tri[:, None, 1, :]
        + L3[None, :, None] * tri[:, None, 2, :]
    )

    norm = np.linalg.norm(xyz, axis=2)

    if np.any(norm <= 0.0):
        raise ValueError("Reference node interpolation hit the zero vector.")

    return R * xyz / norm[:, :, None]


def projected_sphere_mesh_hmin(nodes_xyz: np.ndarray, EToV: np.ndarray) -> float:
    nodes_xyz = np.asarray(nodes_xyz, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    tri = nodes_xyz[EToV]
    e01 = np.linalg.norm(tri[:, 1, :] - tri[:, 0, :], axis=1)
    e12 = np.linalg.norm(tri[:, 2, :] - tri[:, 1, :], axis=1)
    e20 = np.linalg.norm(tri[:, 0, :] - tri[:, 2, :], axis=1)

    return float(np.min(np.minimum(np.minimum(e01, e12), e20)))
