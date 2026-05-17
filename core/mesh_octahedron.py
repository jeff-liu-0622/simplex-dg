import numpy as np


def _add_vertex(vertex_map, VX, VY, x, y, tol_digits=12):
    """
    Add vertex with coordinate deduplication.
    """
    key = (round(float(x), tol_digits), round(float(y), tol_digits))

    if key in vertex_map:
        return vertex_map[key]

    idx = len(VX)
    vertex_map[key] = idx
    VX.append(float(x))
    VY.append(float(y))

    return idx


def _triangle_area_xy(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _subdivide_triangle(vertices, nsub):
    """
    Uniformly subdivide one triangle.

    Parameters
    ----------
    vertices:
        Array-like shape (3,2), ordered counter-clockwise.

    nsub:
        Number of subdivisions per edge.

    Returns
    -------
    points:
        Dict (i,j) -> coordinate.
        Local barycentric grid points:
            p(i,j) = v0 + (i/nsub)(v1-v0) + (j/nsub)(v2-v0)
            i >= 0, j >= 0, i+j <= nsub

    elems:
        List of local small triangles, each as three keys.
    """
    v0 = np.asarray(vertices[0], dtype=float)
    v1 = np.asarray(vertices[1], dtype=float)
    v2 = np.asarray(vertices[2], dtype=float)

    points = {}

    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            xi = i / nsub
            eta = j / nsub

            p = v0 + xi * (v1 - v0) + eta * (v2 - v0)
            points[(i, j)] = p

    elems = []

    for i in range(nsub):
        for j in range(nsub - i):
            # lower triangle
            a = (i, j)
            b = (i + 1, j)
            c = (i, j + 1)
            elems.append([a, b, c])

            # upper triangle
            if i + j <= nsub - 2:
                d = (i + 1, j + 1)
                elems.append([b, d, c])

    return points, elems


def get_octahedral_layout_patch_vertices():
    """
    Return 2D unfolded octahedron patch vertices.

    This layout matches the common cross/diamond arrangement:

        T1: upper-right near center
        T2: upper-right outer
        T3: upper-left near center
        T4: upper-left outer
        T5: lower-left near center
        T6: lower-left outer
        T7: lower-right near center
        T8: lower-right outer

    Each patch is a triangle in the unfolded square [-1,1]^2.

    Patch numbering returned is 1-based in the dictionary keys.
    """
    patches = {
        # upper hemisphere
        1: np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=float),

        2: np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ], dtype=float),

        3: np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ], dtype=float),

        4: np.array([
            [-1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 1.0],
        ], dtype=float),

        # lower hemisphere
        5: np.array([
            [0.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ], dtype=float),

        6: np.array([
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
        ], dtype=float),

        7: np.array([
            [0.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
        ], dtype=float),

        8: np.array([
            [1.0, 0.0],
            [0.0, -1.0],
            [1.0, -1.0],
        ], dtype=float),
    }

    # Ensure all patches are counter-clockwise.
    for pid, verts in patches.items():
        area = _triangle_area_xy(verts[0], verts[1], verts[2])

        if area <= 0.0:
            patches[pid] = verts[[0, 2, 1], :]

    return patches


def create_octahedral_layout_mesh(nsub):
    """
    Create the 2D unfolded octahedron layout mesh.

    Parameters
    ----------
    nsub:
        Uniform subdivisions per edge of each Ti patch.

    Returns
    -------
    VX, VY:
        Global 2D unfolded layout vertices.

    EToV:
        Element-to-vertex connectivity, shape (K,3).

    patch_ids:
        Shape (K,), values 1..8 indicating which Ti each small triangle belongs to.

    local_coords:
        Shape (Nv, 8, 2) is NOT returned because shared layout vertices can belong
        to multiple patches with different local coordinates.

        Instead, this function returns element_local_coords.

    element_local_coords:
        Shape (K,3,2). For each element and each of its three vertices,
        stores local unit-triangle coordinates (xi, eta) within that patch.

    Notes
    -----
    All EToV triangles are counter-clockwise in the unfolded 2D layout.
    """
    if nsub < 1:
        raise ValueError("nsub must be >= 1.")

    patches = get_octahedral_layout_patch_vertices()

    vertex_map = {}
    VX = []
    VY = []
    EToV = []
    patch_ids = []
    element_local_coords = []

    for patch_id in range(1, 9):
        verts = patches[patch_id]

        points, elems = _subdivide_triangle(verts, nsub)

        for elem in elems:
            global_elem = []
            local_elem_coords = []

            for key in elem:
                p = points[key]
                vid = _add_vertex(vertex_map, VX, VY, p[0], p[1])
                global_elem.append(vid)

                i, j = key
                xi = i / nsub
                eta = j / nsub
                local_elem_coords.append([xi, eta])

            # Guarantee positive orientation.
            pts = np.array([[VX[v], VY[v]] for v in global_elem])
            area = _triangle_area_xy(pts[0], pts[1], pts[2])

            if area <= 0.0:
                global_elem = [global_elem[0], global_elem[2], global_elem[1]]
                local_elem_coords = [
                    local_elem_coords[0],
                    local_elem_coords[2],
                    local_elem_coords[1],
                ]

            EToV.append(global_elem)
            patch_ids.append(patch_id)
            element_local_coords.append(local_elem_coords)

    VX = np.array(VX, dtype=float)
    VY = np.array(VY, dtype=float)
    EToV = np.array(EToV, dtype=int)
    patch_ids = np.array(patch_ids, dtype=int)
    element_local_coords = np.array(element_local_coords, dtype=float)

    return VX, VY, EToV, patch_ids, element_local_coords