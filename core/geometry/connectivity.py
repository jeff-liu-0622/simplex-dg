import numpy as np


def build_connectivity(EToV):
    """
    Build element-to-element and element-to-face connectivity.

    Face ordering must match the reference triangle:

        face 0 / edge1: v0 -> v1
        face 1 / edge2: v1 -> v2
        face 2 / edge3: v2 -> v0

    This ordering matches quadrature.py and operators.py:

        edge1: s = -1
        edge2: r + s = 0
        edge3: r = -1
    """
    EToV = np.asarray(EToV, dtype=int)

    K = EToV.shape[0]

    # Important:
    # This must match the reference-edge ordering.
    face_vids = np.array([
        [0, 1],   # face 0: v1 -> v2
        [1, 2],   # face 1: v2 -> v3
        [2, 0],   # face 2: v3 -> v1
    ], dtype=int)

    total_faces = K * 3
    face_data = np.zeros((total_faces, 4), dtype=int)

    idx = 0
    for k in range(K):
        for f in range(3):
            va, vb = EToV[k, face_vids[f]]
            face_data[idx] = [k, f, min(va, vb), max(va, vb)]
            idx += 1

    # Boundary faces point to themselves by default.
    EToE = np.repeat(np.arange(K)[:, None], 3, axis=1)
    EToF = np.repeat(np.arange(3)[None, :], K, axis=0)

    # Sort by the unordered vertex pair.
    sort_idx = np.lexsort((face_data[:, 3], face_data[:, 2]))
    sorted_faces = face_data[sort_idx]

    interior_faces_count = 0

    i = 0
    while i < total_faces - 1:
        same_edge = np.all(sorted_faces[i, 2:] == sorted_faces[i + 1, 2:])

        if same_edge:
            k1, f1 = sorted_faces[i, :2]
            k2, f2 = sorted_faces[i + 1, :2]

            EToE[k1, f1] = k2
            EToF[k1, f1] = f2

            EToE[k2, f2] = k1
            EToF[k2, f2] = f1

            interior_faces_count += 1
            i += 2
        else:
            i += 1

    boundary_faces_count = np.sum(EToE == np.arange(K)[:, None])

    if total_faces != 2 * interior_faces_count + boundary_faces_count:
        raise ValueError(
            "Mesh topology mismatch:\n"
            f"total_faces = {total_faces}\n"
            f"2 * interior_faces = {2 * interior_faces_count}\n"
            f"boundary_faces = {boundary_faces_count}\n"
            "Please check whether EToV contains holes, overlaps, or non-manifold edges."
        )

    return EToE, EToF


def get_engine_fmask(engine):
    """
    Return face-node masks from the ReferenceElement.

    Boundary nodes are ordered as:

        edge1 nodes, edge2 nodes, edge3 nodes, interior nodes
    """
    return [
        np.arange(s.start, s.stop, dtype=int)
        for s in engine.edge_slices
    ]


def apply_periodic_conditions(EToE, EToF, x_nodes, y_nodes, engine):
    """
    Detect boundary faces and connect periodic counterparts.

    This assumes a rectangular domain and structured periodic boundaries.

    Notes
    -----
    This is useful for simple periodic tests.
    For sphere / octahedron patches, we will likely need a different
    boundary-pairing rule.
    """
    EToE = EToE.copy()
    EToF = EToF.copy()

    K = EToE.shape[0]
    Np = engine.num_nodes
    fmask = get_engine_fmask(engine)

    TOL = 1e-8

    x_flat = x_nodes.reshape(-1)
    y_flat = y_nodes.reshape(-1)

    min_x, max_x = np.min(x_flat), np.max(x_flat)
    min_y, max_y = np.min(y_flat), np.max(y_flat)

    boundary_faces = [
        (k, f)
        for k in range(K)
        for f in range(3)
        if EToE[k, f] == k
    ]

    for k1, f1 in boundary_faces:
        if EToE[k1, f1] != k1:
            continue

        ids1 = k1 * Np + fmask[f1]
        cx1 = np.mean(x_flat[ids1])
        cy1 = np.mean(y_flat[ids1])

        target_x = cx1
        target_y = cy1

        if cx1 < min_x + TOL:
            target_x = max_x
        elif cx1 > max_x - TOL:
            target_x = min_x

        if cy1 < min_y + TOL:
            target_y = max_y
        elif cy1 > max_y - TOL:
            target_y = min_y

        for k2, f2 in boundary_faces:
            if k1 == k2:
                continue

            ids2 = k2 * Np + fmask[f2]
            cx2 = np.mean(x_flat[ids2])
            cy2 = np.mean(y_flat[ids2])

            if abs(cx2 - target_x) < TOL and abs(cy2 - target_y) < TOL:
                EToE[k1, f1] = k2
                EToF[k1, f1] = f2

                EToE[k2, f2] = k1
                EToF[k2, f2] = f1

                break

    return EToE, EToF


def build_maps(engine, EToV, EToE, EToF, x_nodes, y_nodes):
    """
    Build vmapM and vmapP.

    vmapM[k, f, :] gives global node ids on the minus side.
    vmapP[k, f, :] gives matched global node ids on the plus side.

    Neighbor face nodes are reversed when the physical face orientation
    is opposite.
    """
    K = EToE.shape[0]
    Np = engine.num_nodes
    fmask = get_engine_fmask(engine)
    Nfp = engine.num_edge_nodes

    node_ids = np.arange(K * Np).reshape(K, Np)

    vmapM = np.zeros((K, 3, Nfp), dtype=int)
    vmapP = np.zeros((K, 3, Nfp), dtype=int)

    x_flat = x_nodes.reshape(-1)
    y_flat = y_nodes.reshape(-1)

    for k in range(K):
        for f in range(3):
            local_face_nodes = fmask[f]

            my_global_nodes = node_ids[k, local_face_nodes]
            vmapM[k, f, :] = my_global_nodes

            k2 = EToE[k, f]
            f2 = EToF[k, f]

            # Boundary face: plus side equals minus side by default.
            # Boundary conditions can overwrite q_plus later.
            if k2 == k:
                vmapP[k, f, :] = my_global_nodes
                continue

            neighbor_face_nodes = fmask[f2]
            neighbor_global_nodes = node_ids[k2, neighbor_face_nodes]

            # Determine physical orientation.
            my_ids_flat = k * Np + local_face_nodes
            nb_ids_flat = k2 * Np + neighbor_face_nodes

            vec1_x = x_flat[my_ids_flat[-1]] - x_flat[my_ids_flat[0]]
            vec1_y = y_flat[my_ids_flat[-1]] - y_flat[my_ids_flat[0]]

            vec2_x = x_flat[nb_ids_flat[-1]] - x_flat[nb_ids_flat[0]]
            vec2_y = y_flat[nb_ids_flat[-1]] - y_flat[nb_ids_flat[0]]

            dot_product = vec1_x * vec2_x + vec1_y * vec2_y

            if dot_product > 0.0:
                vmapP[k, f, :] = neighbor_global_nodes
            else:
                vmapP[k, f, :] = neighbor_global_nodes[::-1]

    return vmapM, vmapP