import numpy as np

from core.rhs import compute_boundary_penalty


def mapped_gradient_split_2d(q, Dr, Ds, rx, sx, ry, sy):
    """
    Compute split-form physical gradients.

    For affine elements:

        d/dx = rx d/dr + sx d/ds
        d/dy = ry d/dr + sy d/ds

    Split form:

        q_x = 1/2 [
            rx q_r + sx q_s
            + D_r(rx q) + D_s(sx q)
        ]

        q_y = 1/2 [
            ry q_r + sy q_s
            + D_r(ry q) + D_s(sy q)
        ]

    Parameters
    ----------
    q:
        Shape (K, Np)

    Dr, Ds:
        Shape (Np, Np)

    rx, sx, ry, sy:
        Shape (K,)

    Returns
    -------
    dqdx, dqdy:
        Shape (K, Np)
    """
    q = np.asarray(q)

    rx_e = rx[:, None]
    sx_e = sx[:, None]
    ry_e = ry[:, None]
    sy_e = sy[:, None]

    qr = (Dr @ q.T).T
    qs = (Ds @ q.T).T

    grad_q_strong_x = rx_e * qr + sx_e * qs
    grad_q_weak_x = (Dr @ (rx_e * q).T).T + (Ds @ (sx_e * q).T).T
    dqdx = 0.5 * (grad_q_strong_x + grad_q_weak_x)

    grad_q_strong_y = ry_e * qr + sy_e * qs
    grad_q_weak_y = (Dr @ (ry_e * q).T).T + (Ds @ (sy_e * q).T).T
    dqdy = 0.5 * (grad_q_strong_y + grad_q_weak_y)

    return dqdx, dqdy


def _expand_volume_quantity_to_nodes(value, shape):
    """
    Expand an elementwise scalar array of shape (K,) or (K,1)
    to volume nodal shape (K,Np).
    """
    value = np.asarray(value)

    if value.ndim == 0:
        return value * np.ones(shape)

    if value.ndim == 1:
        return value[:, None] * np.ones((1, shape[1]))

    if value.ndim == 2 and value.shape[1] == 1:
        return value * np.ones((1, shape[1]))

    return value


def mapped_divergence_split_2d(q, Dr, Ds, xr, xs, yr, ys, J, cx, cy):
    """
    Compute the reference-style split-form mapped divergence.

    Physical advection flux:

        f(q) = (a q, b q)

    Contravariant coefficients:

        alpha = ys a - xs b
        beta  = -yr a + xr b

    Split form:

        div(f) = 1/J * 1/2 [
            D_r(alpha q) + alpha D_r(q) + q D_r(alpha)
            + D_s(beta q) + beta D_s(q) + q D_s(beta)
        ]
    """
    q = np.asarray(q)
    shape = q.shape

    xr_e = _expand_volume_quantity_to_nodes(xr, shape)
    xs_e = _expand_volume_quantity_to_nodes(xs, shape)
    yr_e = _expand_volume_quantity_to_nodes(yr, shape)
    ys_e = _expand_volume_quantity_to_nodes(ys, shape)
    J_e = _expand_volume_quantity_to_nodes(J, shape)

    a = _expand_volume_quantity_to_nodes(cx, shape)
    b = _expand_volume_quantity_to_nodes(cy, shape)

    alpha = ys_e * a - xs_e * b
    beta = -yr_e * a + xr_e * b

    qr = (Dr @ q.T).T
    qs = (Ds @ q.T).T

    alpha_q = alpha * q
    beta_q = beta * q

    Dr_alpha_q = (Dr @ alpha_q.T).T
    Ds_beta_q = (Ds @ beta_q.T).T

    Dr_alpha = (Dr @ alpha.T).T
    Ds_beta = (Ds @ beta.T).T

    div_r = Dr_alpha_q + alpha * qr + q * Dr_alpha
    div_s = Ds_beta_q + beta * qs + q * Ds_beta

    return 0.5 * (div_r + div_s) / J_e


def _edge_boundary_values(engine, q, edge_id):
    """
    Extract boundary values of q on one edge.

    Boundary nodes are ordered as:

        edge0, edge1, edge2, interior

    q shape:
        (K, Np)

    Returns:
        q_edge shape (K, Nfp)
    """
    edge_slice = engine.edge_slices[edge_id]
    return q[:, edge_slice]


def _expand_face_quantity_to_nodes(value, Nfp):
    """
    Expand a facewise scalar array of shape (K,) or (K,1)
    to nodal face shape (K,Nfp).

    If already shape (K,Nfp), return unchanged.
    """
    value = np.asarray(value)

    if value.ndim == 1:
        return value[:, None] * np.ones((1, Nfp))

    if value.ndim == 2 and value.shape[1] == 1:
        return value * np.ones((1, Nfp))

    return value


def compute_split_rhs(q, t, **kwargs):
    """
    Compute SDG split-form RHS for scalar advection.

    PDE:

        q_t + cx q_x + cy q_y = 0

    Volume term:

        rhs_vol = -(cx q_x + cy q_y)

    Surface term:

        + V M^{-1} V^T E^T W_b p

    where

        p = n · (f(q-) - f*)

    and f(q) = V q.

    Required kwargs
    ---------------
    engine:
        ReferenceElement

    xr, xs, yr, ys, rx, sx, ry, sy, J:
        Geometry metrics, shape (K,). xr, xs, yr, ys are used by the
        mapped divergence split form.

    nx, ny:
        Outward unit normals, shape (K,3)

    edge_lengths:
        Physical edge lengths, shape (K,3)

    vmapM, vmapP:
        Global maps, shape (K,3,Nfp)

    cx, cy:
        Constant physical advection velocity

    x_nodes, y_nodes:
        Shape (K,Np), only needed if q_exact is provided

    Optional kwargs
    ---------------
    q_exact:
        Boundary/exact solution function q_exact(x,y,t)

    lift_mode:
        'physical' or 'exact_trace'

        physical:
            Use neighbor values on interior faces and exact values only on inflow
            physical boundaries.

        exact_trace:
            Override qP on every face using exact solution. Useful for manufactured
            diagnostic tests, but not a real DG interface exchange.

    tau:
        0 = upwind
        1 = central

    flux_type:
        "central", "upwind", or "lf". If omitted, defaults to upwind.

    alpha_lf:
        Dissipation multiplier for LF flux.
    """
    engine = kwargs["engine"]

    rx = kwargs["rx"]
    sx = kwargs["sx"]
    ry = kwargs["ry"]
    sy = kwargs["sy"]
    J = kwargs["J"]
    xr = kwargs.get("xr", sy * J)
    xs = kwargs.get("xs", -ry * J)
    yr = kwargs.get("yr", -sx * J)
    ys = kwargs.get("ys", rx * J)

    nx = kwargs["nx"]
    ny = kwargs["ny"]
    edge_lengths = kwargs["edge_lengths"]

    vmapM = kwargs["vmapM"]
    vmapP = kwargs["vmapP"]

    cx = kwargs["cx"]
    cy = kwargs["cy"]

    q_exact = kwargs.get("q_exact", None)
    lift_mode = kwargs.get("lift_mode", "physical")
    tau = kwargs.get("tau", None)
    flux_type = kwargs.get("flux_type", None)
    alpha_lf = kwargs.get("alpha_lf", 1.0)

    if flux_type is None and tau is None:
        flux_type = "upwind"

    q = np.asarray(q)

    if q.ndim != 2:
        raise ValueError("q must have shape (K, Np).")

    K, Np = q.shape

    if Np != engine.num_nodes:
        raise ValueError(
            f"q has Np={Np}, but engine.num_nodes={engine.num_nodes}."
        )

    # ------------------------------------------------------------
    # 1. Volume split-form RHS
    # ------------------------------------------------------------
    rhs = -mapped_divergence_split_2d(
        q=q,
        Dr=engine.Dr,
        Ds=engine.Ds,
        xr=xr,
        xs=xs,
        yr=yr,
        ys=ys,
        J=J,
        cx=cx,
        cy=cy,
    )

    # ------------------------------------------------------------
    # 2. Surface SDG penalty
    # ------------------------------------------------------------
    q_flat = q.reshape(-1)

    for f in range(3):
        edge_slice = engine.edge_slices[f]
        Nfp = engine.num_edge_nodes

        # Interior values q-
        qM = _edge_boundary_values(engine, q, f)

        # Exterior / neighbor values q+
        qP = q_flat[vmapP[:, f, :]]

        # --------------------------------------------------------
        # Boundary condition handling
        # --------------------------------------------------------
        is_boundary = np.all(vmapM[:, f, :] == vmapP[:, f, :], axis=1)

        if q_exact is not None:
            x_nodes = kwargs["x_nodes"]
            y_nodes = kwargs["y_nodes"]

            if lift_mode == "exact_trace":
                # Diagnostic mode: overwrite qP on every face.
                x_face = x_nodes[:, edge_slice]
                y_face = y_nodes[:, edge_slice]
                qP = q_exact(x_face, y_face, t)

            else:
                # Physical mode: only overwrite inflow physical boundaries.
                ndotV_face = cx * nx[:, f] + cy * ny[:, f]
                inflow = is_boundary & (ndotV_face < -1e-12)

                if np.any(inflow):
                    x_bnd = x_nodes[inflow, edge_slice]
                    y_bnd = y_nodes[inflow, edge_slice]
                    qP[inflow, :] = q_exact(x_bnd, y_bnd, t)

        # --------------------------------------------------------
        # Penalty p = n · (f - f*)
        # --------------------------------------------------------
        nx_face = _expand_face_quantity_to_nodes(nx[:, f], Nfp)
        ny_face = _expand_face_quantity_to_nodes(ny[:, f], Nfp)

        u_face = cx * np.ones_like(qM)
        v_face = cy * np.ones_like(qM)

        p_face = compute_boundary_penalty(
            q_minus=qM,
            q_plus=qP,
            nx=nx_face,
            ny=ny_face,
            u=u_face,
            v=v_face,
            tau=tau,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
        )

        # --------------------------------------------------------
        # Lift face penalty element-by-element.
        #
        # engine.lift_boundary_penalty expects all three faces packed as:
        #     [edge0, edge1, edge2]
        #
        # For this loop, only one edge is nonzero.
        # --------------------------------------------------------
        p_all = np.zeros((K, engine.num_boundary_nodes))
        p_all[:, edge_slice] = p_face

        for k in range(K):
            rhs[k, :] += engine.lift_boundary_penalty(
                p_all[k, :],
                edge_lengths=edge_lengths[k, :],
            ) / J[k]

    return rhs
