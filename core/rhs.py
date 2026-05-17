import numpy as np


def compute_upwind_flux(q_minus, q_plus, nx, ny, u, v):
    """
    Compute scalar upwind numerical flux for advection.

    Physical flux:
        f(q) = V q = (u q, v q)

    Normal numerical flux:
        n · f* =
            0.5 * (n·V) * (q^- + q^+)
            + 0.5 * |n·V| * (q^- - q^+)

    Parameters
    ----------
    q_minus:
        Interior boundary value q^-.

    q_plus:
        Exterior / neighbor / boundary value q^+.

    nx, ny:
        Outward unit normal components.

    u, v:
        Velocity components at boundary nodes.

    Returns
    -------
    flux_star_n:
        Normal numerical flux n · f*.
    """
    ndotV = nx * u + ny * v

    return (
        0.5 * ndotV * (q_minus + q_plus)
        + 0.5 * np.abs(ndotV) * (q_minus - q_plus)
    )


def _normalize_flux_type(flux_type):
    if flux_type is None:
        return "upwind"

    return str(flux_type).lower()


def compute_normal_flux(
    q_minus,
    q_plus,
    nx,
    ny,
    u,
    v,
    flux_type="upwind",
    alpha_lf=1.0,
    tau=None,
):
    """
    Compute scalar normal numerical flux n dot f* for advection.

    flux_type:
        "central" -> C = 0
        "upwind"  -> C = |n dot V|
        "lf"      -> C = alpha_lf |n dot V|

    tau:
        Backward-compatible old interface. If tau is not None and the
        requested flux is not LF, use C = (1 - tau) |n dot V|.
    """
    ndotV = nx * u + ny * v
    flux_type = _normalize_flux_type(flux_type)

    if tau is not None and flux_type not in (
        "lf",
        "lax-friedrichs",
        "lax_friedrichs",
        "rusanov",
    ):
        C = (1.0 - tau) * np.abs(ndotV)
    elif flux_type in ("central", "centered"):
        C = 0.0
    elif flux_type == "upwind":
        C = np.abs(ndotV)
    elif flux_type in ("lf", "lax-friedrichs", "lax_friedrichs", "rusanov"):
        C = alpha_lf * np.abs(ndotV)
    else:
        raise ValueError(
            "Unknown flux_type "
            f"{flux_type!r}. Expected 'central', 'upwind', or 'lf'."
        )

    return 0.5 * ndotV * (q_minus + q_plus) + 0.5 * C * (q_minus - q_plus)


def compute_boundary_penalty(
    q_minus,
    q_plus,
    nx,
    ny,
    u,
    v,
    tau=None,
    flux_type="upwind",
    alpha_lf=1.0,
):
    """
    Compute SDG boundary penalty p.

    In the SDG notes, the penalty is:

        p = n · (f(q^-) - f*(q^-, q^+))

    where

        f(q^-) = V q^-

    and

        n · f* =
            0.5 * (n·V) * (q^- + q^+)
            + 0.5 * (1 - tau) * |n·V| * (q^- - q^+)

    tau = 0 gives upwind flux.
    tau = 1 gives central flux.

    Returns
    -------
    p:
        Boundary penalty vector.
    """
    ndotV = nx * u + ny * v

    flux_internal_n = ndotV * q_minus

    flux_star_n = compute_normal_flux(
        q_minus=q_minus,
        q_plus=q_plus,
        nx=nx,
        ny=ny,
        u=u,
        v=v,
        tau=tau,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
    )

    return flux_internal_n - flux_star_n


def compute_split_volume_rhs(engine, q, u, v):
    """
    Compute SDG split-form volume RHS on reference triangle.

    Scheme:

        q_t =
          -1/2 [ Dr(u q) + Ds(v q) ]
          -1/2 [ u Dr(q) + v Ds(q) ]
          -1/2 [ Dr(u) + Ds(v) ] q

    Here Dr and Ds differentiate with respect to reference coordinates r,s.

    Therefore u and v must be the contravariant / reference-coordinate
    velocities corresponding to r,s.

    Parameters
    ----------
    engine:
        ReferenceElement object.

    q:
        Nodal solution values, shape (Np,).

    u, v:
        Velocity components in reference coordinates, shape (Np,).

    Returns
    -------
    rhs_vol:
        Volume contribution to q_t.
    """
    Dr = engine.Dr
    Ds = engine.Ds

    uq = u * q
    vq = v * q

    rhs_vol = (
        -0.5 * (Dr @ uq + Ds @ vq)
        -0.5 * (u * (Dr @ q) + v * (Ds @ q))
        -0.5 * ((Dr @ u + Ds @ v) * q)
    )

    return rhs_vol

def compute_volume_divergence(engine, Fx, Fy, rx, sx, ry, sy, J):
    """
    Compute conservative-form physical divergence.

    Formula:

        div F =
        1/J * [
            Dr( J * (rx Fx + ry Fy) )
            + Ds( J * (sx Fx + sy Fy) )
        ]

    Parameters
    ----------
    engine:
        ReferenceElement with Dr, Ds.

    Fx, Fy:
        Physical flux components at nodes.

    rx, sx, ry, sy, J:
        Geometry metrics. Can be scalars or arrays broadcastable to Fx/Fy.

    Returns
    -------
    div_F:
        Numerical divergence at nodes.
    """
    flux_r = J * (rx * Fx + ry * Fy)
    flux_s = J * (sx * Fx + sy * Fy)

    div_F = (engine.Dr @ flux_r + engine.Ds @ flux_s) / J

    return div_F

def compute_sdg_rhs_single_element(
    engine,
    q,
    u,
    v,
    q_plus_boundary,
    nx,
    ny,
    u_boundary=None,
    v_boundary=None,
    tau=None,
    flux_type="upwind",
    alpha_lf=1.0,
    edge_lengths=None,
):
    """
    Compute full SDG RHS for one element:

        RHS = volume split form + lifted boundary penalty

    Boundary penalty:

        V M^{-1} V^T E^T W_b p

    where

        p = n · (f(q^-) - f*)

    Parameters
    ----------
    engine:
        ReferenceElement.

    q:
        Nodal values on this element, shape (num_nodes,).

    u, v:
        Reference-coordinate velocity values at volume nodes.

    q_plus_boundary:
        Exterior / boundary values at boundary nodes,
        shape (num_boundary_nodes,).

    nx, ny:
        Outward normal components at boundary nodes,
        shape (num_boundary_nodes,).

        These normals must match the same coordinate system used by u,v.

    u_boundary, v_boundary:
        Optional velocity values at boundary nodes.
        If None, they are extracted from u, v.

    tau:
        0 = upwind.
        1 = central.

    edge_lengths:
        Optional physical/reference edge lengths passed to engine.lift_boundary_penalty.

    Returns
    -------
    rhs:
        Full RHS q_t, shape (num_nodes,).
    """
    q = np.asarray(q)
    u = np.asarray(u)
    v = np.asarray(v)

    if q.shape[0] != engine.num_nodes:
        raise ValueError(f"q must have length {engine.num_nodes}.")

    rhs_vol = compute_split_volume_rhs(engine, q, u, v)

    q_minus_boundary = engine.boundary_values(q)

    if u_boundary is None:
        u_boundary = engine.boundary_values(u)
    if v_boundary is None:
        v_boundary = engine.boundary_values(v)

    p_boundary = compute_boundary_penalty(
        q_minus=q_minus_boundary,
        q_plus=q_plus_boundary,
        nx=nx,
        ny=ny,
        u=u_boundary,
        v=v_boundary,
        tau=tau,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
    )

    rhs_surf = engine.lift_boundary_penalty(
        p_boundary,
        edge_lengths=edge_lengths,
    )

    return rhs_vol + rhs_surf
