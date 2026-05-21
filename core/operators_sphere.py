import numpy as np

from core.rhs_sphere import compute_sphere_surface_penalty
from core.operators import compute_manifold_volume_rhs_fast
"""
Projected 3D manifold sphere operators.

This module is for the projected octahedron / 3D manifold route. It does not
implement the SDG T1-T8 patch-coordinate route.
"""


def compute_manifold_skew_volume_rhs(engine, geometry, V3D, q):
    """
    Compute one-element manifold split-form volume RHS.

    Returns
    -------
    rhs_Jq:
        J-weighted volume RHS, i.e. J * q_t(volume).

    rhs_q:
        Physical nodal RHS q_t(volume).

    u_tilde, v_tilde:
        Contravariant velocity components.
    """
    J = geometry["J"]
    a_contra_1 = geometry["a_contra_1"]
    a_contra_2 = geometry["a_contra_2"]

    q = np.asarray(q)
    J = np.asarray(J)

    if q.ndim != 1:
        raise ValueError(f"q must have shape (Np,), got {q.shape}.")
    if J.shape != q.shape:
        raise ValueError(f"J shape {J.shape} must match q shape {q.shape}.")
    if np.any(J <= 1.0e-14):
        raise ValueError("Degenerate manifold geometry: J too small.")

    u_tilde = np.sum(a_contra_1 * V3D, axis=1)
    v_tilde = np.sum(a_contra_2 * V3D, axis=1)

    Dr_q = engine.Dr @ q
    Ds_q = engine.Ds @ q

    div_Jv = engine.Dr @ (J * u_tilde) + engine.Ds @ (J * v_tilde)

    rhs_Jq = (
        -0.5 * (
            engine.Dr @ (J * u_tilde * q)
            + engine.Ds @ (J * v_tilde * q)
        )
        -0.5 * (J * u_tilde * Dr_q + J * v_tilde * Ds_q)
        -0.5 * q * div_Jv
    )

    rhs_q = rhs_Jq / J

    return rhs_Jq, rhs_q, u_tilde, v_tilde


def compute_sphere_rhs(
    q,
    t,
    *,
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    surface_mode="local",
):
    del t

    q = np.asarray(q)

    volume_rhs = compute_manifold_volume_rhs_fast(q, state)

    state["q"] = q
    state["volume_rhs"] = volume_rhs

    surface = compute_sphere_surface_penalty(
        state,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        surface_mode=surface_mode,
    )

    return volume_rhs + surface["surface_rhs"]
