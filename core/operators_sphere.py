import numpy as np

from core.rhs_sphere import compute_sphere_surface_penalty

"""
Projected 3D manifold sphere operators.

This module is for the projected octahedron / 3D manifold route. It does not
implement the SDG T1-T8 patch-coordinate route.
"""


def compute_manifold_skew_volume_rhs(engine, geometry, V3D, q):
    """
    Compute the manifold skew-symmetric volume RHS diagnostic.

    This is volume-only: no surface flux, no sphere RHS object, and no time
    integration are introduced here.
    """
    J = geometry["J"]
    a_contra_1 = geometry["a_contra_1"]
    a_contra_2 = geometry["a_contra_2"]

    u_tilde = np.sum(a_contra_1 * V3D, axis=1)
    v_tilde = np.sum(a_contra_2 * V3D, axis=1)

    Dr_q = engine.Dr @ q
    Ds_q = engine.Ds @ q

    div_Jv = engine.Dr @ (J * u_tilde) + engine.Ds @ (J * v_tilde)

    rhs_vol = (
        -0.5 * (
            engine.Dr @ (J * u_tilde * q)
            + engine.Ds @ (J * v_tilde * q)
        )
        -0.5 * (u_tilde * Dr_q + v_tilde * Ds_q)
        -0.5 * q * div_Jv
    )

    divJv_over_J = div_Jv / J

    return rhs_vol, divJv_over_J, u_tilde, v_tilde


def compute_sphere_rhs(
    q,
    t,
    *,
    state,
    flux_type="upwind",
    alpha_lf=1.0,
    surface_mode="conservative_scaled",
):
    """
    Diagnostic projected-sphere RHS callback for LSRK stages.

    Geometry and velocity are fixed. The current nodal q is supplied by the
    time integrator and is used for both volume and surface terms.
    """
    del t

    engine = state["engine"]
    q = np.asarray(q)
    volume_rhs = np.zeros_like(q)

    for k, geometry in enumerate(state["geometry"]):
        rhs_vol, divJv_over_J, u_local, v_local = compute_manifold_skew_volume_rhs(
            engine=engine,
            geometry=geometry,
            V3D=state["V3D"][k],
            q=q[k],
        )
        volume_rhs[k, :] = divJv_over_J
        state["u_tilde"][k, :] = u_local
        state["v_tilde"][k, :] = v_local

    state["q"] = q
    state["volume_rhs"] = volume_rhs

    surface = compute_sphere_surface_penalty(
        state,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        surface_mode=surface_mode,
    )

    return volume_rhs + surface["surface_rhs"]
