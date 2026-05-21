import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.manifold_metrics import compute_manifold_geometry
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
    projected_sphere_mesh_hmin,
)
from core.operators import build_local_operators
from core.operators_sphere import compute_sphere_rhs
from core.time_integration import lsrk54_step
from core.rhs_sphere import build_face_exchange_cache
from core.rhs_sphere import physical_conormal_flux_on_face

def reference_style_omega(u0=1.0, alpha0=0):
    return u0 * np.array(
        [-np.sin(alpha0), 0.0, np.cos(alpha0)],
        dtype=float,
    )


def solid_body_velocity(xyz, u0=1.0, alpha0=0):
    omega = reference_style_omega(u0=u0, alpha0=alpha0)

    return np.cross(omega[None, :], xyz)


def rotate_xyz_rodrigues(xyz, omega, t):
    omega = np.asarray(omega, dtype=float).reshape(3)
    speed = float(np.linalg.norm(omega))
    if speed == 0.0 or t == 0.0:
        return np.asarray(xyz, dtype=float).copy()

    axis = omega / speed
    angle = speed * float(t)
    c = np.cos(angle)
    s = np.sin(angle)
    dot = xyz @ axis
    cross = np.cross(axis[None, :], xyz)

    return xyz * c + cross * s + axis[None, :] * dot[:, None] * (1.0 - c)


def gaussian_bell_reference_style(
    xyz,
    R=1.0,
    center_xyz=(1.0, 0.0, 0.0),
    width=1.0 / np.sqrt(10.0),
):
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    center = R * center / np.linalg.norm(center)
    dot = np.clip((xyz @ center) / (R * R), -1.0, 1.0)
    distance = R * np.arccos(dot)

    return np.exp(-((distance / float(width)) ** 2))


def exact_gaussian_bell_reference_style(
    xyz,
    t,
    u0=1.0,
    alpha0=0,
    R=1.0,
    center_xyz=(1.0, 0.0, 0.0),
    width=1.0 / np.sqrt(10.0),
):
    omega = reference_style_omega(u0=u0, alpha0=alpha0)
    xyz0 = rotate_xyz_rodrigues(xyz, omega=omega, t=-float(t))

    return gaussian_bell_reference_style(
        xyz0,
        R=R,
        center_xyz=center_xyz,
        width=width,
    )


def weighted_integral(state, values):
    engine = state["engine"]
    total = 0.0

    for k, geometry in enumerate(state["geometry"]):
        total += engine.area * np.sum(engine.w_s * geometry["J"] * values[k])

    return float(total)


def weighted_l2_error(state, error):
    area = weighted_integral(state, np.ones_like(error))

    return np.sqrt(weighted_integral(state, error * error) / area)


def build_reference_style_state(
    nsub,
    order=4,
    R=1.0,
    u0=1.0,
    alpha0=-np.pi / 4.0,
    center_xyz=(1.0, 0.0, 0.0),
    width=1.0 / np.sqrt(10.0),
):
    engine = build_local_operators(N=order, n=order, rule="table1")

    _, _, _, EToV, patch_ids, nodes_xyz = create_projected_octahedron_sphere_mesh(
        nsub=nsub,
        R=R,
    )

    EToE, EToF = build_connectivity(EToV)

    element_xyz = map_reference_nodes_to_projected_sphere(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        r=engine.r,
        s=engine.s,
        R=R,
    )

    K = EToV.shape[0]
    Np = engine.num_nodes

    geometries = []
    velocities = []

    q = np.zeros((K, Np))

    # ------------------------------------------------------------
    # Volume RHS cache
    # ------------------------------------------------------------
    J_array = np.zeros((K, Np))
    u_tilde = np.zeros((K, Np))
    v_tilde = np.zeros((K, Np))
    J_u = np.zeros((K, Np))
    J_v = np.zeros((K, Np))
    div_Jv = np.zeros((K, Np))

    for k in range(K):
        xyz = element_xyz[k]

        geometry = compute_manifold_geometry(engine, xyz)
        V3D = solid_body_velocity(xyz, u0=u0, alpha0=alpha0)

        geometries.append(geometry)
        velocities.append(V3D)

        q[k, :] = gaussian_bell_reference_style(
            xyz,
            R=R,
            center_xyz=center_xyz,
            width=width,
        )

        # ------------------------------------------------------------
        # Precompute geometry / velocity quantities.
        # These do not change during time stepping.
        # ------------------------------------------------------------
        J = geometry["J"]
        a_contra_1 = geometry["a_contra_1"]
        a_contra_2 = geometry["a_contra_2"]

        u = np.sum(a_contra_1 * V3D, axis=1)
        v = np.sum(a_contra_2 * V3D, axis=1)

        J_array[k, :] = J
        u_tilde[k, :] = u
        v_tilde[k, :] = v
        J_u[k, :] = J * u
        J_v[k, :] = J * v

        div_Jv[k, :] = engine.Dr @ (J * u) + engine.Ds @ (J * v)

    state = {
        "engine": engine,
        "EToE": EToE,
        "EToF": EToF,
        "patch_ids": patch_ids,
        "xyz": element_xyz,
        "geometry": geometries,
        "V3D": np.asarray(velocities),

        # ------------------------------------------------------------
        # Cached geometry / velocity arrays
        # ------------------------------------------------------------
        "u_tilde": u_tilde,
        "v_tilde": v_tilde,
        "J_array": J_array,
        "J_u": J_u,
        "J_v": J_v,
        "div_Jv": div_Jv,

        "q": q,
        "volume_rhs": np.zeros_like(q),
        "h": projected_sphere_mesh_hmin(nodes_xyz, EToV),
        "R": R,
        "u0": u0,
        "alpha0": alpha0,
        "center_xyz": center_xyz,
        "width": width,
    }

    # ------------------------------------------------------------
    # Face exchange cache.
    # This avoids recomputing face node matching during every RHS call.
    # ------------------------------------------------------------
    state["face_cache"] = build_face_exchange_cache(state)
    Nfp = engine.num_edge_nodes
    vn_face = np.zeros((K, 3, Nfp))

    face_nodes_M = state["face_cache"]["face_nodes_M"]

    for k in range(K):
        for f in range(3):
            nodes = face_nodes_M[k, f]
            vn_face[k, f, :] = physical_conormal_flux_on_face(
                state,
                elem_id=k,
                face_id=f,
                nodes=nodes,
            )

    state["vn_face"] = vn_face
    # ------------------------------------------------------------
    # Boundary lift cache.
    #
    # Old local surface did:
    #     lifted = engine.lift_boundary_penalty(p_boundary, edge_lengths=np.ones(3))
    #
    # That is equivalent to:
    #     lifted = (Wb * p_boundary) @ LIFT_B.T
    #
    # where:
    #     LIFT_B = first boundary columns of V M^{-1} V^T
    #     Wb     = boundary quadrature weights
    # ------------------------------------------------------------
    Nb = engine.num_boundary_nodes

    LIFT = engine.V @ engine.invM_modal @ engine.V.T
    LIFT_B = LIFT[:, :Nb]

    Wb = engine.boundary_weight_diag(edge_lengths=np.ones(3))

    state["LIFT_B"] = LIFT_B
    state["Wb"] = Wb

    return state

def rate(previous_error, current_error, previous_h, current_h):
    if previous_error is None:
        return None

    return np.log(previous_error / current_error) / np.log(previous_h / current_h)


def cfl_dt_from_state(state, cfl=0.25):
    engine = state["engine"]
    N = int(getattr(engine, "N", 4))
    h_min = float(state["h"])
    max_speed = float(np.max(np.linalg.norm(state["V3D"], axis=2)))

    if max_speed <= 0.0:
        dt_raw = np.inf
    else:
        dt_raw = float(cfl * h_min / (((N + 1) ** 2) * max_speed))

    return {
        "CFL": float(cfl),
        "N": N,
        "h_min": h_min,
        "max_speed": max_speed,
        "dt_raw": dt_raw,
    }


def run_reference_style_short_time_case(
    nsub,
    final_time=5.0e-2,
    cfl=1.0,
):
    state = build_reference_style_state(nsub=nsub)
    dt_info = cfl_dt_from_state(state, cfl=cfl)
    steps = max(1, int(np.ceil(final_time / dt_info["dt_raw"])))
    dt = float(final_time / steps)

    q_initial = state["q"].copy()
    q = q_initial.copy()
    res = np.zeros_like(q)
    t = 0.0
    num_steps = 0

    mass_initial = weighted_integral(state, q_initial)
    energy_initial = weighted_integral(state, q_initial * q_initial)

    while t < final_time - 1.0e-15:
        dt_step = min(dt, final_time - t)
        q, res = lsrk54_step(
            q,
            res,
            t,
            dt_step,
            compute_sphere_rhs,
            state=state,
            flux_type="upwind",
            surface_mode="local_fast",
        )
        t += dt_step
        num_steps += 1

    q_exact = np.zeros_like(q)
    for k, xyz in enumerate(state["xyz"]):
        q_exact[k, :] = exact_gaussian_bell_reference_style(
            xyz,
            t=final_time,
            u0=state["u0"],
            alpha0=state["alpha0"],
            R=state["R"],
            center_xyz=state["center_xyz"],
            width=state["width"],
        )

    state["q"] = q
    error = q - q_exact
    mass_final = weighted_integral(state, q)
    energy_final = weighted_integral(state, q * q)

    return {
        "nsub": nsub,
        "K": int(q.shape[0]),
        "h": float(state["h"]),
        "CFL": dt_info["CFL"],
        "h_min": dt_info["h_min"],
        "max_speed": dt_info["max_speed"],
        "dt_raw": dt_info["dt_raw"],
        "dt": dt,
        "num_steps": num_steps,
        "L2_error": float(weighted_l2_error(state, error)),
        "max_error": float(np.max(np.abs(error))),
        "mass_error": float(mass_final - mass_initial),
        "energy_change": float(energy_final - energy_initial),
        "has_nonfinite": bool(not np.all(np.isfinite(q))),
    }


def run_reference_style_table(final_time, title, cfl=1.0):
    levels = [2, 4, 8, 16, 32]
    previous = None
    results = []

    print("\n" + "=" * 132, flush=True)
    print(
        f"{title} | projected 3D manifold | LSRK54 | upwind",
        flush=True,
    )
    print("=" * 132, flush=True)
    print(
        "q0=exp(-(dist/width)^2), width=1/sqrt(10), center=(1,0,0), "
        f"Omega=(-sin(-pi/4),0,cos(-pi/4)), T_final={final_time:g}",
        flush=True,
    )
    print(
        f"{'nsub':>8s} {'K':>8s} {'h':>13s} {'CFL':>8s} "
        f"{'h_min':>13s} {'max_speed':>12s} {'dt':>12s} "
        f"{'steps':>8s} {'L2_error':>16s} {'L2_rate':>10s} "
        f"{'max_error':>16s} {'max_rate':>10s} "
        f"{'mass_error':>16s} {'energy_change':>16s}",
        flush=True,
    )
    print("-" * 132, flush=True)

    for nsub in levels:
        row = run_reference_style_short_time_case(
            nsub=nsub,
            final_time=final_time,
            cfl=cfl,
        )
        L2_rate = rate(
            None if previous is None else previous["L2_error"],
            row["L2_error"],
            None if previous is None else previous["h"],
            row["h"],
        )
        max_rate = rate(
            None if previous is None else previous["max_error"],
            row["max_error"],
            None if previous is None else previous["h"],
            row["h"],
        )

        print(
            f"{row['nsub']:8d} {row['K']:8d} {row['h']:13.6e} "
            f"{row['CFL']:8.3f} "
            f"{row['h_min']:13.6e} "
            f"{row['max_speed']:12.6e} "
            f"{row['dt']:12.6e} {row['num_steps']:8d} "
            f"{row['L2_error']:16.6e} "
            f"{'---' if L2_rate is None else f'{L2_rate:.4f}':>10s} "
            f"{row['max_error']:16.6e} "
            f"{'---' if max_rate is None else f'{max_rate:.4f}':>10s} "
            f"{row['mass_error']:16.6e} "
            f"{row['energy_change']:16.6e}",
            flush=True,
        )

        results.append(row)
        previous = row

    print("-" * 132, flush=True)

    for row in results:
        assert not row["has_nonfinite"]
        assert np.isfinite(row["L2_error"])
        assert np.isfinite(row["max_error"])
        assert np.isfinite(row["mass_error"])
        assert np.isfinite(row["energy_change"])

    assert results[-1]["L2_error"] < results[0]["L2_error"]
    assert results[-1]["max_error"] < results[0]["max_error"]


def test_sphere_gaussian_bell_reference_style_t05_cfl1():
    run_reference_style_table(
        final_time=5.0e-1,
        title="sphere Gaussian bell reference-style T=0.5 CFL=1 diagnostic",
        cfl=1.0,
    )


def run_all_tests():
    test_sphere_gaussian_bell_reference_style_t05_cfl1()
    print("sphere Gaussian bell reference-style T=0.5 CFL=1 diagnostic passed")


if __name__ == "__main__":
    run_all_tests()
