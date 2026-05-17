import numpy as np

from core.operators import build_local_operators
from core.mesh import create_unit_triangle_mesh
from core.geometry.sphere_velocity import (
    solid_body_contravariant_velocity_face1_regularized,
)


def map_reference_nodes_to_subelement(r, s, tri_vertices):
    """
    Map reference triangle nodes (r,s) to one subelement
    in parent unit-triangle coordinates (xi, eta).

    Reference triangle:
        (-1,-1), (1,-1), (-1,1)

    Unit triangle coordinates on one subelement:
        tri_vertices[0] -> local vertex 0
        tri_vertices[1] -> local vertex 1
        tri_vertices[2] -> local vertex 2
    """
    a = 0.5 * (r + 1.0)
    b = 0.5 * (s + 1.0)

    v0 = tri_vertices[0]
    v1 = tri_vertices[1]
    v2 = tri_vertices[2]

    xi = v0[0] + a * (v1[0] - v0[0]) + b * (v2[0] - v0[0])
    eta = v0[1] + a * (v1[1] - v0[1]) + b * (v2[1] - v0[1])

    return xi, eta


def local_reference_to_parent_metrics(tri_vertices):
    """
    For affine map:

        (r,s) -> (xi, eta)

    compute inverse metrics:

        d/dxi  = r_xi  d/dr + s_xi  d/ds
        d/deta = r_eta d/dr + s_eta d/ds

    The affine map is:

        xi  = xi0  + 0.5(r+1)(xi1-xi0)  + 0.5(s+1)(xi2-xi0)
        eta = eta0 + 0.5(r+1)(eta1-eta0) + 0.5(s+1)(eta2-eta0)
    """
    v0 = tri_vertices[0]
    v1 = tri_vertices[1]
    v2 = tri_vertices[2]

    xi_r = 0.5 * (v1[0] - v0[0])
    xi_s = 0.5 * (v2[0] - v0[0])

    eta_r = 0.5 * (v1[1] - v0[1])
    eta_s = 0.5 * (v2[1] - v0[1])

    J = xi_r * eta_s - xi_s * eta_r

    if J <= 0.0:
        raise ValueError(f"Non-positive local Jacobian: J={J}")

    r_xi = eta_s / J
    s_xi = -eta_r / J

    r_eta = -xi_s / J
    s_eta = xi_r / J

    return r_xi, s_xi, r_eta, s_eta


def compute_face1_divergence_error(
    nsub,
    N=4,
    alpha=np.pi / 4.0,
    exclude_pole_touching_elements=False,
):
    """
    Compute divergence error on the first octahedron face.

    Continuous target:

        d_xi u^xi + d_eta u^eta = 0

    Parameters
    ----------
    nsub:
        Number of uniform subdivisions per edge of the first octahedron face.

    N:
        Polynomial degree used by the local differentiation matrix.

    alpha:
        Solid-body rotation tilt angle.

    exclude_pole_touching_elements:
        If True, skip elements whose vertices touch rho = xi + eta = 0,
        i.e. the north pole of the first octahedron face.

    Returns
    -------
    max_err:
        Max norm of divergence error over all retained elements.

    l2_err:
        Weighted L2 average of divergence error over all retained elements.

    kept_count:
        Number of retained subelements.

    skipped_count:
        Number of skipped subelements.
    """
    engine = build_local_operators(N=N, n=N, rule="table1")

    r = engine.r
    s = engine.s

    VX, VY, EToV = create_unit_triangle_mesh(nsub)

    max_err = 0.0
    l2_num = 0.0
    l2_den = 0.0

    kept_count = 0
    skipped_count = 0

    for elem in EToV:
        tri_vertices = np.column_stack([VX[elem], VY[elem]])

        # rho = xi + eta.
        # rho = 0 is the north pole on the first face.
        rho_vertices = np.sum(tri_vertices, axis=1)

        if exclude_pole_touching_elements:
            if np.min(rho_vertices) < 1.0e-14:
                skipped_count += 1
                continue

        kept_count += 1

        xi, eta = map_reference_nodes_to_subelement(
            r,
            s,
            tri_vertices,
        )

        u_xi, u_eta = solid_body_contravariant_velocity_face1_regularized(
            xi,
            eta,
            R=1.0,
            u0=1.0,
            alpha=alpha,
        )

        r_xi, s_xi, r_eta, s_eta = local_reference_to_parent_metrics(
            tri_vertices
        )

        duxi_dxi = r_xi * (engine.Dr @ u_xi) + s_xi * (engine.Ds @ u_xi)
        dueta_deta = r_eta * (engine.Dr @ u_eta) + s_eta * (engine.Ds @ u_eta)

        div = duxi_dxi + dueta_deta

        elem_max_err = np.max(np.abs(div))
        max_err = max(max_err, elem_max_err)

        # Area of each subtriangle in the parent unit triangle.
        # Unit triangle total area = 1/2.
        # There are nsub^2 subtriangles.
        local_area = 0.5 / (nsub**2)

        l2_num += local_area * np.sum(engine.w_s * div**2)
        l2_den += local_area * np.sum(engine.w_s)

    if l2_den == 0.0:
        return np.nan, np.nan, kept_count, skipped_count

    l2_err = np.sqrt(l2_num / l2_den)

    return max_err, l2_err, kept_count, skipped_count


def run_one_case(exclude_pole_touching_elements):
    """
    Run h-refinement divergence diagnostic for one setting:
        include pole elements
    or
        exclude pole-touching elements.
    """
    N = 4
    alpha = np.pi / 4.0

    if exclude_pole_touching_elements:
        title = "exclude elements touching north pole"
        nsubs = [2, 4, 8, 16, 32]
    else:
        title = "include all elements"
        nsubs = [1, 2, 4, 8, 16]

    print("\n" + "=" * 94)
    print("First octahedron face divergence h-refinement test")
    print("regularized A^{-1}, q=1, solid-body velocity")
    print(f"mode: {title}")
    print("=" * 94)

    prev_max = None
    prev_l2 = None

    print(
        f"{'nsub':>8s} "
        f"{'h':>12s} "
        f"{'max |div|':>16s} "
        f"{'rate':>8s} "
        f"{'L2 div':>16s} "
        f"{'rate':>8s} "
        f"{'kept':>8s} "
        f"{'skip':>8s}"
    )
    print("-" * 94)

    for nsub in nsubs:
        h = 1.0 / nsub

        max_err, l2_err, kept_count, skipped_count = compute_face1_divergence_error(
            nsub=nsub,
            N=N,
            alpha=alpha,
            exclude_pole_touching_elements=exclude_pole_touching_elements,
        )

        if not np.isfinite(max_err) or not np.isfinite(l2_err):
            max_rate = "-"
            l2_rate = "-"
        elif prev_max is None:
            max_rate = "-"
            l2_rate = "-"
        else:
            max_rate_val = np.log(prev_max / max_err) / np.log(2.0)
            l2_rate_val = np.log(prev_l2 / l2_err) / np.log(2.0)

            max_rate = f"{max_rate_val:.3f}"
            l2_rate = f"{l2_rate_val:.3f}"

        print(
            f"{nsub:8d} "
            f"{h:12.4e} "
            f"{max_err:16.6e} "
            f"{max_rate:>8s} "
            f"{l2_err:16.6e} "
            f"{l2_rate:>8s} "
            f"{kept_count:8d} "
            f"{skipped_count:8d}"
        )

        if np.isfinite(max_err) and np.isfinite(l2_err):
            prev_max = max_err
            prev_l2 = l2_err

    print("-" * 94)


def run_h_refinement_test():
    """
    Run both diagnostics:

    1. Include all elements.
       This usually exposes the pole singular / non-smooth behavior.

    2. Exclude elements touching the north pole.
       This checks whether the divergence error is mainly localized
       at the pole.
    """
    run_one_case(exclude_pole_touching_elements=False)
    run_one_case(exclude_pole_touching_elements=True)

    print("\n✅ face1 divergence h-refinement diagnostics finished")


if __name__ == "__main__":
    run_h_refinement_test()