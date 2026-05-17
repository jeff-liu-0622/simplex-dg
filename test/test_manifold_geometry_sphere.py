import numpy as np

from core.mesh_octahedron import create_octahedral_layout_mesh
from core.geometry.sphere_mapping import map_unit_triangle_to_sphere
from core.operators import build_local_operators


def map_reference_nodes_to_subelement(r, s, tri_vertices):
    """
    Map reference triangle nodes (r,s) to one subelement in the parent
    unit-triangle coordinates (xi, eta).
    """
    a = 0.5 * (r + 1.0)
    b = 0.5 * (s + 1.0)

    v0 = tri_vertices[0]
    v1 = tri_vertices[1]
    v2 = tri_vertices[2]

    xi = v0[0] + a * (v1[0] - v0[0]) + b * (v2[0] - v0[0])
    eta = v0[1] + a * (v1[1] - v0[1]) + b * (v2[1] - v0[1])

    return xi, eta


def compute_manifold_geometry(engine, xyz_nodes):
    """
    Compute local manifold geometry directly from 3D nodal coordinates.

    This is a diagnostic helper only. It intentionally does not build a
    sphere RHS or alter the planar DG implementation.
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


def build_octahedral_sphere_diagnostics(nsub=4, order=4, R=1.0):
    engine = build_local_operators(N=order, n=order, rule="table1")
    _, _, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

    diagnostics = []

    for k in range(EToV.shape[0]):
        patch0 = patch_ids[k] - 1
        tri_vertices = element_local_coords[k]

        xi, eta = map_reference_nodes_to_subelement(
            engine.r,
            engine.s,
            tri_vertices,
        )

        X, Y, Z = map_unit_triangle_to_sphere(xi, eta, patch0, R=R)
        xyz_nodes = np.column_stack([X, Y, Z])

        geometry = compute_manifold_geometry(engine, xyz_nodes)
        diagnostics.append((xyz_nodes, geometry))

    return engine, diagnostics


def test_sphere_nodes_lie_on_radius_R():
    R = 1.7
    _, diagnostics = build_octahedral_sphere_diagnostics(nsub=4, order=4, R=R)

    for k, (xyz_nodes, _) in enumerate(diagnostics):
        radius = np.linalg.norm(xyz_nodes, axis=1)
        err = np.max(np.abs(radius - R))
        assert err < 1.0e-12, f"element {k}: radius error = {err:.3e}"


def test_covariant_basis_surface_jacobian_and_normal_are_valid():
    _, diagnostics = build_octahedral_sphere_diagnostics(nsub=4, order=4, R=1.0)

    for k, (_, geometry) in enumerate(diagnostics):
        a1 = geometry["a1"]
        a2 = geometry["a2"]
        J = geometry["J"]
        n = geometry["n"]

        assert np.all(np.isfinite(a1)), f"element {k}: non-finite a1"
        assert np.all(np.isfinite(a2)), f"element {k}: non-finite a2"
        assert np.all(np.isfinite(J)), f"element {k}: non-finite J"
        assert np.all(J > 0.0), f"element {k}: non-positive J"
        assert np.all(np.isfinite(n)), f"element {k}: non-finite normal"

        normal_length_error = np.max(np.abs(np.linalg.norm(n, axis=1) - 1.0))
        assert normal_length_error < 1.0e-12, (
            f"element {k}: normal length error = {normal_length_error:.3e}"
        )


def test_contravariant_basis_is_biorthogonal_to_covariant_basis():
    _, diagnostics = build_octahedral_sphere_diagnostics(nsub=4, order=4, R=1.0)

    max_error = 0.0

    for _, geometry in diagnostics:
        a1 = geometry["a1"]
        a2 = geometry["a2"]
        ac1 = geometry["a_contra_1"]
        ac2 = geometry["a_contra_2"]

        errors = [
            np.abs(np.sum(ac1 * a1, axis=1) - 1.0),
            np.abs(np.sum(ac1 * a2, axis=1)),
            np.abs(np.sum(ac2 * a1, axis=1)),
            np.abs(np.sum(ac2 * a2, axis=1) - 1.0),
        ]

        max_error = max(max_error, max(np.max(err) for err in errors))

    assert max_error < 1.0e-12, (
        f"contravariant/covariant biorthogonality error too large: {max_error:.3e}"
    )


def test_global_surface_area_approximates_sphere_area():
    R = 1.0
    engine, diagnostics = build_octahedral_sphere_diagnostics(nsub=4, order=4, R=R)

    total_area = 0.0

    for _, geometry in diagnostics:
        total_area += engine.area * np.sum(engine.w_s * geometry["J"])

    exact_area = 4.0 * np.pi * R**2
    relative_error = abs(total_area - exact_area) / exact_area

    print(
        "manifold geometry sphere area: "
        f"computed={total_area:.12e}, exact={exact_area:.12e}, "
        f"relerr={relative_error:.6e}"
    )

    assert relative_error < 1.0e-3, (
        f"sphere area relative error too large: {relative_error:.3e}"
    )


def run_all_tests():
    print("\n" + "=" * 80)
    print("Manifold sphere geometry diagnostic")
    print("=" * 80)

    test_sphere_nodes_lie_on_radius_R()
    print("sphere nodes lie on radius R")

    test_covariant_basis_surface_jacobian_and_normal_are_valid()
    print("covariant basis, surface Jacobian, and normal are valid")

    test_contravariant_basis_is_biorthogonal_to_covariant_basis()
    print("contravariant basis is biorthogonal to covariant basis")

    test_global_surface_area_approximates_sphere_area()
    print("global surface area approximates 4*pi*R^2")

    print("=" * 80)
    print("test_manifold_geometry_sphere.py passed")


if __name__ == "__main__":
    run_all_tests()
