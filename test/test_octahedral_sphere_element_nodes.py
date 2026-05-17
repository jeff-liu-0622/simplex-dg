import numpy as np

from core.mesh_octahedron import create_octahedral_layout_mesh
from core.geometry.sphere_mapping import map_unit_triangle_to_sphere
from core.operators import build_local_operators


def reference_to_unit_triangle(r, s):
    a = 0.5 * (r + 1.0)
    b = 0.5 * (s + 1.0)
    return a, b


def map_reference_nodes_to_subelement(r, s, tri_vertices):
    a, b = reference_to_unit_triangle(r, s)

    v0 = tri_vertices[0]
    v1 = tri_vertices[1]
    v2 = tri_vertices[2]

    xi = v0[0] + a * (v1[0] - v0[0]) + b * (v2[0] - v0[0])
    eta = v0[1] + a * (v1[1] - v0[1]) + b * (v2[1] - v0[1])

    return xi, eta


def test_all_high_order_nodes_lie_on_sphere():
    nsub = 5
    order = 4
    R = 1.3

    VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

    engine = build_local_operators(N=order, n=order, rule="table1")
    r = engine.r
    s = engine.s

    for k in range(EToV.shape[0]):
        patch0 = patch_ids[k] - 1
        tri_vertices = element_local_coords[k]

        xi, eta = map_reference_nodes_to_subelement(r, s, tri_vertices)
        X, Y, Z = map_unit_triangle_to_sphere(xi, eta, patch0, R=R)

        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y))
        assert np.all(np.isfinite(Z))

        err = np.max(np.abs(X**2 + Y**2 + Z**2 - R**2))
        assert err < 1.0e-12, f"element {k}: radius error = {err:.3e}"


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 octahedral sphere element nodes 測試")
    print("=" * 72)

    test_all_high_order_nodes_lie_on_sphere()
    print("✅ all high-order DG nodes lie on sphere")

    print("=" * 72)
    print("🎉 test_octahedral_sphere_element_nodes.py 全部通過")


if __name__ == "__main__":
    run_all_tests()