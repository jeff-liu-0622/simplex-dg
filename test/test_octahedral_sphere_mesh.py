import numpy as np

from core.mesh_octahedron import create_octahedral_layout_mesh
from core.geometry.sphere_mapping import map_unit_triangle_to_sphere


def triangle_area_3d(p0, p1, p2):
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def test_all_vertices_on_sphere():
    for nsub in [1, 2, 4, 5]:
        VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

        R = 1.7

        for k in range(EToV.shape[0]):
            pid = patch_ids[k] - 1
            loc = element_local_coords[k]

            xi = loc[:, 0]
            eta = loc[:, 1]

            X, Y, Z = map_unit_triangle_to_sphere(xi, eta, pid, R=R)

            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(Y))
            assert np.all(np.isfinite(Z))

            err = np.max(np.abs(X**2 + Y**2 + Z**2 - R**2))
            assert err < 1.0e-12, f"nsub={nsub}, elem={k}, radius error={err:.3e}"


def test_all_spherical_triangles_have_positive_area():
    for nsub in [1, 2, 4, 5]:
        VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

        R = 1.0
        areas = []

        for k in range(EToV.shape[0]):
            pid = patch_ids[k] - 1
            loc = element_local_coords[k]

            xi = loc[:, 0]
            eta = loc[:, 1]

            X, Y, Z = map_unit_triangle_to_sphere(xi, eta, pid, R=R)

            p0 = np.array([X[0], Y[0], Z[0]])
            p1 = np.array([X[1], Y[1], Z[1]])
            p2 = np.array([X[2], Y[2], Z[2]])

            area = triangle_area_3d(p0, p1, p2)
            areas.append(area)

        areas = np.array(areas)
        assert np.all(areas > 0.0), f"nsub={nsub}: found non-positive spherical triangle area"


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 octahedral sphere mesh 測試")
    print("=" * 72)

    test_all_vertices_on_sphere()
    print("✅ all mapped vertices lie on sphere")

    test_all_spherical_triangles_have_positive_area()
    print("✅ all mapped spherical triangles have positive area")

    print("=" * 72)
    print("🎉 test_octahedral_sphere_mesh.py 全部通過")


if __name__ == "__main__":
    run_all_tests()