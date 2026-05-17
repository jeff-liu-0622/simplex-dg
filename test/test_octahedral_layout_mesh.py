import numpy as np

from core.mesh_octahedron import create_octahedral_layout_mesh


def triangle_area(x, y):
    x0, x1, x2 = x
    y0, y1, y2 = y

    return 0.5 * (
        (x1 - x0) * (y2 - y0)
        - (x2 - x0) * (y1 - y0)
    )


def test_element_count():
    for nsub in [1, 2, 3, 4, 5]:
        VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

        expected = 8 * nsub**2
        actual = EToV.shape[0]

        assert actual == expected, (
            f"nsub={nsub}: expected {expected} elements, got {actual}"
        )


def test_patch_counts():
    for nsub in [1, 2, 4, 5]:
        VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

        for pid in range(1, 9):
            count = np.sum(patch_ids == pid)
            expected = nsub**2

            assert count == expected, (
                f"patch {pid}, nsub={nsub}: expected {expected}, got {count}"
            )


def test_positive_area():
    for nsub in [1, 2, 4, 5]:
        VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

        areas = []

        for elem in EToV:
            x = VX[elem]
            y = VY[elem]

            areas.append(triangle_area(x, y))

        areas = np.array(areas)

        assert np.all(areas > 0.0), "Found non-positive area element."


def test_total_area():
    """
    The unfolded layout is the square [-1,1]^2, total area 4.
    The 8 triangular patches each have area 1/2.
    """
    for nsub in [1, 2, 4, 5]:
        VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

        total_area = 0.0

        for elem in EToV:
            x = VX[elem]
            y = VY[elem]
            total_area += triangle_area(x, y)

        assert abs(total_area - 4.0) < 1.0e-12, (
            f"nsub={nsub}: total_area={total_area}"
        )


def test_local_coords_inside_unit_triangle():
    for nsub in [1, 2, 4, 5]:
        VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

        xi = element_local_coords[:, :, 0]
        eta = element_local_coords[:, :, 1]

        assert np.all(xi >= -1.0e-14)
        assert np.all(eta >= -1.0e-14)
        assert np.all(xi + eta <= 1.0 + 1.0e-14)


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 octahedral unfolded layout mesh 測試")
    print("=" * 72)

    test_element_count()
    print("✅ element count correct")

    test_patch_counts()
    print("✅ patch element counts correct")

    test_positive_area()
    print("✅ all elements have positive area")

    test_total_area()
    print("✅ total unfolded area = 4")

    test_local_coords_inside_unit_triangle()
    print("✅ local coordinates inside unit triangle")

    print("=" * 72)
    print("🎉 test_octahedral_layout_mesh.py 全部通過")


if __name__ == "__main__":
    run_all_tests()