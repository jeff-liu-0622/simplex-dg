import numpy as np

from core.mesh import create_unit_triangle_mesh


def triangle_area(x, y):
    x0, x1, x2 = x
    y0, y1, y2 = y

    return 0.5 * (
        (x1 - x0) * (y2 - y0)
        - (x2 - x0) * (y1 - y0)
    )


def test_triangle_count():
    for n_div in [1, 2, 3, 4, 8]:
        VX, VY, EToV = create_unit_triangle_mesh(n_div)

        expected = n_div**2
        actual = EToV.shape[0]

        assert actual == expected, (
            f"n_div={n_div}: expected {expected} triangles, got {actual}"
        )


def test_all_elements_positive_area():
    for n_div in [1, 2, 3, 4, 8]:
        VX, VY, EToV = create_unit_triangle_mesh(n_div)

        areas = []

        for elem in EToV:
            x = VX[elem]
            y = VY[elem]

            areas.append(triangle_area(x, y))

        areas = np.array(areas)

        assert np.all(areas > 0.0), (
            f"n_div={n_div}: found non-positive triangle area"
        )


def test_total_area():
    for n_div in [1, 2, 3, 4, 8]:
        VX, VY, EToV = create_unit_triangle_mesh(n_div)

        total_area = 0.0

        for elem in EToV:
            x = VX[elem]
            y = VY[elem]

            total_area += triangle_area(x, y)

        assert abs(total_area - 0.5) < 1.0e-12, (
            f"n_div={n_div}: total_area={total_area}"
        )


def test_vertices_inside_unit_triangle():
    for n_div in [1, 2, 3, 4, 8]:
        VX, VY, EToV = create_unit_triangle_mesh(n_div)

        assert np.all(VX >= -1.0e-14)
        assert np.all(VY >= -1.0e-14)
        assert np.all(VX + VY <= 1.0 + 1.0e-14)


def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 unit triangle mesh 測試")
    print("=" * 72)

    test_triangle_count()
    print("✅ triangle count correct")

    test_all_elements_positive_area()
    print("✅ all triangle areas positive")

    test_total_area()
    print("✅ total area = 1/2")

    test_vertices_inside_unit_triangle()
    print("✅ all vertices inside unit triangle")

    print("=" * 72)
    print("🎉 test_unit_triangle_mesh.py 全部通過")


if __name__ == "__main__":
    run_all_tests()