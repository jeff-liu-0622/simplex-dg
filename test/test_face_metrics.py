import numpy as np

from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics


def test_reference_triangle_metrics():
    """
    Triangle:
        v0 = (-1,-1)
        v1 = ( 1,-1)
        v2 = (-1, 1)

    This is exactly the reference triangle.
    """
    x = np.array([[-1.0, 1.0, -1.0]])
    y = np.array([[-1.0, -1.0, 1.0]])

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(x, y)

    assert abs(J[0] - 1.0) < 1e-12
    assert abs(xr[0] - 1.0) < 1e-12
    assert abs(xs[0] - 0.0) < 1e-12
    assert abs(yr[0] - 0.0) < 1e-12
    assert abs(ys[0] - 1.0) < 1e-12
    assert abs(rx[0] - 1.0) < 1e-12
    assert abs(sx[0] - 0.0) < 1e-12
    assert abs(ry[0] - 0.0) < 1e-12
    assert abs(sy[0] - 1.0) < 1e-12


def test_reference_triangle_face_metrics():
    x = np.array([[-1.0, 1.0, -1.0]])
    y = np.array([[-1.0, -1.0, 1.0]])

    nx, ny, edge_lengths, sJ = compute_face_metrics(x, y)

    expected_lengths = np.array([2.0, 2.0 * np.sqrt(2.0), 2.0])

    expected_nx = np.array([0.0, 1.0 / np.sqrt(2.0), -1.0])
    expected_ny = np.array([-1.0, 1.0 / np.sqrt(2.0), 0.0])

    assert np.linalg.norm(edge_lengths[0] - expected_lengths, ord=np.inf) < 1e-12
    assert np.linalg.norm(nx[0] - expected_nx, ord=np.inf) < 1e-12
    assert np.linalg.norm(ny[0] - expected_ny, ord=np.inf) < 1e-12

    assert np.linalg.norm(sJ[0] - 0.5 * expected_lengths, ord=np.inf) < 1e-12


def run_all_tests():
    test_reference_triangle_metrics()
    test_reference_triangle_face_metrics()
    print("🎉 test_face_metrics.py 全部測試通過")


if __name__ == "__main__":
    run_all_tests()