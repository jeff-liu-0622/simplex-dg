import numpy as np

from core.cfl import compute_advection_dt, compute_advection_dt_from_velocity


def test_compute_advection_dt_constant_speed():
    dt = compute_advection_dt(h=0.25, N=4, speed=1.0, cfl=0.05)
    exact = 0.05 * 0.25 / 16.0

    assert abs(dt - exact) < 1e-15


def test_compute_advection_dt_zero_speed():
    dt = compute_advection_dt(h=0.25, N=4, speed=0.0, cfl=0.05)

    assert np.isinf(dt)


def test_compute_advection_dt_from_velocity():
    u = np.array([1.0, 2.0, 0.0])
    v = np.array([0.0, 0.0, 3.0])

    dt = compute_advection_dt_from_velocity(h=0.5, N=4, u=u, v=v, cfl=0.05)

    exact = 0.05 * 0.5 / (16.0 * 3.0)

    assert abs(dt - exact) < 1e-15


def run_all_tests():
    test_compute_advection_dt_constant_speed()
    test_compute_advection_dt_zero_speed()
    test_compute_advection_dt_from_velocity()

    print("🎉 test_cfl.py 全部測試通過")


if __name__ == "__main__":
    run_all_tests()