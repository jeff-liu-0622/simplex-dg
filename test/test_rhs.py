import numpy as np

from core import build_local_operators
from core.rhs import (
    compute_split_volume_rhs,
    compute_boundary_penalty,
    compute_sdg_rhs_single_element,
)


def reference_boundary_normals(engine):
    """
    Reference triangle:
        edge1: s = -1
        edge2: r + s = 0
        edge3: r = -1

    Outward unit normals in (r,s):
        edge1: (0, -1)
        edge2: (1/sqrt(2), 1/sqrt(2))
        edge3: (-1, 0)
    """
    Nfp = engine.num_edge_nodes

    n1 = np.tile(np.array([0.0, -1.0]), (Nfp, 1))
    n2 = np.tile(np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]), (Nfp, 1))
    n3 = np.tile(np.array([-1.0, 0.0]), (Nfp, 1))

    normals = np.vstack([n1, n2, n3])

    return normals[:, 0], normals[:, 1]


def test_split_volume_rhs_zero_velocity():
    print("\n=== RHS Test 1: zero velocity ===")

    engine = build_local_operators(N=3, n=4, rule="table1")

    q = np.ones(engine.num_nodes)
    u = np.zeros(engine.num_nodes)
    v = np.zeros(engine.num_nodes)

    rhs = compute_split_volume_rhs(engine, q, u, v)

    err = np.linalg.norm(rhs, ord=np.inf)
    print(f"zero velocity RHS error = {err:.2e}")

    assert err < 1e-12


def test_split_volume_rhs_constant_solution_constant_velocity():
    print("\n=== RHS Test 2: constant solution + constant velocity ===")

    engine = build_local_operators(N=3, n=4, rule="table1")

    q = np.ones(engine.num_nodes)
    u = np.ones(engine.num_nodes) * 2.0
    v = np.ones(engine.num_nodes) * -0.5

    rhs = compute_split_volume_rhs(engine, q, u, v)

    err = np.linalg.norm(rhs, ord=np.inf)
    print(f"constant volume RHS error = {err:.2e}")

    assert err < 1e-12


def test_boundary_penalty_zero_when_same_state():
    print("\n=== RHS Test 3: boundary penalty zero if q-=q+ ===")

    engine = build_local_operators(N=3, n=4, rule="table1")
    nx, ny = reference_boundary_normals(engine)

    q_minus = np.ones(engine.num_boundary_nodes)
    q_plus = np.ones(engine.num_boundary_nodes)

    u = np.ones(engine.num_boundary_nodes) * 1.3
    v = np.ones(engine.num_boundary_nodes) * -0.7

    p = compute_boundary_penalty(
        q_minus=q_minus,
        q_plus=q_plus,
        nx=nx,
        ny=ny,
        u=u,
        v=v,
        tau=0.0,
    )

    err = np.linalg.norm(p, ord=np.inf)
    print(f"same-state boundary penalty error = {err:.2e}")

    assert err < 1e-12


def test_full_rhs_constant_solution_same_boundary():
    print("\n=== RHS Test 4: full RHS constant solution with same boundary ===")

    engine = build_local_operators(N=3, n=4, rule="table1")
    nx, ny = reference_boundary_normals(engine)

    q = np.ones(engine.num_nodes)
    u = np.ones(engine.num_nodes) * 0.8
    v = np.ones(engine.num_nodes) * -0.4

    q_plus_boundary = np.ones(engine.num_boundary_nodes)

    rhs = compute_sdg_rhs_single_element(
        engine=engine,
        q=q,
        u=u,
        v=v,
        q_plus_boundary=q_plus_boundary,
        nx=nx,
        ny=ny,
        tau=0.0,
    )

    err = np.linalg.norm(rhs, ord=np.inf)
    print(f"full constant RHS error = {err:.2e}")

    assert err < 1e-11


def test_boundary_penalty_inflow_sign_1d_like():
    """
    用 edge3 測 sign。

    edge3 outward normal is (-1, 0).
    Choose velocity u = 1, v = 0.

    Then n·V = -1 < 0, so this is inflow.

    Upwind flux should use q_plus.
    If q_minus = 0 and q_plus = 1:

        n·f_internal = (n·V) q_minus = 0
        n·f_star     = (n·V) q_plus  = -1

    Therefore:
        p = n·(f - f*) = 0 - (-1) = +1

    So p on edge3 should be positive.
    """
    print("\n=== RHS Test 5: inflow sign check ===")

    engine = build_local_operators(N=3, n=4, rule="table1")
    Nfp = engine.num_edge_nodes

    nx, ny = reference_boundary_normals(engine)

    q_minus = np.zeros(engine.num_boundary_nodes)
    q_plus = np.zeros(engine.num_boundary_nodes)

    # Only edge3 has exterior value 1
    edge3 = engine.edge_slices[2]
    q_plus[edge3] = 1.0

    u = np.ones(engine.num_boundary_nodes)
    v = np.zeros(engine.num_boundary_nodes)

    p = compute_boundary_penalty(
        q_minus=q_minus,
        q_plus=q_plus,
        nx=nx,
        ny=ny,
        u=u,
        v=v,
        tau=0.0,
    )

    p_edge3 = p[edge3]

    print(f"edge3 penalty min = {np.min(p_edge3):.2e}")
    print(f"edge3 penalty max = {np.max(p_edge3):.2e}")

    assert np.all(p_edge3 > 0.0)


def run_all_tests():
    test_split_volume_rhs_zero_velocity()
    test_split_volume_rhs_constant_solution_constant_velocity()
    test_boundary_penalty_zero_when_same_state()
    test_full_rhs_constant_solution_same_boundary()
    test_boundary_penalty_inflow_sign_1d_like()

    print("\n🎉 test_rhs.py 全部測試通過")


if __name__ == "__main__":
    run_all_tests()