from math import factorial

import numpy as np

from core.basis import build_vandermonde2d
from core.operators import build_local_operators


def _reference_monomial_integral(a: int, b: int) -> float:
    """Integral of xi**a eta**b over the r,s reference triangle."""
    return 4.0 * factorial(a) * factorial(b) / factorial(a + b + 2)


def _reference_boundary_matrices(engine):
    normals = (
        (0.0, -1.0),
        (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
        (-1.0, 0.0),
    )

    Br = np.zeros((engine.num_nodes, engine.num_nodes))
    Bs = np.zeros_like(Br)

    for f, (nr, ns) in enumerate(normals):
        Bf = engine.edge_lengths[f] * np.diag(engine.w_e)
        Rf = engine.R_faces[f]
        Br += nr * (Rf.T @ Bf @ Rf)
        Bs += ns * (Rf.T @ Bf @ Rf)

    return Br, Bs


def test_table2_n4_quadrature_exact_through_degree_8():
    engine = build_local_operators(N=4, n=4, rule="table2")

    assert engine.num_nodes == 16
    assert engine.num_basis == 15
    assert engine.num_edge_nodes == 5
    assert np.isclose(np.sum(engine.w_s), 1.0)
    assert np.isclose(np.sum(engine.w_e), 1.0)
    assert np.all(engine.w_s > 0.0)
    assert np.all(engine.w_e > 0.0)

    for total_degree in range(9):
        for a in range(total_degree + 1):
            b = total_degree - a
            numerical = engine.area * np.sum(
                engine.w_s * engine.xi**a * engine.eta**b
            )
            exact = _reference_monomial_integral(a, b)
            assert abs(numerical - exact) < 5.0e-13


def test_table2_projection_differentiation_and_trace_are_polynomial_exact():
    engine = build_local_operators(N=4, n=4, rule="table2")

    P = engine.invM_modal @ (engine.area * engine.V.T @ engine.W)
    np.testing.assert_allclose(P @ engine.V, np.eye(engine.num_basis), atol=5e-13)
    np.testing.assert_allclose(engine.Dr @ engine.V, engine.Vr, atol=5e-13)
    np.testing.assert_allclose(engine.Ds @ engine.V, engine.Vs, atol=5e-13)

    V_face = np.vstack([
        build_vandermonde2d(4, engine.face_r[f], engine.face_s[f])[0]
        for f in range(3)
    ])
    np.testing.assert_allclose(engine.R_all @ engine.V, V_face, atol=5e-13)


def test_table2_projected_operators_satisfy_diagonal_norm_sbp():
    engine = build_local_operators(N=4, n=4, rule="table2")
    H = engine.area * engine.W
    Br, Bs = _reference_boundary_matrices(engine)

    np.testing.assert_allclose(H @ engine.Dr + engine.Dr.T @ H, Br, atol=1e-12)
    np.testing.assert_allclose(H @ engine.Ds + engine.Ds.T @ H, Bs, atol=1e-12)


def test_table1_boundary_extraction_remains_unchanged():
    engine = build_local_operators(N=4, n=4, rule="table1")
    u = np.arange(engine.num_nodes, dtype=float)

    np.testing.assert_array_equal(
        engine.boundary_values(u),
        u[: engine.num_boundary_nodes],
    )
