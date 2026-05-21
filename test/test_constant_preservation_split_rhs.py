import numpy as np

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.geometry.connectivity import (
    build_connectivity,
    apply_periodic_conditions,
    build_maps,
)
from core.operators_split import compute_split_rhs


def weighted_l2_norm(field, J, weights):
    numerator = np.sum(J[:, None] * weights[None, :] * field**2)
    denominator = np.sum(J[:, None] * weights[None, :])
    return np.sqrt(numerator / denominator)


def build_periodic_affine_case(N_poly=4, n_quad=4, NX=4, NY=4, cx=1.0, cy=1.0):
    engine = build_local_operators(N_poly, n_quad, rule="table1")

    VX, VY, EToV = create_square_mesh(NX, NY)
    EToV_x = VX[EToV]
    EToV_y = VY[EToV]

    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)
    nx, ny, edge_lengths, _ = compute_face_metrics(EToV_x, EToV_y)

    r, s = engine.r, engine.s
    x_nodes = 0.5 * (
        -(r + s) * EToV_x[:, 0:1]
        + (1.0 + r) * EToV_x[:, 1:2]
        + (1.0 + s) * EToV_x[:, 2:3]
    )
    y_nodes = 0.5 * (
        -(r + s) * EToV_y[:, 0:1]
        + (1.0 + r) * EToV_y[:, 1:2]
        + (1.0 + s) * EToV_y[:, 2:3]
    )

    EToE, EToF = build_connectivity(EToV)
    EToE, EToF = apply_periodic_conditions(EToE, EToF, x_nodes, y_nodes, engine)
    vmapM, vmapP = build_maps(engine, EToV, EToE, EToF, x_nodes, y_nodes)

    kwargs = {
        "engine": engine,
        "xr": xr,
        "xs": xs,
        "yr": yr,
        "ys": ys,
        "rx": rx,
        "sx": sx,
        "ry": ry,
        "sy": sy,
        "J": J,
        "nx": nx,
        "ny": ny,
        "edge_lengths": edge_lengths,
        "vmapM": vmapM,
        "vmapP": vmapP,
        "cx": cx,
        "cy": cy,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "lift_mode": "physical",
    }

    return engine, kwargs


def run_constant_preservation_case(tau):
    engine, kwargs = build_periodic_affine_case()
    q = np.ones((kwargs["J"].shape[0], engine.num_nodes))

    rhs = compute_split_rhs(q, t=0.0, tau=tau, **kwargs)
    max_norm = np.max(np.abs(rhs))
    l2_norm = weighted_l2_norm(rhs, kwargs["J"], engine.w_s)

    return max_norm, l2_norm


def test_constant_preservation_upwind():
    max_norm, l2_norm = run_constant_preservation_case(tau=0.0)

    assert max_norm < 1.0e-11
    assert l2_norm < 1.0e-11


def test_constant_preservation_central():
    max_norm, l2_norm = run_constant_preservation_case(tau=1.0)

    assert max_norm < 1.0e-11
    assert l2_norm < 1.0e-11


def run_all_tests():
    print("constant preservation | planar affine mesh | q=1 | cx=1, cy=1")
    print("-" * 72)
    print(f"{'tau':>8s} {'flux':>10s} {'max_norm':>16s} {'weighted_L2':>16s}")
    print("-" * 72)

    for tau, name in [(0.0, "upwind"), (1.0, "central")]:
        max_norm, l2_norm = run_constant_preservation_case(tau=tau)
        print(f"{tau:8.1f} {name:>10s} {max_norm:16.8e} {l2_norm:16.8e}")

        if max_norm >= 1.0e-11 or l2_norm >= 1.0e-11:
            raise AssertionError(
                "constant preservation failed: "
                f"tau={tau}, max_norm={max_norm:.6e}, weighted_L2={l2_norm:.6e}"
            )

    print("-" * 72)
    print("constant preservation passed")


if __name__ == "__main__":
    run_all_tests()
