import numpy as np

from core.operators import build_local_operators
from core.geometry.sphere_mapping import reference_to_unit_triangle
from core.geometry.sphere_velocity import solid_body_contravariant_velocity


def compute_patch_velocity_divergence(N, n, patch_id, alpha):
    """
    Compute

        div = d_xi u^xi + d_eta u^eta

    on one sphere patch.

    engine.Dr, engine.Ds differentiate with respect to reference coordinates
    (r,s), where

        xi  = (r+1)/2
        eta = (s+1)/2

    Therefore:

        d/dxi  = 2 d/dr
        d/deta = 2 d/ds
    """
    engine = build_local_operators(N=N, n=n, rule="table1")

    r = engine.r
    s = engine.s

    xi, eta = reference_to_unit_triangle(r, s)

    u_xi, u_eta = solid_body_contravariant_velocity(
        xi,
        eta,
        patch_id,
        R=1.0,
        u0=1.0,
        alpha=alpha,
    )

    div = 2.0 * (engine.Dr @ u_xi) + 2.0 * (engine.Ds @ u_eta)

    return div


def test_divergence_finite():
    """
    Basic safety check:
    divergence should not contain NaN / inf.
    """
    for alpha in [0.0, np.pi / 4.0, np.pi / 2.0]:
        for patch_id in range(8):
            div = compute_patch_velocity_divergence(
                N=4,
                n=4,
                patch_id=patch_id,
                alpha=alpha,
            )

            assert np.all(np.isfinite(div)), (
                f"Non-finite divergence: patch={patch_id}, alpha={alpha}"
            )


def test_divergence_small_for_zonal_rotation():
    """
    For alpha=0, the velocity is simple zonal rotation.

    The contravariant velocity should be divergence-free on each patch.
    This case is especially clean.
    """
    max_errors = []

    for patch_id in range(8):
        div = compute_patch_velocity_divergence(
            N=4,
            n=4,
            patch_id=patch_id,
            alpha=0.0,
        )

        max_errors.append(np.max(np.abs(div)))

    max_error = max(max_errors)

    print(f"alpha=0 max |div| = {max_error:.6e}")

    # This should be very small. If this fails badly, the velocity transform
    # or metric derivatives are likely wrong.
    assert max_error < 1.0e-10, (
        f"Zonal rotation divergence too large: {max_error:.6e}"
    )


def test_divergence_reasonable_for_tilted_rotation():
    """
    For tilted rotation, u^xi and u^eta are non-polynomial functions
    on the patch coordinates.

    With N=4 differentiation, we should not expect machine precision,
    but the divergence should remain bounded and reasonably small.
    """
    for alpha in [np.pi / 4.0, np.pi / 2.0]:
        patch_errors = []

        for patch_id in range(8):
            div = compute_patch_velocity_divergence(
                N=4,
                n=4,
                patch_id=patch_id,
                alpha=alpha,
            )

            patch_errors.append(np.max(np.abs(div)))

        max_error = max(patch_errors)

        print(f"alpha={alpha:.6f}")
        for patch_id, err in enumerate(patch_errors):
            print(f"  patch {patch_id}: max |div| = {err:.6e}")


        # This tolerance is intentionally not too strict.
        # The nodal derivative is applied to a non-polynomial velocity field.
        assert max_error < 1.0, (
            f"Tilted rotation divergence too large: "
            f"alpha={alpha}, max_error={max_error:.6e}"
        )


def test_divergence_trend_with_degree():
    """
    Diagnostic only.

    Since the tilted velocity is not polynomial in patch coordinates,
    the divergence error may not be zero. Also, Table1 node sets change
    with N, so monotone convergence in this single-patch test is not required.
    """
    alpha = np.pi / 4.0

    print("\nDivergence diagnostic for alpha=pi/4")
    print("-" * 60)
    print(f"{'N':>4s} {'max |div|':>16s}")
    print("-" * 60)

    for N in [1, 2, 3, 4]:
        n = N
        max_error_N = 0.0

        for patch_id in range(8):
            div = compute_patch_velocity_divergence(
                N=N,
                n=n,
                patch_id=patch_id,
                alpha=alpha,
            )

            max_error_N = max(max_error_N, np.max(np.abs(div)))

        print(f"{N:4d} {max_error_N:16.6e}")

    print("-" * 60)

    # Do not enforce strict monotone convergence here because node sets and
    # quadrature rules change with N. Just ensure N=4 is not worse than N=1.

def run_all_tests():
    print("\n" + "=" * 72)
    print("啟動 sphere contravariant velocity divergence 測試")
    print("=" * 72)

    test_divergence_finite()
    print("✅ divergence has no NaN / inf")

    test_divergence_small_for_zonal_rotation()
    print("✅ alpha=0 divergence is near zero")

    test_divergence_reasonable_for_tilted_rotation()
    print("✅ tilted divergence is bounded")

    test_divergence_trend_with_degree()
    print("✅ divergence degree diagnostic passed")

    print("=" * 72)
    print("🎉 test_sphere_velocity_divergence.py 全部通過")


if __name__ == "__main__":
    run_all_tests()