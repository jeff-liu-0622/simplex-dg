import numpy as np

from test.test_sphere_full_rhs_smooth_snapshot import build_projected_sphere_smooth_state
from test.test_sphere_lsrk_short_sanity import compute_sphere_full_rhs_for_state


def build_sphere_rhs_operator(
    surface_mode,
    flux_type,
    alpha_lf=1.0,
    nsub=2,
    order=2,
):
    state = build_projected_sphere_smooth_state(nsub=nsub, order=order)
    shape = state["q"].shape
    ndof = int(np.prod(shape))
    operator = np.zeros((ndof, ndof), dtype=float)

    for col in range(ndof):
        q = np.zeros(shape, dtype=float)
        q.flat[col] = 1.0
        rhs = compute_sphere_full_rhs_for_state(
            q,
            0.0,
            state,
            flux_type=flux_type,
            alpha_lf=alpha_lf,
            surface_mode=surface_mode,
        )
        operator[:, col] = np.asarray(rhs, dtype=float).ravel()

    return operator, state


def spectrum_summary(operator, surface_mode, flux_type, alpha_lf):
    eigenvalues = np.linalg.eigvals(operator)
    real_parts = np.real(eigenvalues)
    magnitudes = np.abs(eigenvalues)
    order = np.argsort(real_parts)[::-1]

    return {
        "surface_mode": surface_mode,
        "flux_type": flux_type,
        "alpha_lf": alpha_lf,
        "operator": operator,
        "eigenvalues": eigenvalues,
        "max_real_eigenvalue": float(np.max(real_parts)),
        "min_real_eigenvalue": float(np.min(real_parts)),
        "spectral_radius": float(np.max(magnitudes)),
        "num_positive_real": int(np.count_nonzero(real_parts > 1.0e-10)),
        "top_eigenvalues": eigenvalues[order[:10]],
        "all_finite": bool(
            np.all(np.isfinite(operator))
            and np.all(np.isfinite(real_parts))
            and np.all(np.isfinite(np.imag(eigenvalues)))
        ),
    }


def run_spectrum_case(surface_mode, flux_type, alpha_lf=1.0, nsub=2, order=2):
    operator, state = build_sphere_rhs_operator(
        surface_mode=surface_mode,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
        nsub=nsub,
        order=order,
    )
    summary = spectrum_summary(
        operator=operator,
        surface_mode=surface_mode,
        flux_type=flux_type,
        alpha_lf=alpha_lf,
    )
    summary["ndof"] = int(operator.shape[0])
    summary["nsub"] = nsub
    summary["order"] = order
    summary["num_elements"] = int(state["q"].shape[0])
    summary["num_nodes_per_element"] = int(state["q"].shape[1])
    return summary


def format_complex(value):
    return f"{value.real:.6e}{value.imag:+.6e}i"


def test_sphere_operator_spectrum_diagnostic():
    surface_modes = ["old", "conservative", "conservative_scaled"]
    flux_cases = [
        ("central", 1.0),
        ("upwind", 1.0),
        ("lf", 1.0),
        ("lf", 1.5),
    ]
    results = []

    print("\n" + "=" * 172)
    print("Projected sphere semi-discrete operator spectrum diagnostic")
    print("=" * 172)
    print("Builds q_t = L q by applying the sphere RHS callback to basis vectors.")
    print("Configuration: projected sphere topology, nsub=2, N=2; no time integration.")
    print(
        f"{'surface_mode':>22s} {'flux':>10s} {'alpha':>8s} "
        f"{'ndof':>7s} {'max_real':>15s} {'min_real':>15s} "
        f"{'spectral_radius':>17s} {'num_Re>1e-10':>14s}"
    )
    print("-" * 172)

    for surface_mode in surface_modes:
        for flux_type, alpha_lf in flux_cases:
            result = run_spectrum_case(
                surface_mode=surface_mode,
                flux_type=flux_type,
                alpha_lf=alpha_lf,
            )
            results.append(result)
            print(
                f"{surface_mode:>22s} {flux_type:>10s} {alpha_lf:8.4f} "
                f"{result['ndof']:7d} "
                f"{result['max_real_eigenvalue']:15.6e} "
                f"{result['min_real_eigenvalue']:15.6e} "
                f"{result['spectral_radius']:17.6e} "
                f"{result['num_positive_real']:14d}"
            )

    print("-" * 172)
    print("Top 10 eigenvalues by real part")
    for result in results:
        print(
            f"\nmode={result['surface_mode']}, "
            f"flux={result['flux_type']}, alpha={result['alpha_lf']:.4f}"
        )
        for rank, value in enumerate(result["top_eigenvalues"], start=1):
            print(f"  {rank:2d}: {format_complex(value)}")

    print("\n" + "-" * 172)
    print("Upwind vs LF(alpha=1) operator differences")
    for surface_mode in surface_modes:
        upwind = next(
            row
            for row in results
            if row["surface_mode"] == surface_mode and row["flux_type"] == "upwind"
        )
        lf1 = next(
            row
            for row in results
            if row["surface_mode"] == surface_mode
            and row["flux_type"] == "lf"
            and abs(row["alpha_lf"] - 1.0) < 1.0e-14
        )
        max_diff = float(np.max(np.abs(upwind["operator"] - lf1["operator"])))
        print(f"{surface_mode:>22s}: max |L_upwind - L_lf1| = {max_diff:.6e}")
        assert max_diff < 1.0e-12

    print("=" * 172)

    ndof_values = {row["ndof"] for row in results}
    assert len(ndof_values) == 1
    assert next(iter(ndof_values)) > 0
    for result in results:
        assert result["all_finite"]
        assert np.isfinite(result["max_real_eigenvalue"])
        assert np.isfinite(result["min_real_eigenvalue"])
        assert np.isfinite(result["spectral_radius"])
        assert result["spectral_radius"] > 0.0


def run_all_tests():
    test_sphere_operator_spectrum_diagnostic()
    print("test_sphere_operator_spectrum_diagnostic.py passed")


if __name__ == "__main__":
    run_all_tests()
