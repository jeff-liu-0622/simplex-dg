import numpy as np
import matplotlib.pyplot as plt

from core.time_integration import lsrk54_step


def rhs_decay(q, t, lam=1.0):
    """
    Test ODE:

        q_t = -lambda q

    Exact solution:

        q(t) = exp(-lambda t) q(0)
    """
    return -lam * q


def exact_decay(q0, t, lam=1.0):
    return np.exp(-lam * t) * q0


def run_lsrk_convergence_test(show_plot=True):
    print("\n" + "=" * 60)
    print("啟動 LSRK54 時間收斂測試")
    print("=" * 60)

    # ------------------------------------------------------------
    # Test problem:
    #
    #     q_t = -lambda q
    #
    # This isolates the time integrator only.
    # No spatial discretization, no DG, no flux, no boundary condition.
    # ------------------------------------------------------------
    lam = 1.0
    T_final = 1.0

    q0 = np.array([
        1.0,
        2.0,
        -0.5,
        3.0,
    ])

    dts = np.array([
        1.0 / 10.0,
        1.0 / 20.0,
        1.0 / 40.0,
        1.0 / 80.0,
        1.0 / 160.0,
    ])

    errors = []

    print("-" * 60)

    for dt in dts:
        steps = int(round(T_final / dt))
        actual_dt = T_final / steps

        q = q0.copy()
        res = np.zeros_like(q)

        t = 0.0

        for _ in range(steps):
            q, res = lsrk54_step(
                q,
                res,
                t,
                actual_dt,
                rhs_decay,
                lam=lam,
            )
            t += actual_dt

        q_exact = exact_decay(q0, T_final, lam=lam)

        err = np.linalg.norm(q - q_exact, ord=np.inf)
        errors.append(err)

        print(
            f"dt = {actual_dt:.6e} | "
            f"steps = {steps:5d} | "
            f"error = {err:.6e}"
        )

    errors = np.array(errors)

    print("-" * 60)

    # ------------------------------------------------------------
    # Compute observed convergence orders
    # ------------------------------------------------------------
    orders = []

    for i in range(1, len(errors)):
        order = np.log(errors[i - 1] / errors[i]) / np.log(dts[i - 1] / dts[i])
        orders.append(order)

    print("觀察到的收斂階數：")

    for i, order in enumerate(orders, start=1):
        print(
            f"  dt {dts[i-1]:.3e} -> {dts[i]:.3e}: "
            f"order = {order:.4f}"
        )

    # LSRK54 should be 4th order.
    # The coarsest dt may be pre-asymptotic, so check the last two orders.
    assert orders[-1] > 3.8, (
        f"LSRK54 convergence order too low: observed order = {orders[-1]:.4f}"
    )

    print("\n✅ LSRK54 時間收斂測試通過")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if show_plot:
        plt.figure(figsize=(8, 6))

        plt.loglog(
            dts,
            errors,
            "o-",
            linewidth=2,
            markersize=8,
            label="LSRK54 error",
        )

        ref = errors[0] * (dts / dts[0]) ** 4

        plt.loglog(
            dts,
            ref,
            "k--",
            linewidth=2,
            label=r"reference $O(\Delta t^4)$",
        )

        plt.xlabel("Time step dt")
        plt.ylabel("Max absolute error")
        plt.title("LSRK54 Time Convergence Test")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("=" * 60)


if __name__ == "__main__":
    run_lsrk_convergence_test(show_plot=True)