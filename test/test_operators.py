import numpy as np

from core import build_local_operators


def test_differentiation_matrices():
    """
    測試微分矩陣 Dr 和 Ds 是否能精確計算多項式導數。

    Dr, Ds 是針對 reference triangle:
        (-1,-1), (1,-1), (-1,1)
    上的 r, s 座標微分。
    """
    print("\n" + "=" * 60)
    print("啟動微分矩陣精確度測試")
    print("=" * 60)

    for rule in ["table1", "table2"]:
        print(f"\n--- 測試 [{rule.upper()}] (N=3, n=4) ---")

        engine = build_local_operators(N=3, n=4, rule=rule)

        r, s = engine.r, engine.s
        Dr, Ds = engine.Dr, engine.Ds

        # Test 0: constant function
        u0 = np.ones_like(r)

        err_r0 = np.max(np.abs(Dr @ u0))
        err_s0 = np.max(np.abs(Ds @ u0))

        print(f"  常數函數誤差:   dr = {err_r0:.2e}, ds = {err_s0:.2e}")

        assert err_r0 < 1e-12, f"[{rule}] 常數函數 r 偏導數錯誤"
        assert err_s0 < 1e-12, f"[{rule}] 常數函數 s 偏導數錯誤"

        # Test 1: linear function
        u1 = r + s

        err_r1 = np.max(np.abs(Dr @ u1 - 1.0))
        err_s1 = np.max(np.abs(Ds @ u1 - 1.0))

        print(f"  線性函數誤差:   dr = {err_r1:.2e}, ds = {err_s1:.2e}")

        assert err_r1 < 1e-12, f"[{rule}] 線性函數 r 偏導數錯誤"
        assert err_s1 < 1e-12, f"[{rule}] 線性函數 s 偏導數錯誤"

        # Test 2: quadratic function
        u2 = r**2 + 2.0 * s

        err_r2 = np.max(np.abs(Dr @ u2 - 2.0 * r))
        err_s2 = np.max(np.abs(Ds @ u2 - 2.0))

        print(f"  二次函數誤差:   dr = {err_r2:.2e}, ds = {err_s2:.2e}")

        assert err_r2 < 1e-12, f"[{rule}] 二次函數 r 偏導數錯誤"
        assert err_s2 < 1e-12, f"[{rule}] 二次函數 s 偏導數錯誤"

        # Test 3: mixed quadratic function
        u3 = 1.0 + r + 2.0 * s + r**2 + r * s

        exact_dr = 1.0 + 2.0 * r + s
        exact_ds = 2.0 + r

        err_r3 = np.max(np.abs(Dr @ u3 - exact_dr))
        err_s3 = np.max(np.abs(Ds @ u3 - exact_ds))

        print(f"  混合二次誤差:   dr = {err_r3:.2e}, ds = {err_s3:.2e}")

        assert err_r3 < 1e-12, f"[{rule}] 混合二次函數 r 偏導數錯誤"
        assert err_s3 < 1e-12, f"[{rule}] 混合二次函數 s 偏導數錯誤"

        print(f"✅ [{rule.upper()}] 微分矩陣驗證通過")


def test_boundary_penalty_lift_shape_table1():
    """
    測試 SDG boundary penalty lift 的輸入輸出 shape。
    只能用 table1，因為 table2 沒有 edge weights。
    """
    print("\n" + "=" * 60)
    print("啟動 boundary penalty lift shape 測試")
    print("=" * 60)

    engine = build_local_operators(N=3, n=4, rule="table1")

    Nb = engine.num_boundary_nodes

    p = np.ones(Nb)
    lifted = engine.lift_boundary_penalty(p)

    print(f"  boundary nodes = {Nb}")
    print(f"  lifted shape   = {lifted.shape}")

    assert lifted.shape == (engine.num_nodes,)

    print("✅ boundary penalty lift shape 測試通過")


def test_boundary_penalty_lift_zero_table1():
    """
    如果 boundary penalty 是 0，lift 後也應該是 0。
    """
    print("\n" + "=" * 60)
    print("啟動 boundary penalty lift zero 測試")
    print("=" * 60)

    engine = build_local_operators(N=3, n=4, rule="table1")

    p = np.zeros(engine.num_boundary_nodes)
    lifted = engine.lift_boundary_penalty(p)

    err = np.linalg.norm(lifted, ord=np.inf)

    print(f"  zero lift error = {err:.2e}")

    assert err < 1e-14

    print("✅ boundary penalty lift zero 測試通過")


def test_boundary_penalty_lift_table2_should_fail():
    """
    table2 沒有 edge quadrature weights，因此不應該能做簡單 SDG lift。
    """
    print("\n" + "=" * 60)
    print("啟動 table2 boundary lift 應失敗測試")
    print("=" * 60)

    engine = build_local_operators(N=3, n=4, rule="table2")

    # table2 通常沒有 boundary nodes，所以這裡建立一個假的 p。
    # 重點是 lift_boundary_penalty 應該因 w_e is None 而 raise ValueError。
    p = np.ones(engine.num_boundary_nodes)

    try:
        engine.lift_boundary_penalty(p)
    except ValueError:
        print("✅ table2 正確拒絕 boundary penalty lift")
        return

    raise AssertionError("table2 should not support lift_boundary_penalty.")


def run_all_tests():
    test_differentiation_matrices()
    test_boundary_penalty_lift_shape_table1()
    test_boundary_penalty_lift_zero_table1()
    test_boundary_penalty_lift_table2_should_fail()

    print("\n" + "=" * 60)
    print("🎉 test_operators.py 全部測試通過")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()