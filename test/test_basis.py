import numpy as np

# 享受套件化的極簡引入
from core import get_reference_nodes, build_vandermonde2d, build_local_operators

def test_basis_orthogonality():
    """
    驗證: 2.0 * V^T * W * V = I (單位矩陣)
    這是在測試 PKD 正交多項式與高斯積分點的完美配合。
    """
    # 測試配置： (rule, quadrature_n, polynomial_p)
    # 數學原理：要讓 V^T W V 完全等於單位矩陣，基底相乘的最高階數 (2p)
    # 必須小於等於積分規則的極限精確度。
    test_cases = [
        # ==========================================
        # 測試 Table 1 (極限精度 2n-1)
        # 條件： 2p <= 2n-1 => p <= n - 1
        # ==========================================
        ("table1", 2, 1),  # 2(2)-1 = 3 >= 2(1)
        ("table1", 3, 2),  # 2(3)-1 = 5 >= 2(2)
        ("table1", 4, 3),  # 2(4)-1 = 7 >= 2(3)
        
        # ==========================================
        # 測試 Table 2 (極限精度 2n) -> 馬力更強！
        # 條件： 2p <= 2n => p <= n
        # (同樣的積分階數 n，Table 2 能積更高階的波浪)
        # ==========================================
        ("table2", 1, 1),  # 2(1) = 2 >= 2(1)
        ("table2", 2, 2),  # 2(2) = 4 >= 2(2)
        ("table2", 3, 3),  # 2(3) = 6 >= 2(3)
        ("table2", 4, 4),  # 2(4) = 8 >= 2(4)
    ]

    print("\n" + "="*50)
    print("啟動正交性嚴格測試 (Orthogonality Test)")
    print("="*50)

    for rule, n, p in test_cases:
        print(f"測試 [{rule.upper():<6}] 積分 n={n}, 支援基底階數 p={p} ...", end=" ")
        
        # 1. 取得積分點與權重
        nodes = get_reference_nodes(n, rule=rule)
        r = nodes["r"]
        s = nodes["s"]
        w_s = nodes["w_s"]
        
        # 2. 建立范德蒙矩陣 V
        V, _, _ = build_vandermonde2d(p, r, s)
        
        # 3. 建構基底空間的質量矩陣
        W_matrix = np.diag(w_s)
        M_basis = 2.0 * (V.T @ W_matrix @ V)
        
        # 4. 驗證是否等於單位矩陣
        identity = np.eye(M_basis.shape[0])
        max_err = np.linalg.norm(M_basis - identity, ord=np.inf)
        
        if max_err < 1e-13:
            print(f"✅ 通過! (誤差: {max_err:.2e})")
        else:
            print(f"❌ 失敗! (誤差: {max_err:.2e})")
            raise AssertionError(f"[{rule}] n={n}, p={p} 基底正交性驗證失敗！")

def test_engine_initialization():
    """
    驗證我們剛剛寫好的 OOP 物理引擎是否能正確吃下這兩種 Table
    """
    print("\n" + "="*50)
    print("啟動物理引擎初始化測試 (Engine Init Test)")
    print("="*50)

    try:
        # 測試引擎切換 Table 1
        engine_t1 = build_local_operators(N=3, n=4, rule="table1")
        print(f"✅ Table 1 引擎初始化成功! (包含邊界點數: {engine_t1.num_edge_nodes})")

        # 測試引擎切換 Table 2
        engine_t2 = build_local_operators(N=3, n=4, rule="table2")
        print(f"✅ Table 2 引擎初始化成功! (純內部點，邊界權重為: {engine_t2.w_e})")
        
    except Exception as e:
        print(f"❌ 引擎初始化失敗: {e}")
        raise

if __name__ == "__main__":
    test_basis_orthogonality()
    test_engine_initialization()
    print("\n🎉 所有核心測試完美通過！系統狀態：極度健康。")