import numpy as np
from math import factorial

# 享受套件化引入
from core import get_reference_nodes

def exact_triangle_integral(a, b):
    """計算 ∫_T xi^a eta^b dA 的精確解析解"""
    return factorial(a) * factorial(b) / factorial(a + b + 2)

def test_quadrature_accuracy():
    """驗證 Table 1 與 Table 2 積分點的面積分與線積分精確度"""
    
    print("\n" + "="*50)
    print("啟動高斯積分精確度測試 (Quadrature Accuracy Test)")
    print("="*50)

    for rule in ["table1", "table2"]:
        print(f"\n--- 測試 [{rule.upper()}] ---")
        
        for n in [1, 2, 3, 4]:
            nodes = get_reference_nodes(n, rule=rule)
            xi, eta, w_s = nodes["xi"], nodes["eta"], nodes["w_s"]
            
            area = 0.5  # 參考三角形面積 (在 xi, eta 座標系下)
            
            # ==========================================
            # 驗證 1: 面積分精確度 (Volume Integral)
            # 數學極限：Table 1 是 2n-1 階，Table 2 是 2n 階
            # ==========================================
            target_vol_degree = 2 * n - 1 if rule == "table1" else 2 * n
            
            max_vol_err = 0.0
            for a in range(target_vol_degree + 1):
                for b in range(target_vol_degree + 1 - a):
                    exact = exact_triangle_integral(a, b)
                    num = area * np.sum(w_s * (xi**a) * (eta**b))
                    max_vol_err = max(max_vol_err, abs(num - exact))
            
            print(f"[{rule.upper()}] n={n} | 面積分精度 (至 {target_vol_degree} 階): 最大誤差 = {max_vol_err:.3e}")
            assert max_vol_err < 1e-13, f"[{rule}] n={n} 面積分驗證失敗！"

            # ==========================================
            # 驗證 2: 邊界線積分精確度 (Edge Integral)
            # ⚠️ 注意：Table 2 是純內部點，沒有線積分，所以只測 Table 1
            # ==========================================
            if rule == "table1":
                w_e = nodes["w_e"]
                num_edge_nodes = nodes["num_edge_nodes"]
                
                target_edge_degree = 2 * n + 1
                t = xi[:num_edge_nodes]  # 取第一條邊上的點 (eta=0)
                
                max_edge_err = 0.0
                for m in range(target_edge_degree + 1):
                    exact = 1.0 / (m + 1.0)  # ∫_0^1 t^m dt
                    num = np.sum(w_e * (t**m))
                    max_edge_err = max(max_edge_err, abs(num - exact))
                    
                print(f"[{rule.upper()}] n={n} | 線積分精度 (至 {target_edge_degree} 階): 最大誤差 = {max_edge_err:.3e}")
                assert max_edge_err < 1e-13, f"[{rule}] n={n} 線積分驗證失敗！"
                
    print("\n✅ 所有積分精度測試完美通過！")

if __name__ == "__main__":
    test_quadrature_accuracy()