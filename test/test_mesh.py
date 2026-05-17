import numpy as np

# 享受套件化引入
from core import build_local_operators
from core.mesh import map_reference_to_physical, compute_physical_derivatives

def test_geometric_mapping():
    print("\n" + "="*50)
    print("啟動幾何映射與連鎖律微分測試 (Geometric Mapping Test)")
    print("="*50)
    
    # 🌟 雙引擎測試：Table 1 與 Table 2 都要能完美適應幾何變形
    for rule in ["table1", "table2"]:
        print(f"\n--- 測試 [{rule.upper()}] (N=3, n=4) ---")
        
        # 1. 取得參考三角形的工具 (直接呼叫 OOP 引擎)
        engine = build_local_operators(N=3, n=4, rule=rule)
        
        # 享受極致優雅的屬性提取
        r, s, w_s = engine.r, engine.s, engine.w_s
        Dr, Ds = engine.Dr, engine.Ds
        
        # 2. 定義一個真實世界的三角形：底為 3，高為 4 的直角三角形
        # 頂點分別在 (0,0), (3,0), (0,4)
        v1, v2, v3 = (0.0, 0.0), (3.0, 0.0), (0.0, 4.0)
        
        # 3. 進行幾何映射
        geo = map_reference_to_physical(r, s, v1, v2, v3)
        x, y, J = geo["x"], geo["y"], geo["J"]
        
        # --- 驗證 A: 物理面積積分 ---
        # H&W 標準三角形面積是 2.0，所以數值積分要乘 2.0
        numerical_area = 2.0 * np.sum(w_s * abs(J)) 
        
        print(f"  理論面積: 6.0, 數值積分面積: {numerical_area:.3f}")
        assert abs(numerical_area - 6.0) < 1e-13, f"[{rule}] 幾何 Jacobian 面積轉換失敗！"

        # --- 驗證 B: 物理空間的連鎖律微分 ---
        # 給定一個真實物理空間的函數 u(x, y) = 2*x + 5*y
        # 理論偏導數應該是：du/dx = 2.0, du/dy = 5.0
        u = 2.0 * x + 5.0 * y
        
        ux, uy = compute_physical_derivatives(Dr, Ds, geo["rx"], geo["sx"], geo["ry"], geo["sy"], u)
        
        err_x = np.max(np.abs(ux - 2.0))
        err_y = np.max(np.abs(uy - 5.0))
        print(f"  物理空間斜率誤差: du/dx 誤差 = {err_x:.2e}, du/dy 誤差 = {err_y:.2e}")
        
        assert err_x < 1e-13, f"[{rule}] 物理空間 x 方向微分失敗！"
        assert err_y < 1e-13, f"[{rule}] 物理空間 y 方向微分失敗！"
        
        print(f"✅ [{rule.upper()}] 幾何映射與微分驗證完美通過！")

if __name__ == "__main__":
    test_geometric_mapping()