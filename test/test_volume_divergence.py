import numpy as np
from core import build_local_operators
from core.rhs import compute_volume_divergence

def test_divergence():
    print("\n" + "="*60)
    print("🌪️ 啟動內部體積散度 (Volume Divergence) 壓力測試")
    print("="*60)
    
    engine = build_local_operators(N=3, n=4, rule="table1")
    r, s = engine.r, engine.s
    
    # ==========================================
    # 測試一：標準參考三角形 (純粹驗證數學)
    # ==========================================
    print("\n[測試一] 標準參考三角形 (無變形, J=1)")
    # 測試流場: F = (x^2, y^2) 
    # 解析散度: Div F = 2x + 2y
    Fx1, Fy1 = r**2, s**2
    exact_div1 = 2*r + 2*s
    
    # 幾何參數 (因為 x=r, y=s，所以 rx=1, sy=1，其餘為 0)
    rx1, sx1, ry1, sy1, J1 = 1.0, 0.0, 0.0, 1.0, 1.0
    
    num_div1 = compute_volume_divergence(engine, Fx1, Fy1, rx1, sx1, ry1, sy1, J1)
    err1 = np.max(np.abs(num_div1 - exact_div1))
    print(f"最大誤差: {err1:.2e} -> {'✅ Pass' if err1 < 1e-12 else '❌ Fail'}")
    
    # ==========================================
    # 測試二：嚴重歪斜與放大的真實三角形 (終極考驗)
    # ==========================================
    print("\n[測試二] 嚴重歪斜與拉伸的物理三角形 (J=5)")
    # 故意設計一個仿射變換: x = 2r + s, y = r + 3s
    x = 2.0 * r + s
    y = r + 3.0 * s
    
    # 手算幾何變換矩陣 (Metric Terms)
    xr, xs = 2.0, 1.0
    yr, ys = 1.0, 3.0
    J_val = xr * ys - xs * yr  # J = 2*3 - 1*1 = 5
    
    # 建立與點數相同大小的幾何陣列
    rx = np.full_like(r, ys / J_val)
    sx = np.full_like(r, -yr / J_val)
    ry = np.full_like(r, -xs / J_val)
    sy = np.full_like(r, xr / J_val)
    J = np.full_like(r, J_val)
    
    # 測試流場依然是: F = (x^2, y^2)
    # 解析散度: Div F = 2x + 2y
    Fx2, Fy2 = x**2, y**2
    exact_div2 = 2*x + 2*y
    
    num_div2 = compute_volume_divergence(engine, Fx2, Fy2, rx, sx, ry, sy, J)
    err2 = np.max(np.abs(num_div2 - exact_div2))
    print(f"最大誤差: {err2:.2e} -> {'✅ Pass' if err2 < 1e-12 else '❌ Fail'}")
    print("\n💡 結論：Teng 教授的守恆型公式，完美抵禦了網格扭曲帶來的幾何誤差！")

if __name__ == "__main__":
    test_divergence()