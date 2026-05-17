import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 呼叫您打造的頂級 OOP 引擎！
from core import build_local_operators, build_vandermonde2d

def smooth_bump(r, s):
    """測試波浪：高斯波包"""
    return np.exp(-5.0 * (r**2 + s**2))

def get_uniform_dense_grid(N_dense):
    """生成用來畫平滑等高線的『密集觀測網格』"""
    r, s = [], []
    for i in range(N_dense + 1):
        for j in range(N_dense + 1 - i):
            r.append(-1.0 + 2.0 * i / N_dense)
            s.append(-1.0 + 2.0 * j / N_dense)
    return np.array(r), np.array(s)

def test_engine_with_table2_nodes():
    N = 3  # 多項式階數 (波浪的解析度)
    n = 4  # 積分階數 
    
    # ==========================================
    # 1. 取得 Table 2 節點與 OOP 引擎
    # ==========================================
    # 🌟 魔法在此：把 rule 換成 "table2"，引擎就會自動載入純內部的高階神聖點！
    engine = build_local_operators(N=N, n=n, rule="table2")
    
    # 取出 Table 2 的座標
    r_nodes = engine.r
    s_nodes = engine.s
    
    # 算一下真實波浪在這些點上的高度
    u_nodes = smooth_bump(r_nodes, s_nodes)
    
    # ==========================================
    # 2. 模態轉換 (Nodal -> Modal)
    # ==========================================
    # 同樣使用完美的 Galerkin 投影 (面積權重 w_s 已經自動替換成 Table 2 的版本了)
    u_hat = engine.get_modal_coeffs(u_nodes)
    
    # ==========================================
    # 3. 模態重構/外插 (Modal -> Nodal 密集網格)
    # ==========================================
    dense_N = 40  # 撒下 800多個密集點來畫圖
    r_dense, s_dense = get_uniform_dense_grid(dense_N)
    
    V_dense, _, _ = build_vandermonde2d(N, r_dense, s_dense)
    
    # 矩陣相乘，算出預測值
    u_interp = V_dense @ u_hat
    u_exact = smooth_bump(r_dense, s_dense)
    
    # 計算誤差場
    error_field = u_interp - u_exact
    
    # ==========================================
    # 4. 開始繪圖
    # ==========================================
    fig, ax = plt.subplots(figsize=(7, 6))
    triang = mtri.Triangulation(r_dense, s_dense)
    
    # 畫出誤差等高線
    levels = np.linspace(np.min(error_field), np.max(error_field), 25)
    contour = ax.tricontourf(triang, error_field, levels=levels, cmap='coolwarm')
    fig.colorbar(contour, ax=ax, format="%.2e")
    
    # 把 Table 2 的點畫上去！
    ax.scatter(r_nodes, s_nodes, color='black', s=50, zorder=5, label=f'Table 2 Nodes (n={n})')
    
    # 標上 K0 ~ K_last 的文字
    for i in range(len(r_nodes)):
        ax.text(r_nodes[i]+0.03, s_nodes[i]+0.03, f'K{i}', fontsize=9, fontweight='bold', color='black')
    
    # 畫外框
    ax.plot([-1, 1, -1, -1], [-1, -1, 1, -1], 'k-', lw=2)
    
    # 標題顯示最大誤差！
    max_err = np.max(np.abs(error_field))
    ax.set_title(f"OOP Engine with Table 2 (N={N}, n={n})\nMax Galerkin Error: {max_err:.2e}", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_engine_with_table2_nodes()