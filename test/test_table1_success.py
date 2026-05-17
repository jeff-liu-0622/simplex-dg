import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 1. 呼叫您打造的頂級 OOP 引擎！
from core import build_local_operators, build_vandermonde2d

# 🌟 2. 從您的兵器庫匯入測試案例 (拿掉原本寫死的 smooth_bump)
from core.cases import gaussian_bell

def get_uniform_dense_grid(N_dense):
    """生成用來畫平滑等高線的『密集觀測網格』 (留在測試腳本內，保持 core 純潔)"""
    r, s = [], []
    for i in range(N_dense + 1):
        for j in range(N_dense + 1 - i):
            r.append(-1.0 + 2.0 * i / N_dense)
            s.append(-1.0 + 2.0 * j / N_dense)
    return np.array(r), np.array(s)

def test_engine_with_table1_nodes():
    N = 3  # 多項式階數 (對應您圖片中的 10 個波浪自由度)
    n = 4  # 積分階數 (負責提供高精度的點位與面積權重 w_s)
    
    # ==========================================
    # 1. 取得 Table 1 節點與 OOP 引擎
    # ==========================================
    engine = build_local_operators(N=N, n=n, rule="table1")
    r_nodes = engine.r
    s_nodes = engine.s
    
    # 🌟 3. 使用套件庫的高斯波包！
    u_nodes = gaussian_bell(r_nodes, s_nodes)
    
    # ==========================================
    # 2. 模態轉換 (Nodal -> Modal)
    # ==========================================
    u_hat = engine.get_modal_coeffs(u_nodes)
    
    # ==========================================
    # 3. 模態重構/外插 (Modal -> Nodal 密集網格)
    # ==========================================
    dense_N = 40  
    r_dense, s_dense = get_uniform_dense_grid(dense_N)
    V_dense, _, _ = build_vandermonde2d(N, r_dense, s_dense)
    
    u_interp = V_dense @ u_hat
    
    # 🌟 4. 計算誤差時，也要呼叫套件庫的高斯波包
    u_exact = gaussian_bell(r_dense, s_dense)
    error_field = u_interp - u_exact
    
    # ==========================================
    # 4. 開始繪圖
    # ==========================================
    fig, ax = plt.subplots(figsize=(7, 6))
    triang = mtri.Triangulation(r_dense, s_dense)
    
    levels = np.linspace(np.min(error_field), np.max(error_field), 25)
    contour = ax.tricontourf(triang, error_field, levels=levels, cmap='coolwarm')
    fig.colorbar(contour, ax=ax, format="%.2e")
    
    ax.scatter(r_nodes, s_nodes, color='black', s=50, zorder=5, label=f'Table 1 Nodes (n={n})')
    
    for i in range(len(r_nodes)):
        ax.text(r_nodes[i]+0.05, s_nodes[i]+0.05, f'K{i}', fontsize=10, fontweight='bold', color='black')
    
    ax.plot([-1, 1, -1, -1], [-1, -1, 1, -1], 'k-', lw=2)
    
    max_err = np.max(np.abs(error_field))
    
    # 標題也順便更新，讓您知道現在跑的是哪個 case
    ax.set_title(f"Case: Gaussian Bell | Table 1 (N={N}, n={n})\nMax Galerkin Error: {max_err:.2e}", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_engine_with_table1_nodes()