import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 呼叫您的神級 OOP 引擎
from core import build_local_operators, build_vandermonde2d
from core.cases import gaussian_bell
def smooth_bump(r, s):
    """測試波浪：高斯波包"""
    return np.exp(-5.0 * (r**2 + s**2))

def get_uniform_dense_grid(N_dense):
    r, s = [], []
    for i in range(N_dense + 1):
        for j in range(N_dense + 1 - i):
            r.append(-1.0 + 2.0 * i / N_dense)
            s.append(-1.0 + 2.0 * j / N_dense)
    return np.array(r), np.array(s)

def plot_3d_error(rule="table2"):
    # 🌟 設定為與目標圖片相同的 N=4
    N = 4
    n = 4
    
    # 1. 啟動 OOP 引擎
    engine = build_local_operators(N=N, n=n, rule=rule)
    r_nodes, s_nodes = engine.r, engine.s
    
    # 換回真實的物理波浪！
    u_nodes = gaussian_bell(r_nodes, s_nodes)
    u_hat = engine.get_modal_coeffs(u_nodes)
    
    # 2. 模態重構 (Dense Grid)
    dense_N = 50  # 開高一點讓曲面更滑順
    r_dense, s_dense = get_uniform_dense_grid(dense_N)
    V_dense, _, _ = build_vandermonde2d(N, r_dense, s_dense)
    
    u_interp = V_dense @ u_hat
    u_exact = gaussian_bell(r_dense, s_dense) 
    
    # 計算誤差 (此時誤差大約在 +-0.4，是非常宏觀平滑的波浪)
    error_field = u_interp - u_exact
    max_err = np.max(np.abs(error_field))
    
    # ==========================================
    # 3. 繪圖
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    triang = mtri.Triangulation(r_dense, s_dense)
    
    # 畫出 3D 誤差表面
    surf = ax.plot_trisurf(r_dense, s_dense, error_field, 
                           triangles=triang.triangles, 
                           cmap='viridis', # 您可以改成 'Blues_r' 會更像目標圖
                           linewidth=0.2, 
                           edgecolor='black',
                           alpha=0.85)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, format="%.2f", label="u_recon - u_exact")
    
    # 畫出底部的參考三角形外框 (橘色線條)
    ax.plot([-1, 1, -1, -1], [-1, -1, 1, -1], zdir='z', zs=np.min(error_field), color='darkorange', lw=2)
    
    # 將節點畫在底部的平面上
    z_nodes = np.full_like(r_nodes, np.min(error_field))
    ax.scatter(r_nodes, s_nodes, z_nodes, color='tab:blue', s=50, zorder=5)
    
    # 設定標題與座標軸
    ax.set_title(f"3D error field: {rule.capitalize()} order {N}, N={N}, case=gaussian bell", fontsize=14)
    ax.set_xlabel('r axis')
    ax.set_ylabel('s axis')
    ax.set_zlabel('Error')
    
    # 設定一個能看清楚整個波浪起伏的視角
    ax.view_init(elev=25, azim=-55)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3d_error(rule="table2")