import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# 呼叫您的神級引擎
from core.basis import build_vandermonde2d

def smooth_bump(r, s):
    return np.exp(-5.0 * (r**2 + s**2))

def get_uniform_nodes(N):
    """產生 Table 2 均勻點 (製造可見的誤差)"""
    r, s = [], []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            r.append(-1.0 + 2.0 * i / N)
            s.append(-1.0 + 2.0 * j / N)
    return np.array(r), np.array(s)

def plot_3d_with_reference_floor():
    N = 4  # 使用 N=4 的均勻點，製造約 0.46 的邊界誤差
    
    # 1. 取得節點與計算係數
    r_nodes, s_nodes = get_uniform_nodes(N)
    u_nodes = smooth_bump(r_nodes, s_nodes)
    
    V_nodes, _, _ = build_vandermonde2d(N, r_nodes, s_nodes)
    u_hat, _, _, _ = np.linalg.lstsq(V_nodes, u_nodes, rcond=None)
    
    # 2. 密集網格與誤差計算
    dense_N = 40
    r_dense, s_dense = get_uniform_nodes(dense_N)
    V_dense, _, _ = build_vandermonde2d(N, r_dense, s_dense)
    
    u_interp = V_dense @ u_hat
    u_exact = smooth_bump(r_dense, s_dense)
    error_field = u_interp - u_exact
    
    # ==========================================
    # 開始繪製「雙層 3D 圖」
    # ==========================================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    triang = mtri.Triangulation(r_dense, s_dense)
    
    # 【第一層】：畫出底部的基準三角形 (Z=0 的完美零誤差地板)
    z_zero = np.zeros_like(r_dense)
    ax.plot_trisurf(r_dense, s_dense, z_zero, 
                    triangles=triang.triangles, 
                    color='gray', alpha=0.3, edgecolor='none')
    
    # 畫出地板的黑色三角形邊框，讓它更明顯
    ax.plot([-1, 1, -1, -1], [-1, -1, 1, -1], [0, 0, 0, 0], 'k--', lw=2, label='Z=0 Ground (Zero Error)')

    # 【第二層】：畫出真實的誤差曲面
    surf = ax.plot_trisurf(r_dense, s_dense, error_field, 
                           triangles=triang.triangles, 
                           cmap='coolwarm', 
                           linewidth=0.1, 
                           edgecolor='black',
                           alpha=0.8)
    
    # 【魔法連線】：畫出那 15 個節點，並用垂直線連到地板上
    # 先算出那 15 個節點在重構後的真實誤差高度
    error_at_nodes = (V_nodes @ u_hat) - smooth_bump(r_nodes, s_nodes)
    
    for i in range(len(r_nodes)):
        # 畫出紅色的點 (懸浮在誤差曲面上)
        ax.scatter(r_nodes[i], s_nodes[i], error_at_nodes[i], color='red', s=40, zorder=5)
        # 畫垂直輔助線 (從地面的 0 連到空中的誤差高度)
        ax.plot([r_nodes[i], r_nodes[i]], [s_nodes[i], s_nodes[i]], [0, error_at_nodes[i]], 
                'r:', lw=2)

    # 加上 Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, format="%.2f", label="Error Magnitude")
    
    # 設定視角與標籤
    ax.set_title(f"3D Error Surface vs Z=0 Floor (Table 2, N={N})\nObserve the gap at the boundaries!", fontsize=16)
    ax.set_xlabel('r axis')
    ax.set_ylabel('s axis')
    ax.set_zlabel('Error')
    
    # 設定 Z 軸的比例，讓您可以清楚看到山峰與山谷
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=20, azim=-35) # 稍微低一點的視角，最適合看高度差
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3d_with_reference_floor()