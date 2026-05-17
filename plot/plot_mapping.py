import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 呼叫您無敵的 OOP 引擎
from core import build_local_operators

def plot_2d_to_3d_mapping():
    """
    視覺化：將 2D 參考三角形上的節點，映射到 3D 實體三角形 (正八面體的一個面)
    """
    # 1. 啟動 OOP 引擎 (設定 N=3, n=4, 預設使用 Table 1)
    engine = build_local_operators(N=3, n=4, rule="table2")
    r = engine.r
    s = engine.s
    Np = len(r)

    # 2. 定義 3D 空間中的三角形頂點 (類似正八面體的一個面)
    v1_3d = np.array([1.0, 0.0, 0.0])
    v2_3d = np.array([0.0, 1.0, 0.0])
    v3_3d = np.array([0.0, 0.0, 1.0])

    # 3. 進行 3D 幾何映射 (重心座標轉換)
    L1 = -0.5 * (r + s)
    L2 = 0.5 * (1.0 + r)
    L3 = 0.5 * (1.0 + s)

    x_3d = L1 * v1_3d[0] + L2 * v2_3d[0] + L3 * v3_3d[0]
    y_3d = L1 * v1_3d[1] + L2 * v2_3d[1] + L3 * v3_3d[1]
    z_3d = L1 * v1_3d[2] + L2 * v2_3d[2] + L3 * v3_3d[2]

    # --- 開始繪圖 ---
    fig = plt.figure(figsize=(14, 6))

    # ==========================================
    # 左圖：2D 參考三角形 (Reference Element)
    # ==========================================
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([-1, 1, -1, -1], [-1, -1, 1, -1], color='#4A90E2', lw=2)
    ax1.scatter(r, s, color='red', s=80, zorder=5)
    
    for i in range(Np):
        ax1.text(r[i]+0.05, s[i]+0.05, str(i+1), fontsize=12, fontweight='bold')

    ax1.set_title("2D Reference Triangle (r, s)", fontsize=14, pad=15)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlabel('r axis', fontsize=12)
    ax1.set_ylabel('s axis', fontsize=12)

    # ==========================================
    # 右圖：3D 物理三角形 (Physical Element in 3D)
    # ==========================================
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot([v1_3d[0], v2_3d[0], v3_3d[0], v1_3d[0]], 
             [v1_3d[1], v2_3d[1], v3_3d[1], v1_3d[1]], 
             [v1_3d[2], v2_3d[2], v3_3d[2], v1_3d[2]], color='#50E3C2', lw=2)
    
    ax2.scatter(x_3d, y_3d, z_3d, color='red', s=80, depthshade=False)
    
    for i in range(Np):
        ax2.text(x_3d[i], y_3d[i], z_3d[i]+0.05, str(i+1), fontsize=12, fontweight='bold')

    ax2.set_title("3D Physical Element (x, y, z)", fontsize=14, pad=15)
    ax2.view_init(elev=30, azim=45) 
    ax2.set_xlabel('X axis', fontsize=12)
    ax2.set_ylabel('Y axis', fontsize=12)
    ax2.set_zlabel('Z axis', fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_2d_to_3d_mapping()