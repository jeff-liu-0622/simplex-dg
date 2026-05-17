import numpy as np
import matplotlib.pyplot as plt
from core import build_local_operators, build_vandermonde2d

def verify_projection_matrices():
    print("\n=== 黑箱破解：驗證 Galerkin 投影矩陣的數學真理 ===")
    
    N = 4 
    n = 4 
    
    # 1. 取得引擎與 V 矩陣
    engine = build_local_operators(N=N, n=n, rule="table1")
    r, s = engine.r, engine.s
    V, _, _ = build_vandermonde2d(N, r, s)
    
    Np = len(r)          # 點數：22
    Nmodes = V.shape[1]  # 基底數：15
    
    # ==========================================
    # 實驗一：驗證「模態轉換無損性」 (15x15 單位矩陣)
    # ==========================================
    # 我們用迴圈把 V 矩陣的 15 個行向量 (每一行是一個基底波浪)，逐一餵給引擎
    M_modal_identity = np.zeros((Nmodes, Nmodes))
    for i in range(Nmodes):
        M_modal_identity[:, i] = engine.get_modal_coeffs(V[:, i])
    
    diag_values = np.diag(M_modal_identity)
    off_diag_max = np.max(np.abs(M_modal_identity - np.diag(diag_values)))
    
    print(f"【實驗一：基底函數的模態投影 (應為 15x15 單位矩陣)】")
    print(f"矩陣大小: {M_modal_identity.shape}")
    print(f"對角線平均值: {np.mean(diag_values):.4f} (理論值為 1.0)")
    print(f"非對角線最大誤差: {off_diag_max:.2e} (理論值為 0.0)")
    
    # ==========================================
    # 實驗二：萃取 22x22 的投影矩陣 P
    # ==========================================
    # 產生 22x22 的單位矩陣 (代表 22 個獨立的脈衝測試)
    I_22 = np.eye(Np)
    projector = np.zeros((Nmodes, Np))
    
    # 逐一把這 22 個脈衝餵給引擎，萃取出底層的轉換算子
    for i in range(Np):
        projector[:, i] = engine.get_modal_coeffs(I_22[:, i])
        
    P_matrix = V @ projector
    
    print(f"\n【實驗二：投影矩陣 P = V * (M^-1 * V^T * W)】")
    print(f"矩陣大小: {P_matrix.shape}")
    print(f"P 是不是單位矩陣？ -> {'是' if np.allclose(P_matrix, np.eye(Np)) else '絕對不是！'}")
    
    # ==========================================
    # 畫圖視覺化
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左圖：完美的 15x15 藍色對角線
    cax1 = axes[0].matshow(M_modal_identity, cmap='Blues')
    axes[0].set_title(f"Modal Identity (15x15)\nengine.get_modal_coeffs(V)", pad=20, fontweight='bold')
    fig.colorbar(cax1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 右圖：妥協的 22x22 紅色濾波器
    cax2 = axes[1].matshow(P_matrix, cmap='Reds')
    axes[1].set_title(f"Projection Matrix P (22x22)\nV @ engine.get_modal_coeffs(I)", pad=20, fontweight='bold')
    fig.colorbar(cax2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_projection_matrices()