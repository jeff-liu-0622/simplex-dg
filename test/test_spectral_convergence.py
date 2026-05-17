import numpy as np
import matplotlib.pyplot as plt
from core import build_local_operators

def test_spectral_convergence():
    print("\n" + "="*60)
    print("📈 Nodal DG 引擎：頻譜收斂性測試 (Spectral Convergence)")
    print("測試函數: u(r, s) = sin(0.1*π*r) * cos(0.1*π*s)")
    print("="*60)
    
    # 配合您 Table 1 的極限，我們只測到 N=4
    N_list = [1, 2, 3, 4]
    
    errors_t1 = []
    errors_t2 = []
    
    for rule, errors_list in zip(["table1", "table2"], [errors_t1, errors_t2]):
        print(f"\n計算 {rule} 的誤差中...")
        for N in N_list:
            # 讓 n=N，確保不會呼叫到不存在的 n=5 節點表
            engine = build_local_operators(N=N, n=N, rule=rule)
            r, s = engine.r, engine.s
            
            # 🌟 修正：配合那份 Notebook 的頻率 (0.1*pi)
            freq = 0.1 * np.pi
            u = np.sin(freq * r) * np.cos(freq * s)
            
            # 解析微分
            du_dr_exact = freq * np.cos(freq * r) * np.cos(freq * s)
            du_ds_exact = -freq * np.sin(freq * r) * np.sin(freq * s)
            
            # 引擎微分
            du_dr_num = engine.Dr @ u
            du_ds_num = engine.Ds @ u
            
            # 計算最大誤差
            err_r = np.max(np.abs(du_dr_num - du_dr_exact))
            err_s = np.max(np.abs(du_ds_num - du_ds_exact))
            max_err = max(err_r, err_s)
            
            errors_list.append(max_err)
            print(f"  N={N}, Max Error = {max_err:.4e}")

    # ==========================================
    # 畫圖：加入 r 前綴，解決 SyntaxWarning
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    plt.semilogy(N_list, errors_t1, 'bo-', linewidth=2, markersize=8, label='Table 1 (Boundary/Interior)')
    plt.semilogy(N_list, errors_t2, 'rs-', linewidth=2, markersize=8, label='Table 2 (Pure Interior)')
    
    # 加上 r 前綴，告訴 Python 這是 Raw String
    plt.title(r"Spectral Convergence of Differentiation Matrices" + "\n" + r"$u = \sin(0.1\pi r)\cos(0.1\pi s)$", fontsize=14, fontweight='bold')
    plt.xlabel("Polynomial Order (N)", fontsize=12)
    plt.ylabel(r"Max Error ($L_\infty$)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xticks(N_list)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_spectral_convergence()