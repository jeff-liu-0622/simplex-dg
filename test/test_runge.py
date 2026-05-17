import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

# ==========================================
# 1. 測試函式 (魔王：龍格函數)
# ==========================================
def runge_function(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# ==========================================
# 2. 真實的高斯積分點 (神聖節點的 1D 化身)
# ==========================================
def get_table2_nodes(N):
    """
    真實 Table 2 (1D 化身)：高斯-勒讓德點 (Gauss-Legendre Nodes)
    特徵：純內部點，完全不會碰到邊界 -1 與 1。積分精度最高。
    """
    # leggauss 會回傳 (節點, 權重)，我們畫圖只需要節點 [0]
    nodes, weights = leggauss(N + 1)
    return nodes

def get_table1_nodes(N):
    """
    真實 Table 1 (1D 化身)：切比雪夫-洛巴托點 (Chebyshev-Lobatto Nodes)
    特徵：強制包含邊界點 (-1, 1)，點距向兩側漸密。
    """
    j = np.arange(N + 1)
    return -np.cos(j * np.pi / N)

# ==========================================
# 3. 主程式：神仙打架 (Table 2 vs Table 1)
# ==========================================
def run_divine_interpolation_test(N=14):
    x_dense = np.linspace(-1, 1, 1000)
    y_true = runge_function(x_dense)

    # 取得兩種「神聖」模式的節點
    x_table2 = get_table2_nodes(N)
    x_table1 = get_table1_nodes(N)

    y_table2 = runge_function(x_table2)
    y_table1 = runge_function(x_table1)

    # 計算多項式 (因為節點都很神聖，所以不會跳 RankWarning 警告了！)
    p_table2 = np.polyfit(x_table2, y_table2, N)
    p_table1 = np.polyfit(x_table1, y_table1, N)

    y_eval_table2 = np.polyval(p_table2, x_dense)
    y_eval_table1 = np.polyval(p_table1, x_dense)

    # 計算最大誤差
    err_table2 = np.max(np.abs(y_eval_table2 - y_true))
    err_table1 = np.max(np.abs(y_eval_table1 - y_true))

    # --- 繪圖 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"The Divine Nodes Interpolation (N={N})", fontsize=16)

    # 左圖：Table 2 (純內部點)
    ax1.plot(x_dense, y_true, 'k--', lw=2, label="True Function")
    ax1.plot(x_dense, y_eval_table2, 'r-', lw=2, label="Poly (Table 2 / GL)")
    ax1.scatter(x_table2, y_table2, color='red', s=50, zorder=5, label="Nodes (No Boundary)")
    ax1.set_title(f"Table 2 Style (Gauss-Legendre)\nMax Error: {err_table2:.2e}", color='red')
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.5)
    ax1.legend()

    # 右圖：Table 1 (含邊界點)
    ax2.plot(x_dense, y_true, 'k--', lw=2, label="True Function")
    ax2.plot(x_dense, y_eval_table1, 'g-', lw=2, label="Poly (Table 1 / Lobatto)")
    ax2.scatter(x_table1, y_table1, color='green', s=50, zorder=5, label="Nodes (Has Boundary)")
    ax2.set_title(f"Table 1 Style (Chebyshev-Lobatto)\nMax Error: {err_table1:.2e}", color='green')
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.5)
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    # 就算把 N 開到 20，這兩組神仙點也絕對不會爆炸！
    run_divine_interpolation_test(N=16)