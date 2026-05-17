import numpy as np
import matplotlib.pyplot as plt

from core.geometry.connectivity import build_connectivity


def build_test_mesh():
    """建立一個 2x2 正方形切出來的 8 個三角形網格。"""
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)

    X, Y = np.meshgrid(x, y)
    VX = X.flatten()
    VY = Y.flatten()

    EToV = []

    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            n1 = n0 + 1
            n2 = n0 + 3
            n3 = n2 + 1

            # 兩個三角形都保持逆時針。
            EToV.append([n0, n1, n3])
            EToV.append([n3, n2, n0])

    return VX, VY, np.array(EToV, dtype=int)


def check_connectivity_counts(EToV, EToE):
    """
    2x2 square mesh:
        4 squares
        8 triangles
        total directed faces = 8 * 3 = 24

    Unique edges:
        horizontal grid edges: 3 rows * 2 segments = 6
        vertical grid edges:   3 cols * 2 segments = 6
        diagonals:             4
        total unique edges = 16

    Boundary edges:
        outer square has 8 boundary segments

    Interior unique edges:
        16 - 8 = 8

    Directed interior faces:
        8 * 2 = 16

    Directed boundary faces:
        8
    """
    K = EToV.shape[0]
    total_directed_faces = 3 * K

    boundary_count = np.sum(EToE == np.arange(K)[:, None])
    interior_directed_count = total_directed_faces - boundary_count

    print(f"  total directed faces      = {total_directed_faces}")
    print(f"  boundary directed faces   = {boundary_count}")
    print(f"  interior directed faces   = {interior_directed_count}")

    assert total_directed_faces == 24
    assert boundary_count == 8
    assert interior_directed_count == 16


def check_neighbor_symmetry(EToE, EToF):
    """
    如果 k 的 face f 指向 k2 的 face f2，
    那麼 k2 的 face f2 必須指回 k 的 face f。
    """
    K = EToE.shape[0]

    for k in range(K):
        for f in range(3):
            k2 = EToE[k, f]
            f2 = EToF[k, f]

            if k2 == k:
                continue

            assert EToE[k2, f2] == k, (
                f"Neighbor symmetry failed: "
                f"({k},{f}) -> ({k2},{f2}), "
                f"but reverse is ({EToE[k2, f2]},{EToF[k2, f2]})"
            )

            assert EToF[k2, f2] == f, (
                f"Face symmetry failed: "
                f"({k},{f}) -> ({k2},{f2}), "
                f"but reverse face is {EToF[k2, f2]}"
            )


def plot_mesh_connectivity(VX, VY, EToV, EToE):
    """
    將網格與連通性畫出來。

    Face ordering must match connectivity.py:

        face 0: v0 -> v1
        face 1: v1 -> v2
        face 2: v2 -> v0
    """
    plt.figure(figsize=(8, 8))

    K = EToV.shape[0]

    face_vids = [
        [0, 1],
        [1, 2],
        [2, 0],
    ]

    for k in range(K):
        vids = EToV[k]
        x = VX[vids]
        y = VY[vids]

        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])

        plt.plot(x_closed, y_closed, "k-", lw=1, alpha=0.5)

        cx, cy = np.mean(x), np.mean(y)

        plt.text(
            cx,
            cy,
            str(k),
            color="red",
            fontsize=14,
            weight="bold",
            ha="center",
            va="center",
        )

        for f in range(3):
            neighbor = EToE[k, f]

            if neighbor != k:
                n_vids = EToV[neighbor]
                nx = np.mean(VX[n_vids])
                ny = np.mean(VY[n_vids])

                plt.annotate(
                    "",
                    xy=(nx, ny),
                    xytext=(cx, cy),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="blue",
                        alpha=0.6,
                        lw=1.5,
                        shrinkA=20,
                        shrinkB=20,
                    ),
                )
            else:
                va = vids[face_vids[f][0]]
                vb = vids[face_vids[f][1]]

                plt.plot(
                    [VX[va], VX[vb]],
                    [VY[va], VY[vb]],
                    color="orange",
                    lw=4,
                )

    plt.title("Nodal DG Mesh Connectivity Map (EToE)", fontsize=16, pad=20)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_all_tests(show_plot=True):
    print("🚀 啟動網格拓撲測試...")

    VX, VY, EToV = build_test_mesh()
    print(f"✅ 成功生成 {EToV.shape[0]} 個三角形網格")

    EToE, EToF = build_connectivity(EToV)
    print("✅ 拓撲矩陣 EToE, EToF 計算完成")

    print("🔍 檢查 connectivity 數量...")
    check_connectivity_counts(EToV, EToE)
    print("✅ connectivity 數量正確")

    print("🔍 檢查 neighbor symmetry...")
    check_neighbor_symmetry(EToE, EToF)
    print("✅ neighbor symmetry 正確")

    if show_plot:
        print("📊 正在繪製拓撲連線圖...")
        plot_mesh_connectivity(VX, VY, EToV, EToE)

    print("🎉 test_connectivity.py 全部測試通過")


if __name__ == "__main__":
    run_all_tests(show_plot=True)