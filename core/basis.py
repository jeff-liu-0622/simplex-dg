import numpy as np
from math import gamma
from typing import Tuple

def jacobi_p(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """計算標準正交的雅可比多項式 P_N^{(alpha,beta)}(x)"""
    x = np.asarray(x, dtype=float).reshape(-1)
    PL = np.zeros((N + 1, x.size), dtype=float)

    # 初始值 P_0
    gamma0 = (2.0 ** (alpha + beta + 1.0) / (alpha + beta + 1.0) *
              gamma(alpha + 1.0) * gamma(beta + 1.0) / gamma(alpha + beta + 1.0))
    PL[0, :] = 1.0 / np.sqrt(gamma0)
    if N == 0: return PL[0, :]

    # 初始值 P_1
    gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
    PL[1, :] = (((alpha + beta + 2.0) * x / 2.0) + (alpha - beta) / 2.0) / np.sqrt(gamma1)
    if N == 1: return PL[1, :]

    # 遞迴關係式計算高階
    aold = 2.0 / (2.0 + alpha + beta) * np.sqrt((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0))
    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        anew = 2.0 / (h1 + 2.0) * np.sqrt((i + 1.0) * (i + 1.0 + alpha + beta) * (i + 1.0 + alpha) * (i + 1.0 + beta) / ((h1 + 1.0) * (h1 + 3.0)))
        bnew = -(alpha ** 2 - beta ** 2) / (h1 * (h1 + 2.0))
        PL[i + 1, :] = ((x - bnew) * PL[i, :] - aold * PL[i - 1, :]) / anew
        aold = anew
    return PL[N, :]

def grad_jacobi_p(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """計算雅可比多項式的導數"""
    x = np.asarray(x, dtype=float).reshape(-1)
    if N == 0: return np.zeros_like(x)
    return np.sqrt(N * (N + alpha + beta + 1.0)) * jacobi_p(x, alpha + 1.0, beta + 1.0, N - 1)

def rstoab(r: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """將三角形座標 (r,s) 映射到退化矩形座標 (a,b)"""
    r = np.asarray(r, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    a = np.empty_like(r)
    mask = np.abs(1.0 - s) > 1e-14
    a[mask] = 2.0 * (1.0 + r[mask]) / (1.0 - s[mask]) - 1.0
    a[~mask] = -1.0 # 處理頂點奇點
    return a, s

def simplex2d_p(a: np.ndarray, b: np.ndarray, i: int, j: int) -> np.ndarray:
    """2D 三角形正交基底函數 psi_{ij}(a,b)"""
    h1 = jacobi_p(a, 0.0, 0.0, i)
    h2 = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    return np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i

def grad_simplex2d_p(a: np.ndarray, b: np.ndarray, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """2D 三角形正交基底函數的導數 d/dr 與 d/ds"""
    fa = jacobi_p(a, 0.0, 0.0, i)
    dfa = grad_jacobi_p(a, 0.0, 0.0, i)
    gb = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    dgb = grad_jacobi_p(b, 2.0 * i + 1.0, 0.0, j)

    # 連鎖律轉換導數
    dmodedr = dfa * gb
    if i > 0: dmodedr *= (0.5 * (1.0 - b)) ** (i - 1)

    dmodeds = dfa * (gb * (0.5 * (1.0 + a)))
    if i > 0: dmodeds *= (0.5 * (1.0 - b)) ** (i - 1)

    tmp = dgb * (0.5 * (1.0 - b)) ** i
    if i > 0: tmp -= 0.5 * i * gb * (0.5 * (1.0 - b)) ** (i - 1)
    dmodeds += fa * tmp

    scale = 2.0 ** (i + 0.5)
    return scale * dmodedr, scale * dmodeds

def build_vandermonde2d(N: int, r: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """建構范德蒙矩陣 V 以及梯度矩陣 Vr, Vs"""
    Np = (N + 1) * (N + 2) // 2
    r, s = np.asarray(r), np.asarray(s)
    V = np.zeros((r.size, Np), dtype=float)
    Vr = np.zeros((r.size, Np), dtype=float)
    Vs = np.zeros((r.size, Np), dtype=float)

    a, b = rstoab(r, s)
    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            V[:, sk] = simplex2d_p(a, b, i, j)
            Vr[:, sk], Vs[:, sk] = grad_simplex2d_p(a, b, i, j)
            sk += 1
    return V, Vr, Vs