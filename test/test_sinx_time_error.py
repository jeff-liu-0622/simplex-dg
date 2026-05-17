import numpy as np
import matplotlib.pyplot as plt

from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.rhs import compute_surface_rhs_matrix_free
from core.time_integration import lsrk54_step 

# ==========================================
# 幾何配對雷達 (與之前完全相同)
# ==========================================
def build_pure_geometric_maps(x_nodes, y_nodes, fmask):
    K, Np = x_nodes.shape
    Nfaces = 3
    Nfp = len(fmask[0])
    vmapM = np.zeros((K, Nfaces, Nfp), dtype=int)
    vmapP = np.zeros((K, Nfaces, Nfp), dtype=int)
    x_flat, y_flat = x_nodes.flatten(), y_nodes.flatten()
    
    for k in range(K):
        for f in range(Nfaces):
            vmapM[k, f, :] = k * Np + fmask[f]
            
    face_centers = {}
    for k in range(K):
        for f in range(Nfaces):
            m_idx = vmapM[k, f, :]
            cx, cy = np.mean(x_flat[m_idx]), np.mean(y_flat[m_idx])
            face_centers[(k, f)] = (round(cx, 8), round(cy, 8))
            
    center_to_face = {}
    for (k, f), coords in face_centers.items():
        if coords not in center_to_face: center_to_face[coords] = []
        center_to_face[coords].append((k, f))

    for k1 in range(K):
        for f1 in range(Nfaces):
            coords = face_centers[(k1, f1)]
            neighbors = [nf for nf in center_to_face[coords] if nf != (k1, f1)]
            if not neighbors:
                vmapP[k1, f1, :] = vmapM[k1, f1, :]
            else:
                k2, f2 = neighbors[0]
                nodes1, nodes2 = vmapM[k1, f1, :], vmapM[k2, f2, :]
                for i, n1 in enumerate(nodes1):
                    dist = (x_flat[nodes2] - x_flat[n1])**2 + (y_flat[nodes2] - y_flat[n1])**2
                    vmapP[k1, f1, i] = nodes2[np.argmin(dist)]
    return vmapM, vmapP

# ==========================================
# 解析解 (🌟 確保波浪與風向對齊 X 軸)
# ==========================================
def q_exact(x, y, t, cx=1.0):
    return np.sin(x - cx * t)

# ==========================================
# 引擎包裝函數 (專門給 LSRK 呼叫)
# ==========================================
def compute_full_rhs(q, t, **kwargs):
    engine = kwargs['engine']
    rx, sx, ry, sy = kwargs['rx'], kwargs['sx'], kwargs['ry'], kwargs['sy']
    nx, ny, sJ, J = kwargs['nx'], kwargs['ny'], kwargs['sJ'], kwargs['J']
    vmapM, vmapP = kwargs['vmapM'], kwargs['vmapP']
    cx, cy = kwargs['cx'], kwargs['cy']
    x_nodes, y_nodes = kwargs['x_nodes'], kwargs['y_nodes']

    qr, qs = (engine.Dr @ q.T).T, (engine.Ds @ q.T).T
    rhs = -(cx * (rx[:, None]*qr + sx[:, None]*qs) + cy * (ry[:, None]*qr + sy[:, None]*qs))

    for f in range(3):
        qM = q.flatten()[vmapM[:, f, :]]
        qP = q.flatten()[vmapP[:, f, :]]
        
        is_boundary = np.all(vmapM[:, f, :] == vmapP[:, f, :], axis=1)
        if np.any(is_boundary):
            xf_bnd = x_nodes.flatten()[vmapM[is_boundary, f, :]]
            yf_bnd = y_nodes.flatten()[vmapM[is_boundary, f, :]]
            qP[is_boundary, :] = q_exact(xf_bnd, yf_bnd, t, cx=cx)
            
        rhs += compute_surface_rhs_matrix_free(
            u_minus=qM, u_plus=qP, 
            nx=nx[:, f:f+1], ny=ny[:, f:f+1], cx=cx, cy=cy, 
            w_face=engine.w_s[engine.fmask[f]], w_vol=engine.w_s, 
            sJ=sJ[:, f:f+1], J=J[:, None], fmask=engine.fmask[f], 
            tau=1.0 # 🌟 保持迎風阻力
        )
    return rhs

# ==========================================
# 🏁 測試主程式 (🌟 參考解模式)
# ==========================================
def run_lsrk_pure_time_test():
    print("🤝 啟動 LSRK54 【純時間收斂】測試 (Reference Mode)...")
    
    N, n = 4, 4 
    NX, NY = 8, 8      # 粗網格也沒關係，因為空間誤差會被抵消
    cx, cy = 1.0, 0.0  # 🌟 風向沿著 X 軸吹
    T_final = 0.04     # 測試總時間
    
    # --- 幾何與算子初始化 ---
    engine = build_local_operators(N, n, rule="table1")
    TOL = 1e-12
    f1_raw = np.where(abs(engine.s + 1.0) < TOL)[0]
    f2_raw = np.where(abs(engine.r + engine.s) < TOL)[0]
    f3_raw = np.where(abs(engine.r + 1.0) < TOL)[0]
    engine.fmask = [f1_raw[np.argsort(engine.r[f1_raw])], f2_raw[np.argsort(-engine.r[f2_raw])], f3_raw[np.argsort(-engine.s[f3_raw])]]
    
    VX, VY, EToV = create_square_mesh(NX, NY)
    EToV_coords_x, EToV_coords_y = VX[EToV], VY[EToV]
    rx, sx, ry, sy, J = compute_volume_metrics(EToV_coords_x, EToV_coords_y)
    nx, ny, sJ = compute_face_metrics(EToV_coords_x, EToV_coords_y)
    r, s = engine.r, engine.s
    x_nodes = 0.5 * (-(r+s)*EToV_coords_x[:,0:1] + (1+r)*EToV_coords_x[:,1:2] + (1+s)*EToV_coords_x[:,2:3])
    y_nodes = 0.5 * (-(r+s)*EToV_coords_y[:,0:1] + (1+r)*EToV_coords_y[:,1:2] + (1+s)*EToV_coords_y[:,2:3])
    vmapM, vmapP = build_pure_geometric_maps(x_nodes, y_nodes, engine.fmask)
    
    kwargs = {
        'engine': engine, 'rx': rx, 'sx': sx, 'ry': ry, 'sy': sy, 'J': J,
        'nx': nx, 'ny': ny, 'sJ': sJ, 'vmapM': vmapM, 'vmapP': vmapP,
        'cx': cx, 'cy': cy, 'x_nodes': x_nodes, 'y_nodes': y_nodes
    }

    # 🌟 建立一個獨立的推進函數，確保每次都從 t=0 乾淨開跑
    def solve_to_time(dt_step):
        t = 0.0
        q = q_exact(x_nodes, y_nodes, t, cx=cx)
        res = np.zeros_like(q)
        steps = int(round(T_final / dt_step))
        for _ in range(steps):
            q, res = lsrk54_step(q, res, t, dt_step, compute_full_rhs, **kwargs)
            t += dt_step
        return q

    # --------------------------------------------------
    # 🌟 步驟 1：計算「無時間誤差」的參考解 (超級小的 dt)
    # --------------------------------------------------
    dt_ref = 0.000125  # 極端小的時間步
    print(f"⏳ 正在計算參考解 (dt = {dt_ref:.6f})，請稍候...")
    q_ref = solve_to_time(dt_ref)
    print("✅ 參考解計算完畢！\n")

    # --------------------------------------------------
    # 🌟 步驟 2：測試正常 dt，並減去參考解
    # --------------------------------------------------
    dts = [0.004, 0.002, 0.001, 0.0005]
    errors = []

    print("-" * 50)
    for dt in dts:
        q_test = solve_to_time(dt)
        
        # 🔥 魔法就在這一行：減去參考解 (抵消空間誤差)，而不是精確解！
        pure_time_err = np.max(np.abs(q_test - q_ref))
        
        errors.append(pure_time_err)
        print(f" dt = {dt:.4f} | 執行 {int(round(T_final/dt)):3d} 步 | 純時間誤差 = {pure_time_err:.4e}")
    print("-" * 50)

    # --------------------------------------------------
    # 🌟 步驟 3：畫出完美的斜率圖
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.loglog(dts, errors, 'o-b', linewidth=2, markersize=8, label='Pure Time Error (Ref Mode)')
    
    # 繪製理論的 O(dt^4) 參考線
    ref_slope = [errors[0] * (dt / dts[0])**4 for dt in dts]
    plt.loglog(dts, ref_slope, 'k--', linewidth=2, label='Theoretical O(dt^4) Slope')
    
    plt.xlabel('Time Step (dt)', fontsize=12)
    plt.ylabel('Max Absolute Error (vs Reference)', fontsize=12)
    plt.title('LSRK54 Time Convergence (Isolated Temporal Error)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lsrk_pure_time_test()