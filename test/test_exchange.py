import numpy as np
from core.operators import build_local_operators
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.rhs import compute_surface_rhs_matrix_free

# ==========================================
# 🌟 絕對雷達：純幾何配對通訊錄 (無須 EToE/EToF)
# ==========================================
def build_pure_geometric_maps(x_nodes, y_nodes, fmask):
    """無視所有拓撲編號，直接在物理空間中尋找距離重合的點進行配對"""
    K = x_nodes.shape[0]
    Nfaces = 3
    Nfp = len(fmask[0])
    
    vmapM = np.zeros((K, Nfaces, Nfp), dtype=int)
    vmapP = np.zeros((K, Nfaces, Nfp), dtype=int)
    
    x_flat = x_nodes.flatten()
    y_flat = y_nodes.flatten()
    
    # 1. 建立本機邊界
    for k in range(K):
        for f in range(Nfaces):
            vmapM[k, f, :] = k * x_nodes.shape[1] + fmask[f]
            
    # 2. 暴力幾何匹配
    for k1 in range(K):
        for f1 in range(Nfaces):
            nodes1 = vmapM[k1, f1, :]
            cx1 = np.mean(x_flat[nodes1])
            cy1 = np.mean(y_flat[nodes1])
            
            match_k, match_f = k1, f1
            # 尋找中心點距離小於 1e-8 的面
            for k2 in range(K):
                for f2 in range(Nfaces):
                    if k1 == k2 and f1 == f2: continue
                    nodes2 = vmapM[k2, f2, :]
                    cx2 = np.mean(x_flat[nodes2])
                    cy2 = np.mean(y_flat[nodes2])
                    if (cx1 - cx2)**2 + (cy1 - cy2)**2 < 1e-10:
                        match_k, match_f = k2, f2
                        break
                        
            if match_k == k1:
                # 找不到鄰居 -> 外部邊界
                vmapP[k1, f1, :] = vmapM[k1, f1, :]
            else:
                # 找到鄰居 -> 點對點鎖定配對
                nodes2 = vmapM[match_k, match_f, :]
                for i, n1 in enumerate(nodes1):
                    dist = (x_flat[nodes2] - x_flat[n1])**2 + (y_flat[nodes2] - y_flat[n1])**2
                    vmapP[k1, f1, i] = nodes2[np.argmin(dist)]
                    
    return vmapM, vmapP

# ==========================================
# 解析解 (N=2)
# ==========================================
def q_exact(x, y, t, cx=1.0, cy=1.0):
    """只沿著 Y 軸推進的正弦波 (x 方向完全沒變化)"""
    return np.sin(y - cy * t)

def qt_exact(x, y, t, cx=1.0, cy=1.0):
    """解析解的時間變化率: - cx(dq/dx) - cy(dq/dy)"""
    dq_dx = np.zeros_like(x)  # x 方向斜率永遠為 0
    dq_dy = np.cos(y - cy * t)
    return -cx * dq_dx - cy * dq_dy
# ==========================================
# 測試主程式
# ==========================================
def run_exchange_test():
    print("🤝 啟動 Nodal DG 引擎全網格通訊測試 (Pure Geometric Exchange)...")
    
    N, n = 4, 4
    cx, cy, t0 = 1.0, 1.0, 0.0
    
    engine = build_local_operators(N, n, rule="table1")
    
    # 嚴格 H&W 排序 fmask
    TOL = 1e-12
    f1_raw = np.where(abs(engine.s + 1.0) < TOL)[0]
    f2_raw = np.where(abs(engine.r + engine.s) < TOL)[0]
    f3_raw = np.where(abs(engine.r + 1.0) < TOL)[0]
    
    engine.fmask = [
        f1_raw[np.argsort(engine.r[f1_raw])],  
        f2_raw[np.argsort(-engine.r[f2_raw])], 
        f3_raw[np.argsort(-engine.s[f3_raw])]  
    ]
    
    # 建立網格
    VX = np.array([0.0, 1.0, 0.0, 1.0])
    VY = np.array([0.0, 0.0, 1.0, 1.0])
    EToV = np.array([[0, 1, 2], [3, 2, 1]]) 
    K = EToV.shape[0]
    EToV_coords_x, EToV_coords_y = VX[EToV], VY[EToV]

    # 計算幾何與座標
    rx, sx, ry, sy, J = compute_volume_metrics(EToV_coords_x, EToV_coords_y)
    nx, ny, sJ = compute_face_metrics(EToV_coords_x, EToV_coords_y)
    r, s = engine.r, engine.s
    x_nodes = 0.5 * (-(r + s) * EToV_coords_x[:, 0:1] + (1 + r) * EToV_coords_x[:, 1:2] + (1 + s) * EToV_coords_x[:, 2:3])
    y_nodes = 0.5 * (-(r + s) * EToV_coords_y[:, 0:1] + (1 + r) * EToV_coords_y[:, 1:2] + (1 + s) * EToV_coords_y[:, 2:3])
    
    # 🌟 呼叫純幾何雷達 (不依賴任何外部拓撲)
    vmapM, vmapP = build_pure_geometric_maps(x_nodes, y_nodes, engine.fmask)
    
    q_current = q_exact(x_nodes, y_nodes, t0, cx, cy)
    qt_target = qt_exact(x_nodes, y_nodes, t0, cx, cy)

    qr, qs = (engine.Dr @ q_current.T).T, (engine.Ds @ q_current.T).T
    volume_rhs = -(cx * (rx[:, None]*qr + sx[:, None]*qs) + cy * (ry[:, None]*qr + sy[:, None]*qs))

    total_surface_rhs = np.zeros_like(q_current)
    max_interior_mismatch = 0.0
    
    for f in range(3):
        qM = q_current.flatten()[vmapM[:, f, :]]
        qP = q_current.flatten()[vmapP[:, f, :]]
        
        # 邊界判定：如果本機點等於對面點，代表是外圍邊界
        is_boundary = np.all(vmapM[:, f, :] == vmapP[:, f, :], axis=1)
        
        if np.any(is_boundary):
            xf_bnd = x_nodes.flatten()[vmapM[is_boundary, f, :]]
            yf_bnd = y_nodes.flatten()[vmapM[is_boundary, f, :]]
            qP[is_boundary, :] = q_exact(xf_bnd, yf_bnd, t0, cx, cy)
            
        interior_mask = ~is_boundary
        if np.any(interior_mask):
            xf_ours = x_nodes.flatten()[vmapM[interior_mask, f, :]]
            yf_ours = y_nodes.flatten()[vmapM[interior_mask, f, :]]
            qP_exact = q_exact(xf_ours, yf_ours, t0, cx, cy)
            mismatch = np.max(np.abs(qP[interior_mask, :] - qP_exact))
            max_interior_mismatch = max(max_interior_mismatch, mismatch)
            
        total_surface_rhs += compute_surface_rhs_matrix_free(
            u_minus=qM, u_plus=qP, 
            nx=nx[:, f:f+1], ny=ny[:, f:f+1], cx=cx, cy=cy, 
            w_face=engine.w_e if engine.w_e is not None else engine.w_s[engine.fmask[f]], 
            w_vol=engine.w_s, sJ=sJ[:, f:f+1], J=J[:, None], 
            fmask=engine.fmask[f], tau=0.0
        )

    total_rhs = volume_rhs + total_surface_rhs
    err_total = total_rhs - qt_target
    
    print("\n📊 測試結果報告 (Pure Geometric Exchange):")
    print("-" * 50)
    print(f"  Max Interior Mismatch : {max_interior_mismatch:.4e}")
    print(f"  Max Error (Total RHS) : {np.max(np.abs(err_total)):.4e}")
    print("-" * 50)
    
    if max_interior_mismatch < 1e-12 and np.max(np.abs(err_total)) < 1e-10:
        print("✅ 測試完美通過！引擎算子與物理通量已達機器極限！")
    else:
        print("❌ 誤差依然存在。")

if __name__ == "__main__":
    run_exchange_test()