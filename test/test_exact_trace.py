import numpy as np
from core.operators import build_local_operators
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.rhs import compute_surface_rhs_matrix_free
from core.mesh import create_square_mesh
# ==========================================
# 1. 定義上帝視角的解析解 (Y方向正弦波測試)
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
# 2. 測試主程式
# ==========================================
def run_exact_trace_test():
    print("🛡️ 啟動 Nodal DG 引擎 MMS 隔離除錯測試 (Exact Trace Test)...")
    
    # 物理與數值參數
    N = 4      # 多項式階數
    n = 4      # 積分階數
    cx, cy = 1.0, 1.0
    t0 = 0.0
    
    # 1. 建立參考三角形引擎 (包含 Dr, Ds, w_s)
    engine = build_local_operators(N, n, rule="table1")
    

    # 🔧 在這裡調整您的網格密度！
    NX, NY = 64,64  
    VX, VY, EToV = create_square_mesh(NX, NY)
    K = EToV.shape[0]
    
    EToV_coords_x = VX[EToV]
    EToV_coords_y = VY[EToV]
    # --------------------------------------------------------

    # 3. 計算幾何變換矩陣
    xr, xs, yr, ys, rx, sx, ry, sy, J = compute_volume_metrics(EToV_coords_x, EToV_coords_y)
    nx, ny, sJ = compute_face_metrics(EToV_coords_x, EToV_coords_y)
    
    # 4. 映射節點到實體空間 (x, y)
    # 使用重心座標公式: x = 0.5*(-(r+s)*v1 + (1+r)*v2 + (1+s)*v3)
    r, s = engine.r, engine.s
    x_nodes = 0.5 * (-(r + s) * EToV_coords_x[:, 0:1] + (1 + r) * EToV_coords_x[:, 1:2] + (1 + s) * EToV_coords_x[:, 2:3])
    y_nodes = 0.5 * (-(r + s) * EToV_coords_y[:, 0:1] + (1 + r) * EToV_coords_y[:, 1:2] + (1 + s) * EToV_coords_y[:, 2:3])
    
    # 5. 計算當前的物理狀態 q 與目標答案 qt
    q_current = q_exact(x_nodes, y_nodes, t0, cx, cy)
    qt_target = qt_exact(x_nodes, y_nodes, t0, cx, cy)

    # ------------------------------------------------
    # 步驟 C: 引擎運算 (體積項 + 表面項)
    # ------------------------------------------------
    
    # 1. 計算體積散度 (使用連鎖律: rx, sx, ry, sy)
    # 運算為 (K, Np) 格式
    qr = (engine.Dr @ q_current.T).T
    qs = (engine.Ds @ q_current.T).T
    dq_dx = rx[:, None] * qr + sx[:, None] * qs
    dq_dy = ry[:, None] * qr + sy[:, None] * qs
    volume_rhs = -(cx * dq_dx + cy * dq_dy)

    # 2. 計算表面項 (Exact Trace 模式)
    # 提取邊界節點座標以計算上帝邊界值 (Exact Trace)
    fmask = [
        np.where(abs(s + 1.0) < 1e-12)[0], # Face 1
        np.where(abs(r + s) < 1e-12)[0],   # Face 2
        np.where(abs(r + 1.0) < 1e-12)[0]  # Face 3
    ]
    
    total_surface_rhs = np.zeros_like(q_current)
    
    for f in range(3):
        # 提取我方邊界值 (q_minus)
        qM = q_current[:, fmask[f]]
        
        # 提取邊界實體座標
        xf = x_nodes[:, fmask[f]]
        yf = y_nodes[:, fmask[f]]
        
        # 計算上帝標準答案 (Exact Trace)
        q_boundary_exact = q_exact(xf, yf, t0, cx, cy)
        
        # 執行 Matrix-Free 抬升
       # 執行 Matrix-Free 抬升
        total_surface_rhs += compute_surface_rhs_matrix_free(
            u_minus=qM, 
            u_plus=None, 
            nx=nx[:, f:f+1], ny=ny[:, f:f+1], 
            cx=cx, cy=cy, 
            
            # 使用專屬的邊界權重 (w_e)，若無則降級使用 w_s
            w_face=engine.w_e if engine.w_e is not None else engine.w_s[fmask[f]], 
            w_vol=engine.w_s,            
            sJ=sJ[:, f:f+1], 
            J=J[:, None], 
            
            # 傳入 fmask，讓引擎知道要把 5 個點插回 22 個點的哪裡！
            fmask=fmask[f],              
            
            tau=0.0, 
            exact_trace=q_boundary_exact
        )

    # 總變化率
    total_rhs = volume_rhs + total_surface_rhs

    # ------------------------------------------------
    # 步驟 D: 誤差分析
    # ------------------------------------------------
    err_vol = volume_rhs - qt_target
    err_total = total_rhs - qt_target
    
    max_err_vol = np.max(np.abs(err_vol))
    max_err_total = np.max(np.abs(err_total))
    
    print("\n📊 測試結果報告 (Exact Trace Test):")
    print("-" * 50)
    print(f"  Max Error (Volume only) : {max_err_vol:.4e}")
    print(f"  Max Error (Total RHS)   : {max_err_total:.4e}")
    print("-" * 50)
    
    if max_err_total < 1e-10:
        print("✅ 測試完美通過！您的引擎具備機器極限的精準度！")
    else:
        print("❌ 誤差過大。請檢查 LIFT 權重或 nx, ny 是否對齊 H&W 順序。")

if __name__ == "__main__":
    run_exact_trace_test()