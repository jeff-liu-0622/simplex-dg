import numpy as np

# 載入您的核心引擎模組
from core.operators import build_local_operators
from core.mesh import create_square_mesh
from core.geometry.face_metrics import compute_volume_metrics, compute_face_metrics
from core.operators_split import compute_split_rhs
# 🌟 直接徵用原廠的工業級連通表產生器！
from core.geometry.connectivity import build_connectivity
# ==========================================
# 🧱 1. 工業級網格拓樸系統 (Topology)
# ==========================================

def make_periodic_EToE(EToV, EToE, EToF, x_nodes, y_nodes, engine):
    """竄改拓樸地圖：將左右、上下物理邊界強行接合 (小精靈無限結界)"""
    K = EToV.shape[0]
    TOL = 1e-8
    
    bnd_faces = []
    for k in range(K):
        for f in range(3):
            if EToE[k, f] == k:
                bnd_faces.append((k, f))
                
    x_flat, y_flat = x_nodes.flatten(), y_nodes.flatten()
    min_x, max_x = np.min(x_flat), np.max(x_flat)
    min_y, max_y = np.min(y_flat), np.max(y_flat)
    
    for k1, f1 in bnd_faces:
        if EToE[k1, f1] != k1: continue 
        
        m1 = k1 * x_nodes.shape[1] + engine.fmask[f1]
        cx1, cy1 = np.mean(x_flat[m1]), np.mean(y_flat[m1])
        
        tcx, tcy = cx1, cy1
        if cx1 < min_x + TOL: tcx = max_x
        elif cx1 > max_x - TOL: tcx = min_x
        if cy1 < min_y + TOL: tcy = max_y
        elif cy1 > max_y - TOL: tcy = min_y
        
        for k2, f2 in bnd_faces:
            if k1 == k2 and f1 == f2: continue
            m2 = k2 * x_nodes.shape[1] + engine.fmask[f2]
            cx2, cy2 = np.mean(x_flat[m2]), np.mean(y_flat[m2])
            
            if abs(cx2 - tcx) < TOL and abs(cy2 - tcy) < TOL:
                EToE[k1, f1], EToF[k1, f1] = k2, f2
                EToE[k2, f2], EToF[k2, f2] = k1, f1
                break
                
    return EToE, EToF, min_x, max_x

def build_maps(engine, EToV, EToE, EToF):
    """建立全域邊界索引 vmapM 與 vmapP (自帶完美 [::-1] 翻轉對齊)"""
    K = EToV.shape[0]
    Np = engine.Np
    Nfp = len(engine.fmask[0])
    fmask = engine.fmask 
    face_vids = np.array([[1, 2], [2, 0], [0, 1]])
    
    node_ids = np.arange(K * Np).reshape(K, Np)
    vmapM = np.zeros((K, 3, Nfp), dtype=int)
    vmapP = np.zeros((K, 3, Nfp), dtype=int)
    
    for k in range(K):
        for f in range(3):
            vmapM[k, f, :] = node_ids[k, fmask[f]]
            k2, f2 = EToE[k, f], EToF[k, f]
            
            if k == k2: 
                vmapP[k, f, :] = vmapM[k, f, :]
            else:
                neighbor_nodes = node_ids[k2, fmask[f2]]
                v1 = EToV[k, face_vids[f, 0]]
                v2 = EToV[k2, face_vids[f2, 0]]
                
                # 🛡️ 破解維度詛咒的翻轉對齊
                if v1 != v2:
                    vmapP[k, f, :] = neighbor_nodes
                else:
                    vmapP[k, f, :] = neighbor_nodes[::-1]
                    
    return vmapM, vmapP

# ==========================================
# ⏱️ 2. 時間變速箱 (Time Integration)
# ==========================================
def perfect_lsrk54_step(q, t, dt, rhs_func, **kwargs):
    """純淨版 LSRK54：保證每次迭代記憶體絕對清空，隔絕浮點數灰塵污染"""
    a = [0.0, -0.4178904745, -1.1921516946, -1.6977846925, -1.5141834443]
    b = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]
    c = [0.0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748]
    
    res = np.zeros_like(q)
    for i in range(5):
        rhs = rhs_func(q, t + c[i]*dt, **kwargs)
        res = a[i]*res + dt*rhs
        q = q + b[i]*res
    return q

# ==========================================
# 🚀 3. 主測試程式 (Main Runner)
# ==========================================
def run_temporal_order_test():
    print("🔬 啟動 Nodal DG 終極引擎：工業級拓樸 + LSRK54 週期收斂分析")
    N, n = 4, 4 
    NX, NY = 4, 4  
    cx, cy = 1.0, 0.0  
    T_final = 0.5
    
    # 建立引擎與基準網格
    engine = build_local_operators(N, n, rule="table1")
    TOL = 1e-12
    f1 = np.where(abs(engine.s + 1.0) < TOL)[0]
    f2 = np.where(abs(engine.r + engine.s) < TOL)[0]
    f3 = np.where(abs(engine.r + 1.0) < TOL)[0]
    engine.fmask = [f1[np.argsort(engine.r[f1])], f2[np.argsort(-engine.r[f2])], f3[np.argsort(-engine.s[f3])]]
    engine.Np = len(engine.r)
    
    VX, VY, EToV = create_square_mesh(NX, NY)
    EToV_x, EToV_y = VX[EToV], VY[EToV]
    rx, sx, ry, sy, J = compute_volume_metrics(EToV_x, EToV_y)
    nx, ny, sJ = compute_face_metrics(EToV_x, EToV_y)
    r, s = engine.r, engine.s
    x_nodes = 0.5 * (-(r+s)*EToV_x[:,0:1] + (1+r)*EToV_x[:,1:2] + (1+s)*EToV_x[:,2:3])
    y_nodes = 0.5 * (-(r+s)*EToV_y[:,0:1] + (1+r)*EToV_y[:,1:2] + (1+s)*EToV_y[:,2:3])
    
    # 🌟 啟動拓樸裝甲與週期結界
    EToE, EToF = build_connectivity(EToV)
    EToE, EToF, min_x, max_x = make_periodic_EToE(EToV, EToE, EToF, x_nodes, y_nodes, engine)
    vmapM, vmapP = build_maps(engine, EToV, EToE, EToF)
    
    # 🌟 自動適應波長，絕對平滑銜接
    domain_width = max_x - min_x
    def q_exact_sinx(x, y, t):
        return np.sin(2.0 * np.pi * (x - cx * t) / domain_width)

    kwargs = {
        'engine': engine, 'rx': rx, 'sx': sx, 'ry': ry, 'sy': sy, 'J': J,
        'nx': nx, 'ny': ny, 'sJ': sJ, 'vmapM': vmapM, 'vmapP': vmapP,
        'cx': cx, 'cy': cy, 'x_nodes': x_nodes, 'y_nodes': y_nodes,
        'lift_mode': 'exact', # 啟動精確反質量矩陣
        'tau': 0.0            # 關閉阻力，讓波浪純粹依靠 SBP 守恆
    }

    def run_simulation(dt):
        q = q_exact_sinx(x_nodes, y_nodes, 0.0)
        t = 0.0
        steps = int(np.round(T_final / dt))
        for _ in range(steps):
            q = perfect_lsrk54_step(q, t, dt, compute_split_rhs, **kwargs)
            t += dt
        return q

    dt_ref = 0.0001
    print(f"⏳ 正在計算高精度參考解 (dt = {dt_ref})...")
    q_ref = run_simulation(dt_ref)
    print("✅ 參考解計算完成！\n")

    # 🌟 完美整除的時間步長
    dt_tests = [0.01, 0.005, 0.0025, 0.00125]
    errors = []
    
    print("-" * 65)
    print(f"{'dt':>10s} | {'步數 (Steps)':>12s} | {'最大誤差 (Max Error)':>20s} | {'收斂斜率 (Rate)':>10s}")
    print("-" * 65)
    
    for i, dt in enumerate(dt_tests):
        q_num = run_simulation(dt)
        err = np.max(np.abs(q_num - q_ref))
        errors.append(err)
        
        steps = int(np.round(T_final / dt))
        rate_str = "---" if i == 0 else f"{np.log2(errors[i-1] / errors[i]):.4f}"
        print(f"{dt:10.4f} | {steps:12d} | {err:20.6e} | {rate_str:>10s}")
    
    print("-" * 65)
    if np.log2(errors[-2] / errors[-1]) > 3.7:
        print("🎯 檢定通過：避開物理與時間干擾後，您的 LSRK54 完美展現理論四階精度！")
    else:
        print("⚠️ 檢定警告：收斂率未達標。")

if __name__ == "__main__":
    run_temporal_order_test()