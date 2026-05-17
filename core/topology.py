import numpy as np

def get_boundary_mask(r, s, tol=1e-10):
    """
    找出所有位在參考三角形邊界上的節點索引。
    加入了 tol 容差，防止浮點數判斷失準。
    """
    # 在參考三角形中，三條邊界分別是 s = -1, r + s = 0, r = -1
    mask = (np.abs(s + 1.0) < tol) | (np.abs(r + s) < tol) | (np.abs(r + 1.0) < tol)
    return np.where(mask)[0]

def build_node_maps(x, y, is_bmap, tol=1e-10):
    """
    建立 vmapM 與 vmapP 的連通性網路。
    加入了歐幾里得距離的 tol 容差判斷。
    """
    Np, K = x.shape
    num_b_nodes = len(is_bmap)
    total_b_nodes = num_b_nodes * K
    
    # 1. 初始化 vmapM (儲存所有邊界點的「全局絕對編號」)
    vmapM = np.zeros(total_b_nodes, dtype=int)
    bx = np.zeros(total_b_nodes)
    by = np.zeros(total_b_nodes)
    
    # 將所有邊界點攤平，記錄他們的物理坐標
    idx = 0
    for k in range(K):
        for i in is_bmap:
            vmapM[idx] = i + k * Np  # 計算全局編號 (F-style)
            bx[idx] = x[i, k]
            by[idx] = y[i, k]
            idx += 1
            
    # 2. 尋找鄰居建立 vmapP (預設先全部牽自己，也就是無鄰居)
    vmapP = vmapM.copy()
    
    # 🌟 魔法在此：用距離 (tol) 判定，而不是嚴格相等
    for i in range(total_b_nodes):
        for j in range(total_b_nodes):
            if i != j:
                # 計算兩個邊界點之間的直線距離
                dist = np.sqrt((bx[i] - bx[j])**2 + (by[i] - by[j])**2)
                if dist < tol:
                    vmapP[i] = vmapM[j]
                    break  # 找到對面牽手的鄰居就跳出！
                    
    return vmapM, vmapP