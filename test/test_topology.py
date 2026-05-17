import numpy as np
from core import build_local_operators
from core.mesh import map_reference_to_physical
from core.topology import get_boundary_mask, build_node_maps

def test_connectivity():
    print("\n" + "="*50)
    print("啟動網格拓撲與 vmapM/vmapP 連通性測試")
    print("="*50)
    
    # 1. 取得 OOP 引擎 (我們必須用 Table 1，因為只有它有邊界點！)
    engine = build_local_operators(N=3, n=4, rule="table1")
    r, s = engine.r, engine.s
    num_edge_nodes = engine.num_edge_nodes  # 引擎會自動告訴我們邊界上有幾個點 (這裡是 5)
    
    # 2. 建構兩個三角形 (K=2) 組成的正方形網格 [0, 1] x [0, 1]
    geo0 = map_reference_to_physical(r, s, (0,0), (1,0), (0,1))
    geo1 = map_reference_to_physical(r, s, (1,1), (0,1), (1,0))
    
    # 將座標組裝成 (Np, K) 的矩陣
    x = np.column_stack((geo0["x"], geo1["x"]))
    y = np.column_stack((geo0["y"], geo1["y"]))
    
    # 3. 找出邊界點並建立 Mapping
    is_bmap = get_boundary_mask(r, s)
    vmapM, vmapP = build_node_maps(x, y, is_bmap)
    
    # --- 驗證邏輯 ---
    connected_nodes = np.sum(vmapM != vmapP)
    
    # 🌟 修正點：對角線上互相牽手的點數，等於單邊節點數 (5) * 2 個三角形 = 10 個點
    expected_connections = num_edge_nodes * 2 
    
    print(f"總邊界點數量: {len(vmapM)}")
    print(f"成功連線的內部節點對數量: {connected_nodes}")
    print(f"無鄰居的外部邊界點數量: {len(vmapM) - connected_nodes}")
    
    assert connected_nodes == expected_connections, f"拓撲連通性失敗：預期 {expected_connections} 點，卻得到 {connected_nodes} 點！"
    print("✅ 網格連通性驗證通過！三角形之間的通訊網路已成功建立。")

if __name__ == "__main__":
    test_connectivity()