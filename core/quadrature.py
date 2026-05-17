import numpy as np
from itertools import permutations
from math import factorial

# 筆記 Table 1 的原始數據 (目前支援 n=1~4)
TABLE1 = {
    1: [("S6", 0.2113248654051871, 0.0, 0.16666666666666667, 0.5000000000000000)],
    2: [
        ("S6", 0.1127016653792583, 0.0, 0.04166666666666666, 0.2777777777777777),
        ("S3", 0.5000000000000000, 0.0, 0.09999999999999999, 0.4444444444444444),
        ("S1", 0.3333333333333333, 0.3333333333333333, 0.45000000000000000, None),
    ],
    3: [
        ("S6", 0.06943184420297367, 0.0, 0.01509901487256561, 0.1739274225687269),
        ("S6", 0.3300094782075718, 0.0, 0.04045654068298990, 0.3260725774312731),
        ("S6", 0.5841571139756568, 0.1870738791912763, 0.11111111111111111, None),
    ],
    4: [
        ("S6", 0.04691007703066797, 0.0, 0.006601315081001592, 0.1184634425280944),
        ("S6", 0.2307653449471584, 0.0, 0.02053045968042892, 0.2393143352496833),
        ("S3", 0.5000000000000000, 0.0, 0.01853708483394990, 0.2844444444444446),
        ("S3", 0.1394337314154536, 0.1394337314154536, 0.10542932962084440, None),
        ("S3", 0.4384239524408185, 0.4384239524408185, 0.12473673228977350, None),
        ("S1", 0.3333333333333333, 0.3333333333333333, 0.09109991119771331, None),
    ]
}
# 筆記 Table 2 的原始數據 (純內部高階積分點，沒有邊界線積分權重 we)
TABLE2 = {
    1: [("S3", 0.1666666666666666,  0.1666666666666666,  0.3333333333333333)],
    2: [("S3", 0.09157621350977067, 0.09157621350977067, 0.1099517436553218),
        ("S3",  0.4459484909159648, 0.4459484909159648, 0.2233815896780115),
    ],
    3:[("S3",0.219429982549783,0.219429982549783, 0.1713331241529809),
       ("S3", 0.480137964112215, 0.480137964112215, 0.08073108959303095),
       ("S6", 0.1416190159239682, 0.0193717243612408, 0.04063455979366068),
    ],
    4:[("S6", 0.7284923929554044, 0.2631128296346379, 0.02723031417443505),
       ("S3", 0.4592925882927232, 0.4592925882927232,0.09509163426728455),
       ("S3", 0.1705693077517602, 0.1705693077517602,0.1032173705347182),
       ("S3", 0.05054722831703096, 0.05054722831703096, 0.03245849762319804),
       ("S1",0.3333333333333333,0.3333333333333333, 0.1443156076777874),
    ]

}
def get_uniform_nodes(N):
    """
    產生 N 階均勻分佈點 (Equidistant Nodes / 也就是均勻的 Table 2)。
    
    【⚠️ 架構師警告】
    這個函數產生的點「沒有」面積積分權重 (w)！
    禁止將此函數用於核心的 Galerkin 投影或物理微積分！
    僅限用來生成畫圖用的密集網格 (Dense Lattice)。
    """
    r, s = [], []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            r.append(-1.0 + 2.0 * i / N)
            s.append(-1.0 + 2.0 * j / N)
            
    return np.array(r), np.array(s)

def _expand_symmetry(sym, b1, b2):
    """根據對稱性 (S1, S3, S6) 展開重心座標"""
    b3 = 1.0 - b1 - b2
    if sym == "S1":
        return [(b1, b2, b3)]
    
    seen = set()
    out = []
    for p in permutations((b1, b2, b3)):
        key = tuple(round(v, 15) for v in p)
        if key not in seen:
            seen.add(key)
            out.append(tuple(float(v) for v in p))
    return out

def get_reference_nodes(n, rule="table1"):
    """
    獲取參考三角形上的積分點配置。
    映射關係：(xi, eta) = (b3, b1)
    
    參數:
      n: 積分法則的階數
      rule: "table1" (含邊界點，精度 2n-1) 或 "table2" (純內部點，精度 2n)
    """
    # 1. 根據使用者的選擇切換表格
    if rule == "table1":
        if n not in TABLE1:
            raise ValueError(f"Table 1 不支援的階數 n={n}")
        table_data = TABLE1[n]
    elif rule == "table2":
        if n not in TABLE2:
            raise ValueError(f"Table 2 不支援的階數 n={n}")
        table_data = TABLE2[n]
    else:
        raise ValueError("rule 參數必須是 'table1' 或 'table2'")

    raw_nodes = []
    
    # 2. 彈性讀取資料 (Table 1 有 5 個元素，Table 2 只有 4 個元素)
    for row in table_data:
        sym = row[0]
        b1 = row[1]
        b2 = row[2]
        ws = row[3]
        we = row[4] if len(row) > 4 else None  # Table 2 沒有 we，自動設為 None
        
        for b1p, b2p, b3p in _expand_symmetry(sym, b1, b2):
            xi, eta = b3p, b1p
            
            # 判斷是否為邊界點
            is_boundary = abs(b1p) < 1e-14 or abs(b2p) < 1e-14 or abs(b3p) < 1e-14
            edge, param = None, None
            
            if is_boundary:
                if abs(b1p) < 1e-14:
                    edge, param = 1, xi        # 邊 1: eta = 0
                elif abs(b2p) < 1e-14:
                    edge, param = 2, eta       # 邊 2: xi + eta = 1
                else:
                    edge, param = 3, -eta      # 邊 3: xi = 0

            raw_nodes.append({
                "xi": xi, "eta": eta, "ws": ws, "we": we,
                "is_boundary": is_boundary, "edge": edge, "param": param
            })

    # 將節點排序：先排邊界點 (Edge 1 -> 2 -> 3)，再排內部點
    # 這是為了確保之後建構提取矩陣 E = [I | 0] 時結構正確
    boundary = [d for d in raw_nodes if d["is_boundary"]]
    interior = [d for d in raw_nodes if not d["is_boundary"]]
    
    edge1 = sorted([d for d in boundary if d["edge"] == 1], key=lambda d: d["param"])
    edge2 = sorted([d for d in boundary if d["edge"] == 2], key=lambda d: d["param"])
    edge3 = sorted([d for d in boundary if d["edge"] == 3], key=lambda d: d["param"])
    
    ordered_nodes = edge1 + edge2 + edge3 + interior
    
    xi = np.array([d["xi"] for d in ordered_nodes])
    eta = np.array([d["eta"] for d in ordered_nodes])
    w_s = np.array([d["ws"] for d in ordered_nodes])
    
    # 3. 安全處理邊界權重：如果是 Table 2，這裡 len(edge1) 會是 0
    if len(edge1) > 0:
        w_e = np.array([d["we"] for d in edge1]) # 邊界權重 (各邊相同)
    else:
        w_e = None
    
    # 轉換至勒讓德多項式計算用的 [-1, 1] 區間
    r = 2.0 * xi - 1.0
    s = 2.0 * eta - 1.0
    
    return {
        "n": n, 
        "rule": rule,           # 紀錄目前是用哪張表
        "xi": xi, "eta": eta, "r": r, "s": s, 
        "w_s": w_s, "w_e": w_e, 
        "num_edge_nodes": len(edge1)
    }