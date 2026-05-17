import numpy as np

def map_reference_to_physical(r, s, v1, v2, v3):
    """
    將標準參考三角形上的點 (r,s) 映射到真實世界物理三角形的座標 (x,y)。
    並計算用於微積分轉換的幾何縮放因子 (Jacobian) 與度量 (Metrics)。
    
    v1, v2, v3: 真實三角形的三個頂點座標，例如 v1=(x1, y1)
    """
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    
    # 1. 座標映射公式 (仿射變換 Affine Mapping)
    x = 0.5 * (-(r + s) * x1 + (1.0 + r) * x2 + (1.0 + s) * x3)
    y = 0.5 * (-(r + s) * y1 + (1.0 + r) * y2 + (1.0 + s) * y3)
    
    # 2. 計算邊界向量 (用來算面積與斜率轉換)
    xr = 0.5 * (x2 - x1)
    xs = 0.5 * (x3 - x1)
    yr = 0.5 * (y2 - y1)
    ys = 0.5 * (y3 - y1)
    
    # 3. 計算 Jacobian (面積縮放因子，也就是筆記中的 |T|)
    J = xr * ys - xs * yr
    
    # 4. 計算空間度量 (Metrics)
    # 這些是微積分連鎖律的係數，用於將 d/dr 轉換為真正的 d/dx
    rx =  ys / J
    sx = -yr / J
    ry = -xs / J
    sy =  xr / J
    
    return {
        "x": x, "y": y,
        "J": J,
        "rx": rx, "sx": sx, "ry": ry, "sy": sy
    }

def compute_physical_derivatives(Dr, Ds, rx, sx, ry, sy, u):
    """
    利用連鎖律，計算真實物理空間中的偏微分 du/dx 與 du/dy
    """
    ur = Dr @ u
    us = Ds @ u
    
    # 連鎖律：du/dx = du/dr * dr/dx + du/ds * ds/dx
    ux = rx * ur + sx * us
    uy = ry * ur + sy * us
    return ux, uy
import numpy as np

def create_square_mesh(nx, ny, length_x=1.0, length_y=1.0):
    """
    生成結構化的 2D 直角三角形網格。
    回傳: VX, VY (頂點座標), EToV (三角形與頂點的連線對應)
    """
    x1d = np.linspace(0, length_x, nx + 1)
    y1d = np.linspace(0, length_y, ny + 1)
    X, Y = np.meshgrid(x1d, y1d)
    VX, VY = X.flatten(), Y.flatten()
    
    EToV = []
    for j in range(ny):
        for i in range(nx):
            v1 = j * (nx + 1) + i     
            v2 = v1 + 1               
            v3 = v1 + (nx + 1)        
            v4 = v3 + 1               
            EToV.append([v1, v2, v3]) # 下三角
            EToV.append([v4, v3, v2]) # 上三角
            
    return VX, VY, np.array(EToV)


def create_unit_triangle_mesh(n_div):
    """
    Create a structured triangular mesh on the unit triangle:

        v0 = (0,0)
        v1 = (1,0)
        v2 = (0,1)

    The triangle is subdivided uniformly with n_div segments per edge.

    Returns
    -------
    VX, VY:
        Vertex coordinates, shape (Nv,).

    EToV:
        Element-to-vertex connectivity, shape (K,3).

    Notes
    -----
    All elements are counter-clockwise.

    Number of small triangles:

        K = n_div^2
    """
    if n_div < 1:
        raise ValueError("n_div must be >= 1.")

    # ------------------------------------------------------------
    # 1. Build vertices
    # ------------------------------------------------------------
    vertex_id = {}
    VX = []
    VY = []

    idx = 0

    for i in range(n_div + 1):
        for j in range(n_div + 1 - i):
            xi = i / n_div
            eta = j / n_div

            vertex_id[(i, j)] = idx
            VX.append(xi)
            VY.append(eta)

            idx += 1

    VX = np.array(VX, dtype=float)
    VY = np.array(VY, dtype=float)

    # ------------------------------------------------------------
    # 2. Build triangles
    # ------------------------------------------------------------
    EToV = []

    for i in range(n_div):
        for j in range(n_div - i):
            # Lower-left small triangle:
            #
            # (i,j) -> (i+1,j) -> (i,j+1)
            #
            # Exists for all j <= n_div-i-1.
            v0 = vertex_id[(i, j)]
            v1 = vertex_id[(i + 1, j)]
            v2 = vertex_id[(i, j + 1)]

            EToV.append([v0, v1, v2])

            # Upper-right small triangle:
            #
            # (i+1,j) -> (i+1,j+1) -> (i,j+1)
            #
            # Exists only if i + j <= n_div - 2.
            if i + j <= n_div - 2:
                v3 = vertex_id[(i + 1, j + 1)]

                # Counter-clockwise ordering.
                EToV.append([v1, v3, v2])

    EToV = np.array(EToV, dtype=int)

    return VX, VY, EToV