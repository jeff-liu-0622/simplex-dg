import numpy as np

def gaussian_bell(x, y):
    """
    (1) 高斯波包 (Gaussian Bell) / Smooth Bump
    特性：無限多次方的非多項式函數。
    用途：測試引擎的截斷誤差 (Truncation Error) 極限，會產生真實物理的微小波紋。
    """
    return np.exp(-5.0 * (x**2 + y**2))

def polynomial_wave(x, y):
    """
    (2) 三次多項式 (Polynomial Wave)
    特性：最高次數為 3 的完美多項式。
    用途：驗證引擎的數學純度。當設定 N=3 時，誤差必須為 10^-15 (機器極限的死寂平原)。
    """
    return 5.0 * x**3 - 2.0 * y**2 + x * y + 1.0

def sine_wave(x, y):
    """
    (3) 正弦波 (Sinusoidal Wave)
    特性：平滑的週期性函數。
    用途：未來測試流體波浪在網格中傳遞、反射、或週期性邊界時的標準教科書案例。
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def constant_state(x, y):
    """
    (4) 靜止水面 (Constant State)
    特性：全域皆為 1.0。
    用途：最基礎的除錯工具。用來檢查矩陣運算是否會無故產生非物理的雜訊 (Free-stream preservation)。
    """
    # 確保回傳的陣列形狀與輸入的 x 相同
    return np.ones_like(x)
def smooth_bump(r, s):
    """測試波浪：與上次一模一樣的高斯波包"""
    return np.exp(-5.0 * (r**2 + s**2))