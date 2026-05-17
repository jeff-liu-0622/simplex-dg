import numpy as np
from core import build_local_operators

def test_all_differentiation_exactness():
    print("\n" + "="*65)
    print("🚀 Nodal DG 引擎微分矩陣 (Dr, Ds) 全面精度驗證")
    print("測試範圍: Table 1 & Table 2, 多項式階數 N = 1 ~ 4")
    print("="*65)
    print(f"{'Rule':<10} {'N':<5} {'Points (Np)':<15} {'Max Error (Dr, Ds)':<20} {'Status'}")
    print("-" * 65)
    
    rules = ["table1", "table2"]
    N_values = [1, 2, 3, 4]
    
    for rule in rules:
        for N in N_values:
            try:
                # 啟動引擎 (n=N 讓積分點階數配合多項式階數)
                engine = build_local_operators(N=N, n=N, rule=rule)
                r, s = engine.r, engine.s
                Np = len(r)
                
                max_err_overall = 0.0
                
                # 窮舉測試所有 i+j <= N 的單項式 (Monomials) r^i * s^j
                for i in range(N + 1):
                    for j in range(N + 1 - i):
                        u = (r**i) * (s**j)
                        
                        # 1. 理論微分值 (純數學偏微分)
                        # 注意：i>0 和 j>0 的判斷是為了避免產生 0 的負次方
                        du_dr_exact = i * (r**(i-1)) * (s**j) if i > 0 else np.zeros_like(r)
                        du_ds_exact = j * (r**i) * (s**(j-1)) if j > 0 else np.zeros_like(s)
                        
                        # 2. 數值微分值 (我們引擎矩陣算出來的結果)
                        du_dr_num = engine.Dr @ u
                        du_ds_num = engine.Ds @ u
                        
                        # 3. 計算此單項式的最大誤差
                        err_r = np.max(np.abs(du_dr_num - du_dr_exact))
                        err_s = np.max(np.abs(du_ds_num - du_ds_exact))
                        
                        max_err_overall = max(max_err_overall, err_r, err_s)
                
                # 只要最大誤差小於 1e-12，就算完美通過！
                status = "✅ Pass" if max_err_overall < 1e-12 else "❌ Fail"
                
                print(f"{rule:<10} {N:<5} {Np:<15} {max_err_overall:<20.2e} {status}")
                
            except Exception as e:
                print(f"{rule:<10} {N:<5} {'ERROR':<15} {str(e):<20} ❌ Fail")

    print("="*65)
    print("💡 結論：只要誤差落在 1e-14 級別，代表該階數下的微分運算完美達到機器精度極限！")

if __name__ == "__main__":
    test_all_differentiation_exactness()