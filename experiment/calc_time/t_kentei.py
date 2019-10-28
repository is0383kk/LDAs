import numpy as np
from scipy import stats
A = np.loadtxt("./ldatime.txt", dtype=np.float32)
B = np.loadtxt( "./dldatime.txt" , dtype=np.float32)
print(A)
print(B)

A_var = np.var(A, ddof=1)  # Aの不偏分散
print("A_var",A_var)
B_var = np.var(B, ddof=1)  # Bの不偏分散
print("B_var",B_var)
A_df = len(A) - 1  # Aの自由度
B_df = len(B) - 1  # Bの自由度
f = A_var / B_var  # F比の値
one_sided_pval1 = stats.f.cdf(f, A_df, B_df)  # 片側検定のp値 1
one_sided_pval2 = stats.f.sf(f, A_df, B_df)   # 片側検定のp値 2
two_sided_pval = min(one_sided_pval1, one_sided_pval2) * 2  # 両側検定のp値

print('F:       ', round(f, 3))
print('p-value: ', round(two_sided_pval, 3))

print(stats.ttest_ind(A, B,equal_var=False))
