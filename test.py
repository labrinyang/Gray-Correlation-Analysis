import pandas as pd

from gray_analysis import gray_corr

a = pd.DataFrame([[1.1, 1.8, 4, 80],
                  [1.2, 1.5, 3, 110],
                  [1.5, 1.3, 5, 100]])
# rename index as A, B, C
a.index = ['A', 'B', 'C']

re = gray_corr(a, [0,1], [2,3], show=True)
print(re)
