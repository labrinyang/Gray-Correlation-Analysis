import pandas as pd

from gray_analysis import gray_corr

a = pd.DataFrame([[8, 9, 8, 7, 5, 2, 9],
                  [7, 8, 7, 5, 7, 3, 8],
                  [9, 7, 9, 6, 6, 4, 7],
                  [6, 8, 8, 8, 4, 3, 6],
                  [8, 6, 6, 9, 8, 3, 8],
                  [8, 9, 5, 7, 6, 4, 8]])
# rename index as A, B, C
a.index = [1, 2, 3, 4, 5, 6]

re = gray_corr(a, positive_list=[0, 1, 2, 3, 3, 4, 5, 6], show=True)
re = pd.DataFrame(re)
re.sort_values(by=0, ascending=False, inplace=True)
print(re)
