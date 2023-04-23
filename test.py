import pandas as pd
import numpy as np
from data_pretreatment import DataPretreatment
from gray_analysis import GrayCorrelation

a = pd.DataFrame([[1.1,1.8,4,80],[1.2,1.5,3,110],[1.5,1.3,5,100]])
print(a.iloc[:, 1])

for i in [0,1]:
    pre = DataPretreatment(a.iloc[:,i], do_forward=True, do_normalize=True)
    pre.pretreatment()
    a.iloc[:,i] = pre.result

for i in [2, 3]:
    pre = DataPretreatment(a.iloc[:,i], do_forward=False, do_normalize=True)
    pre.pretreatment()
    a.iloc[:,i] = pre.result

gray = GrayCorrelation(a)
gray.correlation()
print(gray.result)


