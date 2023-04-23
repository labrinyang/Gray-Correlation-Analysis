import pandas as pd
import numpy as np
class GrayCorrelation:
    """
    灰色关联度综合评价分析法
    请保证每行是一个评价对象，每列是一个评价指标，且每个评价指标完成了同向化与标准化
    """
    def __init__(self, data, target=None, transpose=False, rho=0.5, result=None):
        self.data = np.array(data)
        self.target = target
        self.transpose = transpose
        self.rho = rho
        self.result = result

    def make_target(self):
        """ 生成目标函数 """
        if self.transpose:
            self.data = self.data.T
        else:
            pass

        if self.target is not None:
            dmax = self.data.max(axis=1)
            self.target = np.array(dmax)
        else:
            pass

    def correlation(self):
        """ 计算关联系数 """
        self.make_target()
        data = -(self.data - self.target[:, np.newaxis])
        dmax = data.max().max()
        dmin = data.min().min()

        data = pd.DataFrame(data)
        data.apply(lambda x: ((dmin-self.rho*dmax)/(x-self.rho*dmax)), axis=1)
        self.result = data.mean(axis=1)




