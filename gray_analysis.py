import numpy as np
import pandas as pd

from common.data_pretreatment import DataPretreatment


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

        dmax = (self.data).max(axis=1)
        self.target = np.array(dmax)

    def correlation(self):
        """
        计算关联系数
         """
        self.make_target()
        data = -(self.data - self.target[:, np.newaxis])
        dmax = data.max().max()
        dmin = data.min().min()

        data = pd.DataFrame(data)
        data = data.apply(lambda x: ((dmin+(self.rho)*dmax)/(x+(self.rho)*dmax)), axis=1)
        self.result = data.mean(axis=1)

def gray_corr(data, negtive_list=None, positive_list=None, transpose=False, rho=0.5, show=False):
    """
    # 计算灰色关联度

    data: 数据

    negtive_list: 负向指标列表

    positive_list: 正向指标列表

    transpose: 是否转置(默认为False,请保证每行是一个评价对象，每列是一个评价指标)

    rho: 灰色关联度参数(默认为0.5)

    show: 是否打印带评价对象名的结果(默认为False)

    """
    a=data
    if negtive_list!=None:
        for i in negtive_list:
            pre = DataPretreatment(a.iloc[:,i], do_forward=True, do_normalize=True)
            pre.pretreatment()
            a.iloc[:,i] = pre.result

    if positive_list!=None:
        for i in positive_list:
            pre = DataPretreatment(a.iloc[:,i], do_forward=False, do_normalize=True)
            pre.pretreatment()
            a.iloc[:,i] = pre.result

    gray = GrayCorrelation(a, transpose=transpose, rho=rho)
    gray.correlation()

    if show:
        gray.result.index = data.index
        gray.result.to_csv('gray.csv')
        return gray.result

    return gray.result


