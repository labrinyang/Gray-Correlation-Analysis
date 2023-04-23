""" 这个文件旨在对指标数据进行预处理。包含指标同向化与无量纲化的函数 """

class DataPretreatment:
    """ 数据预处理函数，将数据进行同向化与无量纲化 """
    def __init__(self, data, result=None, do_forward=False, do_normalize=False):
        self.data = data
        self.result = result
        self.do_forward = do_forward
        self.do_normalize = do_normalize

    def apply_forward(self, data):
        """ 同向化函数，将数据转换为正向数据 """
        if self.do_forward:
            data = data.apply(lambda x: 1/x)
        return data

    def apply_normalize(self, data):
        """ 无量纲化函数，将数据转换为无量纲数据 """
        if self.do_normalize:
            m = max(data)
            data = data.apply(lambda x: x/m)
        return data

    def pretreatment(self):
        """ 执行同向化函数 """
        data = self.apply_forward(self.data)
        data = self.apply_normalize(data)
        self.result = data







