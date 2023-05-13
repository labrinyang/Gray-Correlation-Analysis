import unittest
import pandas as pd
from gray_correlation import gray_corr

class TestGrayCorrelation(unittest.TestCase):

    def test_gray_corr(self):
        # 创建一个数据框，其中包含四个指标和五个评价对象
        data = pd.DataFrame({
            'index1': [0.1, 0.2, 0.15, 0.3, 0.25],
            'index2': [0.6, 0.4, 0.7, 0.5, 0.8],
            'index3': [0.7, 0.8, 0.75, 0.85, 0.9],
            'index4': [0.2, 0.3, 0.25, 0.35, 0.4]
        }, index=['object1', 'object2', 'object3', 'object4', 'object5'])

        # 设置负向指标和正向指标
        negtive_list = [0, 2]  # index1 和 index3 是负向指标
        positive_list = [1, 3]  # index2 和 index4 是正向指标

        # 调用 gray_corr 函数，计算灰色关联度
        result = gray_corr(data, negtive_list, positive_list, transpose=False, rho=0.5, store=False, sort=True)

        # 检查结果是否正确。这应该根据你的预期结果来设定
        expected_result = pd.Series({
            'object1': 0.742857,
            'object5': 0.739286,
            'object3': 0.632799,
            'object4': 0.546260,
            'object2': 0.524675
        })
        pd.testing.assert_series_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()
