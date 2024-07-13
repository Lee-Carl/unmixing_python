import numpy as np
import scipy.io as scio
from typing import List, Dict, Any

""" 增加3d的版本，但是感觉不太符合预期 """


class Norm:
    # 归一化方法
    @staticmethod
    def max_norm(Y: np.ndarray):
        """将数据绽放到 [0,1]"""
        if np.max(Y) > 1:
            Y = Y / np.max(Y)
        return Y

    @staticmethod
    def maxmin_norm(data):
        """将数据绽放到 [0,1)"""
        maxi = np.amax(data)
        mini = np.amin(data)
        data_rescaled = (data.astype('float') - mini) / (maxi - mini)
        return data_rescaled

