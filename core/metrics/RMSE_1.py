import numpy as np


def RMSE_1(data_true, data_pred):
    """
    使用此指标的：CyCU
    data_true:真实数据
    data_pred:预测数据

    两者数据必须保持一致;
    RMSE一般计算丰度
    ——————————————————————————————————————————————————————
    分子：像元的维度之积

    计算方式一：
    return np.sqrt(((data_true - data_pred) ** 2).sum() / np.prod(data_true.shape))
    计算方式二：
    dim_product = 1
    for dim in data_true.shape:
        dim_product = dim_product * dim
    return np.sqrt(((data_true - data_pred) ** 2).sum() / dim_product)
    """
    return np.sqrt(((data_true - data_pred) ** 2).sum() / np.prod(data_true.shape))
