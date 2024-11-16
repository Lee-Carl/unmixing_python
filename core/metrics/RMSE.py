import numpy as np
from custom_types import HsiData
from core.wraps import checkShape


@checkShape
def RMSE_1(data_true: HsiData, data_pred: HsiData):
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


@checkShape
def RMSE_2(data_true: HsiData, data_pred: HsiData):  # 也是一种计算RMSE的方式，但与上面的compute_aRMSE不同
    # 使用此指标的方法: PGMSU
    # 初始化一个空列表，用于存储每个数据集的RMSE值
    rmse_values = []
    P = data_true.shape[0]
    # # 循环计算每个数据集的RMSE值
    for i in range(P):
        # rmse = np.sqrt(np.mean((data_true[i] - data_pred[i]) ** 2, axis=0))
        rmse = np.sqrt(np.mean((data_true[i] - data_pred[i]) ** 2, axis=0))
        rmse_values.append(rmse)
    return rmse_values
