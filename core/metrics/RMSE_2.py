import numpy as np


def RMSE_2(data_true, data_pred):  # 也是一种计算RMSE的方式，但与上面的compute_aRMSE不同
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
