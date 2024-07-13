import numpy as np


def SAD(data_true, data_pred):
    """
    端元的shape要求：(L,P)-(波段数，端元数) OR (L,P,N)-(波段数，端元数,像元数)
    像元的shape要求：(L,N)-(波段数，像元数->像元维度之积)
    ————————————————————————————————————————————————————————————————————————————————————————————————————————
    obj:
    M -- 计算端元
    Y -- 计算像元
    ————————————————————————————————————————————————————————————————————————————————————————————————————————
    计算方式一：
    a = (data_true * data_pred).sum(axis=0)
    b = np.linalg.norm(data_true, ord=2, axis=0)
    c = np.linalg.norm(data_pred, ord=2, axis=0)
    return np.arccos(a / (b * c))
    计算方式二：
    a = (data_true * data_pred).sum(axis=0)
    b = np.sqrt(np.sum(data_true ** 2, 0))
    c = np.sqrt(np.sum(data_pred ** 2, 0))
    return np.arccos(a / (b * c))
    """
    a = (data_true * data_pred).sum(axis=0)
    b = np.sqrt(np.sum(data_true ** 2, 0))
    c = np.sqrt(np.sum(data_pred ** 2, 0))
    sad = np.arccos(a / (b * c))
    return sad.mean(), sad
