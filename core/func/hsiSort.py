import numpy as np
from core.metrics import Metrics
from utils import HsiUtil
from custom_types import HsiDataset


def pearson_correlation_coefficient(X, Y):
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    cov_XY = np.mean((X - mean_X) * (Y - mean_Y))
    std_X = np.std(X)
    std_Y = np.std(Y)
    r = cov_XY / (std_X * std_Y)
    return r


def mean(x, y):
    return np.mean((x - y) ** 2)


def sort_Edm_And_Abu(dtrue: HsiDataset, dpred: HsiDataset, case: int = 2,
                     edm_repeat: bool = False, abu_repeat: bool = False,
                     tip: bool = False):
    P, L, N = dtrue.getPLN()
    if len(dtrue.edm.shape) == 3:
        dtrue.edm = HsiUtil.changeDims(dtrue.edm, (L, P, N))
    else:
        dtrue.edm = HsiUtil.changeDims(dtrue.edm, (L, P))

    if len(dpred.edm.shape) == 3:
        dpred.edm = HsiUtil.changeDims(dpred.edm, (L, P, N))
    else:
        dpred.edm = HsiUtil.changeDims(dpred.edm, (L, P))

    if case == 1:
        # 依赖于sortForEndmember
        dpred.edm, dim = sort_edm(dtrue.edm, dpred.edm, P, repeat=edm_repeat)
        dpred.abu = HsiUtil.changeDims(dpred.abu, (P, N))
        dpred.abu = np.take(dpred.abu, dim, axis=0)
    else:
        dpred.E, _ = sort_edm(dtrue.edm, dpred.edm, P, repeat=edm_repeat)
        dpred.A, _ = sort_abu(dtrue.abu, dpred.abu, P, repeat=abu_repeat)

    return dpred


def __get_similarity_matrix(true_, pred_, P, similarity_func, tip=None):
    # 计算相反似度
    # 相似性矩阵
    similarity_matrix = np.zeros((P, P))
    # 相似度比较
    for i in range(P):
        for j in range(P):
            similarity_matrix[i, j] = similarity_func(true_[:, j], pred_[:, i])

    #  是否打印选择
    if tip:
        for i in range(P):
            for j in range(P):
                print(f"{similarity_matrix[i, j]}", end=' ')
            print()
    return similarity_matrix


def __choose_similarity_max(similarity_matrix, P, repeat, tip=None):
    # 对端元进行排序，根据相似性矩阵中的值进行排序；适合的算法：pearson_correlation_coefficient
    dim = [0] * P

    for i in range(P):
        r, c = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)  # 求出每一轮下矩阵的最大值

        # dim[c] = r 含义是第r个预测端元对应第c个真实端元
        dim[c] = r  # 加最大值所对应的端元号
        if not repeat:
            similarity_matrix[r, :] = -np.inf  # 将这一行的端元划掉，不能再选
            similarity_matrix[:, c] = -np.inf  # 将这一列的端元划掉，不能再选
        else:
            similarity_matrix[r, c] = -np.inf  # 将这个端元划掉，不能再选

    # 是否提示端元顺序
    if tip:
        print(f'排序后的顺序: {dim}')
    return dim


def __choose_similarity_min(similarity_matrix, P, repeat, tip=None):
    # 对端元进行排序，根据相似性矩阵中的值进行排序；适合的算法：sad,rmse
    dim = [0] * P
    for i in range(P):
        r, c = np.unravel_index(np.argmin(similarity_matrix, axis=None), similarity_matrix.shape)  # 求出每一轮下矩阵的最小值
        dim[c] = r  # 加最大值所对应的端元号
        if not repeat:
            similarity_matrix[r, :] = np.inf  # 将这一行的端元划掉，不能再选
            similarity_matrix[:, c] = np.inf  # 将这一列的端元划掉，不能再选
        else:
            similarity_matrix[r, c] = np.inf  # 将这个端元划掉，不能再选
    # 是否提示端元顺序
    if tip:
        print(f'排序后的顺序:{dim}')
    return dim


def __sort_framework(true_, pred_, P, func, repeat):
    # 获取相似性矩阵
    similarity_matrix = __get_similarity_matrix(true_, pred_, P, func)

    # 从相似矩阵中进行选择
    if func == pearson_correlation_coefficient:
        dim = __choose_similarity_max(similarity_matrix, P, repeat)
    else:
        dim = __choose_similarity_min(similarity_matrix, P, repeat)

    return dim


def sort_edm(dtrue, dpred, P, func=None, repeat=False):
    """
    Args:
        dtrue:(L,P,N) OR (L,P)
        dpred:(L,P,N) OR (L,P)
        P: 端元个数
        repeat: True
        func: 计算相似度的函数：
    return:
    pred: 预测的数据
    dim: 用于给丰度进行转换的维度，一般写apred = np.take(apred, dim, axis=0), axis对应P所在的维度
    """
    if func is None:
        mt = Metrics()
        func = mt.compute_aSAD
        # func = pearson_correlation_coefficient
    # 兼容性设置
    true_ = dtrue if len(dtrue.shape) == 2 else dtrue[:, :, 1]
    pred_ = dpred if len(dpred.shape) == 2 else dpred[:, :, 1]

    dim = __sort_framework(true_, pred_, P, func, repeat)

    # 完成排序
    pred = np.take(dpred, dim, axis=1)  # 完成排序,dpred[:,dim]或dpred[:,dim,:]

    return pred, dim


def sort_abu(dtrue, dpred, P, func=None, repeat=False):
    """
    dtrue or dpred:
        abundances:(P,N)
    P: 端元个数

    Return:
        dim
        pred
    """
    if func is None:
        func = Metrics.compute_aRMSE_2

    # 转换成(N,P)，为了兼容框架
    true_ = dtrue.T
    pred_ = dpred.T

    dim = __sort_framework(true_, pred_, P, func, repeat)

    # 完成排序
    # pred = np.take(dpred, dim, axis=0)  # 完成排序,其实相当于dpred[dim,:]
    pred = dpred[dim, :]

    return pred, dim
