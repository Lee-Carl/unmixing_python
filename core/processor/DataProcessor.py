import numpy as np
from .Sort import Sort


class DataProcessor:
    def __init__(self, dataset) -> None:
        self.data = dataset

    def getPLN(self):
        return self.data['P'], self.data['L'], self.data['N']

    def getHW(self):
        return self.data['H'], self.data['W']

    @staticmethod
    def transpose(data, shape):
        # 转置
        if data.shape != shape:
            if len(data.shape) == 2:
                data = data.T
            elif len(data.shape) == 3:
                dims = tuple(data.shape.index(e) for e in shape)  # 获取目标shape在data的shape中的次序
                data = data.transpose(dims)  # 转置成目标shape
        return data

    def sort_EndmembersAndAbundances(self, dtrue: dict, dpred: dict, case: int = 2, repeat: bool = False,
                                     edm_repeat: bool = False, abu_repeat: bool = False,
                                     tip: bool = False):
        s = Sort(tip)
        P, L, N = self.getPLN()

        d_true = dtrue['E']
        d_pred = dpred['E']

        if len(d_true.shape) == 3:
            d_true = self.transpose(d_true, (L, P, N))
        else:
            d_true = self.transpose(d_true, (L, P))

        if len(d_pred.shape) == 3:
            d_pred = self.transpose(d_pred, (L, P, N))
        else:
            d_pred = self.transpose(d_pred, (L, P))

        if case == 1:
            # 依赖于sortForEndmember
            dpred['E'], dim = s.sort_edm(d_true, d_pred, P, repeat=repeat)
            dpred['A'] = self.transpose(dpred['A'], (P, N))
            dpred['A'] = np.take(dpred['A'], dim, axis=0)
        else:
            dpred['E'], _ = s.sort_edm(d_true, d_pred, P, repeat=edm_repeat or repeat)
            dpred['A'], _ = s.sort_abu(dtrue['A'], dpred['A'].T, P, repeat=abu_repeat or repeat)

        return dpred

    def getShapeForMetrics(self, data_name):
        # 以下是各数据在计算部分指标前应有形状
        # 不同维度的端元，在计算SAD时需要另外讨论
        P, L, N = self.getPLN()
        if data_name == 'E':
            return L, P
        elif data_name == 'A':
            return P, N
        elif data_name == 'Y':
            return L, N
        else:
            raise Exception("未能匹配到对应的数据类型")

    def check(self, data_true, data_pred, type):
        shape = self.getShapeForMetrics(type)
        d_true = self.transpose(data_true, shape)
        d_pred = self.transpose(data_pred, shape)
        return d_true, d_pred

    def checkShape(self, data_pred: dict):
        P, L, N = self.getPLN()
        if 'Y' in data_pred:
            data_pred['Y'] = self.transpose(data_pred['Y'], (L, N))
        if 'A' in data_pred:
            data_pred['A'] = self.transpose(data_pred['A'], (P, N))
        if 'E' in data_pred:
            if len(data_pred['E'].shape) == 3:
                data_pred['E'] = self.transpose(data_pred['E'], (L, P, N))
            else:
                data_pred['E'] = self.transpose(data_pred['E'], (L, P))
        return data_pred

    @staticmethod
    def generateY(e, a):
        # 作用：用端元矩阵与丰度矩阵生成像元矩阵Y
        # 丰度一般是二维的
        # 端元有二维，也有三维的。假设E2=(198,4),E3=（198，4，10**4），A=（4，10**4）
        E, A = e.copy(), a.copy()
        if A.shape[0] > A.shape[1]:
            A = A.T  # A=(4,10**4)

        if len(E.shape) == 2:
            if E.shape[0] < E.shape[1]:
                E = E.T  # E2=(198,4)
            return E @ A  # y=(198,10**4)
        else:
            A = A.T  # A=(10**4,4)
            A = np.expand_dims(A, axis=1)  # A=(10**4,1,4)
            s = E.shape
            s = sorted(enumerate(s), key=lambda x: x[1])  # e=[(1, 4), (0, 198), (2, 10000)]
            E = E.transpose(s[2][0], s[0][0], s[1][0])  # E3=(10**4,4,198)
            Y = A @ E  # y=(10**4,1,198)
            Y = np.squeeze(Y, axis=1)  # y=(10**4,198)
            Y = Y.T  # y=(198,10**4)
            return Y
