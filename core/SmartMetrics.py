import numpy as np
from .metrics import Metrics
from utils import HsiUtil
from custom_types import HsiDataset, HsiPropertyEnum


class SmartMetrics:
    def __init__(self, dataset: HsiDataset, tip=True) -> None:
        self.tip = tip
        self.data = dataset
        self.mt = Metrics()

    def compute_RMSE(self, A_true, A_pred):
        if A_true.shape != A_pred.shape:
            A_pred = A_pred.T
            assert A_true.shape == A_pred.shape, "两个数据的形状必须保持一致"
        aRMSE = self.mt.compute_aRMSE(data_true=A_true, data_pred=A_pred)
        return aRMSE

    def compute_RMSE_2(self, A_true, A_pred):
        if A_true.shape != A_pred.shape:
            A_pred = A_pred.T
            assert A_true.shape == A_pred.shape, "两个数据的形状必须保持一致"
        aRMSE = self.mt.compute_aRMSE_2(data_true=A_true, data_pred=A_pred)
        return aRMSE

    def compute_RMSE_a2(self, A_true, A_pred):
        if A_true.shape != A_pred.shape:
            A_pred = A_pred.T
            assert A_true.shape == A_pred.shape, "两个数据的形状必须保持一致"
        RMSE = self.mt.compute_RMSE_2(data_true=A_true, data_pred=A_pred)
        return RMSE

    def compute_SAD_3D(self, data_true, data_pred):
        P, L, N = self.data.getPLN()
        if len(data_true.shape) == 3:
            # 真实数据为三维数据
            shape = (L, P, N)
            data_true = HsiUtil.changeDims(data_true, shape)
            data_pred = HsiUtil.changeDims(data_pred, shape)
            aSAD, SAD = self.mt.compute_SAD(data_true=data_true, data_pred=data_pred)
        else:
            # 针对真实数据为二维数据的情况，写了三个计算方式。choose用于切换这些方式
            choose = 1
            if choose == 1:
                shape = (L, P, N)
                D_true = data_true[:, :, np.newaxis] * np.ones(N)  # 方式一：广播方式；内存占用小
                # expanded_data = np.tile(data_true[:, :, np.newaxis].astype(float), self.N) # 方式二：直接复制；内存占用大
                D_true = HsiUtil.changeDims(D_true, shape)
                data_pred = HsiUtil.changeDims(data_pred, shape)
                aSAD, SAD = self.mt.compute_SAD(data_true=D_true, data_pred=data_pred)
            else:
                # 方式三：逐个计算SAD，最后求平均值
                data_true = HsiUtil.changeDims(data_true, (L, P))
                data_pred = HsiUtil.changeDims(data_pred, (N, L, P))
                aSAD, SAD = self.mt.compute_SAD_2(data_true=data_true, data_pred=data_pred)
        return aSAD, SAD

    def compute_SAD(self, E_true, E_pred, prop: HsiPropertyEnum = HsiPropertyEnum.E):
        if len(E_pred.shape) == 2:
            shape = HsiUtil.getShapeForData(prop, self.data)
            E_true = HsiUtil.changeDims(E_true, shape)
            E_pred = HsiUtil.changeDims(E_pred, shape)
            aSAD, SAD = self.mt.compute_SAD(data_true=E_true, data_pred=E_pred)
        elif len(E_pred.shape) == 3:
            aSAD, SAD = self.compute_SAD_3D(data_true=E_true, data_pred=E_pred)
        else:
            raise Exception("无匹配的方法，请检查这两个数据的形状！")
        return aSAD, SAD

    def compute_SRE(self, A_true, A_pred):
        shape = HsiUtil.getShapeForData(HsiPropertyEnum.A, self.data)
        A_true = HsiUtil.changeDims(A_true, shape)
        A_pred = HsiUtil.changeDims(A_pred, shape)
        SRE = self.mt.compute_SRE(A_true, A_pred)
        return SRE
