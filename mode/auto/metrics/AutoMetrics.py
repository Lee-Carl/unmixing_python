from core.init import Norm
from core.SmartMetrics import SmartMetrics
from core.func import hsiSort
from core.load import loadhsi
import scipy.io as sio
import numpy as np


class AutoMetrics:
    def __call__(self, case, file):
        return self.fun2(case, file)

    @staticmethod
    def fun1(case, file):
        data_true = loadhsi(case)
        data_true['Y'] = Norm.max_norm(data_true['Y'])
        am = SmartMetrics(data_true)
        P, L, N = am.dp.getPLN()
        H, W = am.dp.getHW()
        # 找到最新预测数据
        data_pred = sio.loadmat(file)

        # 预测数据
        A_pred = data_pred["A"]
        E_pred = data_pred["E"]

        if 'Y' in data_pred.keys():
            Y_pred = data_pred['Y']
        else:
            Y_pred = am.dp.generateY(e=E_pred, a=A_pred)

        # 真实数据
        A_true = data_true["A"]
        if len(E_pred.shape) == 3 and "E_3d" in data_true.keys():
            E_true = data_true["E_3d"]
        else:
            E_true = data_true["E"]
        Y_true = data_true["Y"]

        E_pred, _ = hsiSort.sort_edm(E_true, E_pred, P)
        A_pred, _ = hsiSort.sort_abu(A_true, A_pred, P)

        # 计算结果
        aRMSE = am.compute_RMSE_2(A_true, A_pred)
        aSAD, SAD = am.compute_SAD(E_true, E_pred)

        RMSE_Y = am.compute_RMSE_2(Y_true, Y_pred)
        SAD_Y = am.compute_SAD(Y_true, Y_pred, type="Y")[0]

        aRMSE2 = am.compute_RMSE(A_true, A_pred)
        # 显示
        content = f"aSAD = {aSAD} | SAD= {SAD}\n" \
                  f"aRMSE = {aRMSE} | aRMSE2 = {aRMSE2}\n" \
                  f"RMSE_Y = {RMSE_Y} | SAD_Y = {SAD_Y}\n"
        print(content)

        return SAD, aSAD, aRMSE, SAD_Y, RMSE_Y, aRMSE2

    @staticmethod
    def fun2(case, file):
        data_true = loadhsi(case)
        data_true['Y'] = Norm.max_norm(data_true['Y'])
        am = SmartMetrics(data_true)
        P, L, N = am.dp.getPLN()
        H, W = am.dp.getHW()
        # 找到最新预测数据
        data_pred = sio.loadmat(file)

        # 预测数据
        A_pred = data_pred["A"]
        E_pred = data_pred["E"]

        if 'Y' in data_pred.keys():
            Y_pred = data_pred['Y']
        else:
            Y_pred = am.dp.generateY(e=E_pred, a=A_pred)

        # 真实数据
        A_true = data_true["A"]
        if len(E_pred.shape) == 3 and "E_3d" in data_true.keys():
            E_true = data_true["E_3d"]
        else:
            E_true = data_true["E"]
        Y_true = data_true["Y"]

        E_pred, _ = hsiSort.sort_edm(E_true, E_pred, P, repeat=True)
        A_pred, _ = hsiSort.sort_abu(A_true, A_pred, P)

        # 计算结果
        aRMSE = am.compute_RMSE_2(A_true, A_pred)
        rmse = am.compute_RMSE_a2(A_true, A_pred)
        aSAD, SAD = am.compute_SAD(E_true, E_pred)
        # SAD = np.mean(SAD, axis=1) # 如果遇到N*L*P的端元，则需要解开这个注释

        RMSE_Y = am.compute_RMSE_2(Y_true, Y_pred)
        SAD_Y = am.compute_SAD(Y_true, Y_pred, type="Y")[0]

        aRMSE2 = am.compute_RMSE(A_true, A_pred)

        # 显示
        content = f"aSAD = {aSAD} | SAD = {SAD}\n" \
                  f"aRMSE = {aRMSE} | RMSE = {rmse}\n" \
                  f"aRMSE2 = {aRMSE2}\n" \
                  f"RMSE_Y = {RMSE_Y} | SAD_Y = {SAD_Y}\n"
        print(content)

        return SAD, aSAD, rmse, aRMSE, SAD_Y, RMSE_Y, aRMSE2
