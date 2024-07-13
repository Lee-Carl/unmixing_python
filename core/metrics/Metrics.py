import numpy as np


# todo:数据统一,只接受numpy数据
class Metrics:
    @staticmethod
    def compute_SAD(data_true, data_pred):
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

    @staticmethod
    def compute_aSAD(data_true, data_pred):
        a = (data_true * data_pred).sum(axis=0)
        b = np.sqrt(np.sum(data_true ** 2, 0))
        c = np.sqrt(np.sum(data_pred ** 2, 0))
        sad = np.arccos(a / (b * c))
        return sad.mean()

    def compute_SAD_2(self, data_true, data_pred):
        """
        :param data_true: (L,P)
        :param data_pred: (N,L,P)
        :return:
        """
        asad = 0
        if np.max(data_true) > 1:
            data_true = data_true / np.max(data_true)
        for e in data_pred:
            if np.max(e) > 1:
                x = e / np.max(e)
            else:
                x = e
            asad += self.compute_SAD(data_true=data_true, data_pred=x)[1]
        asad /= data_pred.shape[0]
        return asad.mean(), asad

    @staticmethod
    def compute_aRMSE(data_true, data_pred):
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

    @staticmethod
    def compute_aRMSE_2(data_true, data_pred):  # 也是一种计算RMSE的方式，但与上面的compute_aRMSE不同
        # 使用此指标的方法: PGMSU
        return np.mean(np.sqrt(np.mean((data_true - data_pred) ** 2, axis=0)))

    @staticmethod
    def compute_RMSE_2(data_true, data_pred):  # 也是一种计算RMSE的方式，但与上面的compute_aRMSE不同
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

    @staticmethod
    def compute_SRE(data_true, data_pred):
        sre = 20 * np.log10(np.linalg.norm(data_true, ord=2) / np.linalg.norm(data_true - data_pred, ord=2))
        return sre
