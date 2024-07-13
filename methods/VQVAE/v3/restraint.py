import math
import os
import scipy.io as sio
import torch
import torch.nn as nn


class restraint:
    @staticmethod
    def SAD(e_true, e_pred):
        sad = torch.mean(torch.acos(
            torch.sum(e_true * e_pred, dim=1) /
            (torch.norm(e_pred, dim=1, p=2) * torch.norm(e_true, dim=1, p=2))
        ))
        return sad

    @staticmethod
    def SAD2(x, output):
        a = torch.sqrt(torch.sum(x ** 2, 0))
        b = torch.sqrt(torch.sum(output ** 2, 0))
        val = torch.acos(torch.sum(x * output, dim=0) / a / b)
        return val.mean()

    @staticmethod
    def SRE(a_true, a_pred):
        sre = 20 * torch.log10(torch.norm(a_true, ord=2) / torch.norm(a_true - a_pred, ord=2))
        return sre

    @staticmethod
    def RMSE(e_true, e_pred):
        return torch.sqrt(torch.mean(torch.pow(torch.norm(e_true - e_pred, dim=1, p=2), 2)))

    @staticmethod
    def l12_norm(inputs):  # L1/2稀疏正则化
        out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
        return out

    @staticmethod
    def nuc_norm(inputs):  # 核范数
        return torch.norm(inputs, p='nuc')


class MinVolumn():  # 最小体积约束MVC
    def __init__(self, band, num_classes, delta=1):
        self.band = band  # 波段数
        self.delta = delta  # 系数
        self.num_classes = num_classes  # 纯净端元数

    def __call__(self, edm):
        edm_result = torch.reshape(edm, (self.band, self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = self.delta * ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss


class SumToOneLoss(nn.Module):  # 和为1约束
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma_reg=1e-7):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        return gamma_reg * loss


class Wspe_Wspa_Loss(nn.Module):
    def forward(self):
        pass

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.Lambda = 6e-3  # 总权重参数

    @staticmethod
    def get_Wspe(m):
        mt = m.t()
        mt_row_nums = mt.shape[0]  # mt矩阵的行数(m矩阵的列数)
        Epsilon = 1e-4
        W_spe = torch.diag(torch.tensor([1. / (torch.norm(mt[i], p=2) + Epsilon) for i in range(mt_row_nums)]))
        return W_spe

    def get_Wspa(self, abu):  # 先对数据进行变换
        abu_row, abu_col = abu.shape
        a = abu.reshape(-1)
        a = a.reshape(100, 100, 9)
        a = a.permute(2, 0, 1)
        t = []
        for i in a:
            t.append(self.get_Wspa2(i))
        a = torch.stack(t, dim=0)  # 还原成数字
        a = a.permute(1, 2, 0)
        a = a.reshape(abu_row, abu_col)
        return a

    def get_Wspa2(self, m):  # 这里是主要的操作；为了这里的扩充矩阵能用上GPU，特意写成了类
        row, col = m.shape  # 获取m的长*宽
        m_new = torch.zeros(row, col)
        # 1.扩充成（row+2,col+2）
        # 先扩充行:增加第一行，最后一行
        r = torch.zeros((1, col))
        m_expand = torch.cat((r, m, r), dim=0)
        # 再扩充列：增加第一列，最后一列
        c = torch.zeros((row + 2, 1))
        m_expand = torch.cat((c, m_expand, c), dim=1)
        # 2.类似卷积操作
        for a in range(0, row):
            for b in range(0, col):
                i = a + 1
                j = b + 1
                m_new[a][b] += (
                        m_expand[i - 1][j] +
                        m_expand[i + 1][j] +
                        m_expand[i][j - 1] +
                        m_expand[i][j + 1]  #
                )
                m_new[a][b] += (1. / math.sqrt(2)) * (
                        m_expand[i - 1][j - 1] +
                        m_expand[i - 1][j + 1] +
                        m_expand[i + 1][j - 1] +
                        m_expand[i + 1][j + 1]  #
                )
                m_new[a][b] /= (4. / math.sqrt(2) + 4.)

        return m_new

    @staticmethod
    def norm1_1(m):
        return torch.abs(m).sum()

    def __call__(self, abu):
        Wspe = self.get_Wspe(abu)
        Wspa = self.get_Wspa(abu).t()
        ww = torch.matmul(Wspe, Wspa).t()  # 权重之间 叉乘
        loss = self.norm1_1(torch.mul(ww, abu))  # 权重与丰度 点乘
        return loss
