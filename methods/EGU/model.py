from torch import nn
import torch
import torch.functional as F
from torch.nn import Module, Sequential, Softmax, \
    Linear, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, \
    MaxPool2d, AvgPool2d, \
    LeakyReLU, Sigmoid, ELU, SELU, Dropout, Dropout2d, Tanh, ReLU
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
act_val = 0.01


class Model(Module):
    def __init__(self, P, L, H):
        super(Model, self).__init__()
        self.P = P
        self.L = L
        self.e_net_12 = Sequential(
            # 1
            Conv2d(L, 256, kernel_size=1, stride=1),
            BatchNorm2d(256, momentum=0.9),
            Dropout2d(p=0.9),
            Tanh(),
            # 2
            Conv2d(256, 128, kernel_size=1, stride=1),
            BatchNorm2d(128, momentum=0.9),
            Tanh(),
        )

        self.ur_net_12 = Sequential(
            # 1
            Conv2d(L, 256, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(256, momentum=0.9),
            Dropout(0.9),
            AvgPool2d(kernel_size=2, stride=2),
            Tanh(),
            # 2
            Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128, momentum=0.9),
            Dropout(0.9),
            AvgPool2d(kernel_size=2, stride=2),
            Tanh(),
        )

        self.share_3 = Sequential(
            Conv2d(128, 32, 1),
            BatchNorm2d(32, momentum=0.9)
        )

        self.ur_net_3 = Sequential(
            AvgPool2d(kernel_size=2, stride=2),
            ReLU()
        )

        self.e_net_3 = ReLU()

        self.e_net_4 = Sequential(
            Conv2d(32, P, 1),
            Softmax(dim=1)
        )

        a = H - 8 * (math.floor(H / 8.0) - 1)  # adapt the ConvTranspose2d

        self.ur_net_4 = Sequential(
            ConvTranspose2d(32, P, kernel_size=int(a), stride=8),
            Softmax(dim=1)
        )

        self.ur_net_de = Sequential(
            ConvTranspose2d(P, 32, 1),
            BatchNorm2d(32, momentum=0.9),
            Sigmoid(),
            ConvTranspose2d(32, 128, 1),
            BatchNorm2d(128, momentum=0.9),
            Sigmoid(),
            ConvTranspose2d(128, 256, 3, padding=1),
            BatchNorm2d(256, momentum=0.9),
            Sigmoid(),
            ConvTranspose2d(256, L, 5, padding=2),
            BatchNorm2d(L, momentum=0.9),
            Sigmoid()
        )

    @staticmethod
    def upsample_(x, upscale=2):
        return x[:, :, :, None, :, None] \
            .expand(-1, -1, -1, upscale, -1, upscale) \
            .reshape(x.size(0), x.size(1), x.size(2) * upscale, x.size(3) * upscale)

    def forward(self, y, vca_edm):
        e1 = self.e_net_12(vca_edm)
        s = self.share_3(e1)
        e2 = self.e_net_3(s)
        abundances_pure = self.e_net_4[0](e2)
        abundances_pure = self.e_net_4[1](abundances_pure)
        abundances_pure = abundances_pure.squeeze()

        ur1 = self.ur_net_12(y)
        s_ur = self.share_3(ur1)
        ur2 = self.ur_net_3(s_ur)
        abundances_mixed = self.ur_net_4(ur2)

        y_hat = self.ur_net_de(abundances_mixed)
        return y_hat, abundances_mixed, abundances_pure
