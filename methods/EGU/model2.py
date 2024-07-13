from torch import nn
import torch
import torch.functional as F
from torch.nn import Module, Sequential, Softmax, \
    Linear, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, \
    MaxPool2d, AvgPool2d, \
    LeakyReLU, Sigmoid, ELU, SELU, Dropout, Dropout2d, Tanh, ReLU, \
    MSELoss
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
act_val = 0.01


def l2(a):
    return torch.sum(torch.norm(a, p=2) ** 2)


class Model(Module):
    def __init__(self, P, L, H):
        super(Model, self).__init__()
        self.P = P
        self.L = L
        self.share_3 = Sequential(
            Conv2d(128, 32, 1, bias=False),
            BatchNorm2d(32, momentum=0.9)
        )

        self.e_net = Sequential(
            # 1
            Conv2d(L, 256, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(256, momentum=0.9),
            Dropout2d(p=0.9),
            Tanh(),
            # 2
            Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(128, momentum=0.9),
            Tanh(),
            # 3
            self.share_3,
            ReLU(),
            # 4
            Conv2d(32, P, 1, bias=False),
            Softmax(dim=1)
        )

        a = H - 8 * (math.floor(H / 8.0) - 1)  # adapt the ConvTranspose2d

        self.ur_net = Sequential(
            # 1
            Conv2d(L, 256, kernel_size=5, stride=1, padding=2, bias=False),
            BatchNorm2d(256, momentum=0.9),
            Dropout(0.9),
            AvgPool2d(kernel_size=2, stride=2),
            Tanh(),
            # 2
            Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(128, momentum=0.9),
            Dropout(0.9),
            AvgPool2d(kernel_size=2, stride=2),
            Tanh(),
            # 3
            self.share_3,
            AvgPool2d(kernel_size=2, stride=2),
            ReLU(),
            # 4
            ConvTranspose2d(32, P, kernel_size=int(a), stride=8, bias=False),
            Softmax(dim=1),
            # de
            ConvTranspose2d(P, 32, 1, bias=False),
            BatchNorm2d(32, momentum=0.9),
            Sigmoid(),
            ConvTranspose2d(32, 128, 1, bias=False),
            BatchNorm2d(128, momentum=0.9),
            Sigmoid(),
            ConvTranspose2d(128, 256, 3, padding=1, bias=False),
            BatchNorm2d(256, momentum=0.9),
            Sigmoid(),
            ConvTranspose2d(256, L, 5, padding=2, bias=False),
            BatchNorm2d(L, momentum=0.9),
            Sigmoid()
        )

    def forward(self, y, vca_edm):
        vca_edm = vca_edm.float()
        l2_loss = 0
        x = 0
        # e_net
        x_pure = 0
        for i, layer in enumerate(self.e_net):
            if i == 0:
                x = layer(vca_edm)
            elif i == 7:
                for la in layer:
                    x = la(x)
                    if isinstance(la, Conv2d) or isinstance(la, ConvTranspose2d):
                        l2_loss += l2(x)
            else:
                x = layer(x)

            if i == 9:
                x_pure = x

            if isinstance(layer, Conv2d) or isinstance(layer, ConvTranspose2d):
                l2_loss += l2(x)

        abundances_pure = x.squeeze().reshape(self.P, -1).transpose(1, 0)  # (5,200,200)->(40000,5)

        # ur_net
        abundances_mixed = 0
        for i, layer in enumerate(self.ur_net):
            if i == 0:
                x = layer(y)
            elif i == 10:
                for la in layer:
                    x = la(x)
                    if isinstance(la, Conv2d) or isinstance(la, ConvTranspose2d):
                        l2_loss += l2(x)
            else:
                x = layer(x)

            if i == 14:
                abundances_mixed = x

            if isinstance(layer, Conv2d) or isinstance(layer, ConvTranspose2d):
                l2_loss += l2(x)

        y_hat = x
        #
        # print(x_pure.shape, abundances_mixed.shape)
        # print(abundances_pure.shape)
        # print(y_hat.shape)
        # print(l2_loss)
        # exit()
        return abundances_pure, abundances_mixed, y_hat, x_pure, l2_loss
