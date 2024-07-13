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

        self.ur_net = Sequential(
            # 1
            Conv2d(L, 256, kernel_size=1, stride=1),
            BatchNorm2d(256, momentum=0.9),
            Dropout(0.9),
            Tanh(),
            # 2
            Conv2d(256, P, kernel_size=1, stride=1),
            BatchNorm2d(P, momentum=0.9),
            Dropout(0.9),
            Sigmoid()
        )

    @staticmethod
    def loss(self, abundances_mixed, l2_loss, **kwargs):
        reg = kwargs.get('reg', 1)
        loss = reg * l2_loss
        return loss

    def forward(self, y, vca_edm):
        l2_loss = 0
        x = 0
        # ur_net
        abundances_mixed = 0
        for i, layer in enumerate(self.ur_net):
            if i == 0:
                x = layer(y)
            else:
                x = layer(x)

            if i == 4:
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
        return abundances_mixed, l2_loss
