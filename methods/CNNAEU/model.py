from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv2d, \
    LeakyReLU, Sigmoid, MaxPool2d, ConvTranspose2d, \
    BatchNorm1d, BatchNorm2d, ELU, SELU, Dropout, Dropout2d, Softmax


class Model(Module):
    def __init__(self, P, L):
        super(Model, self).__init__()
        self.scale = 3.0
        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": True,
        }
        self.p = P
        self.L = L

        self.encoder = nn.Sequential(
            Conv2d(
                self.L,
                48,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            LeakyReLU(**self.lrelu_params),
            BatchNorm2d(48),
            Dropout2d(p=0.2),
            Conv2d(48, self.p, kernel_size=1, bias=False),
            LeakyReLU(**self.lrelu_params),
            BatchNorm2d(self.p),
            Dropout2d(p=0.2),
        )

        self.decoder = Conv2d(
            self.p,
            self.L,
            kernel_size=11,
            padding=5,
            padding_mode="reflect",
            bias=False,
        )

    def forward(self, x):
        code = self.encoder(x)
        abund = F.softmax(code * self.scale, dim=1)
        x_hat = self.decoder(abund)
        return abund, x_hat
