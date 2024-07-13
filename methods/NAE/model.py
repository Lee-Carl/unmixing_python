# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 20:28:55 2018

@author: Mou
"""
import torch
from torch.nn import Module, Sequential, Linear, \
    Conv2d, LeakyReLU, Sigmoid, MaxPool2d, ConvTranspose2d


class Model(Module):
    def __init__(self, L, P):
        super().__init__()  # python2写法：super(mlaem_model, self).__init__()
        self.encoder = Sequential(
            Linear(L, 128),
            LeakyReLU(0.1),
            Linear(128, 64),
            LeakyReLU(0.1),
            Linear(64, 16),
            LeakyReLU(0.1),
            Linear(16, P)
        )

        self.decoder_linearpart = Sequential(
            Linear(P, L, bias=False),
        )

        self.decoder_nonlinearpart = Sequential(
            Linear(L, L, bias=True),
            Sigmoid(),
            Linear(L, L, bias=True)
        )

    def forward(self, x):
        x_latent = self.encoder(x)
        x_latent = x_latent.abs()
        x_latent = x_latent.t() / x_latent.sum(1)
        x_latent = x_latent.t().float()  # 预测的丰度
        x_linear = self.decoder_linearpart(x_latent)
        x_pred = self.decoder_nonlinearpart(x_linear)  # 预测的高光谱图像
        return x_pred, x_latent

    def get_endmember(self, x):
        endmember = self.decoder_linearpart(x)
        return endmember
