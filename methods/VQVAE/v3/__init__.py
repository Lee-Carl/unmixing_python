import numpy as np
import os
import random
import scipy.io as scio
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
from .model3 import Model
from methods import Base
from tqdm import tqdm
from .restraint import restraint


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class VQVAE(Base):
    def __init__(self, params, init):
        super(VQVAE, self).__init__(params, init)

    def run(self, savepath=None, *args, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 导入数据集
        data = self.init.copy()
        Y = data["Y"]  # 像元
        P = data["P"]  # 端元个数
        L = data['L']
        N = data['N']
        H = data['H']
        W = data['W']
        A = data['A']
        E = data['E']
        a_tensor = torch.from_numpy(A).to(device).clone()
        e_tensor = torch.from_numpy(E).to(device).clone()
        params = self.params

        # 超参数
        lr = params['lr']
        EPOCH = params['EPOCH']

        # 生成DataLoader
        y_tensor = torch.from_numpy(Y).to(device)
        y_to_e = torch.from_numpy(Y.reshape(L, 1, H, W)).to(device)
        y_to_a = torch.from_numpy(Y.reshape(1, L, H, W)).to(device)

        # 模型设置
        model = Model(P, L, H, W, 0.9).to(device)
        model.apply(weights_init)

        # optimizer = torch.optim.Adam(
        #     [
        #         {'params': model.encoder_a.parameters()},
        #         {'params': model.encoder_z.parameters()},
        #         {'params': model.decoder.parameters()}
        #     ], lr=lr, weight_decay=1e-5)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        l2 = nn.MSELoss()

        model.decoder[6].weight.data = e_tensor.t()
        # 数据
        losses = []
        model.train()

        prograss = tqdm(range(EPOCH))

        for epoch in prograss:
            y_hat, a_hat, e_hat = model(y_to_e.clone(), y_to_a.clone())
            loss1 = l2(y_hat, y_tensor) # rec
            loss2 = restraint.RMSE(y_hat,y_tensor)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())

        if savepath:
            torch.save(model.state_dict(), savepath + '/VQVAE.pt')

        model.eval()
        with torch.no_grad():
            y_hat, a_hat, e_hat = model(y_to_e.clone(), y_to_a.clone())
            y_hat = y_hat.cpu().numpy()
            e_hat = e_hat.data.cpu().numpy()
            a_hat = a_hat.data.cpu().numpy().T

            return {
                'E': e_hat,
                'A': a_hat,
                'Y': y_hat,
                'loss': losses
            }
