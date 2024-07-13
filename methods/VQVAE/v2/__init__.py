import numpy as np
import os
import random
import scipy.io as scio
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
from .model2 import Model
from methods import Base
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class VQVAE(Base):
    def __init__(self, params, init):
        super(VQVAE, self).__init__(params, init)

    @staticmethod
    def generateDataset(data, patch_size=(50, 50), stride=(1, 1), max_num=250):
        C = data.shape[0]

        # 利用 unfold 函数进行划分
        patches = F.unfold(data, kernel_size=patch_size, stride=stride)

        # 将 patches 变成四维数据
        patches = patches.view(C, patch_size[0], patch_size[1], patches.shape[-1])
        patches = patches.permute(3, 0, 1, 2)
        return patches[:max_num, :, :, :]

    @staticmethod
    def generateDataset2(data, patch_size=(50, 50), stride=(1, 1), max_num=250):
        C, _, H, W = data.shape

        # 利用 torch.narrow 函数进行划分
        patches = []
        for i in range(0, H - patch_size[0] + 1, stride[0]):
            for j in range(0, W - patch_size[1] + 1, stride[1]):
                patch = data[:, i:i + patch_size[0], j:j + patch_size[1]]
                patches.append(patch)
                if len(patches) >= max_num:
                    break
            if len(patches) >= max_num:
                break

        patches = torch.stack(patches)

        return patches

    @staticmethod
    def xavier_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        # 导入数据集
        data = self.init.copy()
        Y = data["Y"]  # 像元
        P = data["P"]  # 端元个数
        L = data['L']
        N = data['N']
        H = data['H']
        W = data['W']

        # 超参数
        params = self.params
        lr = params['lr']
        EPOCH = params['EPOCH']
        batch_size = 15
        patch_size = (40, 40)
        stride = (10, 1)
        max_num = 100

        # 生成DataLoader
        y_tensor = torch.from_numpy(Y.reshape(L, 1, H, W)).clone()
        y_spilt = self.generateDataset(y_tensor, patch_size, stride, max_num)
        dataloader = DataLoader(y_spilt, batch_size=batch_size, shuffle=True)

        # 模型设置
        model = Model(P, L, 0.9).to(device)
        model.apply(self.xavier_init)

        mse = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)

        # 数据
        losses = []
        model.train()

        prograss = range(EPOCH)

        for epoch in prograss:
            for y_data in dataloader:
                y = y_data.clone().to(device)

                y_hat, a_hat, e_hat, vq_loss = model(y)
                # loss_sp = torch.mean(torch.sum(torch.sqrt(torch.abs(a_hat)), dim=1))
                loss = mse(y_hat, y) * 3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses.append(loss.detach().cpu().numpy())

        if savepath:
            torch.save(model.state_dict(), savepath + '/PGMSU.pt')

        y_tensor = torch.from_numpy(Y.reshape(1, L, H, W)).clone().to(device)
        model.eval()
        with torch.no_grad():
            y_hat, a_hat, e_hat, vq_loss = model(y_tensor)
            y_hat = y_hat.cpu().numpy().squeeze(0).reshape(L, -1)
            e_hat = e_hat.data.cpu().numpy().squeeze()
            a_hat = a_hat.cpu().numpy().squeeze(0).reshape(P, -1)

            dpred = {
                'E': e_hat,
                'A': a_hat,
                'Y': y_hat,
                'loss': losses
            }

            # for i in dpred.keys():
            #     print(f'{i}:{dpred[i].shape}')

            return dpred
