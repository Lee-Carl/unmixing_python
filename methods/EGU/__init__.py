import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
import os
from tqdm import tqdm
from .model2 import Model
from ..Base import Base
import torch.nn.functional as F
import numpy as np
import math


class Restraint:
    @staticmethod
    def cross_entropy(pure_layer, abu):
        log_probs = F.log_softmax(pure_layer.squeeze(), dim=1)
        loss = F.nll_loss(log_probs, torch.argmax(abu, dim=0).long())
        return loss

    @staticmethod
    def l2_norm(y_hat, x_mixed):
        return torch.mean((y_hat - x_mixed) ** 2)


class EGU(Base):
    def __init__(self, params, init):
        super(EGU, self).__init__(params=params, init=init)

    @staticmethod
    def random_mini_batches(X1, X2, Y, mini_batch_size, seed):

        m = X1.shape[0]
        m1 = X2.shape[0]
        mini_batches = []
        np.random.seed(seed)

        permutation = list(np.random.permutation(m))
        shuffled_X1 = X1[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

        permutation1 = list(np.random.permutation(m1))
        shuffled_X2 = X2[permutation1, :]

        num_complete_minibatches = math.floor(m1 / mini_batch_size)

        mini_batch_X1 = shuffled_X1
        mini_batch_Y = shuffled_Y

        for k in range(0, num_complete_minibatches):
            mini_batch_X2 = shuffled_X2[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
            mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        data = self.init.copy()
        x_mixed_set = data['Y']
        x_pure_set = data['D']
        A = data['A']
        abu = torch.tensor(A).to(device)
        # 计算每列的和; 将 Abund 的每一列除以对应列的和
        TrLabel = A / np.sum(A, axis=0)

        N = data['N']
        P = data['P']
        L = data['L']
        H = data['H']
        W = data['W']

        params = self.params
        EPOCH = params['EPOCH']
        lr = params['lr']
        weight_decay = params['weight_decay']
        batch_size = N // 10
        reg = params['reg']

        # 定义模型
        model = Model(P, L, H).to(device)

        # 定义数据集
        x_mixed = torch.tensor(x_mixed_set.reshape(1, L, H, W)).to(device)
        x_mixed_enet = torch.tensor(x_mixed_set.transpose(1, 0).reshape(-1, L, 1, 1)).to(device)
        x_pure = torch.tensor(x_pure_set.T.reshape(-1, L, 1, 1)).to(device)

        # 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        op2 = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()  # softmax_cross_entropy_with_logits 对应的 PyTorch 函数是 CrossEntropyLoss
        l2_loss = nn.MSELoss()

        rs = Restraint()
        losses = []

        minibatch_size = N

        progress = range(EPOCH)
        for epoch in progress:
            y = x_mixed.clone()
            y_enet = x_mixed_enet.clone()
            edm_vca = x_pure.clone()

            minibatches = self.random_mini_batches(x_pure_set.T, x_mixed_set.T, TrLabel.T, minibatch_size, seed=0)

            for minibatch in minibatches:
                (batch_x1, batch_x2, batch_y) = minibatch
                batch_x1 = torch.from_numpy(batch_x1).clone().unsqueeze(2).unsqueeze(3).to(device)
                batch_x2 = torch.from_numpy(batch_x2).clone()
                batch_y = torch.from_numpy(batch_y).clone().to(device)
                op2.zero_grad()
                abundances_pure, abundances_mixed, y_hat, x_pure_4, l2_loss = model(y, batch_x1)
                # loss2 = rs.cross_entropy(x_pure_4, batch_y.t()) + reg * l2_loss + rs.l2_norm(y_hat, x_mixed)
                loss2 = reg * l2_loss + rs.l2_norm(y_hat, x_mixed)
                loss2.backward()
                op2.step()

            optimizer.zero_grad()
            abundances_pure, abundances_mixed, y_hat, x_pure_4, l2_loss = model(y, y_enet)
            loss = rs.cross_entropy(x_pure_4, abu) + reg * l2_loss + rs.l2_norm(y_hat, x_mixed)

            progress.set_postfix_str(f"loss={loss:.3e}")
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                # print(loss)
                losses.append(loss)

        model.eval()
        with torch.no_grad():
            abundances_pure, abundances_mixed, y_hat, x_pure_4, l2_loss = model(x_mixed.clone(), x_mixed_enet.clone())
            # abundances_mixed, l2_loss = model(x_mixed.clone(), x_mixed_enet.clone())
            abund = abundances_mixed.cpu().detach().numpy()
            return {
                'A': abund.squeeze().reshape(P, -1),
                "A_3d": abund.squeeze()
            }
