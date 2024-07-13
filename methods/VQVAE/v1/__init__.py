import numpy as np
import os
import random
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import Model, Decoder_NonLinear
from methods import Base
from tqdm import tqdm
from .restraint import Restraint


def createdir(dn):
    if not os.path.exists(dn):
        os.mkdir(dn)


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

    def pretrain_dec_nonlipart(self, hsi):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = self.params['pretrain']

        lr = params.get('lr', 1e-3)
        EPOCH = params.get('EPOCH', 4000)
        hsi = hsi.transpose(1, 0).to(device)

        model = Decoder_NonLinear(hsi.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        progress = tqdm(range(EPOCH))
        for epoch in progress:
            output = model(hsi)
            loss = criterion(output, hsi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model.decoder.state_dict()

    def run(self, savepath: str, *args, **kwargs):
        # GPU First
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load data
        data = self.init.copy()
        P = data["P"]  # the number of endmembers
        L = data['L']
        N = data['N']
        H = data['H']
        W = data['W']

        E = data['E']
        E_init = torch.from_numpy(E).float().to(device)

        # 超参数
        params = self.params
        lr = params['lr']
        EPOCH = params['EPOCH']
        weight_decay = params['weight_decay']

        # 生成训练数据
        Y = data['Y']
        Y2d_tensor = torch.tensor(Y).to(device)
        Y3d_tensor = torch.tensor(Y.reshape(1, L, H, W)).to(device)

        # 模型设置
        model = Model(L, P).to(device)
        model.decoder_linearpart[0].weight.data = E_init
        # 优化器
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': 1e-5},
        ], lr=lr, weight_decay=1e-5)

        # params1 = map(id, model.decoder_linearpart.parameters())
        # params2 = map(id, model.decoder_nonlinearpart.parameters())
        # ignored_params = list(set(params1).union(set(params2)))
        # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        # optimizer = torch.optim.Adam([
        #     {'params': base_params},
        #     {'params': model.decoder_linearpart.parameters(), 'lr': 1e-5},
        #     {'params': model.decoder_nonlinearpart.decoder.parameters(), 'lr': 1e-5},
        # ], lr=lr, weight_decay=weight_decay)

        l2 = nn.MSELoss()

        # pretrain
        filepos = os.path.join(os.path.dirname(__file__), 'VQVAE-pretrain.pth')  # 训练好的样本
        if os.path.exists(filepos):
            model.decoder_nonlinearpart.decoder.load_state_dict(torch.load(filepos))
        else:
            dec_nonlipart = self.pretrain_dec_nonlipart(Y2d_tensor.clone())
            model.decoder_nonlinearpart.decoder.load_state_dict(dec_nonlipart)
            if savepath:
                torch.save(dec_nonlipart, savepath + 'pretrain_decoder_nonlinear.pth')

        # 数据
        losses = []
        model.train()
        progress = tqdm(range(EPOCH))
        for _ in progress:
            y_hat, a_hat, vq_loss = model(Y3d_tensor.clone())

            loss = l2(y_hat, Y2d_tensor.t()) + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_postfix_str(f"loss={loss:.3e}")
            losses.append(loss.detach().cpu().numpy())

        if savepath:
            scio.savemat(savepath + 'loss.mat', {'loss': losses})
            torch.save(model.state_dict(), savepath + 'VQVAE.pt')

        model.eval()
        with torch.no_grad():
            y_hat, a_hat, _ = model(Y3d_tensor)
            y_hat = y_hat.cpu().detach().numpy().T
            a_hat = a_hat.cpu().detach().numpy()
            a_hat = a_hat.squeeze().reshape(P, -1)
            e_hat = model.decoder_linearpart[0].weight.data.cpu().numpy()

        return {
            'E': e_hat,
            'A': a_hat,
            'Y': y_hat,
            'loss': losses
        }
