"""
SUnCNN simple PyTorch implementation
"""

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model2 import Model
import restraint
from utils import extract_edm
from ..Base import Base


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


class SUnCNN(Base):
    def __init__(self, params, init):
        super(SUnCNN, self).__init__(params, init)

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        # Hyperparameters
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = self.params
        niters = params['niters']
        lr = params['lr']
        exp_weight = params['exp_weight']
        noisy_input = {
            0: False,
            1: True
        }[0]

        data = self.init.copy()
        Y = data['Y'].copy()
        D = data['E'].copy()  # 论文中的D就是端元矩阵
        L = data['L']
        P = data['P']
        H = data['H']
        W = data['W']

        model = Model(L, P).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        Y = torch.Tensor(Y)
        Y = Y.view(1, L, H, W)
        Y = Y.to(device)
        D = torch.Tensor(D).to(device)

        noisy_input = torch.rand_like(Y) if noisy_input else Y

        progress = tqdm(range(niters)) if output_display else range(niters)
        out_avg = 0
        minvol = restraint.MinVolumn(L, P)
        # s2w = S2W(H,W)
        for ii in progress:
            optimizer.zero_grad()
            abund = model(noisy_input.detach())
            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * exp_weight + abund.detach() * (1 - exp_weight)
            # Reshape data

            # 以下代码的效果：例如abund->(1,3,50,50)，变成(3,2500);主要用来做loss
            # a = abund.squeeze(0)
            # a = a.reshape(a.shape[0], -1).to(device)

            # 损失函数加这里
            # ypred = D @ abund.view(-1, H * W)
            # apred = abund.view(-1, H * W)
            # epred = extract_edm(ypred.detach().cpu().numpy(), apred.detach().cpu().numpy())
            # epred = torch.tensor(epred)
            loss1 = F.mse_loss(Y.view(-1, H * W), D @ abund.view(-1, H * W))
            # loss2 = minvol(epred)
            loss = loss1

            if output_display:
                progress.set_postfix_str(f"loss={loss.item():.3e}")
            loss.backward()
            optimizer.step()

        A = out_avg.cpu().numpy().reshape(-1, H * W)

        return {
            'A': A
        }
