"""
UnDIP simple PyTorch implementation
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from .model2 import Model
from ..Base import Base
import copy
from utils import extract_edm
import restraint as rs


class UnDIP(Base):
    def __init__(self, params, init):
        super(UnDIP, self).__init__(params, init)

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        params = self.params
        lr = params['lr']
        lr1 = params['lr1']
        lr2 = params['lr2']
        lr3 = params['lr3']
        lr4 = params['lr4']
        lr5 = params['lr5']
        lambda2 = params['lambda2']
        exp_weight = params['exp_weight']
        lambda1 = params['lambda1']
        noisy_input = False

        data = copy.deepcopy(self.init)
        Y = data['Y']
        E = data['E']
        L = data['L']
        P = data['P']
        H = data['H']
        W = data['W']
        niters = params['niters']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Model(L, P).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam([
            {'params': model.layer1.parameters(), 'lr': lr1},
            {'params': model.layer2.parameters(), 'lr': lr2},
            {'params': model.layer3.parameters(), 'lr': lr3},
            {'params': model.layer4.parameters(), 'lr': lr4},
            {'params': model.layer5.parameters(), 'lr': lr5},
        ], lr=lr)
        num_channels, h, w = L, H, W

        Y = torch.Tensor(Y)
        Y = Y.view(1, num_channels, h, w)
        Y = Y.to(device)
        # TODO Investigate requires grad here
        E = torch.Tensor(E).to(device)
        E.requires_grad = False

        noisy_input = Y
        progress = tqdm(range(niters)) if output_display else range(niters)
        out_avg = 0
        losses = []
        s2w = rs.S2WK(device=device, kernel=5)
        for ii in progress:
            optimizer.zero_grad()
            abund = model(noisy_input)

            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * exp_weight + abund.detach() * (
                        1 - exp_weight
                )
            # Reshape data
            abu = abund.unsqueeze(0).reshape(P, -1)
            loss1 = F.mse_loss(Y.view(-1, h * w), E @ abund.view(-1, h * w)) * lambda1
            # loss2 = s2w.compute_spa_loss(abu.clone(), H, W) * lambda2
            # loss3 = rs.norm_l2(abu,beta=1e-5)
            loss = sum([loss1])
            losses.append(loss.detach().cpu().numpy())
            # progress.set_postfix_str(f"loss={loss.item():.3e}")
            loss.backward()
            optimizer.step()

        A = out_avg.cpu().numpy().reshape(-1, h * w)
        E = extract_edm(y=self.init['Y'], a=A)
        return {
            'A': A,
            'E': E,
            'loss': losses
        }
