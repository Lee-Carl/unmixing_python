"""
CNNAEU simple PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.image import extract_patches_2d
from .model import Model
from custom_types import MethodBase
import scipy.io as sio
import core.restraint as rs


class CNNAEU(MethodBase):

    def __init__(self, params, init):
        super().__init__(params, init)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

    @staticmethod
    def CNNAEU_loss(target, output):
        assert target.shape == output.shape

        dot_product = (target * output).sum(dim=1)
        target_norm = target.norm(dim=1)
        output_norm = output.norm(dim=1)
        sad_score = torch.clamp(dot_product / (target_norm * output_norm), -1, 1).acos()
        return sad_score.mean()

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        data = self.init.copy()
        Y = data['Y']
        P = data['P']
        L = data['L']
        N = data['N']
        H = data['H']
        W = data['W']

        params = self.params
        epochs = params['epochs']
        lr = params['lr']
        batch_size = params['batch_size']
        patch_size = params['patch_size']
        lambda1 = params['lambda1']

        # num_patches = int(250 * H * W * L / (307 * 307 * 162))
        num_patches = int(params['num_patches'])

        Y_numpy = Y.reshape((L, H, W)).transpose((1, 2, 0))

        input_patches = extract_patches_2d(
            Y_numpy,
            max_patches=num_patches,
            patch_size=(patch_size, patch_size),
        )
        input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

        # Send model to GPU
        model = Model(P, L).to(self.device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(
            input_patches,
            batch_size=batch_size,
            shuffle=True,
        )
        progress = tqdm(range(epochs)) if output_display else range(epochs)
        losses = []
        for _ in progress:
            losses.clear()
            running_loss = 0
            for ii, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                optimizer.zero_grad()

                _, outputs = model(batch)

                # Reshape data
                loss1 = self.CNNAEU_loss(batch, outputs)  # SAD
                loss2 = rs.l2(batch - outputs, beta=lambda1)
                loss = loss1 + loss2

                running_loss += loss.item()

                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())

        # Get final abundances and endmembers
        model.eval()
        Y_eval = torch.Tensor(Y.reshape((1, L, H, W))).to(self.device)
        abund, _ = model(Y_eval)
        Ahat = abund.detach().cpu().numpy().reshape(P, H * W)
        Ehat = model.decoder.weight.data.mean((2, 3)).detach().cpu().numpy()

        return {
            'E': Ehat,
            'A': Ahat,
            'loss': losses
        }
