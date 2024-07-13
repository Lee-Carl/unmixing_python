"""
CNNAEU simple PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import torch
from sklearn.feature_extraction.image import extract_patches_2d
from .model import Model
from ..Base import Base
import scipy.io as sio
import restraint as rs
from torch.optim import lr_scheduler
import torch.nn as nn
from utils import extract_edm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import restraint as rs
import torch.nn.init as init


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class CNNAEU3(Base):
    def __init__(self, params, init):
        super(CNNAEU3, self).__init__(params=params, init=init)
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

    @staticmethod
    def reset_parameters(d, initializer=None):
        # 用来初始化的，但是效果不好
        for weight in d.parameters():
            if len(weight.shape) < 2:
                torch.nn.init.kaiming_normal_(weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_normal_(weight)

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        # 载入数据
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
        lr1 = params['lr1']
        lr2 = params['lr2']
        lr3 = params['lr3']
        lr4 = params['lr4']
        dp = params['dp']
        weight_decay = params['weight_decay']
        batch_size = params['batch_size']
        patch_size = params['patch_size']
        lambda1 = params['lambda1']
        lambda2 = params['lambda2']
        lambda3 = params['lambda3']
        num_patches = int(params['num_patches'])

        # dataloader
        ## 1
        # Y_numpy = Y.reshape((L, H, W)).transpose((1, 2, 0))
        input_patches = extract_patches_2d(
            Y_numpy,
            max_patches=num_patches,
            patch_size=(patch_size, patch_size),
        )
        # input_patches = input_patches.transpose((0, 3, 1, 2))  # 250,156,40,40

        ## 2
        Y_numpy = Y.reshape((L, H, W))
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转为张量
            transforms.RandomCrop(patch_size),  # 随机裁剪
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
        custom_dataset = CustomDataset(data=Y_numpy, transform=transform)

        dataloader = torch.utils.data.DataLoader(
            custom_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Send model to GPU
        model = Model(P, L, dp).to(self.device)

        # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        optimizer = torch.optim.RMSprop([
            {'params': model.encoder1.parameters()},
            {'params': model.encoder2.parameters(), 'lr': lr},
            {'params': model.encoder3.parameters()},
            {'params': model.decoder.parameters()},
        ], lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)

        # 其他操作
        progress = tqdm(range(epochs)) if output_display else range(epochs)
        aso = rs.SumToOneLoss()
        s2w = rs.S2W()
        losses = []
        mse = nn.MSELoss()
        for _ in progress:
            losses.clear()
            running_loss = 0
            for ii, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                print(batch.shape)
                optimizer.zero_grad()

                abund, outputs = model(batch)

                # Reshape data
                loss1 = self.CNNAEU_loss(batch, outputs) * lambda1  # SAD
                loss2 = mse(batch, outputs) * lambda2
                loss = sum([loss1, loss2])

                running_loss += loss.item()

                loss.backward()
                optimizer.step()
                # scheduler.step()
                losses.append(loss.cpu().detach().numpy())

        # Get final abundances and endmembers
        model.eval()
        with torch.no_grad():
            Y_eval = torch.Tensor(Y.reshape((1, L, H, W))).to(self.device)
            abund, _ = model(Y_eval)
            Ahat = abund.detach().cpu().numpy().reshape(P, H * W)
        Ehat = extract_edm(y=Y.copy(), a=Ahat)

        return {
            'E': Ehat,
            'A': Ahat,
            'loss': losses
        }
