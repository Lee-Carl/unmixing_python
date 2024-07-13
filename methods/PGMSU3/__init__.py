import numpy as np
import os
import random
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
from torch.optim import lr_scheduler
from .model import Model
from ..Base import Base
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def createdir(dn):
    if not os.path.exists(dn):
        os.mkdir(dn)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class PGMSU3(Base):
    def __init__(self, params, init):
        super(PGMSU3, self).__init__(params, init)

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 导入数据集
        data = self.init.copy()
        Y = data["Y"]  # 像元
        EM = data["E"]  # 端元
        P = data["P"]  # 端元个数
        L = data['L']
        H = data['H']
        W = data['W']
        N = data['N']
        batchs = N
        step_size = 100

        # 超参数
        params = self.params
        lr = params['lr']
        epochs = params['epochs']
        z_dim = params['z_dim']
        lambda_kl = params['lambda_kl']
        lambda_sad = params['lambda_sad']
        lambda_vol = params['lambda_vol']
        lambda_vca = params['lambda_vca']
        # weight_decay = params['weight_decay']

        # 生成DataLoader
        train_db = torch.tensor(Y.reshape(1, L, H, W))
        train_db = torch.utils.data.TensorDataset(train_db)
        train_db = torch.utils.data.DataLoader(train_db, batch_size=1, shuffle=False)

        # 初始化权重
        # EM, _, _ = hyperVca(Y.T, P)
        EM = EM.T
        EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype('float32')
        EM = torch.tensor(EM).to(device)  # 1*4*198

        # 模型设置
        model = Model(P, L, z_dim, H, W).to(device)
        model.apply(weights_init)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.9)

        # 数据
        losses = []
        obj1 = []
        obj2 = []
        obj3 = []
        obj4 = []
        obj5 = []
        model.train()
        progress = tqdm(range(epochs)) if output_display else range(epochs)
        for epoch in progress:

            for step, y in enumerate(train_db):
                y = y[0].to(device)
                y_hat, mu, log_var, a, em_tensor = model(y)

                # 1.rec
                loss1 = ((y_hat - y) ** 2).sum() / batchs

                # 2.kl
                kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
                kl_div = kl_div.sum() / batchs
                kl_div = torch.max(kl_div, torch.tensor(0.2).to(device))  # KL balance of VAE
                loss2 = lambda_kl * kl_div

                if epoch < epochs // 3:
                    # pre-train process
                    # loss_vca
                    loss_vca = ((em_tensor - EM) ** 2).sum() / batchs
                    loss3 = loss_vca * lambda_vca
                    loss = sum([loss1, loss2, loss3])
                else:
                    # training process
                    # note:constrain 1 min_vol of EMs
                    em_bar = em_tensor.mean(dim=1, keepdim=True)
                    loss_minvol = ((em_tensor - em_bar) ** 2).sum() / batchs / P / L
                    loss4 = lambda_vol * loss_minvol

                    # note:constrain 2 SAD for same materials
                    em_bar = em_tensor.mean(dim=0, keepdim=True)  # [1,5,198] [1,z_dim,Channel]
                    aa = (em_tensor * em_bar).sum(dim=2)
                    em_bar_norm = (em_bar ** 2).sum(dim=2).sqrt()
                    em_tensor_norm = (em_tensor ** 2).sum(dim=2).sqrt()
                    sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
                    loss_sad = sad.sum() / batchs / P
                    loss5 = loss_sad * lambda_sad

                    loss = sum([loss1, loss2, loss4, loss5])

                    obj1.append(loss1.detach().cpu().numpy())
                    obj2.append(loss2.detach().cpu().numpy())
                    obj3.append(loss4.detach().cpu().numpy())
                    obj4.append(loss5.detach().cpu().numpy())
                    if epoch == epochs - 1:
                        theloss = f'loss_sad:{loss_sad} | loss_minvol:{loss_minvol} | kl_div:{kl_div} | Loss_Rec:{loss1}'
                        print(theloss)

                losses.append(loss.detach().cpu().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            if savepath:
                torch.save(model.state_dict(), savepath + '/PGMSU.pt')

        model.eval()
        with torch.no_grad():
            y_hat, mu, log_var, A, em_hat = model(torch.tensor(Y.reshape(1, L, H, W)).to(device))

            predict_data = {
                'E': em_hat.data.cpu().numpy(),
                'A': A.cpu().numpy().squeeze(0).reshape(P, N),
                'Y': y_hat.cpu().numpy().squeeze(0).reshape(L, N),
                'loss': losses,
                'loss_list': [obj1, obj2, obj3, obj4],
                'loss_name': ['rec', 'kl', 'vol', 'sad']
            }

            return predict_data
