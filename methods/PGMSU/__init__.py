import numpy as np
import os
import random
import scipy.io as scio
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
from .model import Model
from custom_types import MethodBase
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
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class PGMSU(MethodBase):
    def __init__(self, params, init):
        super().__init__(params, init)

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 导入数据集
        data = self.init.copy()
        Y = data["Y"]  # 像元
        EM = data["E"]  # 端元
        P = data["P"]  # 端元个数
        A_true = data["A"]  # 丰度
        Channel = data['L']
        N = data['N']
        batchs = N // 10

        params = self.params

        # 超参数
        lr = params['lr']
        epochs = params['epochs']
        z_dim = params['z_dim']
        seed = params['seed']
        os.environ['PYTHONHASHSEED'] = '0'
        set_seed(seed)

        lambda_kl = params['lambda_kl']
        lambda_sad = params['lambda_sad']
        lambda_vol = params['lambda_vol']

        # 生成DataLoader
        Y = np.transpose(Y)
        train_db = torch.tensor(Y)
        train_db = torch.utils.data.TensorDataset(train_db)
        train_db = torch.utils.data.DataLoader(train_db, batch_size=batchs, shuffle=True)

        # 初始化权重
        # EM, _, _ = hyperVca(Y.T, P)
        EM = EM.T
        EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype('float32')
        EM = torch.tensor(EM).to(device)  # 1*4*198

        # 模型设置
        model = Model(P, Channel, z_dim).to(device)
        model.apply(weights_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 数据
        losses = []
        model.train()

        progress = tqdm(range(epochs)) if output_display else range(epochs)
        for epoch in progress:
            for step, y in enumerate(train_db):
                y = y[0].to(device)

                y_hat, mu, log_var, a, em_tensor = model(y)
                Loss_Rec = ((y_hat - y) ** 2).sum() / y.shape[0]
                kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
                kl_div = kl_div.sum() / y.shape[0]
                # KL balance of VAE
                kl_div = torch.max(kl_div, torch.tensor(0.2).to(device))

                if epoch < epochs // 2:
                    # pre-train process
                    loss_vca = ((em_tensor - EM) ** 2).sum() / y.shape[0]
                    loss = Loss_Rec + lambda_kl * kl_div + 0.1 * loss_vca
                else:
                    # training process
                    # note:constrain 1 min_vol of EMs
                    em_bar = em_tensor.mean(dim=1, keepdim=True)
                    loss_minvol = ((em_tensor - em_bar) ** 2).sum() / y.shape[0] / P / Channel

                    # note:constrain 2 SAD for same materials
                    em_bar = em_tensor.mean(dim=0, keepdim=True)  # [1,5,198] [1,z_dim,Channel]
                    aa = (em_tensor * em_bar).sum(dim=2)
                    em_bar_norm = (em_bar ** 2).sum(dim=2).sqrt()
                    em_tensor_norm = (em_tensor ** 2).sum(dim=2).sqrt()
                    sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
                    loss_sad = sad.sum() / y.shape[0] / P

                    loss = Loss_Rec + lambda_kl * kl_div + lambda_vol * loss_minvol + lambda_sad * loss_sad  # 原方法
                    # progress.set_postfix_str(f"loss_sad:{loss_sad} | loss_minvol:{loss_minvol} | kl_div:{kl_div} | l12_norm:{restraint.l12_norm(a)} | Loss_Rec:{Loss_Rec}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if savepath:
                torch.save(model.state_dict(), savepath + 'PGMSU.pt')
                losses.append(loss.detach().cpu().numpy())
                # if (epoch + 1) % 50 == 0:
                #     scio.savemat(savepath + 'loss.mat', {'loss': losses})
                #     print('epoch:', epoch + 1, ' save results!')

        model.eval()
        with torch.no_grad():
            y_hat, mu, log_var, A, em_hat = model(torch.tensor(Y).to(device))
            A_hat = A.cpu().numpy().T
            A_true = A_true.reshape(P, N)
            dev = np.zeros([P, P])
            for i in range(P):
                for j in range(P):
                    dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
            pos = np.argmin(dev, axis=0)

            # 丰度与端元
            A_hat = A_hat[pos, :]
            em_hat = em_hat[:, pos, :]

            return {
                'E': em_hat.data.cpu().numpy(),
                'A': A_hat.T,
                'Y': y_hat.cpu().numpy(),
                'loss': losses
            }
