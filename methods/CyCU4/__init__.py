import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
import os
from tqdm import tqdm

import restraint
from .model2 import Model
from .loss import SparseKLloss, SumToOneLoss, NonZeroClipper
import restraint as rs
from utils import extract_edm, extract_edm_torch
import scipy.io as sio


# Define Dataset
class MyTrainData(torch.utils.data.Dataset):
    def __init__(self, img, gt, transform=None):
        self.img = img.float()
        self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.gt

    def __len__(self):
        return 1


class CyCU4:
    def __init__(self, params, init):
        self.params = params
        self.init = init

    @staticmethod
    def get_init_dataofModel(dataset_name, net, turns=1, output_display=True):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        script_path = os.path.dirname(os.path.realpath(__file__))
        search_file = os.path.join(script_path, "data", f"{dataset_name}.mat")
        if os.path.exists(search_file):
            if output_display:
                print("已找到缓存，载入中...")

            data = sio.loadmat(f"{search_file}")
            net.encoder[0].weight.data = torch.tensor(data["0"], device=device)
            net.encoder[3].weight.data = torch.tensor(data["1"], device=device)
            net.encoder[6].weight.data = torch.tensor(data["2"], device=device)
            net.encoder[9].weight.data = torch.tensor(data["3"], device=device)

        else:
            if output_display:
                print("正产生随机性...")
            nn.init.kaiming_normal_(net.encoder[0].weight.data)
            nn.init.kaiming_normal_(net.encoder[3].weight.data)
            nn.init.kaiming_normal_(net.encoder[6].weight.data)
            data = {
                '0': net.encoder[0].weight.data.cpu().numpy(),
                '1': net.encoder[3].weight.data.cpu().numpy(),
                '2': net.encoder[6].weight.data.cpu().numpy(),
            }
            sio.savemat(search_file, data)

    def run(self, savepath=None, output_display=True, *args, **kwargs):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load Data
        data = self.init.copy()
        dataset = data["name"]
        P = data['P']
        L = data['L']
        H = data["H"]
        W = data["W"]
        N = data['N']
        col = data['W']

        params = self.params
        LR = params['LR']
        lr1 = params['lr1']
        lr2 = params['lr2']
        lr3 = params['lr3']
        lr4 = params['lr4']
        batch_size = params['batch_size']
        EPOCH = params['EPOCH']
        beta = params['beta']
        delta = params['delta']
        gamma = params['gamma']
        sparse_decay = params['sparse_decay']
        weight_decay_param = params['weight_decay_param']
        step_size = params['step_size']
        turns = params["turns"]

        data = self.init.copy()
        Y = torch.from_numpy(data['Y'])
        A = torch.from_numpy(data['A'])
        M_true = data['E']
        E_VCA_init = torch.from_numpy(data['E']).unsqueeze(2).unsqueeze(3).float()

        Y = torch.reshape(Y, (L, col, col))
        A = torch.reshape(A, (P, col, col))

        train_dataset = MyTrainData(img=Y.clone(), gt=A.clone(), transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        net = Model(P, L).to(device)
        self.get_init_dataofModel(data["src"], net, output_display=output_display, turns=turns)

        model_dict = net.state_dict()
        model_dict['decoder1.0.weight'] = E_VCA_init
        model_dict['decoder2.0.weight'] = E_VCA_init
        net.load_state_dict(model_dict)

        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=LR, weight_decay=weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9)
        apply_clamp_inst1 = NonZeroClipper()

        prograss = tqdm(range(EPOCH)) if output_display else range(EPOCH)
        losses = []
        # loss
        loss_func = nn.MSELoss(reduction='mean')
        criterionSparse = SparseKLloss().to(device)
        s2w = restraint.S2W(device)
        minvol = restraint.MinVolumn(L, P)
        sto = SumToOneLoss().to(device)
        for epoch in prograss:
            for i, (xx, y) in enumerate(train_loader):
                x = xx.to(device)
                abu_est1, re_result1, abu_est2, re_result2 = net(x)
                # 1. l21
                loss1 = rs.l21(re_result1, beta=lr1)
                # 2.空谱加权
                ab = abu_est1.squeeze(0).reshape(P, N)
                loss2 = s2w(ab.clone(), H, W, lambda_=lr2)
                # 3. 重建误差
                loss3 = loss_func(re_result1, x) * lr3
                loss4 = loss_func(x, re_result2) * lr4
                # 5
                loss4 = rs.aRMSE(torch.abs(re_result1 - x), torch.abs(re_result2 - x), beta=lr3)
                # 5
                # loss5 = rs.aRMSE(torch.abs(abu_est1 - y.to(device)), torch.abs(abu_est2 - y.to(device)), beta=lr4)
                # ab = abu_est2.squeeze(0).reshape(P, N)
                # loss4 = s2w(ab.clone(), H, W, lambda_=lr2)
                # loss4 = rs.aSAD(torch.abs(x - re_result1), torch.abs(x - re_result2), beta=lr3)
                # loss4 = sto(abu_est1,gamma_reg=1e-2)
                total_loss = sum([loss1, loss2, loss3, loss4])

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                optimizer.step()

                net.decoder1.apply(apply_clamp_inst1)
                net.decoder2.apply(apply_clamp_inst1)

                losses.append(total_loss.detach().cpu().numpy())

            # scheduler.step()

        net.eval()
        abu_est1, _, _, _ = net(Y.unsqueeze(0).to(device))

        abu_est1 = abu_est1 / (torch.sum(abu_est1, dim=1))
        abu_est1 = torch.reshape(abu_est1.squeeze(0), (P, col, col))
        abu_est1 = abu_est1.cpu().detach().numpy()

        A = abu_est1.reshape(P, N)
        # print(E_VCA_init.shape)
        E = extract_edm(y=data['Y'].copy(), a=A)

        return {
            'A': A,
            'E': E,
            'loss': losses
        }
