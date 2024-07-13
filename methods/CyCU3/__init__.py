import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
import os
from tqdm import tqdm

from .model import Model
from .loss import SparseKLloss, SumToOneLoss, NonZeroClipper

from utils.extract import extract_edm
import utils.restraint as rs
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


class CyCU3:
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
            for id, elem in enumerate([0, 3, 6, 9]):
                net.encoder[elem].weight.data = torch.tensor(data[str(id)], device=device)

        else:
            if output_display:
                print("初始化模型...")
            for _ in range(turns):
                for i in [0, 3, 6, 9]:
                    # kaiming_normal_ and xavier_normal_
                    if dataset_name == 'Samson':
                        nn.init.kaiming_normal_(net.encoder[i].weight.data)
                    else:
                        nn.init.kaiming_normal_(net.encoder[i].weight.data)
                        # nn.init.xavier_normal(net.encoder[i].weight.data)

            data = {
                '0': net.encoder[0].weight.data.cpu().numpy(),
                '1': net.encoder[3].weight.data.cpu().numpy(),
                '2': net.encoder[6].weight.data.cpu().numpy(),
                '3': net.encoder[9].weight.data.cpu().numpy(),
            }
            # sio.savemat(search_file, data)

    def run(self, savepath=None, output_display=True, *args, **kwargs):

        # torch.autograd.set_detect_anomaly(True)
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
        lr5 = params['lr5']
        lr_en = params['lr_en']
        lr_de1 = params['lr_de1']
        batch_size = params['batch_size']
        EPOCH = params['EPOCH']
        beta = params['beta']
        gamma = params['gamma']
        kernel = params['kernel']
        wd = params['wd']
        step_size = params['step_size']
        turns = params["turns"]

        data = self.init.copy()
        Y = torch.from_numpy(data['Y']).float()
        A = torch.from_numpy(data['A'])
        M_true = data['E']
        E_VCA_init = torch.from_numpy(data['E']).unsqueeze(2).unsqueeze(3).float()

        Y = torch.reshape(Y, (L, col, col))
        A = torch.reshape(A, (P, col, col))

        train_dataset = MyTrainData(img=Y.clone(), gt=A.clone(), transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        net = Model(P, L, beta).to(device)
        self.get_init_dataofModel(data["src"][0], net, output_display=output_display, turns=turns)

        model_dict = net.state_dict()
        model_dict['decoder1.0.weight'] = E_VCA_init
        model_dict['decoder2.0.weight'] = E_VCA_init
        net.load_state_dict(model_dict)

        optimizer = torch.optim.Adam([
            {"params": net.encoder.parameters(), 'lr': lr_en},
            {"params": net.decoder1.parameters(), 'lr': lr_de1},
            {"params": net.decoder2.parameters(), 'lr': lr_de1},
        ], lr=LR, weight_decay=wd)
        # optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=LR, weight_decay=weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        apply_clamp_inst1 = NonZeroClipper()

        prograss = tqdm(range(EPOCH)) if output_display else range(EPOCH)
        losses = []
        # loss
        MSE = nn.MSELoss(reduction='mean')
        criterionSparse = SparseKLloss().to(device)
        # s2w = restraint.S2W(device)
        s2w = rs.S2WK(device, kernel=kernel)
        c1 = rs.custom1()
        # s2w = restraint.Spa2W(device, kernel=kernel)
        minvol = rs.MinVolumn(L, P)
        sto = SumToOneLoss().to(device)
        obj1 = []
        obj2 = []
        obj3 = []
        obj4 = []
        obj5 = []
        for epoch in prograss:
            for i, (xx, y) in enumerate(train_loader):
                x = xx.to(device)
                abu_est1, re_result1, abu_est2, re_result2 = net(x)
                # 1. 空谱加权
                aa = abu_est1.squeeze(0)  #
                ab = aa.reshape(P, N)
                loss1 = s2w.compute_spa_loss(ab.clone(), H, W, lambda_=lr1)
                # loss1 = s2w(ab.clone(), H, W, lambda_=lr1)
                # loss1 = s2w(ab.clone(), H, W, lambda_=lr1)
                # loss1 = s2w(ab.clone(), H, W, lambda_=lr1)

                # 2. l21
                # loss2 = c1(re_result1, beta=lr2)
                # loss3 = c1(re_result2, beta=lr3)
                loss2 = rs.norm_l21(re_result1, beta=lr2)
                # loss3 = rs.SAD(x, re_result1, beta=0)

                # 3. 重建误差
                loss4 = MSE(re_result1, x) * lr4
                loss5 = MSE(x, re_result2) * lr5

                # loss5 = rs.norm_tv(aa)

                # 4
                # loss4 = rs.RMSE(abu_est1,abu_est2, beta=1e-7)
                # loss4 = rs.RMSE(torch.abs(re_result1 - x), torch.abs(re_result2 - x), beta=lr3)
                # loss4 = rs.norm_tv(MSE(abu_est1,abu_est2), beta=1e-7)
                # loss5 = loss_func(abu_est1, abu_est2) * -lr5
                # loss5 = rs.aRMSE(torch.abs(abu_est1 - y.to(device)), torch.abs(abu_est2 - y.to(device)), beta=lr4)
                # ab = abu_est2.squeeze(0).reshape(P, N)
                # loss4 = s2w(ab.clone(), H, W, lambda_=lr2)
                # loss4 = rs.SAD(torch.abs(x - re_result1), torch.abs(x - re_result2), beta=lr3)
                # loss4 = rs.SAD(x, re_result1, beta=1e-10)
                # loss4 = sto(abu_est1,gamma_reg=1e-2)
                total_loss = sum([loss1, loss2, loss4, loss5])

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                optimizer.step()

                net.decoder1.apply(apply_clamp_inst1)
                net.decoder2.apply(apply_clamp_inst1)

                obj1.append(loss1.detach().cpu().numpy())
                obj2.append(loss2.detach().cpu().numpy())
                # obj3.append(loss3.detach().cpu().numpy())
                obj4.append(loss4.detach().cpu().numpy())
                obj5.append(loss5.detach().cpu().numpy())

                losses.append(total_loss.detach().cpu().numpy())

            scheduler.step()

        net.eval()
        abu_est1, re_result1, _, _ = net(Y.unsqueeze(0).to(device))

        abu_est1 = abu_est1 / (torch.sum(abu_est1, dim=1))
        abu_est1 = torch.reshape(abu_est1.squeeze(0), (P, col, col))
        abu_est1 = abu_est1.cpu().detach().numpy()

        A = abu_est1.reshape(P, N)

        # E = E_VCA_init.cpu().squeeze(-1).squeeze(-1).detach().numpy() # [156, 3, 1, 1])

        E = extract_edm(y=data['Y'].copy(), a=A)
        # E = E_VCA_init.cpu().squeeze(-1).squeeze(-1).detach().numpy()
        # E = net.decoder2[0].weight.data.squeeze(-1).squeeze(-1).cpu().numpy()
        # Y = re_result1.cpu().detach().numpy()
        # Y = Y.squeeze(0).reshape(L, N)
        # E = extract_edm(y=Y, a=A)

        return {
            'A': A,
            'E': E,
            'loss': losses,
            'loss_list': [obj1, obj2, obj3, obj4, obj5],
            'loss_name': ["spa", "l21", "sad", "mse", "mse"]
        }
