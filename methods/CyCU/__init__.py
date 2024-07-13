import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
import os
from tqdm import tqdm
from .model import Model
from .restraint import SparseKLloss, SumToOneLoss, NonZeroClipper
from utils.extract import extract_edm


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


class CyCU:
    def __init__(self, params, init):
        self.params = params
        self.init = init

    def run(self, savepath=None, output_display=True, *args, **kwargs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Load Data
        data = self.init.copy()
        dataset = data["name"]
        P = data['P']
        L = data['L']
        N = data['N']
        col = data['W']

        params = self.params
        LR = params['LR']
        batch_size = params['batch_size']
        EPOCH = params['EPOCH']
        beta = params['beta']
        delta = params['delta']
        gamma = params['gamma']
        sparse_decay = params['sparse_decay']
        weight_decay_param = params['weight_decay_param']

        data = self.init.copy()
        Y = torch.from_numpy(data['Y'])
        A = torch.from_numpy(data['A'])
        M_true = data['E']
        E_VCA_init = torch.from_numpy(data['E']).unsqueeze(2).unsqueeze(3).float()

        Y = torch.reshape(Y, (L, col, col))
        A = torch.reshape(A, (P, col, col))

        def weights_init(m):
            nn.init.kaiming_normal_(net.encoder[0].weight.data)
            nn.init.kaiming_normal_(net.encoder[4].weight.data)
            nn.init.kaiming_normal_(net.encoder[7].weight.data)

        train_dataset = MyTrainData(img=Y.clone(), gt=A.clone(), transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        net = Model(P, L).to(device)
        net.apply(weights_init)

        criterionSumToOne = SumToOneLoss().to(device)
        criterionSparse = SparseKLloss().to(device)

        model_dict = net.state_dict()
        model_dict['decoder1.0.weight'] = E_VCA_init
        model_dict['decoder2.0.weight'] = E_VCA_init
        net.load_state_dict(model_dict)

        # loss_func = nn.MSELoss(size_average=True, reduce=True, reduction='mean')
        loss_func = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        apply_clamp_inst1 = NonZeroClipper()
        # progress = tqdm(range(EPOCH))
        losses = []
        prograss = tqdm(range(EPOCH)) if output_display else range(EPOCH)
        for epoch in prograss:
            for i, (xx, y) in enumerate(train_loader):
                x = xx.to(device)
                abu_est1, re_result1, abu_est2, re_result2 = net(x)
                loss_sumtoone = criterionSumToOne(abu_est1, gamma_reg=gamma) + \
                                criterionSumToOne(abu_est2, gamma_reg=gamma)

                loss_sparse = criterionSparse(abu_est1, decay=sparse_decay) + \
                              criterionSparse(abu_est2, decay=sparse_decay)

                loss_re = beta * loss_func(re_result1, x) + (1 - beta) * loss_func(x, re_result2)
                # loss_re = loss_func(re_result1, x) + loss_func(x, re_result2)

                loss_abu = delta * loss_func(abu_est1, abu_est2)

                total_loss = loss_re + loss_abu + loss_sumtoone + loss_sparse

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                optimizer.step()

                net.decoder1.apply(apply_clamp_inst1)
                net.decoder2.apply(apply_clamp_inst1)

                losses.append(total_loss.detach().cpu().numpy())
                # if epoch % 10 == 0:
                #     print('Epoch:', epoch, '| i:', i, '| train loss: %.4f' % total_loss.data.numpy(),
                #           '| abu loss: %.4f' % loss_abu.data.numpy(),
                #           '| sumtoone loss: %.4f' % loss_sumtoone.data.numpy(),
                #           '| re loss: %.4f' % loss_re.data.numpy())

            scheduler.step()

        net.eval()
        abu_est1, _, _, _ = net(Y.unsqueeze(0).to(device))

        abu_est1 = abu_est1 / (torch.sum(abu_est1, dim=1))
        abu_est1 = torch.reshape(abu_est1.squeeze(0), (P, col, col))
        abu_est1 = abu_est1.cpu().detach().numpy()

        A = abu_est1.reshape(P, N)

        E = net.decoder2[0].weight.data.squeeze(-1).squeeze(-1)
        # E = E.detach().cpu().numpy()
        # E = extract_edm(y=data['Y'].copy(), a=A)
        # print(E.shape)

        return {
            'A': A,
            'E': E,
            'loss': losses
        }
