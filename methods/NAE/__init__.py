from typing import Any, Dict

from ..Base import Base
from .model import Model
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader
from .pretrain_weight import pretrain_weight, pretrain_dec_nonlipart


class NAE(Base):
    def __init__(self, params, init):
        super().__init__(params, init)

    def run(self, savepath=None, output_display=True, *args: Any, **kwargs: Any) -> Dict:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        init = self.init.copy()
        params = self.params

        EPOCH = params['EPOCH']
        BATCH_SIZE = params['BATCH_SIZE']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        lr1 = params['lr1']
        lr2 = params['lr2']
        num_repeat = params['num_repeat']
        lambda1 = params['lambda1']

        num_endmember = init['P']
        L = init['L']

        W_init = torch.from_numpy(init['E']).float()
        Y = init['Y'].T
        hsi = torch.from_numpy(Y)

        model_name = 'mlaem_model'
        for iter in range(1, num_repeat + 1):
            model = Model(L, num_endmember)
            model.decoder_linearpart[0].weight.data = W_init

            dec_nonlipart = pretrain_dec_nonlipart(Y.copy(), pretrain_lr=params['pretrain_lr'],
                                                   pretrain_EPOCH=params['pretrain_EPOCH'])
            # if savepath:
            #     torch.save(dec_nonlipart, savepath + 'pretrain_decoder_nonlinear.pth')
            model.decoder_nonlinearpart.load_state_dict(dec_nonlipart)

            model = model.to(device)
            criterion = MSELoss()
            losses = []
            # ------------------- Fine Tune ---------------------------------------------
            if model_name == 'mlaem_model':
                params1 = map(id, model.decoder_linearpart.parameters())
                params2 = map(id, model.decoder_nonlinearpart.parameters())
                ignored_params = list(set(params1).union(set(params2)))
                base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
                optimizer = Adam([
                    {'params': base_params},
                    {'params': model.decoder_linearpart.parameters(), 'lr': lr1},
                    {'params': model.decoder_nonlinearpart.parameters(), 'lr': lr2},
                ], lr=learning_rate, weight_decay=weight_decay)
            else:
                ignored_params = list(map(id, model.decoder.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
                optimizer = Adam([
                    {'params': base_params},
                    {'params': model.decoder.parameters(), 'lr': 1e-4}
                ], lr=learning_rate, weight_decay=1e-5)

            vector_all = []
            code_onehot = torch.eye(num_endmember).to(device)
            data_loader = DataLoader(hsi, batch_size=BATCH_SIZE, shuffle=False)
            for epoch in range(1, EPOCH + 1):
                for data in data_loader:
                    pixel = data.to(device)
                    # ===================forward====================
                    output, vector = model(pixel)
                    loss_reconstruction = criterion(output, pixel)
                    loss = loss_reconstruction * lambda1
                    losses.append(loss.cpu().detach().numpy())
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # ===================log========================
                    if epoch == EPOCH - 1:
                        vector_temp = vector.cpu().data
                        vector_temp = vector_temp.numpy()
                        vector_all = np.append(vector_all, vector_temp)

            if savepath:
                torch.save(model.state_dict(), savepath + 'sim_autoencoder.pth')

            if iter == num_repeat:
                vector_all = vector_all.reshape(-1, num_endmember).T
                endmember = model.get_endmember(code_onehot)
                endmember = endmember.cpu().data
                endmember = endmember.numpy().T

                return {
                    'E': endmember,
                    'A': vector_all,
                    'loss': losses
                }
