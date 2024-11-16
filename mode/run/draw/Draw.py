import core.draw as draw
import numpy as np


class Draw:
    def __init__(self, dtrue, dpred, savepath=None):
        self.dtrue = dtrue
        self.dpred = dpred
        self.savepath = savepath

    def __call__(self):
        data = self.dtrue
        data_pred = self.dpred
        savepath = self.savepath
        P, L, N, H, W = data['P'], data['L'], data['N'], data['H'], data['W']
        # 端元对比图
        if 'E' in data_pred.keys():
            if len(data_pred['E'].shape) == 3:
                epred = data_pred['E'][:, :, 1]
                if 'E_3d' in data:
                    etrue = data['E_3d'][:, :, 1]
                else:
                    etrue = data['E']
            else:
                epred = data_pred['E']
                etrue = data['E']
            draw.vs_endmembers_all(etrue, epred, savepath=savepath)
            # draw.vs_endmembers(etrue, epred, savepath=savepath)

        # 丰度对比图
        if 'A' in data_pred.keys():
            apred = data_pred['A'].reshape(P, H, W)
            draw.abundanceMap_all(apred, title="pred", name='pred', savepath=savepath)
            # draw.abundanceMap(apred, name='pred', savepath=savepath)

            atrue = data['A'].reshape(P, H, W)
            draw.abundanceMap_all(atrue, title="true", name='true', savepath=savepath)

            adiff = np.fabs(atrue - apred)
            # draw.abundanceMap_all(adiff, title="diff", name='diff', savepath=savepath)

        if 'loss' in data_pred.keys():
            draw.loss(data_pred['loss'], savepath=savepath)

        if 'loss_list' in data_pred.keys() and 'loss_name' in data_pred.keys():
            draw.loss_sub(losslist=data_pred['loss_list'], namelist=data_pred['loss_name'])
