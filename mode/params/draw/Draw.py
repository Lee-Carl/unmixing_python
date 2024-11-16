import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np


class Draw:
    def __init__(self, out_path):
        self.out_path = out_path
        self.savepath = out_path + "/assets"

    @staticmethod
    def custom_loss(data, dataname, out_path, lam, iter, show=True):
        plt.figure(figsize=(8, 6))
        index = np.argmin(data)
        plt.title(dataname)
        print(f'{dataname}最小值（{np.min(data)}）时：lambda={lam[index]}')
        plt.plot(iter, data)
        for i, label in enumerate(lam):
            plt.text(iter[i] + 0.1, data[i], f'({lam[i]}, {data[i]:.4f})', ha='left', va='bottom', fontsize=6)
        plt.savefig(out_path + f"/{dataname}.png")
        if show:
            plt.show()

    def __call__(self):
        out_path = self.out_path
        tg = sio.loadmat(os.path.join(out_path, "results.mat"))
        lam = tg['lam'][0]
        asad_m = tg['aSAD_E'][0]
        armse_a = tg['aRMSE_A'][0]
        armse_y = tg['aRMSE_Y'][0]
        asad_y = tg['aSAD_Y'][0]
        iter = tg['iter'][0]

        # self.custom_loss(data=armse_y + asad_y, dataname=f"sumy", out_path=self.savepath, lam=lam, iter=iter,show=False)
        self.custom_loss(data=armse_a + asad_m, dataname=f"sum", out_path=self.savepath, lam=lam, iter=iter)
