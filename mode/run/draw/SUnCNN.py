from .BaseDraw import BaseDraw
from core.draw import Draw
import matplotlib.pyplot as plt
import numpy as np


class SUnCNN(BaseDraw):
    def __init__(self, dtrue, dpred, savepath=None):
        super().__init__(dtrue, dpred, savepath)

    def abundancesMap(self, abu, desc):
        # note:画丰度图
        # abu-(P,H,W)
        P, H, W = abu.shape
        # 刻度朝向
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        st = 203
        ed = 204
        for i in range(st, ed + 1):  # 限制画图的数量
            plt.figure(figsize=(8, 6))
            # 画图
            a = plt.imshow(abu[i - 1], cmap='jet', interpolation='none')
            # 颜色范围
            a.set_clim(vmin=0, vmax=1)
            # 布局
            # plt.tight_layout()  # 自适应布局
            # 色度条
            cb = plt.colorbar(fraction=0.0457, pad=0.04)
            cb.set_ticks([i / 10 for i in range(0, 10 + 1, 1)])
            cb.set_ticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"], fontsize=8)
            # 中央标题
            plt.title(desc)
            if self.savepath:
                plt.savefig(f'{self.savepath}/{desc}-{i}.jpg')
        plt.show()

    def abundancesMap_all(self, abu, desc):
        # note:画丰度图
        # abu-(P,H,W)
        P, H, W = abu.shape
        # 刻度朝向
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.figure(figsize=(16, 4))
        st = 201
        ed = 204
        n = ed - st + 1
        x = st - 1
        for i in range(st, ed + 1):  # 限制画图的数量
            plt.subplot(1, n, i - x)
            # 画图
            a = plt.imshow(abu[i - 1], cmap='jet', interpolation='none')
            # 颜色范围
            a.set_clim(vmin=0, vmax=1)
            # 布局
            # plt.tight_layout()  # 自适应布局
            # 色度条
            cb = plt.colorbar(fraction=0.0457, pad=0.04)
            cb.set_ticks([i / 10 for i in range(0, 10 + 1, 1)])
            cb.set_ticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"], fontsize=8)
            # 中央标题
            plt.title(desc)
            if self.savepath:
                plt.savefig(f'{self.savepath}/{desc}-{i}.jpg')
        plt.show()

    def __call__(self):
        dtrue = self.dtrue
        dpred = self.dpred

        P, H, W = dtrue['P'], dtrue['H'], dtrue['W']
        draw = Draw()
        # 丰度对比图
        if 'A' in dpred.keys():
            apred = dpred['A'].reshape(P, H, W)
            self.abundancesMap_all(apred, "pred")

            atrue = dpred['A'].reshape(P, H, W)
            self.abundancesMap_all(atrue, "true")

        if 'loss' in dpred.keys():
            draw.loss(dpred['loss'], savepath=self.savepath)
