import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from core.load import loadhsi
from core import Analyzer
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
from typing import List
from custom_types import ExInfo, DatasetsEnum, HsiDataset


def one_pic_abu(ds: DatasetsEnum, items: List[Analyzer], names: List[str], show: bool = False, todiff: bool = False,
                t: bool = False, withGT: bool = False):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'  # 字体
    font_prop = FontProperties(family='Times New Roman', style='normal', weight='bold', size=14)  # 加粗

    dataset: HsiDataset = loadhsi(ds.name)
    P: int = dataset.bandNum
    methodNums: int = len(items)
    if withGT:
        methodNums += 1
    fig, axs = plt.subplots(P, methodNums, figsize=(12, 9))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, item in enumerate(items):
        a_3d = item.get_differenceMap(True) if todiff else item.get_abundanceMap(True)

        for j in range(0, P):
            # 方法名称
            if j == 0:
                axs[j, i].title.set_text(m)  # 给左上角的子图添加标题
                axs[j, i].title.set_fontproperties(font_prop)
            # 端元名称
            if i == 0:
                if nlist:
                    axs[j, 0].set_ylabel(nlist[j]).set_fontproperties(font_prop)

            # plt.subplot(P, methodNums, i + methodNums * j)
            norm = Normalize(vmin=0, vmax=0.4)  # 这里的vmin和vmax根据您的数据自行设置
            im = a_3d[j] if not t else a_3d[j].T
            axs[j, i].imshow(im, cmap='jet', interpolation='none', norm=norm)
            # if i == methodNums - 1 and j == P - 1:  # 仅在最后一个位置加入colorbar
            #     cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.6])
            #     fig.colorbar(a, cax=cbar_ax)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
        # plt.tight_layout() # 紧凑布局
        # colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.68])
    colorbar = fig.colorbar(axs[0, 0].images[-1], cax=cbar_ax)
    # colorbar_labels
    # ticks = [[i / 10 for i in range(0, 10 + 1, 1)]]
    # colorbar.set_ticks([i / 100 for i in range(0, 30 + 1, 10)])
    # colorbar.set_ticklabels([str(i / 100.0) for i in range(0, 30 + 1, 10)], fontsize=14)
    # colorbar.set_label('Abundance', rotation=270, labelpad=20)
    if show:
        plt.show()
