import matplotlib.pyplot as plt
from core.load import loadhsi
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
from core.init import Norm
from custom_types import DatasetsEnum, HsiDataset, ExInfo
from core import Analyzer
from typing import List


def one_pic_edm(ds: DatasetsEnum, items: List[ExInfo], nameList: List[str], show: bool = False):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'  # 字体
    font_prop = FontProperties(family='Times New Roman', style='normal', weight='bold', size=14)  # 加粗

    nums = len(items)  # 定义方法数
    dataset: HsiDataset = loadhsi(ds.name)
    P: int = dataset.edmNum
    L: int = dataset.bandNum
    fig, axs = plt.subplots(P, nums, figsize=(16, 9))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i, item in enumerate(items):
        # 真实端元图
        result = Analyzer(dataset=item.dataset, method=item.method, path=item.dst)
        edm: np.ndarray = result.get_endmembers()
        if len(edm.shape) == 3:
            edm = edm[:, :, 0]
        for j in range(0, P):
            if j == 0:
                # 第一行上面的方法名称
                axs[j, i].title.set_text(result.method.name)  # 给左上角的子图添加标题
                axs[j, i].title.set_fontproperties(font_prop)
            if i == 0:
                # 在轴外部添加文本
                # axs[j, 0].text(-1.0, 0.8, 'EM #1', fontsize=14, rotation=90, va='top', ha='center')
                # 第一列左侧侧的端元名称
                if nameList:
                    # axs[j, 0].set_ylabel(nlist[j]).set_fontproperties(font_prop)
                    axs[j, 0].set_ylabel(nameList[j])
            axs[j, i].plot(dataset.A[:, j], color="#FF0000", linestyle='-')
            axs[j, i].plot(edm[:, j], color="#0000FF", linestyle='-.')
            # 横轴
            axs[j, i].set_xlim(0, L)
            axs[j, i].set_xticks([_ for _ in range(0, L, 50)])
            axs[j, i].set_xlabel("Bands")
            # 纵轴
            axs[j, i].set_ylim(0, 1)
            axs[j, i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # 设置y轴的刻度标签（不包括0.5，如果您不需要这个值）
            axs[j, i].set_ylabel("Reflectance")
    # plt.tight_layout()  # 紧凑布局
    # 添加图例
    axs[P - 1, nums // 2].legend(bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True,
                                 shadow=True)
    if show:
        plt.show()
