import matplotlib.pyplot as plt
import numpy as np


def abundanceMap_all(abu, name="abundanceMap", title="abundanceMap", show=True,savepath=None):
    # note:画丰度图
    # abu-(P,H,W)
    P, H, W = abu.shape
    fig = plt.figure(figsize=(16, 4))
    # 刻度朝向
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    for i in range(1, P + 1):
        # 窗口
        ax = plt.subplot(1, P, i)
        # 画图
        a = plt.imshow(abu[i - 1].T, cmap='jet', interpolation='none')
        # 颜色范围
        a.set_clim(vmin=0, vmax=1)
        # 关闭中轴线及刻度
        plt.axis('off')
        # 刻度是否显示
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.tick_params(bottom=False, left=False, top=False, right=False)
        # 刻度显示大小
        # plt.xlim(0, sqrtN)
        # plt.ylim(sqrtN, 0)
        # 精调刻度
        plt.xticks(np.arange(0, W + 1, 10), fontsize=8)
        plt.yticks(np.arange(0, H + 1, 10), fontsize=8)
        # 布局
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.2)  # 自定义布局
        # plt.tight_layout()  # 自适应布局
        # 色度条
        cb = plt.colorbar(fraction=0.0457, pad=0.04)
        # cb.set_ticks([i / 10 for i in range(0, 10 + 1, 1)])
        # cb.set_ticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"], fontsize=8)
        cb.set_ticks([i / 10 for i in range(0, 10 + 1, 2)])
        cb.set_ticklabels([str(i / 10.0) for i in range(0, 10 + 1, 2)], fontsize=8)
        # cb.locator = MultipleLocator(0.1)
        # cb.formatter = FormatStrFormatter('%.1f')
        # cb.update_ticks()
    # 中央标题
    fig.suptitle(title)
    if savepath:
        plt.savefig(savepath + f'/{name}.tif')
    if show:
        plt.show()
