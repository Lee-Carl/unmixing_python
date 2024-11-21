import matplotlib.pyplot as plt


def abundanceMap(abu, name: str = "abundaceMap", filetype: str = "tif", savepath: str = None, show: bool = False):
    # note:画丰度图
    # abu-(L,H,W)
    P, H, W = abu.shape
    # 刻度朝向
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    for i in range(1, P + 1):
        fig = plt.figure(figsize=(8, 6))
        # fig = plt.figure(figsize=(16, 12))
        ax = plt.axes()
        # 关闭刻度
        plt.axis("off")
        # 图像
        a = ax.imshow(abu[i - 1], cmap='jet', interpolation='none')
        a.set_clim(vmin=0, vmax=0.18)
        # cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        # cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        # 色度条
        cb = plt.colorbar(a)
        # 1
        cb.set_ticks([i / 100.0 for i in range(0, 18 + 1, 6)])
        cb.set_ticklabels([str(i / 100.0) for i in range(0, 18 + 1, 6)], fontsize=24)
        # 2
        # cb.set_ticks([i / 10 for i in range(0, 10 + 1, 1)])
        # cb.set_ticklabels([str(i / 10.0) for i in range(0, 10 + 1, 1)], fontsize=24)
        # cb.set_ticks([0, 1])
        # 图像刻度
        # plt.xlim(0, W)
        # plt.ylim((H, 0))
        # plt.xticks(np.arange(0,sqrtN+1,10))
        # plt.yticks(np.arange(0,sqrtN+1,10))
        plt.tight_layout()  # 紧凑布局
        if savepath:
            # plt.savefig(f'{savepath}/{name}-{i}.{filetype}', bbox_inches='tight')
            plt.savefig(f'{savepath}/{name}-{i}.{filetype}')
        if show:
            plt.show()
