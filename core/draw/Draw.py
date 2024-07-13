import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距,width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


class Draw:
    def __init__(self, savepath=None):
        self.savepath = savepath

    def confirmSavePath(self, save, savepath):
        # 函数中的savepath为最高优先级
        if savepath:
            return True, savepath
        # 类中的savepath，排在函数中的savepath之后;是否保存看save的情况
        if save and self.savepath:
            return True, self.savepath
        # 如果不保存则返回
        return False, None



    def abundanceMap_AllInOne(self, abu, name="abundanceMap", title="abundanceMap", save=True, savepath=None):
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
            a = plt.imshow(abu[i - 1], cmap='jet', interpolation='none')
            # 颜色范围
            a.set_clim(vmin=0, vmax=1)
            # 关闭中轴线及刻度
            # plt.axis('off')
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
            cb.set_ticks([i / 10 for i in range(0, 10 + 1, 1)])
            cb.set_ticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"], fontsize=8)
            # cb.locator = MultipleLocator(0.1)
            # cb.formatter = FormatStrFormatter('%.1f')
            # cb.update_ticks()
        # 中央标题
        fig.suptitle(title)
        saveflag, savepath = self.confirmSavePath(save, savepath)
        if savepath:
            plt.savefig(savepath + f'/{name}.jpg')
        plt.show()

    def abundanceMap_AllInOne_T(self, abu, name="abundanceMap", title="abundanceMap", save=True, savepath=None):
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
            # plt.axis('off')
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
            plt.subplots_adjust(top=0.8, bottom=0.1, left=0.05, right=0.95, hspace=0.15, wspace=0.2)  # 自定义布局
            # plt.tight_layout()  # 自适应布局
            # 色度条
            cb = plt.colorbar(fraction=0.0457, pad=0.04)
            cb.set_ticks([i / 10 for i in range(0, 10 + 1, 1)])
            cb.set_ticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"], fontsize=8)
            # cb.locator = MultipleLocator(0.1)
            # cb.formatter = FormatStrFormatter('%.1f')
            # cb.update_ticks()
        # 中央标题
        fig.suptitle(title)
        saveflag, savepath = self.confirmSavePath(save, savepath)
        if saveflag:
            plt.savefig(savepath + f'/{name}.jpg')
        plt.show()

    def abundanceMap0(self, abu, name="abundaceMap", filetype="jpg", savepath=None, save=True, show=False):
        # note:画丰度图
        # abu-(L,H,W)
        P, H, W = abu.shape
        # 刻度朝向
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        for i in range(1, P + 1):
            plt.figure(figsize=(8, 6))
            plt.axis('off')
            # 图像
            a = plt.imshow(abu[i - 1].T, cmap='jet', interpolation='none')
            a.set_clim(vmin=0, vmax=1)
            # 色度条
            cb = plt.colorbar(fraction=0.1, pad=0.03)
            cb.set_ticks([i / 10 for i in range(0, 10 + 1, 1)])
            # 图像刻度
            plt.xlim(0, W)
            plt.ylim((H, 0))
            # plt.xticks(np.arange(0, H+1,10))
            # plt.yticks(np.arange(0,W+1,10))
            plt.tight_layout()  # 紧凑布局
            saveflag, savepath = self.confirmSavePath(save, savepath)
            if saveflag:
                plt.savefig(f'{savepath}/{name}-{i}.{filetype}')
            if show:
                plt.show()


if __name__ == '__main__':
    d = Draw("./")
