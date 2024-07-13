import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from sklearn.decomposition import PCA


def PCA_map(data, savepath=None):
    # 一般用来分析混元
    pca = PCA(2)
    y2d = pca.fit_transform(data.T)

    if savepath:
        plt.savefig(savepath + 'pca.png')  # note:fig3
    plt.show()


def scatter_map(y, em_hat, p, savepath=None):
    # 一般用来分析预测的端元
    plt.figure()
    pca = PCA(2)
    y2d = pca.fit_transform(y.T)
    plt.scatter(y2d[:, 0], y2d[:, 1], 5, label='Pixel data', color='grey')
    for i in range(p):
        em2d = pca.transform(np.squeeze(em_hat[:, i, :]))
        plt.scatter(em2d[:, 0], em2d[:, 1], 5, label='EM #' + str(i + 1))

    plt.legend()
    plt.title('Scatter plot of mixed pixels and EMs')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    if savepath:
        plt.savefig(savepath + 'EMs.png')  # note:fig4
    plt.show()


def spectralMap(data, savepath=None):
    plt.plot(data)
    plt.grid('on')
    plt.legend(['pixel 1', 'pixel 2', 'pixel 3', 'pixel 4'])
    if savepath:
        plt.savefig(savepath + 'pixel.png')  # note:fig6
    plt.show()
