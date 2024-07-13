import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def loss_sub(losslist, namelist, savepath=None):
    # note:损失函数图
    lens = len(losslist)
    plt.figure(figsize=(lens * 4, 6))
    for i, loss in enumerate(losslist):
        plt.subplot(1, lens, i + 1)
        plt.loglog(loss)
        plt.title(namelist[i])
    if savepath:
        plt.savefig(savepath + f'/losslist.png')  # note:fig1
    plt.show()
