import matplotlib.pyplot as plt


def loss(loss, name='loss', savepath=None):
    # note:损失函数图
    plt.figure()
    plt.loglog(loss)
    if name:
        plt.title(name)
    if savepath:
        plt.savefig(savepath + f'/{name}.png')  # note:fig1
    plt.show()
