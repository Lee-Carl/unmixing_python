import matplotlib.pyplot as plt


def endmembers_Img1(edm, name="endmembers", savepath=None):
    # note:端元图
    #  前提是，端元必须是L * P
    plt.figure(figsize=(8, 6))
    plt.plot(edm)
    # plt.legend()  # 添加图例
    plt.show()
    if savepath:
        plt.savefig(savepath + f"/{name}.jpg")


def endmembers_Img(edm, name="endmembers", savepath=None):
    # note:端元图
    #  前提是，端元必须是L * P
    plt.figure(figsize=(8, 6))
    # label = ['Tree', 'Water', 'Soil', 'Road'] # jasperridge
    label = ['Soil', 'Tree', 'Water']  # samson
    for i in range(edm.shape[1]):
        plt.plot(edm[:, i], label=f'EM #{i}')
    plt.legend()  # 添加图例
    # plt.legend(loc='upper right')  # 将图例放置在图像右侧
    plt.show()
    if savepath:
        plt.savefig(savepath + f"/{name}.jpg")


