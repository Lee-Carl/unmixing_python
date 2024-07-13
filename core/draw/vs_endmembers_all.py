import matplotlib.pyplot as plt


def vs_endmembers_all(edm1, edm2, name="vs_endmembers", savepath=None):
    # note:端元对比图；前提是，端元必须是L * P
    P = edm1.shape[1]
    plt.figure(figsize=(20, 6))
    for i in range(P):
        plt.subplot(1, P, i + 1)
        """
            color: 标注颜色
            label: 图例名称
        """
        plt.plot(edm1[:, i], color="#FF0000", label="true")
        plt.plot(edm2[:, i], color="#0000FF", label="pred")
        plt.legend()  # 添加图例

    if savepath:
        plt.savefig(savepath + f"/{name}_all.jpg")
    plt.show()
